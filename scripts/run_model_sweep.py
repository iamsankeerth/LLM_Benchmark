"""Sweep orchestrator: one thin coordinator over the per-model pipeline.

Owns the 14-config lifecycle: disk check -> pull -> show/pin/derive ->
eligibility -> preflight -> smoke -> warmups/baseline -> validate ->
summaries -> VERIFY -> unload -> delete (gated) -> disk verify -> next.

Deletion is forbidden until persistence verification succeeds. State in
results/checkpoints/ gives resume across restarts; completed models are
never re-downloaded. Benchmark errors fail closed by default; ineligible
models record an artifact and auto-continue by default.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sqlite3
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml

from analysis.comparison import performance_snapshot
from analysis.failure_modes import TAXONOMY_VERSION, extract_failed_rows
from analysis.reliability import load_spec_statuses
from inference.adapters import (
    get_model_config,
    load_overlays_from_dir,
)
from inference.derive_adapter import (
    ManualPinRequired,
    OverlayExistsError,
    derive_adapter,
    verify_overlay_matches_live,
)
from inference.eligibility import check_eligibility
from inference.ollama_client import (
    GenerationRequest,
    GenerationResult,
    OllamaClientError,
    fetch_ps,
    fetch_show,
    generate,
)
from scripts.check_registry_readiness import load_registry
from scripts.preflight_context import main as preflight_main
from scripts.run_benchmark import run_experiment
from scripts.smoke_inference import main as smoke_main
from scripts.summarize_baseline import main as summarize_main
from storage.db import connect
from storage.execution import derive_execution_id
from storage.sweep import (
    BENCHMARK_ERROR,
    BENCHMARKING,
    COMPLETE,
    COMPLETE_INELIGIBLE,
    DERIVING,
    DOWNLOAD_FAILED,
    DOWNLOADING,
    MANUAL_PIN_REQUIRED as STATE_MANUAL_PIN,
    PREFLIGHT_FAILED,
    PREFLIGHTING,
    SMOKING,
    VERIFY_FAILED,
    VERIFYING,
    disk_free_bytes,
    disk_reclaimed_ok,
    guard_deletion,
    load_sweep_state,
    new_sweep_state,
    ollama_model_present,
    ollama_pull,
    ollama_remove,
    ollama_stop,
    save_sweep_state,
    set_lifecycle,
    verify_execution_persisted,
    wait_until_unloaded,
)

_KNOWN_WINDOWS_BIN = (
    'C:/Users/lenovo/AppData/Local/Programs/Ollama/ollama.exe'
)


def resolve_ollama_bin(explicit: str | None) -> str:
    if explicit:
        return explicit
    env = os.environ.get('OLLAMA_BIN')
    if env:
        return env
    found = shutil.which('ollama')
    if found:
        return found
    return _KNOWN_WINDOWS_BIN


def load_ps_digest(base_url: str, identifier: str) -> tuple[str | None, int | None]:
    """Digest + size of the loaded model from /api/ps (None when absent)."""
    try:
        doc = fetch_ps(base_url)
    except OllamaClientError:
        return None, None
    entries = doc.get('models', [])
    if not isinstance(entries, list):
        return None, None
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        if entry.get('model') == identifier or entry.get('name') == identifier:
            digest = entry.get('digest')
            size = entry.get('size')
            return (
                str(digest) if isinstance(digest, str) else None,
                int(size) if isinstance(size, int) else None,
            )
    return None, None


def write_failure_modes_artifact(
    db_path: str, execution_id: str, statuses: dict[str, str], out_path: Path
) -> None:
    """Deterministic auto-mode decomposition (no human review layer)."""
    conn = connect(db_path)
    try:
        rows = extract_failed_rows(conn, execution_id, statuses)
    finally:
        conn.close()
    det = [r for r in rows if r.task_status == 'READY_DETERMINISTIC']
    judge = [r for r in rows if r.task_status == 'READY_JUDGE']
    det_modes: dict[str, int] = {}
    for row in det:
        det_modes[row.failure_mode] = det_modes.get(row.failure_mode, 0) + 1
    document = {
        'execution_id': execution_id,
        'taxonomy_version': TAXONOMY_VERSION,
        'scope_note': 'auto modes only; substantive review is a separate process',
        'deterministic_fail_rows': len(det),
        'judge_precheck_fail_rows': len(judge),
        'mode_counts_deterministic': det_modes,
        'total_fail_rows': len(rows),
    }
    out_path.write_text(json.dumps(document, indent=2) + '\n', encoding='utf-8')


def write_performance_artifact(
    db_path: str, execution_id: str, out_path: Path
) -> None:
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            'SELECT decode_tok_s, ttft_ms, eval_count, vram_peak_mib FROM runs'
            ' WHERE experiment_id=? AND is_warmup=0',
            (execution_id,),
        ).fetchall()
    finally:
        conn.close()
    snapshot = performance_snapshot(
        [r[0] for r in rows], [r[1] for r in rows],
        [r[2] for r in rows], [r[3] for r in rows],
    )
    snapshot['execution_id'] = execution_id
    out_path.write_text(json.dumps(snapshot, indent=2) + '\n', encoding='utf-8')


def run_one_model(args: argparse.Namespace, entry: dict[str, Any]) -> str:
    """Execute the full lifecycle for one registry entry. Returns outcome."""
    root = Path(__file__).resolve().parent.parent
    model_config_id = str(entry['model_config_id'])
    identifier = str(entry['ollama_identifier'])
    state_path = Path(args.state_file)
    state = load_sweep_state(str(state_path))
    assert state is not None
    base_url: str = args.base_url
    ollama_bin: str = args.ollama_bin
    db_path = str(Path(args.db_dir) / f'{derive_execution_id(args.spec, model_config_id)}.db')
    execution_id = derive_execution_id(args.spec, model_config_id)

    def checkpoint(lifecycle: str, detail: str = '') -> None:
        set_lifecycle(state, model_config_id, lifecycle, detail)
        save_sweep_state(str(state_path), state)
        print(f'[{model_config_id}] {lifecycle} {detail}'.rstrip(), flush=True)

    # --- disk check + pull ---
    free_before = disk_free_bytes(args.db_dir)
    checkpoint(DOWNLOADING, f'free={free_before / 1024**3:.1f}GiB')
    ok, detail = ollama_pull(ollama_bin, identifier)
    if not ok:
        checkpoint(DOWNLOAD_FAILED, detail)
        return DOWNLOAD_FAILED

    # --- show + load probe (digest before any pin decision) ---
    checkpoint(DERIVING)
    show_doc = fetch_show(base_url, identifier)
    try:
        generate(
            base_url,
            GenerationRequest(
                model=identifier, prompt='OK', temperature=0.0,
                num_ctx=512, num_predict=4, raw=False,
            ),
            timeout_s=300.0,
        )
    except OllamaClientError as exc:
        checkpoint(BENCHMARK_ERROR, f'load probe failed: {exc}')
        return BENCHMARK_ERROR
    observed_digest, _ = load_ps_digest(base_url, identifier)
    if observed_digest is None:
        checkpoint(BENCHMARK_ERROR, 'model absent from /api/ps after load probe')
        return BENCHMARK_ERROR
    overlay_path = root / 'configs/adapters' / f'{model_config_id}.json'
    try:
        config = get_model_config(model_config_id)
        adapter_source = 'coded-or-overlay'
    except KeyError:
        config = None
        adapter_source = None
    if config is None:
        def trial_generate(request: GenerationRequest) -> GenerationResult:
            return generate(args.base_url, request, timeout_s=120.0)

        try:
            config = derive_adapter(
                registry_entry=entry, show_doc=show_doc,
                observed_digest=observed_digest, trial_generate=trial_generate,
                overlay_path=overlay_path,
            )
            adapter_source = 'derived'
        except OverlayExistsError:
            load_overlays_from_dir(root / 'configs/adapters')
            config = get_model_config(model_config_id)
            adapter_source = 'overlay-reload'
        except ManualPinRequired as exc:
            checkpoint(STATE_MANUAL_PIN, str(exc))
            return STATE_MANUAL_PIN
    else:
        # Create-once: live metadata must match the pinned overlay/config.
        # Digest binds here (post-probe); stops/template drift fails closed.
        mismatches = verify_overlay_matches_live(config, show_doc, observed_digest)
        if mismatches:
            checkpoint('ADAPTER_MISMATCH', '; '.join(
                f'{m.field}: {m.pinned} != {m.observed}' for m in mismatches))
            return 'ADAPTER_MISMATCH'
    print(f'[{model_config_id}] adapter: {adapter_source} '
          f'mode={config.mode}', flush=True)

    # --- eligibility ---
    eligibility = check_eligibility(
        base_url, identifier, expected_digest=config.ollama_model_digest
    )
    print(f'[{model_config_id}] eligibility: {eligibility.status}', flush=True)
    if not eligibility.eligible:
        artifact = root / 'results/summaries' / f'ineligible-{model_config_id}.json'
        artifact.write_text(json.dumps({
            'execution_id': execution_id, 'model_config_id': model_config_id,
            'eligibility_status': eligibility.status,
            'gpu_residency_ratio': eligibility.gpu_residency_ratio,
            'evidence': eligibility.eligibility_evidence_json,
        }, indent=2) + '\n', encoding='utf-8')
        ollama_stop(ollama_bin, identifier)
        wait_until_unloaded(
            lambda: load_ps_digest(base_url, identifier)[0] is None,
            timeout_s=180.0,
        )
        ollama_remove(ollama_bin, identifier)
        checkpoint(COMPLETE_INELIGIBLE, eligibility.status)
        return COMPLETE_INELIGIBLE

    # --- preflight ---
    checkpoint(PREFLIGHTING)
    preflight_code = preflight_main([
        '--model', model_config_id, '--num-predict', '2048',
        '--base-url', base_url,
        '--out', str(root / 'results/summaries' / f'prompt-tokens-v2-{model_config_id}.json'),
    ])
    if preflight_code != 0:
        checkpoint(PREFLIGHT_FAILED, f'exit={preflight_code}')
        return PREFLIGHT_FAILED

    # --- smoke ---
    checkpoint(SMOKING)
    smoke_db = str(Path(args.db_dir) / f'smoke-v2__{model_config_id}.db')
    smoke_code = smoke_main([
        '--db', smoke_db, '--model', model_config_id,
        '--base-url', base_url, '--config', 'smoke-v2.yaml',
        '--num-predict', '2048',
    ])
    if smoke_code != 0:
        checkpoint(BENCHMARK_ERROR, 'smoke failed')
        return BENCHMARK_ERROR

    # --- baseline (existing resume-safe runner) ---
    checkpoint(BENCHMARKING)
    run_ns = argparse.Namespace(
        config=str(root / 'configs' / f'{args.spec}.yaml'),
        model=model_config_id, db=db_path, base_url=base_url,
        resume=True, execution_id=None, num_predict=None,
        timeout_s=600.0, max_tasks=None,
    )
    bench_code = run_experiment(run_ns)
    if bench_code != 0:
        checkpoint(BENCHMARK_ERROR, f'baseline exit={bench_code}')
        return BENCHMARK_ERROR

    # --- summaries ---
    summarize_main([
        '--db', db_path, '--experiment', execution_id,
        '--config', str(root / 'configs' / f'{args.spec}.yaml'),
        '--out', str(root / 'results/summaries' / f'{execution_id}-capability.json'),
    ])
    write_performance_artifact(
        db_path, execution_id,
        root / 'results/summaries' / f'{execution_id}-performance.json',
    )
    statuses = load_spec_statuses(str(root / 'evals/specs/eval-v1-grading.yaml'))
    write_failure_modes_artifact(
        db_path, execution_id, statuses,
        root / 'results/summaries' / f'{execution_id}-failure-modes.json',
    )

    # --- VERIFY (deletion gate inputs) ---
    checkpoint(VERIFYING)
    summary_dir = root / 'results/summaries'
    manifest_dir = root / 'results/experiment-manifests'
    run_config = yaml.safe_load(
        open(root / 'configs' / f'{args.spec}.yaml', encoding='utf-8')
    )
    trials = int(run_config.get('trials', 3))
    statuses = load_spec_statuses(str(root / 'evals/specs/eval-v1-grading.yaml'))
    expected_det = sum(1 for s in statuses.values() if s == 'READY_DETERMINISTIC')
    expected_judge = sum(1 for s in statuses.values() if s == 'READY_JUDGE')
    verdict = verify_execution_persisted(
        db_path, execution_id, statuses,
        expected_det_tasks=expected_det, expected_judge_tasks=expected_judge,
        trials_per_task=trials,
        required_files={
            'manifest': manifest_dir / f'{execution_id}.json',
            'capability': summary_dir / f'{execution_id}-capability.json',
            'performance': summary_dir / f'{execution_id}-performance.json',
            'preflight': summary_dir / f'prompt-tokens-v2-{model_config_id}.json',
            'failure_modes': summary_dir / f'{execution_id}-failure-modes.json',
            'summary': summary_dir / f'{execution_id}.json',
        },
    )
    state.models[model_config_id].verified = verdict.ok
    save_sweep_state(str(state_path), state)
    if not verdict.ok:
        checkpoint(VERIFY_FAILED, verdict.detail)
        return VERIFY_FAILED

    # --- unload -> delete (gated) -> disk verify ---
    guard_deletion(state.models[model_config_id].verified)
    ollama_stop(ollama_bin, identifier)
    unloaded = wait_until_unloaded(
        lambda: load_ps_digest(base_url, identifier)[0] is None, timeout_s=180.0
    )
    if not unloaded:
        checkpoint(BENCHMARK_ERROR, 'VRAM not released after stop')
        return BENCHMARK_ERROR
    ollama_remove(ollama_bin, identifier)
    if ollama_model_present(ollama_bin, identifier):
        checkpoint(BENCHMARK_ERROR, 'model still present after rm')
        return BENCHMARK_ERROR
    free_after = disk_free_bytes(args.db_dir)
    if not disk_reclaimed_ok(free_before, free_after):
        checkpoint(BENCHMARK_ERROR,
                   f'disk not reclaimed: {free_before} -> {free_after}')
        return BENCHMARK_ERROR
    checkpoint(COMPLETE, f'freed={(free_after - free_before) / 1024**3:+.1f}GiB')
    return COMPLETE


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Sequential model sweep')
    parser.add_argument('--experiment', default='full-baseline-v2')
    parser.add_argument('--registry', default='configs/models-v2.yaml')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--base-url', default='http://127.0.0.1:11434')
    parser.add_argument('--db-dir', default='results/local')
    parser.add_argument('--ollama-bin', default=None)
    parser.add_argument('--state-file', default=None)
    parser.add_argument('--stop-on-ineligible', action='store_true')
    parser.add_argument('--continue-on-benchmark-error', action='store_true')
    parser.add_argument('--continue-on-persistence-error', action='store_true')
    parser.add_argument('--only', default=None, help='run a single model_config_id')
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parent.parent
    args.spec = args.experiment
    args.ollama_bin = resolve_ollama_bin(args.ollama_bin)
    args.state_file = args.state_file or str(
        root / 'results/checkpoints' / f'sweep-{args.experiment}.json'
    )
    registry = load_registry(root / args.registry)
    order = [str(e['model_config_id']) for e in registry.get('models', [])]
    if args.only:
        order = [m for m in order if m == args.only]
        if not order:
            print(f'unknown model {args.only}')
            return 2
    # Overlay adapters participate before any run.
    load_overlays_from_dir(root / 'configs/adapters')
    state = load_sweep_state(args.state_file)
    if state is None:
        state = new_sweep_state(f'{args.experiment}-sweep', order)
        save_sweep_state(args.state_file, state)
    else:
        # Resume: adopt registry order, keep lifecycle memory.
        for mid in order:
            if mid not in state.models:
                from storage.sweep import ModelLifecycle

                state.models[mid] = ModelLifecycle(mid)
        state.order = order
        save_sweep_state(args.state_file, state)
    print(f'sweep: {state.sweep_id} resume_from={state.next_model}', flush=True)
    while True:
        current = state.next_model
        if current is None:
            print('SWEEP COMPLETE', flush=True)
            return 0
        entry = next(e for e in registry['models'] if e['model_config_id'] == current)
        outcome = run_one_model(args, entry)
        if outcome in (COMPLETE, COMPLETE_INELIGIBLE):
            if outcome == COMPLETE_INELIGIBLE and args.stop_on_ineligible:
                print('stopping on ineligible (flag)', flush=True)
                return 0
            continue
        if outcome == BENCHMARK_ERROR and args.continue_on_benchmark_error:
            continue
        if outcome == VERIFY_FAILED and args.continue_on_persistence_error:
            continue
        print(f'sweep stopped: {current} -> {outcome}', flush=True)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
