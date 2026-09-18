"""Sweep orchestrator: one thin coordinator over the per-model pipeline.

Owns the 14-config lifecycle: disk check -> pull -> show/pin/derive ->
eligibility -> preflight -> smoke -> warming_up/benchmarking -> validated
-> summarized -> VERIFY -> unload -> delete (gated) -> disk verify -> next.

Stage controls (--only/--stop-after) pause the same code path unattended
execution uses; resume continues from RESTART_STAGE mapping. Deletion is
forbidden until persistence verification succeeds. State in
results/checkpoints/ gives resume across restarts; completed models are
never re-downloaded. Benchmark errors fail closed by default; ineligible
models record an artifact and auto-continue by default.
"""

from __future__ import annotations

import argparse
import json
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
    register_overlay,
    render_prompt,
)
from inference.derive_adapter import (
    ManualPinRequired,
    OverlayExistsError,
    derive_adapter,
    verify_overlay_matches_live,
)
from inference.eligibility import (
    ELIGIBILITY_MEASUREMENT_ERROR,
    effective_options_for,
    run_canonical_eligibility,
)
from inference.retry import PULL_BACKOFFS, RM_ATTEMPTS
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
from scripts.run_benchmark import (
    MEASUREMENT_FAILED_EXIT,
    PAUSED_EXIT,
    run_experiment,
)
from scripts.smoke_inference import main as smoke_main
from scripts.summarize_baseline import main as summarize_main
from storage.db import connect
from storage.execution import derive_execution_id
from storage.paths import results_path
from storage.pause import (
    PauseFlag,
    PauseRecord,
    consume_pause_request,
    pause_record_to_json,
    pause_requested,
)
from storage.sweep import (
    BENCHMARK_ERROR,
    COMPLETE,
    COMPLETE_INELIGIBLE,
    COMPLETE_INELIGIBLE_RUNTIME_HEADROOM,
    DELETION_FAILED,
    DERIVING,
    DOWNLOAD_FAILED,
    DOWNLOADING,
    MANUAL_PIN_REQUIRED as STATE_MANUAL_PIN,
    MEASUREMENT_FAILED,
    PENDING,
    PREFLIGHT_FAILED,
    PREFLIGHTING,
    RESTART_STAGE,
    SMOKING,
    STAGES,
    STOPPED_STATE,
    VERIFY_FAILED,
    VERIFYING,
    WARMING_UP,
    append_retry_event,
    deletion_verdict,
    disk_free_bytes,
    disk_reclaimed_ok,
    guard_deletion,
    load_sweep_state,
    new_sweep_state,
    ollama_model_present,
    ollama_pull,
    ollama_remove,
    ollama_stop,
    resolve_ollama_bin,
    save_sweep_state,
    set_lifecycle,
    verify_execution_persisted,
    wait_until_unloaded,
)


def _utcnow() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat(timespec='seconds')


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


def remove_and_verify(
    *,
    ollama_bin: str,
    identifier: str,
    free_before: int,
    db_dir: str,
    log_event: Any,
    rm_fn: Any = None,
    present_fn: Any = None,
    disk_fn: Any = None,
) -> tuple[str, str] | None:
    """Remove weights with retries and verify absence + disk reclaim.

    Returns None when fully clean, else (DELETION_FAILED, detail).
    Ghost-state rule: rm success means nothing if list/ps still shows the
    model -> DELETION_FAILED, weights may remain, never advance.
    """
    remove = rm_fn or ollama_remove
    present = present_fn or ollama_model_present
    disk_free = disk_fn or disk_free_bytes
    rm_ok = False
    for rm_attempt in range(1, RM_ATTEMPTS + 1):
        ok, _ = remove(ollama_bin, identifier)
        if ok:
            rm_ok = True
            break
        log_event({
            'stage': 'teardown', 'task_id': None, 'trial': None,
            'attempt': rm_attempt, 'error_type': 'rm_transport',
            'action': 'retry_rm', 'backoff_seconds': 0.0,
        })
    verdict = deletion_verdict(rm_ok, present(ollama_bin, identifier))
    if verdict == DELETION_FAILED:
        return DELETION_FAILED, (
            'rm reported success but model still present (ghost state); '
            'weights may remain; not advancing'
        )
    if verdict != 'DELETED_OK':
        return DELETION_FAILED, f'model removal failed: {verdict}'
    free_after = disk_free(db_dir)
    if not disk_reclaimed_ok(free_before, free_after):
        return DELETION_FAILED, f'disk not reclaimed: {free_before} -> {free_after}'
    return None


def graceful_pause(
    *,
    state: Any,
    state_path: Path,
    checkpoints_dir: Path,
    experiment_spec: str,
    model_config_id: str | None,
    paused_from: str,
    resume_stage: str,
    ollama_bin: str,
    base_url: str,
    identifier: str | None,
    reason: str = 'USER_REQUESTED',
) -> str:
    """Checkpoint, unload (weights kept), record PAUSED, banner, exit path."""
    consume_pause_request(checkpoints_dir, experiment_spec)
    ps_empty = True
    if identifier is not None:
        ollama_stop(ollama_bin, identifier)
        ps_empty = wait_until_unloaded(
            lambda: load_ps_digest(base_url, identifier)[0] is None,
            timeout_s=180.0,
        )
    state.status = 'PAUSED'
    state.pause = json.loads(pause_record_to_json(PauseRecord(
        paused_from=paused_from, resume_stage=resume_stage,
        model_config_id=model_config_id or '', pause_reason=reason,
        weights_retained=True,
    )))
    save_sweep_state(str(state_path), state)
    print('========================================', flush=True)
    print('SWEEP PAUSED SAFELY', flush=True)
    if model_config_id is not None:
        print(f'model: {model_config_id}', flush=True)
    print(f'resume_stage: {resume_stage}', flush=True)
    print(f'/api/ps: {"empty" if ps_empty else "MODEL STILL LOADED"}', flush=True)
    print('weights retained: yes', flush=True)
    if ps_empty:
        print('SAFE TO SHUT DOWN', flush=True)
    else:
        print('NOT SAFE TO SHUT DOWN: model still loaded, investigate',
              flush=True)
    print('========================================', flush=True)
    return 'PAUSED'


def run_one_model(args: argparse.Namespace, entry: dict[str, Any]) -> str:
    """Execute the lifecycle for one entry, honoring resume and --stop-after."""
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
    retry_log_path = root / 'results/logs' / f'transient-retries-{args.spec}.jsonl'
    stop_after: str | None = getattr(args, 'stop_after', None)
    if stop_after is not None and stop_after not in STAGES:
        raise ValueError(f'unknown stage {stop_after!r}; valid: {list(STAGES)}')
    start_stage = RESTART_STAGE.get(
        state.models[model_config_id].lifecycle, 'pulled'
    )
    print(f'[{model_config_id}] resume: lifecycle='
          f'{state.models[model_config_id].lifecycle} start_stage={start_stage}',
          flush=True)

    def checkpoint(lifecycle: str, detail: str = '') -> None:
        set_lifecycle(state, model_config_id, lifecycle, detail)
        save_sweep_state(str(state_path), state)
        print(f'[{model_config_id}] {lifecycle} {detail}'.rstrip(), flush=True)

    def run_stage(stage: str) -> bool:
        return STAGES.index(stage) >= STAGES.index(start_stage)

    def halted(stage: str) -> str | None:
        # Pause outranks step-mode: a shutdown request always wins.
        paused = paused_after(
            stage,
            STAGES[STAGES.index(stage) + 1] if stage != 'complete' else 'complete',
        )
        if paused:
            return paused
        if stop_after == stage:
            checkpoint(STOPPED_STATE[stage], 'stop-after requested')
            return STOPPED_STATE[stage]
        return None

    def paused_after(stage: str, resume_stage: str) -> str | None:
        flag = getattr(args, 'pause_flag', None)
        if (flag is not None and flag.requested) or pause_requested(
            root / 'results/checkpoints', args.spec
        ):
            return graceful_pause(
                state=state, state_path=state_path,
                checkpoints_dir=root / 'results/checkpoints',
                experiment_spec=args.spec, model_config_id=model_config_id,
                paused_from=stage, resume_stage=resume_stage,
                ollama_bin=ollama_bin, base_url=base_url, identifier=identifier,
            )
        return None

    config = None
    free_before = disk_free_bytes(args.db_dir)

    def log_sweep_event(event: dict[str, Any]) -> None:
        record = {
            'timestamp': _utcnow(), 'execution_id': execution_id,
            'model_config_id': model_config_id, **event,
        }
        append_retry_event(str(retry_log_path), record)

    if run_stage('pulled'):
        checkpoint(DOWNLOADING, f'free={free_before / 1024**3:.1f}GiB')
        # Pull transport interruptions retry with backoff (30s/2m/5m);
        # persistent failure stops the sweep (no guessing at identifiers).
        import time as _time

        pulled = False
        pull_detail = ''
        for attempt in range(1 + len(PULL_BACKOFFS)):
            ok, pull_detail = ollama_pull(ollama_bin, identifier)
            if ok:
                pulled = True
                break
            backoff = PULL_BACKOFFS[attempt] if attempt < len(PULL_BACKOFFS) else 0.0
            log_sweep_event({
                'stage': 'download', 'task_id': None, 'trial': None,
                'attempt': attempt + 1, 'error_type': 'pull_transport',
                'action': 'retry_download' if backoff else 'download_failed_stop',
                'backoff_seconds': backoff,
            })
            if backoff > 0:
                _time.sleep(backoff)
        if not pulled:
            checkpoint(DOWNLOAD_FAILED, pull_detail[-300:])
            return DOWNLOAD_FAILED
    halt = halted('pulled')
    if halt:
        return halt

    if run_stage('derived'):
        checkpoint(DERIVING)
        # Resume may land here with weights absent (registry correction,
        # manual cleanup, or eviction): re-pull boundedly, never assume.
        if not ollama_model_present(ollama_bin, identifier):
            print(f'[{model_config_id}] weights absent, re-pulling', flush=True)
            repulled = False
            for attempt in range(1 + len(PULL_BACKOFFS)):
                ok, _ = ollama_pull(ollama_bin, identifier)
                if ok and ollama_model_present(ollama_bin, identifier):
                    repulled = True
                    break
                if attempt < len(PULL_BACKOFFS):
                    import time as _time

                    _time.sleep(PULL_BACKOFFS[attempt])
            if not repulled:
                checkpoint(DOWNLOAD_FAILED, 're-pull failed on resume')
                return DOWNLOAD_FAILED
        overlay_path = root / 'configs/adapters' / f'{model_config_id}.json'
        if overlay_path.exists():
            # Committed overlay takes precedence: no interrogation needed,
            # so a silent /api/show can never block a pinned adapter.
            # Digest binds later at canonical eligibility (expected_digest).
            load_overlays_from_dir(root / 'configs/adapters')
            config = get_model_config(model_config_id)
            adapter_source = 'overlay-pinned'
        else:
            try:
                show_doc = fetch_show(base_url, identifier)
            except Exception as exc:
                # Interrogation impossible and nothing pinned: human decision.
                checkpoint(STATE_MANUAL_PIN,
                           f'model interrogation failed: {type(exc).__name__}: {exc}')
                return STATE_MANUAL_PIN
            # Digest observation only (never measured for eligibility: the
            # canonical stage unloads first, then probes pinned options).
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
            # No committed overlay: derive from interrogation (show_doc is
            # defined in this branch only).
            try:
                config = get_model_config(model_config_id)
                adapter_source = 'coded-or-overlay'
            except KeyError:
                config = None
                adapter_source = None
            if config is None:
                def trial_generate(request: GenerationRequest) -> GenerationResult:
                    return generate(args.base_url, request, timeout_s=120.0)

                def ensure_unloaded() -> bool:
                    ollama_stop(ollama_bin, identifier)
                    return wait_until_unloaded(
                        lambda: load_ps_digest(base_url, identifier)[0] is None,
                        timeout_s=180.0,
                    )

                try:
                    config = derive_adapter(
                        registry_entry=entry, show_doc=show_doc,
                        observed_digest=observed_digest, trial_generate=trial_generate,
                        overlay_path=overlay_path,
                        ensure_unloaded=ensure_unloaded,
                    )
                    adapter_source = 'derived'
                    # The overlay file exists but this process registered
                    # adapters at startup: register the derived config live so
                    # preflight/smoke/benchmark resolve it in-process.
                    register_overlay(config)
                except OverlayExistsError:
                    load_overlays_from_dir(root / 'configs/adapters')
                    config = get_model_config(model_config_id)
                    adapter_source = 'overlay-reload'
                except ManualPinRequired as exc:
                    checkpoint(STATE_MANUAL_PIN, str(exc))
                    return STATE_MANUAL_PIN
            else:
                mismatches = verify_overlay_matches_live(config, show_doc, observed_digest)
                if mismatches:
                    checkpoint('ADAPTER_MISMATCH', '; '.join(
                        f'{m.field}: {m.pinned} != {m.observed}' for m in mismatches))
                    return 'ADAPTER_MISMATCH'
        print(f'[{model_config_id}] adapter: {adapter_source} '
              f'mode={config.mode}', flush=True)
        assert config is not None, 'adapter unresolved'
    else:
        config = get_model_config(model_config_id)
    halt = halted('derived')
    if halt:
        return halt

    if run_stage('eligible'):
        # Canonical eligibility: explicit unload first (a stale or
        # misconfigured load must never be measured), then a probe with the
        # exact pinned options, then the /api/ps verdict.
        ollama_stop(ollama_bin, identifier)
        if not wait_until_unloaded(
            lambda: load_ps_digest(base_url, identifier)[0] is None,
            timeout_s=180.0,
        ):
            checkpoint(BENCHMARK_ERROR, 'could not establish clean state for probe')
            return BENCHMARK_ERROR
        canonical = run_canonical_eligibility(
            base_url=base_url,
            model_identifier=identifier,
            expected_digest=config.ollama_model_digest,
            effective_options=effective_options_for(
                mode=config.mode,
                num_ctx=config.num_ctx,
                num_gpu=config.num_gpu,
                temperature=0.0,
                template_sha256=config.template_sha256,
            ),
            render_prompt=lambda prompt: render_prompt(config, prompt),
        )
        eligibility = canonical.result
        print(f'[{model_config_id}] eligibility: {eligibility.status} '
              f'(residency={eligibility.gpu_residency_ratio})', flush=True)
        if eligibility.status == ELIGIBILITY_MEASUREMENT_ERROR:
            # Absent-after-probe: instrumentation failure, never hardware
            # evidence. Weights retained; sweep stops loudly.
            checkpoint(BENCHMARK_ERROR,
                       f'ELIGIBILITY_MEASUREMENT_ERROR: {eligibility.eligibility_evidence_json}')
            return BENCHMARK_ERROR
        if eligibility.eligible:
            # Second gate: full residency is necessary but not sufficient.
            # The operational canary proves sustained canonical inference
            # (Phi Q4 was 100% resident yet aborted streams at 95 MiB free).
            from inference.eligibility import run_operational_canary
            from inference.sysmon import sample_vram_once

            canary = run_operational_canary(
                base_url=base_url,
                model_identifier=identifier,
                effective_options=canonical.effective_options,
                render_prompt=lambda prompt: render_prompt(config, prompt),
            )
            print(f'[{model_config_id}] canary: '
                  f'{"PASS" if canary.passed else "FAIL " + canary.failure_kind} '
                  f'(eval={canary.eval_count})', flush=True)
            if not canary.passed:
                vram_used, vram_total = sample_vram_once()
                vram_free = (
                    (vram_total - vram_used)
                    if vram_used is not None and vram_total is not None
                    else None
                )
                artifact = root / 'results/summaries' / f'ineligible-{model_config_id}.json'
                artifact.write_text(json.dumps({
                    'execution_id': execution_id,
                    'model_config_id': model_config_id,
                    'eligibility_status': COMPLETE_INELIGIBLE_RUNTIME_HEADROOM,
                    'effective_options': canonical.effective_options.as_dict(),
                    'residency_ratio': eligibility.gpu_residency_ratio,
                    'vram_total_mib': vram_total,
                    'vram_free_post_load_mib': vram_free,
                    'weight_fit': True,
                    'sustained_generation': False,
                    'canary_eval_count': canary.eval_count,
                    'canary_done_reason': canary.done_reason,
                    'canary_failure': canary.failure_kind,
                    'canary_detail': canary.detail,
                    'evidence': eligibility.eligibility_evidence_json,
                }, indent=2) + '\n', encoding='utf-8')
                ollama_stop(ollama_bin, identifier)
                wait_until_unloaded(
                    lambda: load_ps_digest(base_url, identifier)[0] is None,
                    timeout_s=180.0,
                )
                ollama_remove(ollama_bin, identifier)
                checkpoint(COMPLETE_INELIGIBLE_RUNTIME_HEADROOM,
                           f'canary {canary.failure_kind}')
                return COMPLETE_INELIGIBLE_RUNTIME_HEADROOM
        if not eligibility.eligible:
            artifact = root / 'results/summaries' / f'ineligible-{model_config_id}.json'
            artifact.write_text(json.dumps({
                'execution_id': execution_id, 'model_config_id': model_config_id,
                'eligibility_status': eligibility.status,
                'effective_options': canonical.effective_options.as_dict(),
                'rendered_prompt_sha256': canonical.rendered_prompt_sha256,
                'model_size_bytes': eligibility.model_size_bytes,
                'size_vram_bytes': eligibility.size_vram_bytes,
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
    else:
        # Resume path: no pre-check here by design. A cold process cannot
        # judge residency without loading, and any load it performs would
        # be the wrong configuration to measure. run_experiment always runs
        # 2 fresh warmups (canonical load) followed by its own eligibility
        # gate, which refuses with exit 3 when not fully resident. That
        # gate is the single authority; duplicating it here caused a false
        # INELIGIBLE verdict on a stale 512-ctx default load (Phi incident).
        pass
    halt = halted('eligible')
    if halt:
        return halt

    if run_stage('preflighted'):
        checkpoint(PREFLIGHTING)
        preflight_code = preflight_main([
            '--model', model_config_id, '--num-predict', '2048',
            '--base-url', base_url,
            '--out', str(root / 'results/summaries' / f'prompt-tokens-v2-{model_config_id}.json'),
        ])
        if preflight_code != 0:
            checkpoint(PREFLIGHT_FAILED, f'exit={preflight_code}')
            return PREFLIGHT_FAILED
    halt = halted('preflighted')
    if halt:
        return halt

    if run_stage('smoked'):
        checkpoint(SMOKING)
        smoke_db = str(Path(args.db_dir) / f'smoke-v2__{model_config_id}.db')
        smoke_code = smoke_main([
            '--db', smoke_db, '--model', model_config_id,
            '--base-url', base_url, '--config', 'smoke-v2.yaml',
            '--num-predict', '2048', '--resume',
        ])
        if smoke_code != 0:
            checkpoint(BENCHMARK_ERROR, 'smoke failed')
            return BENCHMARK_ERROR
    halt = halted('smoked')
    if halt:
        return halt

    if run_stage('benchmarked'):
        # Warmups are bound to this loaded session (internal warming_up state):
        # the runner records 2 fresh warmups every invocation and re-warms
        # automatically on mid-run reload evidence.
        checkpoint(WARMING_UP)
        run_ns = argparse.Namespace(
            config=str(root / 'configs' / f'{args.spec}.yaml'),
            model=model_config_id, db=db_path, base_url=base_url,
            resume=True, execution_id=None, num_predict=None,
            timeout_s=600.0, max_tasks=None, ollama_bin=ollama_bin,
            pause_flag=getattr(args, 'pause_flag', None),
        )
        bench_code = run_experiment(run_ns)
        if bench_code == PAUSED_EXIT:
            # Runner paused after a committed row (unloaded there, weights
            # kept): record sweep-level PAUSED without touching anything.
            state.status = 'PAUSED'
            state.pause = json.loads(pause_record_to_json(PauseRecord(
                paused_from='BENCHMARKING', resume_stage='benchmarked',
                model_config_id=model_config_id, weights_retained=True,
            )))
            save_sweep_state(str(state_path), state)
            consume_pause_request(root / 'results/checkpoints', args.spec)
            print('========================================', flush=True)
            print('SWEEP PAUSED SAFELY', flush=True)
            print(f'model: {model_config_id}', flush=True)
            print('resume_stage: benchmarked', flush=True)
            print('weights retained: yes', flush=True)
            print('SAFE TO SHUT DOWN', flush=True)
            print('========================================', flush=True)
            return 'PAUSED'
        if bench_code == MEASUREMENT_FAILED_EXIT:
            # Persistent measurement failure: no row written, weights kept,
            # sweep STOPS. The V2 contract is never weakened to absorb it.
            checkpoint(MEASUREMENT_FAILED, 'persistent generation failure')
            return MEASUREMENT_FAILED
        if bench_code != 0:
            checkpoint(BENCHMARK_ERROR, f'baseline exit={bench_code}')
            return BENCHMARK_ERROR
    halt = halted('benchmarked')
    if halt:
        return halt

    statuses = load_spec_statuses(str(root / 'evals/specs/eval-v1-grading.yaml'))
    run_config = yaml.safe_load(
        open(root / 'configs' / f'{args.spec}.yaml', encoding='utf-8')
    )
    trials = int(run_config.get('trials', 3))
    expected_det = sum(1 for s in statuses.values() if s == 'READY_DETERMINISTIC')
    expected_judge = sum(1 for s in statuses.values() if s == 'READY_JUDGE')
    summary_dir = root / 'results/summaries'
    manifest_dir = root / 'results/experiment-manifests'

    if run_stage('validated'):
        counts = verify_execution_persisted(
            db_path, execution_id, statuses,
            expected_det_tasks=expected_det, expected_judge_tasks=expected_judge,
            trials_per_task=trials, required_files={},
        )
        row_checks = {k: v for k, v in counts.checks.items() if not k.startswith('file:')}
        if not all(row_checks.values()):
            checkpoint(BENCHMARK_ERROR, f'row validation: {counts.detail}')
            return BENCHMARK_ERROR
    halt = halted('validated')
    if halt:
        return halt

    if run_stage('summarized'):
        summarize_main([
            '--db', db_path, '--experiment', execution_id,
            '--config', str(root / 'configs' / f'{args.spec}.yaml'),
            '--out', str(summary_dir / f'{execution_id}-capability.json'),
        ])
        write_performance_artifact(
            db_path, execution_id,
            summary_dir / f'{execution_id}-performance.json',
        )
        write_failure_modes_artifact(
            db_path, execution_id, statuses,
            summary_dir / f'{execution_id}-failure-modes.json',
        )
    halt = halted('summarized')
    if halt:
        return halt

    if run_stage('verified'):
        checkpoint(VERIFYING)
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
    halt = halted('verified')
    if halt:
        return halt

    if run_stage('unloaded'):
        guard_deletion(state.models[model_config_id].verified)
        ollama_stop(ollama_bin, identifier)
        unloaded = wait_until_unloaded(
            lambda: load_ps_digest(base_url, identifier)[0] is None, timeout_s=180.0
        )
        if not unloaded:
            checkpoint(BENCHMARK_ERROR, 'VRAM not released after stop')
            return BENCHMARK_ERROR
    halt = halted('unloaded')
    if halt:
        return halt

    if run_stage('deleted'):
        guard_deletion(state.models[model_config_id].verified)
        failure = remove_and_verify(
            ollama_bin=ollama_bin, identifier=identifier,
            free_before=free_before, db_dir=args.db_dir,
            log_event=log_sweep_event,
        )
        if failure is not None:
            outcome, detail = failure
            checkpoint(outcome, detail)
            return outcome
    halt = halted('deleted')
    if halt:
        return halt

    checkpoint(COMPLETE, 'lifecycle complete')
    return COMPLETE


def maybe_write_family_report(
    args: argparse.Namespace,
    registry: dict[str, Any],
    just_completed_id: str,
) -> None:
    """After a terminal model, emit missing anchor-paired family reports.

    Non-blocking and idempotent (existing reports are never regenerated).
    Ineligible anchors/members yield NOT_AVAILABLE records, never a pairing.
    """
    from scripts.compare_models import main as compare_main

    root = Path(__file__).resolve().parent.parent
    members = [
        e for e in registry.get('models', [])
        if str(e.get('family', '')) == next(
            m.get('family', '') for m in registry.get('models', [])
            if str(m.get('model_config_id')) == just_completed_id
        )
    ]
    anchors = [m for m in members if str(m.get('quantization')) == 'Q4_K_M']
    if not anchors:
        return
    anchor_id = str(anchors[0]['model_config_id'])
    state = load_sweep_state(args.state_file)
    if state is None:
        return
    summary_dir = root / 'results/summaries'
    for member in members:
        member_id = str(member['model_config_id'])
        if member_id == anchor_id:
            continue
        report_path = summary_dir / f'{anchor_id}-vs-{member_id}.json'
        if report_path.exists():
            continue
        anchor_lc = state.models.get(anchor_id)
        member_lc = state.models.get(member_id)
        anchor_done = anchor_lc is not None and anchor_lc.lifecycle == COMPLETE
        member_done = member_lc is not None and member_lc.lifecycle == COMPLETE
        if anchor_done and member_done:
            base_exe = derive_execution_id(args.spec, anchor_id)
            cand_exe = derive_execution_id(args.spec, member_id)
            compare_main([
                '--db-base', str(Path(args.db_dir) / f'{base_exe}.db'),
                '--db-candidate', str(Path(args.db_dir) / f'{cand_exe}.db'),
                '--base', base_exe, '--candidate', cand_exe,
                '--spec', args.spec,
                '--out', str(report_path),
            ])
            print(f'family report: {report_path.name}', flush=True)
        elif member_done and not anchor_done:
            report_path.write_text(json.dumps({
                'experiment_spec_id': args.spec,
                'base_execution_id': derive_execution_id(args.spec, anchor_id),
                'candidate_execution_id': derive_execution_id(args.spec, member_id),
                'comparison_status': 'NOT_AVAILABLE',
                'reason': 'CONFIG_INELIGIBLE_GPU_ONLY',
            }, indent=2) + '\n', encoding='utf-8')
            print(f'family report (not available): {report_path.name}', flush=True)


def write_sweep_summary(
    args: argparse.Namespace, registry: dict[str, Any]
) -> Path:
    """Final sweep artifact with terminal counts."""
    root = Path(__file__).resolve().parent.parent
    state = load_sweep_state(args.state_file)
    order = [str(e['model_config_id']) for e in registry.get('models', [])]
    attempted = [
        mid for mid in order
        if state is not None and mid in state.models
        and state.models[mid].lifecycle != PENDING
    ]
    complete = list(state.completed) if state else []
    ineligible = list(state.ineligible) if state else []
    failed = list(state.failed) if state else []
    document = {
        'experiment_spec_id': args.spec,
        'registry_count': len(order),
        'attempted': len(attempted),
        'complete': len(complete),
        'complete_ineligible': len(ineligible),
        'failed_integrity': len(failed),
        'completed_ids': sorted(complete),
        'ineligible_ids': sorted(ineligible),
        'failed_ids': sorted(failed),
        'sweep_status': 'COMPLETE',
    }
    out = results_path(root, 'summaries', f'sweep-{args.spec}-complete.json')
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(document, indent=2) + '\n', encoding='utf-8')
    return out


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
    parser.add_argument('--stop-after', default=None,
                        help=f'step mode: stop after one of {list(STAGES)}')
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
    pause_flag = PauseFlag()
    pause_flag.install_sigint_handler()
    args.pause_flag = pause_flag
    try:
        return _sweep_loop(args, registry, root)
    finally:
        pause_flag.uninstall_sigint_handler()


def _sweep_loop(
    args: argparse.Namespace, registry: dict[str, Any], root: Path
) -> int:
    state = load_sweep_state(args.state_file)
    if state is None:
        order = [str(e['model_config_id']) for e in registry.get('models', [])]
        state = new_sweep_state(f'{args.experiment}-sweep', order)
        save_sweep_state(args.state_file, state)
    else:
        # Resume: adopt registry order, keep lifecycle memory.
        order = [str(e['model_config_id']) for e in registry.get('models', [])]
        for mid in order:
            if mid not in state.models:
                from storage.sweep import ModelLifecycle

                state.models[mid] = ModelLifecycle(mid)
        state.order = order
        if state.status == 'PAUSED':
            paused = state.pause or {}
            print(f"resuming from PAUSED (was: {paused.get('paused_from')}, "
                  f'model {paused.get("model_config_id")})', flush=True)
            state.status = 'RUNNING'
        save_sweep_state(args.state_file, state)
    return _run_loop(args, registry, root, state)


def _run_loop(
    args: argparse.Namespace,
    registry: dict[str, Any],
    root: Path,
    state: Any,
) -> int:
    print(f'sweep: {state.sweep_id} resume_from={state.next_model}', flush=True)
    stopped_states = set(STOPPED_STATE.values())
    terminal_states = {'COMPLETE', 'COMPLETE_INELIGIBLE',
                       COMPLETE_INELIGIBLE_RUNTIME_HEADROOM}
    checkpoints_dir = root / 'results/checkpoints'
    while True:
        # The state FILE is authoritative: reload every iteration so a
        # terminal outcome can never select the same model again.
        fresh = load_sweep_state(args.state_file)
        if fresh is not None:
            state = fresh
        # A pending request pauses before starting new work (never mid-unit).
        # The current model's identifier is passed so a lingering load is
        # still evicted and verified (stop is a no-op when absent).
        if pause_requested(checkpoints_dir, args.spec) or (
            getattr(args, 'pause_flag', None) is not None
            and args.pause_flag.requested
        ):
            current = state.next_model
            loop_identifier: str | None = None
            if current is not None:
                loop_entry = next(
                    (e for e in registry.get('models', [])
                     if str(e.get('model_config_id')) == current),
                    None,
                )
                if loop_entry is not None:
                    loop_identifier = str(loop_entry.get('ollama_identifier'))
            graceful_pause(
                state=state, state_path=Path(args.state_file),
                checkpoints_dir=checkpoints_dir,
                experiment_spec=args.spec, model_config_id=current,
                paused_from='SCHEDULER',
                resume_stage='pulled' if current else 'complete',
                ollama_bin=args.ollama_bin, base_url=args.base_url,
                identifier=loop_identifier,
            )
            return 0
        current = state.next_model
        if current is None:
            summary_path = write_sweep_summary(args, registry)
            print(f'SWEEP COMPLETE: {summary_path.name}', flush=True)
            print('========================================', flush=True)
            print('FULL BASELINE V2 SWEEP COMPLETE', flush=True)
            print(f"{len(registry.get('models', []))} / "
                  f"{len(registry.get('models', []))} configurations attempted",
                  flush=True)
            print('========================================', flush=True)
            return 0
        if args.only and state.models[current].lifecycle in terminal_states:
            print(f'sweep paused: {current} already terminal '
                  f'({state.models[current].lifecycle})', flush=True)
            return 0
        entry = next(e for e in registry['models'] if e['model_config_id'] == current)
        try:
            outcome = run_one_model(args, entry)
        except Exception as exc:
            # Last-line integrity net: an unhandled per-model crash must
            # persist as BENCHMARK_ERROR and stop, never kill state silently.
            checkpoint_state = load_sweep_state(args.state_file)
            if checkpoint_state is not None:
                set_lifecycle(checkpoint_state, current, BENCHMARK_ERROR,
                              f'unhandled: {type(exc).__name__}: {exc}')
                save_sweep_state(args.state_file, checkpoint_state)
            print(f'sweep stopped: {current} -> BENCHMARK_ERROR '
                  f'(unhandled {type(exc).__name__})', flush=True)
            return 1
        if outcome == 'PAUSED':
            return 0  # banner already printed by the pausing layer
        if outcome in stopped_states:
            print(f'sweep paused: {current} -> {outcome}', flush=True)
            return 0
        if outcome in (COMPLETE, COMPLETE_INELIGIBLE,
                        COMPLETE_INELIGIBLE_RUNTIME_HEADROOM):
            # Stepped mode: an explicit --stop-after complete pauses even on
            # natural completion. Otherwise COMPLETE always advances.
            if getattr(args, 'stop_after', None) == 'complete' and outcome == COMPLETE:
                print(f'sweep paused: {current} -> COMPLETE (stepped)', flush=True)
                return 0
            if outcome in (COMPLETE_INELIGIBLE,
                           COMPLETE_INELIGIBLE_RUNTIME_HEADROOM) \
                    and args.stop_on_ineligible:
                print('stopping on ineligible (flag)', flush=True)
                return 0
            # Non-blocking family reports: generated, recorded, never gated.
            maybe_write_family_report(args, registry, current)
            continue
        if outcome == BENCHMARK_ERROR and args.continue_on_benchmark_error:
            continue
        if outcome == VERIFY_FAILED and args.continue_on_persistence_error:
            continue
        # MEASUREMENT_FAILED, DELETION_FAILED and all other integrity
        # outcomes always stop: the contract outranks continuity.
        print(f'sweep stopped: {current} -> {outcome}', flush=True)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
