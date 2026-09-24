"""Benchmark runner: dataset -> inference -> profiler -> grader -> SQLite.

Joins frozen executable tasks (prompts) with frozen grading-spec entries
(graders + status) at runtime. Commits after every generation; --resume
skips exactly the COMPLETE (model, task, trial, config-hash) tuples.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evals.graders.engine import GraderResult, grade_output
from evals.verdicts import ERROR, REFUSED_NOT_EXECUTABLE, reduce_verdict
from inference.adapters import (
    ModelConfig,
    get_model_config,
    model_config_hash,
    render_prompt,
    rendered_prompt_sha256,
)
from inference.eligibility import check_eligibility
from inference.ollama_client import (
    GenerationRequest,
    GenerationResult,
    OllamaClientError,
    generate,
)
from inference.profiler import (
    RELOAD_EVIDENCE_LOAD_DURATION_MS,
    ProfiledMetrics,
    derive_metrics,
    needs_rewarm,
)
from inference.retry import GENERATION_RETRIES, error_kind, is_retryable
from inference.sysmon import SystemSample, SystemSampler, sample_vram_once
from storage.db import (
    RunRecord,
    completed_identities,
    connect,
    create_experiment,
    effective_generation_config,
    fetch_measured,
    init_schema,
    insert_run,
    run_config_hash,
)
from storage.execution import (
    CREATE,
    LEGACY_READ_ONLY,
    MISMATCH,
    RESUME,
    derive_execution_id,
    ensure_provenance_table,
    record_execution_provenance,
    resolve_execution,
)
from storage.manifest import (
    build_manifest,
    collect_live_environment,
    git_commit,
    hash_experiment_config,
)
from storage.paths import results_path
from storage.pause import PauseFlag, consume_pause_request, pause_requested
from storage.sweep import (
    append_retry_event,
    ollama_stop,
    resolve_ollama_bin,
    wait_until_unloaded,
)
from analysis.temperature_study import validate_temperature_study_run

WARMUP_PROMPTS = ('Return only the word OK.', 'Return only the digit 7.')
WARMUP_VERDICT = 'WARMUP'
# Exit code: persistent measurement failure (no row written, STOP sweep).
MEASUREMENT_FAILED_EXIT = 6
# Exit code: graceful pause completed (row committed, model unloaded).
PAUSED_EXIT = 7
# Backoff between same-identity generation retries (seconds).
GENERATION_RETRY_BACKOFFS = (5.0, 15.0)


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def load_executable_tasks(path: Path) -> dict[str, str]:
    tasks: dict[str, str] = {}
    with open(path, encoding='utf-8') as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row: Any = json.loads(line)
            tasks[str(row['id'])] = str(row['prompt'])
    return tasks


def load_spec_entries(path: Path) -> dict[str, dict[str, Any]]:
    spec: Any = yaml.safe_load(open(path, encoding='utf-8'))
    entries: dict[str, dict[str, Any]] = {}
    for task_id, entry in spec['tasks'].items():
        entries[str(task_id)] = {
            'grading_status': str(entry['grading_status']),
            'graders': list(entry.get('graders') or []),
        }
    return entries


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ollama_version(base_url: str) -> str | None:
    try:
        response = requests.get(base_url.rstrip('/') + '/api/version', timeout=15)
    except requests.RequestException:
        return None
    if response.status_code != 200:
        return None
    try:
        return str(response.json().get('version'))
    except ValueError:
        return None


def build_request(
    config_id: str,
    rendered: str,
    temperature: float,
    num_predict: int,
) -> GenerationRequest:
    config = get_model_config(config_id)
    return GenerationRequest(
        model=config.ollama_identifier,
        prompt=rendered,
        temperature=temperature,
        num_ctx=config.num_ctx,
        num_predict=num_predict,
        stop=config.stop_tokens,
        raw=(config.mode == 'raw'),
        think=config.think,
        num_gpu=config.num_gpu,
    )


def insert_measured_row(
    conn: sqlite3.Connection,
    *,
    base: dict[str, Any],
    text: str,
    thinking: str,
    done_reason: str | None,
    metrics: ProfiledMetrics,
    sampler_ram: tuple[float | None, float | None],
    vram_pre: tuple[float | None, float | None, float | None],
    vram_post_mib: float | None,
    verdict: str,
    details: list[GraderResult],
    status: str,
    error: str | None,
    eligibility_status: str | None,
    residency_ratio: float | None,
    evidence_json: str | None,
    ollama_ver: str | None,
    model_digest: str,
) -> None:
    detail_docs = [
        {'grader_type': r.grader_type, 'passed': r.passed,
         'detail': r.detail, 'violations': r.violations}
        for r in details
    ]
    record = RunRecord(
        experiment_id=str(base['experiment_id']),
        model_config_id=str(base['model_config_id']),
        task_id=str(base['task_id']),
        trial=int(base['trial']),
        run_kind=str(base['run_kind']),
        run_config_hash=str(base['run_config_hash']),
        is_warmup=bool(base['is_warmup']),
        prompt=str(base['prompt']),
        rendered_prompt_sha256=str(base['rendered_prompt_sha256']),
        temperature=float(base['temperature']),
        num_ctx=int(base['num_ctx']),
        num_predict=int(base['num_predict']),
        template_sha256=str(base['template_sha256']),
        grader_verdict=verdict,
        status=status,
        started_at_utc=str(base['started_at_utc']),
        ended_at_utc=str(base['ended_at_utc']),
        raw_output=text,
        thinking_output=thinking,
        done_reason=done_reason,
        num_gpu=base.get('num_gpu'),
        stop_tokens=list(base.get('stop_tokens', [])),
        think=str(base.get('think', 'default')),
        ttft_ms=metrics.ttft_ms,
        client_e2e_ms=metrics.client_e2e_ms,
        server_total_duration_ms=metrics.server_total_duration_ms,
        server_load_duration_ms=metrics.server_load_duration_ms,
        prompt_eval_count=metrics.prompt_eval_count,
        prompt_eval_cached_count=metrics.prompt_eval_cached_count,
        prompt_eval_uncached_count=metrics.prompt_eval_uncached_count,
        prompt_cache_ratio=metrics.prompt_cache_ratio,
        prompt_eval_duration_ms=metrics.prompt_eval_duration_ms,
        eval_count=metrics.eval_count,
        eval_duration_ms=metrics.eval_duration_ms,
        prefill_compute_tok_s=metrics.prefill_compute_tok_s,
        prefill_cache_state=metrics.prefill_cache_state,
        decode_tok_s=metrics.decode_tok_s,
        decode_ms_per_token=metrics.decode_ms_per_token,
        client_overhead_ms=metrics.client_overhead_ms,
        ram_baseline_mb=sampler_ram[0],
        ram_peak_mb=sampler_ram[1],
        vram_baseline_mib=vram_pre[0],
        vram_peak_mib=vram_pre[1],
        vram_post_mib=vram_post_mib,
        vram_total_mib=vram_pre[2],
        grader_details_json=json.dumps(detail_docs),
        error=error,
        eligibility_status=eligibility_status,
        gpu_residency_ratio=residency_ratio,
        eligibility_evidence_json=evidence_json,
        ollama_version=ollama_ver,
        model_digest=model_digest,
    )
    insert_run(conn, record)
    conn.commit()


def _load_probe(base_url: str, model_identifier: str) -> None:
    """Unrecorded single-token generation: forces model load so the artifact
    digest is observable before any benchmark write. Result discarded."""
    generate(
        base_url,
        GenerationRequest(
            model=model_identifier, prompt='OK', temperature=0.0,
            num_ctx=512, num_predict=4, raw=False,
        ),
        timeout_s=300.0,
    )


def _run_warmup_trial(
    conn: sqlite3.Connection,
    *,
    experiment_id: str,
    config: ModelConfig,
    config_hash: str,
    temperature: float,
    num_predict: int,
    warmup_prompt: str,
    base_url: str,
    timeout_s: float,
    ollama_ver: str | None,
) -> ProfiledMetrics:
    """Run, profile, persist and commit ONE warmup generation.

    Trial numbers continue from persisted history (collision-free resume),
    but warmth is proven only by fresh completion in this invocation.
    """
    trial_no = int(conn.execute(
        "SELECT COUNT(*) FROM runs WHERE experiment_id=? AND model_config_id=?"
        " AND run_kind='WARMUP'",
        (experiment_id, config.config_id),
    ).fetchone()[0]) + 1
    started = utcnow()
    rendered = render_prompt(config, warmup_prompt)
    with SystemSampler() as sampler:
        result = generate(
            base_url,
            GenerationRequest(
                model=config.ollama_identifier,
                prompt=rendered,
                temperature=temperature,
                num_ctx=config.num_ctx,
                num_predict=num_predict,
                stop=config.stop_tokens,
                raw=(config.mode == 'raw'),
                think=config.think,
                num_gpu=config.num_gpu,
            ),
            timeout_s=timeout_s,
        )
    metrics = derive_metrics(result)
    sample = sampler.sample()
    post, _ = sample_vram_once()
    insert_measured_row(
        conn,
        base={
            'experiment_id': experiment_id, 'model_config_id': config.config_id,
            'task_id': f'WARMUP-{trial_no}', 'trial': trial_no,
            'run_kind': 'WARMUP', 'run_config_hash': config_hash,
            'is_warmup': True, 'prompt': warmup_prompt,
            'rendered_prompt_sha256': rendered_prompt_sha256(rendered),
            'temperature': temperature, 'num_ctx': config.num_ctx,
            'num_predict': num_predict,
            'template_sha256': config.template_sha256,
            'num_gpu': config.num_gpu,
            'stop_tokens': list(config.stop_tokens),
            'think': repr(config.think),
            'started_at_utc': started, 'ended_at_utc': utcnow(),
        },
        text=result.text, thinking=result.thinking,
        done_reason=result.done_reason, metrics=metrics,
        sampler_ram=(sample.ram_baseline_mb, sample.ram_peak_mb),
        vram_pre=(sample.vram_baseline_mib, sample.vram_peak_mib,
                  sample.vram_total_mib),
        vram_post_mib=post, verdict=WARMUP_VERDICT, details=[],
        status='COMPLETE', error=None, eligibility_status=None,
        residency_ratio=None, evidence_json=None, ollama_ver=ollama_ver,
        model_digest=config.ollama_model_digest,
    )
    print(f'warmup {trial_no}: {len(result.text)} chars, '
          f'decode={metrics.decode_tok_s:.1f} tok/s' if metrics.decode_tok_s else
          f'warmup {trial_no}: {len(result.text)} chars', flush=True)
    return metrics


class MeasurementFailed(Exception):
    """Persistent measurement failure: no row written, sweep must STOP."""

    def __init__(self, identity: str, reason: str) -> None:
        super().__init__(f'{identity}: {reason}')
        self.identity = identity
        self.reason = reason


def _reload_canonical(
    *,
    ollama_bin: str,
    base_url: str,
    config: ModelConfig,
    temperature: float,
    timeout_s: float,
) -> None:
    """Unload, reload with the EXACT pinned options, re-verify identity.

    The reload probe uses the adapter's template/mode, num_ctx, num_gpu and
    temperature (num_predict=4 keeps it cheap; residency follows num_ctx).
    A default-options probe here would repeat the 512-context incident:
    measuring a misconfigured load as if it were the benchmark config.
    Raises MeasurementFailed on any fault: the session cannot be trusted.
    """
    model_identifier = config.ollama_identifier
    ollama_stop(ollama_bin, model_identifier)
    if not wait_until_unloaded(
        lambda: _ps_absent(base_url, model_identifier), timeout_s=120.0
    ):
        raise MeasurementFailed(
            model_identifier, 'model did not unload for recovery reload'
        )
    probe_prompt = (
        render_prompt(config, 'OK')
        if config.mode == 'raw'
        else 'OK'
    )
    try:
        generate(
            base_url,
            GenerationRequest(
                model=model_identifier, prompt=probe_prompt, temperature=temperature,
                num_ctx=config.num_ctx, num_predict=4, num_gpu=config.num_gpu,
                stop=config.stop_tokens, raw=(config.mode == 'raw'),
                think=config.think,
            ),
            timeout_s=timeout_s,
        )
    except OllamaClientError as exc:
        raise MeasurementFailed(
            model_identifier, f'reload probe failed: {exc}'
        ) from exc
    reverified = check_eligibility(
        base_url, model_identifier, expected_digest=config.ollama_model_digest
    )
    if not reverified.eligible:
        raise MeasurementFailed(
            model_identifier,
            f'post-reload identity not eligible: {reverified.status}',
        )


def _ps_absent(base_url: str, model_identifier: str) -> bool:
    from inference.ollama_client import fetch_ps

    try:
        doc = fetch_ps(base_url)
    except OllamaClientError:
        return False
    entries = doc.get('models', [])
    if not isinstance(entries, list):
        return False
    return not any(
        isinstance(entry, dict)
        and (entry.get('model') == model_identifier
             or entry.get('name') == model_identifier)
        for entry in entries
    )


def _generate_with_recovery(
    *,
    make_request: Any,
    task_id: str,
    trial: int,
    base_url: str,
    timeout_s: float,
    ollama_bin: str,
    model_identifier: str,
    config: ModelConfig,
    temperature: float,
    rewarm: Any,
    log_event: Any,
    sleep_fn: Any = None,
) -> tuple[GenerationResult, SystemSample, float | None]:
    """Generate one measured row with bounded transport recovery.

    Same-identity retries (GENERATION_RETRIES), then unload/reload with
    exact-identity re-verification, 2 fresh warmups, one final attempt.
    Persistent failure raises MeasurementFailed WITHOUT writing any row.
    Non-retryable faults raise through for the ERROR-row path.
    Partial output from timed-out attempts is discarded (generate() raises
    before returning anything; nothing reaches the grader or the DB).
    """
    import time as _time

    sleeper = sleep_fn or _time.sleep
    identity = f'{task_id} t{trial}'
    last_error: OllamaClientError | None = None
    attempts = 1 + GENERATION_RETRIES
    for attempt in range(1, attempts + 1):
        try:
            with SystemSampler() as sampler:
                result = generate(base_url, make_request(), timeout_s=timeout_s)
            sample = sampler.sample()
            post, _ = sample_vram_once()
            return result, sample, post
        except OllamaClientError as exc:
            if not is_retryable(exc):
                raise
            last_error = exc
            backoff = GENERATION_RETRY_BACKOFFS[attempt - 1] \
                if attempt - 1 < len(GENERATION_RETRY_BACKOFFS) else 0.0
            log_event({
                'task_id': task_id, 'trial': trial, 'attempt': attempt,
                'error_type': error_kind(exc),
                'action': 'retry_same_identity', 'backoff_seconds': backoff,
            })
            if backoff > 0:
                sleeper(backoff)
    assert last_error is not None
    log_event({
        'task_id': task_id, 'trial': trial, 'attempt': attempts + 1,
        'error_type': error_kind(last_error),
        'action': 'unload_reload_reverify', 'backoff_seconds': 0.0,
    })
    try:
        _reload_canonical(
            ollama_bin=ollama_bin, base_url=base_url,
            config=config, temperature=temperature, timeout_s=timeout_s,
        )
    except MeasurementFailed as exc:
        raise MeasurementFailed(identity, exc.reason) from exc
    except Exception as exc:
        # Recovery machinery itself broken: the session cannot be trusted.
        raise MeasurementFailed(
            identity, f'recovery reload failed: {type(exc).__name__}: {exc}'
        ) from exc
    rewarm()
    try:
        with SystemSampler() as sampler:
            result = generate(base_url, make_request(), timeout_s=timeout_s)
        sample = sampler.sample()
        post, _ = sample_vram_once()
        log_event({
            'task_id': task_id, 'trial': trial, 'attempt': attempts + 2,
            'error_type': '', 'action': 'final_attempt_ok', 'backoff_seconds': 0.0,
        })
        return result, sample, post
    except OllamaClientError as exc:
        if is_retryable(exc):
            log_event({
                'task_id': task_id, 'trial': trial, 'attempt': attempts + 2,
                'error_type': error_kind(exc),
                'action': 'final_attempt_failed', 'backoff_seconds': 0.0,
            })
            raise MeasurementFailed(identity, f'final attempt failed: {exc}') from exc
        raise


def _pause_if_requested(
    *,
    pause_flag: PauseFlag | None,
    checkpoints_dir: Path,
    experiment_spec: str,
    base_url: str,
    ollama_bin: str,
    model_identifier: str,
) -> bool:
    """Row-boundary pause check. The completed row is already committed.

    On request: unload the model (weights retained), consume the request,
    report honestly. Returns True when the caller must stop (PAUSED_EXIT).
    """
    requested = (pause_flag is not None and pause_flag.requested) or pause_requested(
        checkpoints_dir, experiment_spec
    )
    if not requested:
        return False
    ollama_stop(ollama_bin, model_identifier)
    ps_empty = wait_until_unloaded(
        lambda: _ps_absent(base_url, model_identifier), timeout_s=180.0
    )
    consume_pause_request(checkpoints_dir, experiment_spec)
    state = 'verified (ps empty)' if ps_empty else 'UNVERIFIED (ps non-empty)'
    print(f'pause: row committed, model unload {state}, weights retained',
          flush=True)
    return True


def run_experiment(args: argparse.Namespace) -> int:
    root = Path(__file__).resolve().parent.parent
    run_config = yaml.safe_load(open(args.config, encoding='utf-8'))
    # Identity split: the frozen YAML names the experimental CONTRACT;
    # execution_id names one concrete execution of spec x model.
    experiment_spec_id: str = str(run_config['experiment'])
    model_config_id: str = str(args.model)
    execution_id: str = (
        str(args.execution_id)
        if getattr(args, 'execution_id', None)
        else derive_execution_id(experiment_spec_id, model_config_id)
    )
    experiment_id = execution_id  # operational identity from here on
    run_kind: str = str(run_config.get('run_kind', 'BASELINE'))
    temperature: float = float(run_config.get('temperature', 0.0))
    trials: int = int(run_config.get('trials', 1))
    # Canonical ceiling lives in the frozen contract; CLI overrides explicitly.
    cli_predict = getattr(args, 'num_predict', None)
    num_predict: int = (
        int(cli_predict) if cli_predict is not None
        else int(run_config.get('num_predict', 512))
    )
    task_ids: list[str] = [str(t) for t in run_config['task_ids']]
    if args.max_tasks is not None:
        task_ids = task_ids[: args.max_tasks]

    config = get_model_config(args.model)
    tasks = load_executable_tasks(root / 'evals/datasets/eval-v1/executable-v1.jsonl')
    spec_entries = load_spec_entries(root / 'evals/specs/eval-v1-grading.yaml')
    validate_temperature_study_run(
        run_config, root=root, model_config_id=args.model, task_ids=task_ids,
        temperature=temperature, trials=trials, num_predict=num_predict,
        grading_statuses={
            task_id: str(entry['grading_status'])
            for task_id, entry in spec_entries.items()
        },
    )

    effective = effective_generation_config(
        ollama_identifier=config.ollama_identifier,
        quantization=config.quantization,
        mode=config.mode,
        temperature=temperature,
        num_ctx=config.num_ctx,
        num_predict=num_predict,
        num_gpu=config.num_gpu,
        stop_tokens=list(config.stop_tokens),
        think=config.think,
        template_sha256=config.template_sha256,
    )
    config_hash = run_config_hash(effective)

    db_path = Path(args.db)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = connect(str(db_path))
    init_schema(conn)
    ensure_provenance_table(conn)

    experiment_config = {
        'config_file': Path(args.config).name,
        'experiment_spec_id': experiment_spec_id,
        'run_kind': run_kind,
        'trials': trials,
        'task_ids': task_ids,
    }
    experiment_config_hash = hash_experiment_config(experiment_config)
    declared_config_hash = model_config_hash(config)

    # Load probe (unrecorded): the artifact digest must be observed BEFORE
    # any write, so a weights change under the same tag fails closed here.
    _load_probe(args.base_url, config.ollama_identifier)
    probe_eligibility = check_eligibility(
        args.base_url, config.ollama_identifier,
        expected_digest=config.ollama_model_digest,
    )
    observed_digest = probe_eligibility.observed_digest
    observed_size = probe_eligibility.model_size_bytes

    resolution = resolve_execution(
        conn, execution_id,
        experiment_spec_id=experiment_spec_id,
        model_config_id=model_config_id,
        experiment_config_hash=experiment_config_hash,
        model_config_hash=declared_config_hash,
        model_artifact_digest=observed_digest,
    )
    if resolution.decision == MISMATCH:
        print(f'EXECUTION REFUSED: {resolution.detail}', flush=True)
        conn.close()
        return 4
    if resolution.decision == LEGACY_READ_ONLY:
        print(f'EXECUTION REFUSED: {resolution.detail}', flush=True)
        conn.close()
        return 5
    if resolution.decision == CREATE:
        create_experiment(
            conn, experiment_id=execution_id, name=execution_id,
            config_hash=config_hash,
            config_yaml=open(args.config, encoding='utf-8').read(),
            created_at_utc=utcnow(),
        )
        record_execution_provenance(
            conn, execution_id=execution_id,
            experiment_spec_id=experiment_spec_id,
            model_config_id=model_config_id,
            experiment_config_hash=experiment_config_hash,
            model_config_hash=declared_config_hash,
            model_artifact_digest=observed_digest,
            created_at_utc=utcnow(),
        )
    elif resolution.decision == RESUME and not args.resume:
        print(f'execution {execution_id} exists; use --resume to continue', flush=True)
        conn.close()
        return 2
    print(f'execution: {execution_id} ({resolution.decision})', flush=True)

    ollama_ver = ollama_version(args.base_url)

    # Warm-ups are bound to THIS loaded session: 2 fresh warmups complete in
    # this invocation before any measured generation. Persisted history is
    # used only for trial numbering, never as proof of warmth.
    fresh_warmups = 0
    for warmup_prompt in WARMUP_PROMPTS:
        _run_warmup_trial(
            conn, experiment_id=experiment_id, config=config,
            config_hash=config_hash, temperature=temperature,
            num_predict=num_predict, warmup_prompt=warmup_prompt,
            base_url=args.base_url, timeout_s=args.timeout_s,
            ollama_ver=ollama_ver,
        )
        fresh_warmups += 1
    if fresh_warmups != len(WARMUP_PROMPTS):
        raise RuntimeError(
            f'measured rows require {len(WARMUP_PROMPTS)} fresh warmups, '
            f'got {fresh_warmups}'
        )

    eligibility = check_eligibility(
        args.base_url, config.ollama_identifier,
        expected_digest=config.ollama_model_digest,
    )
    print(f'eligibility: {eligibility.status} '
          f'(residency={eligibility.gpu_residency_ratio})', flush=True)
    if not eligibility.eligible:
        print('GPU-only rule: refusing to benchmark', flush=True)
        return 3

    freeze = yaml.safe_load(
        open(root / 'evals/specs/eval-v1-grading.freeze.json', encoding='utf-8')
    )
    manifest = build_manifest(
        app_git_commit=git_commit(str(root)),
        eval_freeze_hash=str(freeze['artifacts']['eval-v1-grading.yaml']),
        grading_spec_hash=str(freeze['artifacts']['eval-v1-grading.yaml']),
        dataset_hash=str(freeze['artifacts']['executable-v1.jsonl']),
        experiment_spec_id=experiment_spec_id,
        execution_id=execution_id,
        model_config_id=model_config_id,
        experiment_config_hash=experiment_config_hash,
        model_config_hash=declared_config_hash,
        model_artifact_digest=observed_digest,
        model_artifact_size_bytes=observed_size,
        model_identifier=config.ollama_identifier,
        quantization=config.quantization,
        template_sha256=config.template_sha256,
        temperature=temperature, num_ctx=config.num_ctx,
        num_predict=num_predict, num_gpu=config.num_gpu,
        stop_tokens=config.stop_tokens, think=config.think,
        thinking_source='explicit_config' if config.think is not None else 'model_default',
        reload_evidence_load_duration_ms=RELOAD_EVIDENCE_LOAD_DURATION_MS,
        live=collect_live_environment(
            ollama_version=ollama_ver, started_at_utc=utcnow()),
    )
    manifest_path = results_path(root, 'experiment-manifests', f'{experiment_id}.json')
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(manifest.to_json() + '\n', encoding='utf-8')
    print(f'manifest: {manifest_path.name}', flush=True)

    done = completed_identities(conn, experiment_id, run_kind)
    ran = 0
    skipped = 0
    retry_log_path = results_path(
        root, 'logs', f'transient-retries-{experiment_spec_id}.jsonl'
    )
    retry_events: list[dict[str, Any]] = []
    ollama_bin = resolve_ollama_bin(getattr(args, 'ollama_bin', None))

    def _log_retry(event: dict[str, Any]) -> None:
        record = {
            'timestamp': utcnow(), 'execution_id': experiment_id,
            'model_config_id': args.model, 'stage': 'baseline', **event,
        }
        retry_events.append(record)
        append_retry_event(str(retry_log_path), record)

    def _rewarm_now() -> None:
        nonlocal fresh_warmups
        for warmup_prompt in WARMUP_PROMPTS:
            _run_warmup_trial(
                conn, experiment_id=experiment_id, config=config,
                config_hash=config_hash, temperature=temperature,
                num_predict=num_predict, warmup_prompt=warmup_prompt,
                base_url=args.base_url, timeout_s=args.timeout_s,
                ollama_ver=ollama_ver,
            )
            fresh_warmups += 1

    def _row_pause() -> bool:
        return _pause_if_requested(
            pause_flag=getattr(args, 'pause_flag', None),
            checkpoints_dir=root / 'results/checkpoints',
            experiment_spec=experiment_spec_id,
            base_url=args.base_url,
            ollama_bin=ollama_bin,
            model_identifier=config.ollama_identifier,
        )

    for task_id in task_ids:
        if task_id not in tasks or task_id not in spec_entries:
            print(f'{task_id}: unknown task, skipping', flush=True)
            continue
        entry = spec_entries[task_id]
        rendered = render_prompt(config, tasks[task_id])
        rendered_hash = rendered_prompt_sha256(rendered)
        for trial in range(1, trials + 1):
            if (args.model, task_id, trial, config_hash) in done:
                skipped += 1
                continue
            base = {
                'experiment_id': experiment_id, 'model_config_id': args.model,
                'task_id': task_id, 'trial': trial, 'run_kind': run_kind,
                'run_config_hash': config_hash, 'is_warmup': False,
                'prompt': tasks[task_id], 'rendered_prompt_sha256': rendered_hash,
                'temperature': temperature, 'num_ctx': config.num_ctx,
                'num_predict': num_predict,
                'template_sha256': config.template_sha256,
                'num_gpu': config.num_gpu,
                'stop_tokens': list(config.stop_tokens),
                'think': repr(config.think),
            }
            status = str(entry['grading_status'])
            if status in ('PENDING_SPECIFICATION', 'PENDING_REVIEW'):
                base['started_at_utc'] = utcnow()
                base['ended_at_utc'] = utcnow()
                empty = ProfiledMetrics(
                    ttft_ms=None, client_e2e_ms=0.0,
                    server_total_duration_ms=None,
                    server_load_duration_ms=None,
                    prompt_eval_duration_ms=None, eval_duration_ms=None,
                    prompt_eval_count=None, prompt_eval_cached_count=None,
                    prompt_eval_uncached_count=None, prompt_cache_ratio=None,
                    eval_count=None, prefill_compute_tok_s=None,
                    prefill_cache_state='UNKNOWN', decode_tok_s=None,
                    decode_ms_per_token=None, client_overhead_ms=None,
                )
                insert_measured_row(
                    conn, base=base, text='', thinking='', done_reason=None,
                    metrics=empty, sampler_ram=(None, None),
                    vram_pre=(None, None, None), vram_post_mib=None,
                    verdict=REFUSED_NOT_EXECUTABLE, details=[],
                    status='COMPLETE', error=None,
                    eligibility_status=eligibility.status,
                    residency_ratio=eligibility.gpu_residency_ratio,
                    evidence_json=eligibility.eligibility_evidence_json,
                    ollama_ver=ollama_ver,
                    model_digest=config.ollama_model_digest,
                )
                print(f'{task_id} t{trial}: REFUSED_NOT_EXECUTABLE', flush=True)
                ran += 1
                if _pause_if_requested(
                    pause_flag=getattr(args, 'pause_flag', None),
                    checkpoints_dir=root / 'results/checkpoints',
                    experiment_spec=experiment_spec_id,
                    base_url=args.base_url,
                    ollama_bin=ollama_bin,
                    model_identifier=config.ollama_identifier,
                ):
                    conn.close()
                    return PAUSED_EXIT
                continue
            base['started_at_utc'] = utcnow()
            try:
                result, sampler, post = _generate_with_recovery(
                    make_request=lambda: build_request(
                        args.model, rendered, temperature, num_predict
                    ),
                    task_id=task_id,
                    trial=trial,
                    base_url=args.base_url,
                    timeout_s=args.timeout_s,
                    ollama_bin=ollama_bin,
                    model_identifier=config.ollama_identifier,
                    config=config,
                    temperature=temperature,
                    rewarm=_rewarm_now,
                    log_event=_log_retry,
                )
                sample = sampler
            except MeasurementFailed as exc:
                # Persistent measurement failure: NO row is written (the
                # contract forbids synthetic ERROR rows), weights are kept,
                # and the sweep must STOP. Exit code 6 signals this path.
                print(f'{task_id} t{trial}: MEASUREMENT_FAILED {exc.reason}',
                      flush=True)
                conn.close()
                return MEASUREMENT_FAILED_EXIT
            except OllamaClientError as exc:
                base['ended_at_utc'] = utcnow()
                empty = ProfiledMetrics(
                    ttft_ms=None, client_e2e_ms=0.0,
                    server_total_duration_ms=None,
                    server_load_duration_ms=None,
                    prompt_eval_duration_ms=None, eval_duration_ms=None,
                    prompt_eval_count=None, prompt_eval_cached_count=None,
                    prompt_eval_uncached_count=None, prompt_cache_ratio=None,
                    eval_count=None, prefill_compute_tok_s=None,
                    prefill_cache_state='UNKNOWN', decode_tok_s=None,
                    decode_ms_per_token=None, client_overhead_ms=None,
                )
                insert_measured_row(
                    conn, base=base, text='', thinking='', done_reason=None,
                    metrics=empty, sampler_ram=(None, None),
                    vram_pre=(None, None, None), vram_post_mib=None,
                    verdict=ERROR, details=[], status='ERROR',
                    error=f'{type(exc).__name__}: {exc}',
                    eligibility_status=eligibility.status,
                    residency_ratio=eligibility.gpu_residency_ratio,
                    evidence_json=eligibility.eligibility_evidence_json,
                    ollama_ver=ollama_ver,
                    model_digest=config.ollama_model_digest,
                )
                print(f'{task_id} t{trial}: ERROR {type(exc).__name__}', flush=True)
                ran += 1
                if _row_pause():
                    conn.close()
                    return PAUSED_EXIT
                continue
            base['ended_at_utc'] = utcnow()
            metrics = derive_metrics(result)
            try:
                details = grade_output(result.text, entry['graders'])
                verdict = reduce_verdict(status, details)
            except Exception as exc:
                # Grading must never kill an experiment: record the failure
                # as an ERROR row (retried on resume) and keep going. Any
                # ERROR verdict demands investigation before interpreting
                # results.
                insert_measured_row(
                    conn, base=base, text=result.text,
                    thinking=result.thinking,
                    done_reason=result.done_reason, metrics=metrics,
                    sampler_ram=(sample.ram_baseline_mb, sample.ram_peak_mb),
                    vram_pre=(sample.vram_baseline_mib, sample.vram_peak_mib,
                              sample.vram_total_mib),
                    vram_post_mib=post, verdict=ERROR, details=[],
                    status='ERROR',
                    error=f'grading failure {type(exc).__name__}: {exc}',
                    eligibility_status=eligibility.status,
                    residency_ratio=eligibility.gpu_residency_ratio,
                    evidence_json=eligibility.eligibility_evidence_json,
                    ollama_ver=ollama_ver,
                    model_digest=config.ollama_model_digest,
                )
                print(f'{task_id} t{trial}: ERROR grading {type(exc).__name__}: {exc}',
                      flush=True)
                ran += 1
                if _row_pause():
                    conn.close()
                    return PAUSED_EXIT
                continue
            insert_measured_row(
                conn, base=base, text=result.text, thinking=result.thinking,
                done_reason=result.done_reason, metrics=metrics,
                sampler_ram=(sample.ram_baseline_mb, sample.ram_peak_mb),
                vram_pre=(sample.vram_baseline_mib, sample.vram_peak_mib,
                          sample.vram_total_mib),
                vram_post_mib=post, verdict=verdict, details=details,
                status='COMPLETE', error=None,
                eligibility_status=eligibility.status,
                residency_ratio=eligibility.gpu_residency_ratio,
                evidence_json=eligibility.eligibility_evidence_json,
                ollama_ver=ollama_ver,
                model_digest=config.ollama_model_digest,
            )
            decode = (f'{metrics.decode_tok_s:.1f} tok/s'
                      if metrics.decode_tok_s is not None else 'decode=n/a')
            print(f'{task_id} t{trial}: {verdict} '
                  f'({len(result.text)} chars, {decode}, '
                  f'cache={metrics.prefill_cache_state})', flush=True)
            ran += 1
            if _row_pause():
                conn.close()
                return PAUSED_EXIT
            if needs_rewarm(metrics.server_load_duration_ms):
                # Mid-run reload evidence (eviction): the loaded session is
                # no longer the warmed one. Record 2 fresh warmups before
                # the next measured generation.
                print(f'{task_id} t{trial}: reload evidence '
                      f'(load={metrics.server_load_duration_ms:.0f}ms > '
                      f'{RELOAD_EVIDENCE_LOAD_DURATION_MS:.0f}ms) -> re-warming',
                      flush=True)
                for warmup_prompt in WARMUP_PROMPTS:
                    _run_warmup_trial(
                        conn, experiment_id=experiment_id, config=config,
                        config_hash=config_hash, temperature=temperature,
                        num_predict=num_predict, warmup_prompt=warmup_prompt,
                        base_url=args.base_url, timeout_s=args.timeout_s,
                        ollama_ver=ollama_ver,
                    )
                    fresh_warmups += 1

    measured = fetch_measured(conn, experiment_id)
    verdicts: dict[str, int] = {}
    for row in measured:
        verdicts[row['grader_verdict']] = verdicts.get(row['grader_verdict'], 0) + 1
    retry_by_error: dict[str, int] = {}
    for event in retry_events:
        key = str(event.get('error_type') or 'unknown')
        retry_by_error[key] = retry_by_error.get(key, 0) + 1
    summary = {
        'experiment_spec_id': experiment_spec_id,
        'execution_id': execution_id,
        'model_config_id': args.model,
        'run_kind': run_kind, 'ran_this_invocation': ran,
        'skipped_resume': skipped, 'measured_rows': len(measured),
        'verdicts': verdicts,
        'transient_retries': {
            'events': len(retry_events),
            'by_error': retry_by_error,
            'log': str(retry_log_path),
        },
    }
    summary_path = results_path(root, 'summaries', f'{experiment_id}.json')
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(summary, indent=2), flush=True)
    conn.close()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Run a LocalLLM Lab benchmark')
    parser.add_argument('--config', required=True, help='experiment YAML file')
    parser.add_argument('--model', required=True, help='model config id')
    parser.add_argument('--db', required=True, help='SQLite path (results/local/)')
    parser.add_argument('--base-url', default='http://127.0.0.1:11434')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--execution-id', default=None,
                        help='explicit execution id (default: <spec>__<model>);'
                             ' required with __rerun-NN suffix for fresh reruns')
    parser.add_argument('--num-predict', type=int, default=None,
                        help='explicit override; default comes from the frozen contract yaml')
    parser.add_argument('--timeout-s', type=float, default=300.0)
    parser.add_argument('--max-tasks', type=int, default=None)
    parser.add_argument('--ollama-bin', default=None,
                        help='ollama executable for recovery reload (default: auto-resolve)')
    args = parser.parse_args(argv)
    from storage.pause import PauseFlag

    pause_flag = PauseFlag()
    pause_flag.install_sigint_handler()
    args.pause_flag = pause_flag
    try:
        return run_experiment(args)
    finally:
        pause_flag.uninstall_sigint_handler()


if __name__ == '__main__':
    raise SystemExit(main())
