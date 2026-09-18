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
    get_model_config,
    model_config_hash,
    render_prompt,
    rendered_prompt_sha256,
)
from inference.eligibility import check_eligibility
from inference.ollama_client import (
    GenerationRequest,
    OllamaClientError,
    generate,
)
from inference.profiler import ProfiledMetrics, derive_metrics
from inference.sysmon import SystemSampler, sample_vram_once
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

WARMUP_PROMPTS = ('Return only the word OK.', 'Return only the digit 7.')
WARMUP_VERDICT = 'WARMUP'


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

    # Warm-ups: 2 per invocation (per model load), continuing trial numbers
    # so reruns never collide on the WARMUP identity.
    warmup_done = conn.execute(
        "SELECT COUNT(*) FROM runs WHERE experiment_id=? AND model_config_id=?"
        " AND run_kind='WARMUP'",
        (experiment_id, args.model),
    ).fetchone()[0]
    for offset, warmup_prompt in enumerate(WARMUP_PROMPTS, start=1):
        trial_no = int(warmup_done) + offset
        started = utcnow()
        rendered = render_prompt(config, warmup_prompt)
        with SystemSampler() as sampler:
            result = generate(
                args.base_url,
                build_request(args.model, rendered, temperature, num_predict),
                timeout_s=args.timeout_s,
            )
        metrics = derive_metrics(result)
        sample = sampler.sample()
        post, _ = sample_vram_once()
        insert_measured_row(
            conn,
            base={
                'experiment_id': experiment_id, 'model_config_id': args.model,
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
        live=collect_live_environment(
            ollama_version=ollama_ver, started_at_utc=utcnow()),
    )
    manifest_path = root / 'results/experiment-manifests' / f'{experiment_id}.json'
    manifest_path.write_text(manifest.to_json() + '\n', encoding='utf-8')
    print(f'manifest: {manifest_path.name}', flush=True)

    done = completed_identities(conn, experiment_id, run_kind)
    ran = 0
    skipped = 0
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
                continue
            base['started_at_utc'] = utcnow()
            try:
                with SystemSampler() as sampler:
                    result = generate(
                        args.base_url,
                        build_request(args.model, rendered, temperature,
                                      num_predict),
                        timeout_s=args.timeout_s,
                    )
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
                continue
            base['ended_at_utc'] = utcnow()
            metrics = derive_metrics(result)
            sample = sampler.sample()
            post, _ = sample_vram_once()
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

    measured = fetch_measured(conn, experiment_id)
    verdicts: dict[str, int] = {}
    for row in measured:
        verdicts[row['grader_verdict']] = verdicts.get(row['grader_verdict'], 0) + 1
    summary = {
        'experiment_spec_id': experiment_spec_id,
        'execution_id': execution_id,
        'model_config_id': args.model,
        'run_kind': run_kind, 'ran_this_invocation': ran,
        'skipped_resume': skipped, 'measured_rows': len(measured),
        'verdicts': verdicts,
    }
    summary_path = root / 'results/summaries' / f'{experiment_id}.json'
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
    args = parser.parse_args(argv)
    return run_experiment(args)


if __name__ == '__main__':
    raise SystemExit(main())
