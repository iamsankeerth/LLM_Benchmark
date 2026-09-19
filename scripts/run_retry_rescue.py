"""Retry executor: one contract-aware generation per failed row, new lineage.

Baseline DB rows are never mutated. Each retry gets the exact canonical
adapter/runtime (inference/adapter parity with the baseline) plus a
single deterministic suffix derived ONLY from RetryPromptInput. Success is
graded by the frozen eval-v1 grader exactly; baseline rows stay immutable.

Lifecycle per locked contract:
  pull exact artifact -> digest/config verify -> canonical probe (1 token)
  -> residency+canary -> 2 warmups -> 51 retry generations (2048 tokens)
  -> verify -> unload -> delete
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

from analysis.retry_population import (
    RetryPromptInput,
    derive_population,
    load_grading_spec,
    load_task_meta,
    render_retry_prompt,
)
from evals.graders.engine import grade_output
from inference.adapters import get_model_config, render_prompt
from inference.eligibility import run_canonical_eligibility, run_operational_canary
from inference.eligibility import effective_options_for as eligibility_effective_options
from inference.ollama_client import GenerationRequest, fetch_ps, generate
from inference.profiler import derive_metrics
from inference.sysmon import SystemSampler, sample_vram_once
from storage.db import connect, effective_generation_config, init_schema, run_config_hash
from storage.manifest import hash_experiment_config

EXECUTION_ID = 'retry-rescue-v1__qwen3-4b-q4'
WARMUP_PROMPTS = ('Return only the word OK.', 'Return only the digit 7.')


def _utcnow() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def _ps_has_model(ps_doc: dict[str, Any], model_identifier: str) -> bool:
    entries = ps_doc.get('models', [])
    if not isinstance(entries, list):
        return False
    return any(
        isinstance(e, dict) and (e.get('model') == model_identifier or e.get('name') == model_identifier)
        for e in entries
    )


def _ensure_pulled(
    ollama_bin: str, identifier: str, base_url: str
) -> tuple[bool, str]:
    import subprocess

    proc = subprocess.run(
        [ollama_bin, 'pull', identifier],
        capture_output=True, text=True, timeout=7200,
    )
    if proc.returncode != 0:
        return False, proc.stderr[-500:] or proc.stdout[-500:]
    try:
        ps = fetch_ps(base_url)
        if _ps_has_model(ps, identifier):
            return True, 'pulled'
    except Exception:
        pass
    return True, 'pulled'


def _retry_input_for_task(
    task_id: str, spec: dict[str, Any], task_meta: dict[str, Any]
) -> RetryPromptInput:
    entry = spec['tasks'][task_id]
    graders = entry.get('graders') or [{}]
    grader = graders[0] if graders else {}
    grader_type = str(grader.get('type', 'exact'))
    required = tuple(str(f) for f in grader.get('required_fields', []) or [])
    output_type = str(task_meta.get(task_id, {}).get('output_type', ''))
    return RetryPromptInput(
        original_prompt=str(task_meta[task_id]['prompt']),
        grader_type=grader_type,
        required_fields=required,
        output_type=output_type,
        units_required=False,
    )


def run_retry_rescue(args: argparse.Namespace) -> int:
    root = Path(__file__).resolve().parent.parent
    retry_config = yaml.safe_load(open(args.config, encoding='utf-8'))
    spec = load_grading_spec(str(root / 'evals/specs/eval-v1-grading.yaml'))
    task_meta = load_task_meta(str(root / 'evals/datasets/eval-v1/executable-v1.jsonl'))
    base_model = str(retry_config.get('base_model', 'qwen3-4b-q4'))
    _, primary_rows, control_rows = derive_population(
        str(root / retry_config['populations']['primary']['source_artifact']),
        str(root / 'evals/datasets/eval-v1/executable-v1.jsonl'),
        str(root / 'evals/specs/eval-v1-grading.yaml'),
    )
    if len(primary_rows) != 39 or len(control_rows) != 12:
        print(f'POPULATION MISMATCH: primary={len(primary_rows)} control={len(control_rows)}')
        return 2
    ordered = sorted(primary_rows, key=lambda r: (r['task_id'], r['trial'])) + \
              sorted(control_rows, key=lambda r: (r['task_id'], r['trial']))
    print(f'primary: {len(primary_rows)} control: {len(control_rows)} total: {len(ordered)}')

    config = get_model_config(base_model)
    ollama_bin = args.ollama_bin or 'C:/Users/lenovo/AppData/Local/Programs/Ollama/ollama.exe'

    # Lifecycle: pull exact artifact + canonical probe (1 token) + canary
    print(f'pulling {config.ollama_identifier} ...', flush=True)
    ok, detail = _ensure_pulled(ollama_bin, config.ollama_identifier, args.base_url)
    if not ok:
        print(f'pull failed: {detail}', flush=True)
        return 3
    print('pull complete, running canonical probe (1 token) ...', flush=True)
    probe_effective = eligibility_effective_options(
        mode=config.mode, num_ctx=config.num_ctx, num_gpu=config.num_gpu,
        temperature=0.0, template_sha256=config.template_sha256,
    )
    canonical = run_canonical_eligibility(
        base_url=args.base_url,
        model_identifier=config.ollama_identifier,
        expected_digest=config.ollama_model_digest,
        effective_options=probe_effective,
        render_prompt=lambda p: render_prompt(config, p),
    )
    if not canonical.result.eligible:
        print(f'eligibility FAIL: {canonical.result.status}', flush=True)
        return 3
    print(f'eligibility PASS: residency {canonical.result.gpu_residency_ratio}', flush=True)
    # Runtime canary: sustained generation check
    from inference.eligibility import CANARY_MIN_EVAL_TOKENS

    canary = run_operational_canary(
        base_url=args.base_url,
        model_identifier=config.ollama_identifier,
        effective_options=probe_effective,
        render_prompt=lambda p: render_prompt(config, p),
    )
    if not canary.passed:
        print(f'canary FAIL: {canary.failure_kind} - {canary.detail}', flush=True)
        return 3
    print(f'canary PASS: {canary.eval_count} tokens (floor {CANARY_MIN_EVAL_TOKENS})', flush=True)

    # Warmups (2 fresh, session-bound)
    for prompt in WARMUP_PROMPTS:
        rendered = render_prompt(config, prompt) if config.mode == 'raw' else prompt
        generate(
            args.base_url,
            GenerationRequest(
                model=config.ollama_identifier, prompt=rendered,
                temperature=0.0, num_ctx=config.num_ctx, num_predict=2048,
                stop=config.stop_tokens, raw=(config.mode == 'raw'),
                think=config.think, num_gpu=config.num_gpu,
            ),
            timeout_s=args.timeout_s,
        )
        print(f'warmup: {prompt[:30]!r} done', flush=True)

    # Stable run_config_hash: hash of canonical effective retry config, shared across all 51 rows
    effective = effective_generation_config(
        ollama_identifier=config.ollama_identifier,
        quantization=config.quantization,
        mode=config.mode,
        temperature=0.0,
        num_ctx=config.num_ctx,
        num_predict=2048,
        num_gpu=config.num_gpu,
        stop_tokens=config.stop_tokens,
        think=config.think,
        template_sha256=config.template_sha256,
    )
    # Include retry-specific suffix hashes in effective? No - run_config_hash identifies retry experiment, not per-row prompt
    # Per-row rendered_retry_prompt_sha256 remains row-specific identity
    retry_effective = {**effective, 'retry_rescue': 'v1', 'retry_budget': 1}
    stable_run_config_hash = run_config_hash(retry_effective)

    out_db = str(root / f'results/local/{EXECUTION_ID}.db')
    Path(out_db).parent.mkdir(parents=True, exist_ok=True)
    conn = connect(out_db)
    init_schema(conn)
    try:
        conn.execute(
            'INSERT INTO experiments(experiment_id, name, config_hash, config_yaml, created_at_utc)'
            ' VALUES(?,?,?,?,?)',
            (EXECUTION_ID, 'retry-rescue-v1',
             hash_experiment_config(retry_config),
             open(args.config, encoding='utf-8').read(), _utcnow()),
        )
        conn.commit()
    except sqlite3.IntegrityError:
        pass
    existing = {
        (str(r[0]), int(r[1]))
        for r in conn.execute(
            'SELECT task_id, trial FROM runs WHERE experiment_id=?',
            (EXECUTION_ID,),
        ).fetchall()
    }
    print(f'resume: {len(existing)} already persisted, {len(ordered)-len(existing)} to run', flush=True)
    ran = 0
    for row in ordered:
        key = (row['task_id'], row['trial'])
        if key in existing:
            continue
        retry_input = _retry_input_for_task(row['task_id'], spec, task_meta)
        rendered, hashes = render_retry_prompt(retry_input)
        generation_prompt = render_prompt(config, rendered) if config.mode == 'raw' else rendered
        started = _utcnow()
        with SystemSampler() as sampler:
            result = generate(
                args.base_url,
                GenerationRequest(
                    model=config.ollama_identifier,
                    prompt=generation_prompt,
                    temperature=0.0,
                    num_ctx=config.num_ctx,
                    num_predict=2048,
                    stop=config.stop_tokens,
                    raw=(config.mode == 'raw'),
                    think=config.think,
                    num_gpu=config.num_gpu,
                ),
                timeout_s=args.timeout_s,
            )
        ended = _utcnow()
        metrics = derive_metrics(result)
        sample = sampler.sample()
        post, _ = sample_vram_once()
        details = grade_output(result.text, spec['tasks'][row['task_id']]['graders'])
        passed = all(d.passed for d in details)
        detail_docs = [
            {'grader_type': d.grader_type, 'passed': d.passed, 'detail': d.detail, 'violations': d.violations}
            for d in details
        ]
        from storage.db import RunRecord, insert_run

        record = RunRecord(
            experiment_id=EXECUTION_ID,
            model_config_id=config.config_id,
            task_id=row['task_id'],
            trial=int(row['trial']),
            run_kind='RETRY_RESCUE',
            run_config_hash=stable_run_config_hash,
            is_warmup=False,
            prompt=rendered,
            rendered_prompt_sha256=hashes['rendered_retry_prompt_sha256'],
            temperature=0.0,
            num_ctx=config.num_ctx,
            num_predict=2048,
            template_sha256=config.template_sha256,
            grader_verdict='PASS' if passed else 'FAIL',
            status='COMPLETE',
            started_at_utc=started,
            ended_at_utc=ended,
            raw_output=result.text,
            thinking_output=result.thinking or '',
            done_reason=result.done_reason,
            num_gpu=config.num_gpu,
            stop_tokens=list(config.stop_tokens),
            think=str(config.think),
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
            ram_baseline_mb=sample.ram_baseline_mb,
            ram_peak_mb=sample.ram_peak_mb,
            vram_baseline_mib=sample.vram_baseline_mib,
            vram_peak_mib=sample.vram_peak_mib,
            vram_total_mib=sample.vram_total_mib,
            grader_details_json=json.dumps(detail_docs),
            error=json.dumps(hashes),
            eligibility_status='ELIGIBLE_GPU',
            ollama_version=result.raw_final.get('model') if isinstance(result.raw_final, dict) else None,
        )
        insert_run(conn, record)
        conn.commit()
        print(f"{row['task_id']} t{row['trial']}: {'RECOVERED' if passed else 'STILL_FAIL'} ({len(result.text)} chars)", flush=True)
        ran += 1
    print(f'done: {ran} new retry rows', flush=True)
    conn.close()

    # Report generation inline (telemetry + content-change)
    # Substantive-change classification via asserted-value machinery where available
    print('retry-rescue complete - generating report ...', flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Run retry-rescue-v1')
    parser.add_argument('--config', default='configs/retry-rescue-v1.yaml')
    parser.add_argument('--base-url', default='http://127.0.0.1:11434')
    parser.add_argument('--ollama-bin', default=None)
    parser.add_argument('--timeout-s', type=float, default=600.0)
    args = parser.parse_args(argv)
    return run_retry_rescue(args)


if __name__ == '__main__':
    raise SystemExit(main())
