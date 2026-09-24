"""Execute retry-rescue-v2 constrained decoding after explicit authorization.

This runner is intentionally separate from V1: C uses the sealed original
prompt and an Ollama response-format constraint, never a retry suffix.
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

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis.retry_population import load_grading_spec, load_task_meta
from analysis.retry_rescue_v2 import (
    V2Contract,
    constraint_sha256,
    load_v2_contract,
    reject_retry_suffix,
    v2_run_config_hash,
    validate_c_resume,
)
from evals.graders.engine import grade_output
from inference.adapters import get_model_config, model_config_hash, render_prompt, rendered_prompt_sha256
from inference.eligibility import effective_options_for, run_canonical_eligibility, run_operational_canary
from inference.ollama_client import GenerationRequest, generate
from inference.profiler import derive_metrics
from inference.sysmon import SystemSampler, sample_vram_once
from scripts.probe_retry_rescue_v2 import _teardown
from storage.db import (
    RunRecord,
    connect,
    create_experiment,
    init_schema,
    insert_run,
    migrate_v2_run_columns,
)
from storage.sweep import disk_free_bytes, ollama_model_present, ollama_pull, resolve_ollama_bin


ROOT = Path(__file__).resolve().parent.parent
WARMUP_PROMPTS = ('Return only the word OK.', 'Return only the digit 7.')


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


def _readonly_baseline_prompts(
    db_path: Path, execution_id: str, identities: set[tuple[str, int]],
) -> dict[tuple[str, int], tuple[str, str, str]]:
    uri = f'file:{db_path.resolve().as_posix()}?mode=ro'
    conn = sqlite3.connect(uri, uri=True)
    try:
        rows = conn.execute(
            'SELECT task_id, trial, prompt, rendered_prompt_sha256, template_sha256 '
            'FROM runs WHERE experiment_id=? AND is_warmup=0', (execution_id,),
        ).fetchall()
    finally:
        conn.close()
    selected = {
        (str(task_id), int(trial)): (str(prompt), str(rendered_hash), str(template_hash))
        for task_id, trial, prompt, rendered_hash, template_hash in rows
        if (str(task_id), int(trial)) in identities
    }
    if set(selected) != identities:
        raise ValueError('sealed A prompt identity mismatch')
    return selected


def verify_prompt_invariance(
    *, contract: V2Contract, task_meta: dict[str, Any], config: Any,
    baseline_db: Path, baseline_execution: str,
) -> dict[tuple[str, int], tuple[str, str, str]]:
    """Prove C starts from the same source prompt/template as sealed A."""
    identities = set(contract.primary_identities + contract.control_identities)
    baseline = _readonly_baseline_prompts(baseline_db, baseline_execution, identities)
    expected: dict[tuple[str, int], tuple[str, str, str]] = {}
    for task_id, trial in sorted(identities):
        prompt = str(task_meta[task_id]['prompt'])
        reject_retry_suffix(prompt)
        rendered = render_prompt(config, prompt) if config.mode == 'raw' else prompt
        expected_row = (prompt, rendered_prompt_sha256(rendered), config.template_sha256)
        if baseline[(task_id, trial)] != expected_row:
            raise ValueError(f'{task_id} t{trial}: prompt invariance mismatch against sealed A')
        expected[(task_id, trial)] = expected_row
    return expected


def _record(
    *, execution_id: str, config: Any, task_id: str, trial: int, prompt: str,
    rendered_hash: str, original_hash: str, run_hash: str, verdict: str, status: str,
    started: str, ended: str, result: Any | None, response_kind: str | None,
    response_hash: str | None, details: list[dict[str, Any]], sampler: Any | None,
    post_vram: float | None,
) -> RunRecord:
    metrics = derive_metrics(result) if result is not None else None
    return RunRecord(
        experiment_id=execution_id, model_config_id=config.config_id, task_id=task_id,
        trial=trial, run_kind='WARMUP' if response_kind is None else 'RETRY_RESCUE',
        run_config_hash=run_hash, is_warmup=response_kind is None, prompt=prompt,
        rendered_prompt_sha256=rendered_hash, original_prompt_sha256=original_hash,
        response_format_kind=response_kind, response_format_sha256=response_hash,
        temperature=0.0, num_ctx=config.num_ctx, num_predict=2048,
        template_sha256=config.template_sha256, grader_verdict=verdict, status=status,
        started_at_utc=started, ended_at_utc=ended,
        raw_output='' if result is None else result.text,
        thinking_output='' if result is None else result.thinking,
        done_reason=None if result is None else result.done_reason,
        num_gpu=config.num_gpu, stop_tokens=list(config.stop_tokens), think=str(config.think),
        grader_details_json=json.dumps(details),
        ttft_ms=None if metrics is None else metrics.ttft_ms,
        client_e2e_ms=None if metrics is None else metrics.client_e2e_ms,
        server_total_duration_ms=None if metrics is None else metrics.server_total_duration_ms,
        server_load_duration_ms=None if metrics is None else metrics.server_load_duration_ms,
        prompt_eval_count=None if metrics is None else metrics.prompt_eval_count,
        prompt_eval_cached_count=None if metrics is None else metrics.prompt_eval_cached_count,
        prompt_eval_uncached_count=None if metrics is None else metrics.prompt_eval_uncached_count,
        prompt_cache_ratio=None if metrics is None else metrics.prompt_cache_ratio,
        prompt_eval_duration_ms=None if metrics is None else metrics.prompt_eval_duration_ms,
        eval_count=None if metrics is None else metrics.eval_count,
        eval_duration_ms=None if metrics is None else metrics.eval_duration_ms,
        prefill_compute_tok_s=None if metrics is None else metrics.prefill_compute_tok_s,
        prefill_cache_state=None if metrics is None else metrics.prefill_cache_state,
        decode_tok_s=None if metrics is None else metrics.decode_tok_s,
        decode_ms_per_token=None if metrics is None else metrics.decode_ms_per_token,
        client_overhead_ms=None if metrics is None else metrics.client_overhead_ms,
        ram_baseline_mb=None if sampler is None else sampler.ram_baseline_mb,
        ram_peak_mb=None if sampler is None else sampler.ram_peak_mb,
        vram_baseline_mib=None if sampler is None else sampler.vram_baseline_mib,
        vram_peak_mib=None if sampler is None else sampler.vram_peak_mib,
        vram_post_mib=post_vram,
        vram_total_mib=None if sampler is None else sampler.vram_total_mib,
        model_digest=config.ollama_model_digest,
    )


def run_v2(args: argparse.Namespace) -> int:
    config_path = ROOT / args.config
    run_config = yaml.safe_load(config_path.read_text(encoding='utf-8'))
    if not isinstance(run_config, dict):
        raise ValueError('V2 config must be a mapping')
    contract = load_v2_contract(str(config_path), str(ROOT / run_config['capability_matrix']))
    config = get_model_config(str(run_config['base_model']))
    spec_path = ROOT / 'evals/specs/eval-v1-grading.yaml'
    dataset_path = ROOT / 'evals/datasets/eval-v1/executable-v1.jsonl'
    spec_hash = hashlib.sha256(spec_path.read_bytes()).hexdigest()
    run_hash = v2_run_config_hash(
        model_config_id=config.config_id, model_config_sha256=model_config_hash(config),
        temperature=0.0, num_ctx=config.num_ctx,
        num_predict=int(run_config['num_predict']), template_sha256=config.template_sha256,
        grader_spec_sha256=spec_hash, population_sha256=contract.population_sha256,
        format_mapping_sha256=contract.format_mapping_sha256,
    )
    task_meta = load_task_meta(str(dataset_path))
    prompts = verify_prompt_invariance(
        contract=contract, task_meta=task_meta, config=config,
        baseline_db=ROOT / 'results/local/full-baseline-v2__qwen3-4b-q4.db',
        baseline_execution=str(run_config['base_execution']),
    )
    expected = set(contract.primary_identities + contract.control_identities)
    out_db = ROOT / 'results/local/retry-rescue-v2__qwen3-4b-q4.db'
    ollama_bin = resolve_ollama_bin(args.ollama_bin)
    free_before = disk_free_bytes(ROOT)
    pulled = False
    conn: sqlite3.Connection | None = None
    try:
        ok, detail = ollama_pull(ollama_bin, config.ollama_identifier)
        if not ok or not ollama_model_present(ollama_bin, config.ollama_identifier):
            raise RuntimeError(f'pull failed: {detail}')
        pulled = True
        options = effective_options_for(
            mode=config.mode, num_ctx=config.num_ctx,
            num_gpu=config.num_gpu, temperature=0.0,
            template_sha256=config.template_sha256,
            stop_tokens=tuple(config.stop_tokens), think=config.think,
        )
        eligibility = run_canonical_eligibility(
            base_url=args.base_url, model_identifier=config.ollama_identifier,
            expected_digest=config.ollama_model_digest, effective_options=options,
            render_prompt=lambda prompt: render_prompt(config, prompt), timeout_s=args.timeout_s,
        )
        if not eligibility.result.eligible:
            raise RuntimeError(f'eligibility failed: {eligibility.result.status}')
        canary = run_operational_canary(
            base_url=args.base_url, model_identifier=config.ollama_identifier,
            effective_options=eligibility.effective_options,
            render_prompt=lambda prompt: render_prompt(config, prompt), timeout_s=args.timeout_s,
        )
        if not canary.passed:
            raise RuntimeError(f'canary failed: {canary.failure_kind}')
        conn = connect(str(out_db))
        init_schema(conn)
        migrate_v2_run_columns(conn)
        try:
            create_experiment(conn, experiment_id=str(run_config['execution_id']),
                              name='retry-rescue-v2', config_hash=run_hash,
                              config_yaml=config_path.read_text(encoding='utf-8'),
                              created_at_utc=_utcnow())
        except sqlite3.IntegrityError:
            pass
        completed = validate_c_resume(conn, str(run_config['execution_id']), expected,
                                      run_hash, contract.formats)
        warmup_count = int(conn.execute(
            'SELECT COUNT(*) FROM runs WHERE experiment_id=? AND is_warmup=1',
            (str(run_config['execution_id']),),
        ).fetchone()[0])
        if warmup_count not in (0, len(WARMUP_PROMPTS)):
            raise ValueError('partial V2 warmup set prevents resume')
        for index, prompt in enumerate(WARMUP_PROMPTS, start=1) if warmup_count == 0 else ():
            rendered = render_prompt(config, prompt) if config.mode == 'raw' else prompt
            started = _utcnow()
            result = generate(args.base_url, GenerationRequest(
                model=config.ollama_identifier, prompt=rendered, temperature=0.0,
                num_ctx=config.num_ctx, num_predict=2048, stop=config.stop_tokens,
                raw=config.mode == 'raw', think=config.think, num_gpu=config.num_gpu,
            ), timeout_s=args.timeout_s)
            insert_run(conn, _record(
                execution_id=str(run_config['execution_id']), config=config,
                task_id=f'__warmup_{index}', trial=1, prompt=prompt,
                rendered_hash=rendered_prompt_sha256(rendered), original_hash=_sha256_text(prompt),
                run_hash=run_hash, verdict='PASS', status='COMPLETE', started=started,
                ended=_utcnow(), result=result, response_kind=None, response_hash=None,
                details=[], sampler=None, post_vram=None,
            ))
            conn.commit()
        spec = load_grading_spec(str(spec_path))
        for task_id, trial in sorted(expected):
            if (task_id, trial) in completed:
                continue
            prompt, rendered_hash, _ = prompts[(task_id, trial)]
            response_kind, response_format = contract.formats[task_id]
            rendered = render_prompt(config, prompt) if config.mode == 'raw' else prompt
            started = _utcnow()
            with SystemSampler() as sampler:
                result = generate(args.base_url, GenerationRequest(
                    model=config.ollama_identifier, prompt=rendered, temperature=0.0,
                    num_ctx=config.num_ctx, num_predict=2048, stop=config.stop_tokens,
                    raw=config.mode == 'raw', think=config.think, num_gpu=config.num_gpu,
                    format=response_format,
                ), timeout_s=args.timeout_s)
            post_vram, _ = sample_vram_once()
            details = grade_output(result.text, spec['tasks'][task_id]['graders'])
            detail_docs = [
                {'grader_type': detail.grader_type, 'passed': detail.passed,
                 'detail': detail.detail, 'violations': detail.violations}
                for detail in details
            ]
            insert_run(conn, _record(
                execution_id=str(run_config['execution_id']), config=config,
                task_id=task_id, trial=trial, prompt=prompt, rendered_hash=rendered_hash,
                original_hash=_sha256_text(prompt), run_hash=run_hash,
                verdict='PASS' if all(detail.passed for detail in details) else 'FAIL',
                status='COMPLETE', started=started, ended=_utcnow(), result=result,
                response_kind=response_kind, response_hash=constraint_sha256(response_format),
                details=detail_docs, sampler=sampler.sample(), post_vram=post_vram,
            ))
            conn.commit()
        if validate_c_resume(conn, str(run_config['execution_id']), expected,
                             run_hash, contract.formats) != expected:
            raise RuntimeError('C persistence verification failed')
        return 0
    finally:
        if conn is not None:
            conn.close()
        if pulled:
            cleanup = _teardown(args.base_url, ollama_bin, config.ollama_identifier, free_before)
            if cleanup['cleanup_status'] != 'MODEL_STATE_VERIFIED':
                raise RuntimeError(f'cleanup failed: {cleanup}')


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/retry-rescue-v2.yaml')
    parser.add_argument('--base-url', default='http://127.0.0.1:11434')
    parser.add_argument('--ollama-bin', default=None)
    parser.add_argument('--timeout-s', type=float, default=600.0)
    return run_v2(parser.parse_args(argv))


if __name__ == '__main__':
    raise SystemExit(main())
