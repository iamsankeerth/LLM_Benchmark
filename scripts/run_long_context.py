"""Run the standalone full-context long-context-v1 suite."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis.long_context import load_long_context_tasks, summarize_long_context
from evals.graders.engine import grade_output
from evals.verdicts import reduce_verdict
from inference.adapters import get_model_config, render_prompt, rendered_prompt_sha256
from inference.ollama_client import GenerationRequest, GenerationResult, OllamaClientError, generate
from inference.profiler import ProfiledMetrics, derive_metrics
from storage.db import (
    RunRecord,
    connect,
    create_experiment,
    effective_generation_config,
    init_schema,
    insert_run,
    run_config_hash,
)
from storage.execution import ensure_provenance_table, record_execution_provenance
from storage.manifest import hash_experiment_config
from storage.sweep import ollama_stop, resolve_ollama_bin, wait_until_unloaded


class LongContextError(RuntimeError):
    """A long-context run cannot preserve its fixed contract."""


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def _request(model_id: str, prompt: str, *, num_predict: int) -> GenerationRequest:
    config = get_model_config(model_id)
    return GenerationRequest(
        model=config.ollama_identifier,
        prompt=render_prompt(config, prompt),
        temperature=0.0,
        num_ctx=config.num_ctx,
        num_predict=num_predict,
        stop=config.stop_tokens,
        raw=config.mode == 'raw',
        think=config.think,
        num_gpu=config.num_gpu,
    )


def _record(
    *,
    execution_id: str,
    model_id: str,
    task_id: str,
    trial: int,
    run_kind: str,
    is_warmup: bool,
    prompt: str,
    rendered: str,
    result: GenerationResult,
    metrics: ProfiledMetrics,
    config_hash: str,
    verdict: str,
    details: list[dict[str, Any]],
    model_digest: str,
) -> RunRecord:
    return RunRecord(
        experiment_id=execution_id,
        model_config_id=model_id,
        task_id=task_id,
        trial=trial,
        run_kind=run_kind,
        run_config_hash=config_hash,
        is_warmup=is_warmup,
        prompt=prompt,
        rendered_prompt_sha256=rendered_prompt_sha256(rendered),
        temperature=0.0,
        num_ctx=get_model_config(model_id).num_ctx,
        num_predict=256,
        template_sha256=get_model_config(model_id).template_sha256,
        grader_verdict=verdict,
        status='COMPLETE',
        started_at_utc=_utcnow(),
        ended_at_utc=_utcnow(),
        raw_output=result.text,
        thinking_output=result.thinking,
        done_reason=result.done_reason,
        num_gpu=get_model_config(model_id).num_gpu,
        stop_tokens=list(get_model_config(model_id).stop_tokens),
        think=repr(get_model_config(model_id).think),
        ttft_ms=metrics.ttft_ms,
        client_e2e_ms=metrics.client_e2e_ms,
        server_total_duration_ms=metrics.server_total_duration_ms,
        server_load_duration_ms=metrics.server_load_duration_ms,
        prompt_eval_duration_ms=metrics.prompt_eval_duration_ms,
        eval_duration_ms=metrics.eval_duration_ms,
        prompt_eval_count=metrics.prompt_eval_count,
        prompt_eval_cached_count=metrics.prompt_eval_cached_count,
        prompt_eval_uncached_count=metrics.prompt_eval_uncached_count,
        prompt_cache_ratio=metrics.prompt_cache_ratio,
        prefill_compute_tok_s=metrics.prefill_compute_tok_s,
        prefill_cache_state=metrics.prefill_cache_state,
        eval_count=metrics.eval_count,
        decode_tok_s=metrics.decode_tok_s,
        decode_ms_per_token=metrics.decode_ms_per_token,
        client_overhead_ms=metrics.client_overhead_ms,
        grader_details_json=json.dumps(details),
        model_digest=model_digest,
    )


def _wait_model_absent(base_url: str, model_identifier: str) -> bool:
    from inference.ollama_client import fetch_ps
    try:
        document = fetch_ps(base_url)
    except Exception:
        return False
    entries = document.get('models', [])
    return not any(
        isinstance(entry, dict)
        and (entry.get('model') == model_identifier or entry.get('name') == model_identifier)
        for entry in entries
    )


def run_model(
    root: Path,
    *,
    model_id: str,
    db_path: Path,
    base_url: str = 'http://127.0.0.1:11434',
    timeout_s: float = 300.0,
    generate_fn: Callable[..., GenerationResult] = generate,
) -> dict[str, Any]:
    config_document = yaml.safe_load(
        (root / 'configs/long-context-v1.yaml').read_text(encoding='utf-8')
    )
    tasks = load_long_context_tasks(root)
    if len(tasks) != int(config_document['task_count']):
        raise LongContextError('task count does not match frozen config')
    model = get_model_config(model_id)
    execution_id = f'long-context-v1__{model_id}'
    conn = connect(str(db_path))
    init_schema(conn)
    ensure_provenance_table(conn)
    existing = conn.execute(
        'SELECT 1 FROM experiments WHERE experiment_id=?', (execution_id,)
    ).fetchone()
    if existing is not None:
        conn.close()
        raise LongContextError(f'execution already exists: {execution_id}')
    ollama_bin = resolve_ollama_bin(None)
    stopped, detail = ollama_stop(ollama_bin, model.ollama_identifier)
    if not stopped or not wait_until_unloaded(
        lambda: _wait_model_absent(base_url, model.ollama_identifier), timeout_s=180.0
    ):
        conn.close()
        raise LongContextError(f'could not establish cold model state: {detail}')
    try:
        probe = generate_fn(
            base_url,
            _request(model_id, 'OK', num_predict=1),
            timeout_s=timeout_s,
        )
        if probe.prompt_eval_count is None:
            raise LongContextError('canonical probe did not return prompt token count')
        observed_digest = model.ollama_model_digest
        effective = effective_generation_config(
            ollama_identifier=model.ollama_identifier,
            quantization=model.quantization,
            mode=model.mode,
            temperature=0.0,
            num_ctx=model.num_ctx,
            num_predict=256,
            num_gpu=model.num_gpu,
            stop_tokens=list(model.stop_tokens),
            think=model.think,
            template_sha256=model.template_sha256,
        )
        config_hash = run_config_hash(effective)
        experiment_config = {
            'experiment_spec_id': 'long-context-v1',
            'dataset_sha256': hashlib.sha256(
                (root / 'evals/datasets/long-context-v1/tasks.jsonl').read_bytes()
            ).hexdigest(),
            'grading_spec_sha256': hashlib.sha256(
                (root / 'evals/specs/long-context-v1-grading.yaml').read_bytes()
            ).hexdigest(),
            'task_ids': [task.task_id for task in tasks],
            'trials': 3,
            'num_predict': 256,
        }
        experiment_config_hash = hash_experiment_config(experiment_config)
        create_experiment(
            conn, experiment_id=execution_id, name=execution_id,
            config_hash=config_hash,
            config_yaml=(root / 'configs/long-context-v1.yaml').read_text(encoding='utf-8'),
            created_at_utc=_utcnow(),
        )
        record_execution_provenance(
            conn, execution_id=execution_id, experiment_spec_id='long-context-v1',
            model_config_id=model_id, experiment_config_hash=experiment_config_hash,
            model_config_hash=hashlib.sha256(
                json.dumps(model.__dict__, sort_keys=True, default=str).encode('utf-8')
            ).hexdigest(),
            model_artifact_digest=observed_digest, created_at_utc=_utcnow(),
        )
        for task in tasks:
            task_probe = generate_fn(
                base_url, _request(model_id, task.prompt, num_predict=1),
                timeout_s=timeout_s,
            )
            if task_probe.prompt_eval_count is None:
                raise LongContextError(f'{task.task_id}: missing prompt token count')
            if task_probe.prompt_eval_count + 256 > model.num_ctx:
                raise LongContextError(
                    f'{task.task_id}: prompt headroom is insufficient'
                )
        for trial, (prompt, verdict) in enumerate(
            (('Return only OK.', 'WARMUP'), ('Return only 7.', 'WARMUP')), start=1
        ):
            rendered = render_prompt(model, prompt)
            result = generate_fn(
                base_url, _request(model_id, prompt, num_predict=256),
                timeout_s=timeout_s,
            )
            metrics = derive_metrics(result)
            insert_run(conn, _record(
                execution_id=execution_id, model_id=model_id,
                task_id=f'WARMUP-{trial}', trial=trial, run_kind='WARMUP',
                is_warmup=True, prompt=prompt, rendered=rendered, result=result,
                metrics=metrics, config_hash=config_hash, verdict=verdict, details=[],
                model_digest=observed_digest,
            ))
            conn.commit()
        measured_rows: list[dict[str, Any]] = []
        for task in tasks:
            rendered = render_prompt(model, task.prompt)
            for trial in range(1, 4):
                result = generate_fn(
                    base_url, _request(model_id, task.prompt, num_predict=256),
                    timeout_s=timeout_s,
                )
                if result.prompt_eval_count is not None and result.prompt_eval_count + 256 > model.num_ctx:
                    raise LongContextError(f'{task.task_id}#{trial}: measured prompt exceeded context')
                metrics = derive_metrics(result)
                results = grade_output(result.text, list(task.graders))
                verdict = reduce_verdict('READY_DETERMINISTIC', results)
                details = [
                    {'grader_type': r.grader_type, 'passed': r.passed,
                     'detail': r.detail, 'violations': r.violations}
                    for r in results
                ]
                insert_run(conn, _record(
                    execution_id=execution_id, model_id=model_id,
                    task_id=task.task_id, trial=trial, run_kind='LONG_CONTEXT',
                    is_warmup=False, prompt=task.prompt, rendered=rendered,
                    result=result, metrics=metrics, config_hash=config_hash,
                    verdict=verdict, details=details, model_digest=observed_digest,
                ))
                conn.commit()
                measured_rows.append({
                    'task_id': task.task_id, 'trial': trial,
                    'grader_verdict': verdict,
                })
        summary = summarize_long_context(measured_rows, tasks)
        summary.update({
            'execution_id': execution_id,
            'model_config_id': model_id,
            'model_digest': observed_digest,
            'request_accounting': {
                'canonical_probe': 1,
                'task_preflight_probes': len(tasks),
                'warmups': 2,
                'measured': len(tasks) * 3,
                'total': 1 + len(tasks) + 2 + len(tasks) * 3,
            },
        })
        return summary
    except Exception:
        ollama_stop(ollama_bin, model.ollama_identifier)
        raise
    finally:
        conn.close()


def main(argv: list[str] | None = None) -> int:
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description='Run long-context-v1')
    parser.add_argument('--model', required=True)
    parser.add_argument('--db', required=True)
    parser.add_argument('--base-url', default='http://127.0.0.1:11434')
    parser.add_argument('--timeout-s', type=float, default=300.0)
    args = parser.parse_args(argv)
    try:
        result = run_model(
            root, model_id=args.model, db_path=Path(args.db),
            base_url=args.base_url, timeout_s=args.timeout_s,
        )
    except (LongContextError, OllamaClientError, OSError, ValueError) as exc:
        print(f'LONG-CONTEXT REFUSED: {exc}', flush=True)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
