"""Run Coding v1 model outputs through the isolated OCI boundary."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis.coding import extract_code, load_fixture_manifest, static_check
from inference.adapters import get_model_config, render_prompt
from inference.ollama_client import GenerationRequest, GenerationResult, generate
from inference.sandbox import SandboxRequest, SandboxUnavailable, run_isolated_tests
from storage.coding import open_coding_db, record_coding_run


class CodingRunError(RuntimeError):
    """Coding v1 cannot run without pinned isolated worker images."""


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


def _runtime(manifest: dict[str, Any], language: str) -> tuple[str, str]:
    runtime = manifest['runtimes']['python' if language == 'python' else 'node']
    digest = runtime.get('digest')
    if not isinstance(digest, str) or not digest:
        raise CodingRunError(
            f'{language} runtime image digest is not pinned; live run refused'
        )
    return str(runtime['reference']), digest


def run_coding_model(
    root: Path,
    *,
    model_id: str,
    db_path: Path,
    generate_fn: Callable[..., GenerationResult] = generate,
    sandbox_runner: Callable[..., dict[str, Any]] = run_isolated_tests,
    base_url: str = 'http://127.0.0.1:11434',
    timeout_s: float = 300.0,
) -> dict[str, Any]:
    manifest = load_fixture_manifest(
        root / 'evals/fixtures/coding-v1/fixture-manifest.json'
    )
    extraction_policy = json.loads(
        (root / 'evals/fixtures/coding-v1/extraction-policy.json').read_text(encoding='utf-8')
    )
    resource_policy = json.loads(
        (root / 'evals/fixtures/coding-v1/resource-policy.json').read_text(encoding='utf-8')
    )
    _runtime(manifest, 'python')
    _runtime(manifest, 'javascript')
    execution_id = f'coding-v1__{model_id}'
    conn = open_coding_db(db_path)
    counts = {'rows': 0, 'passes': 0, 'static_failures': 0, 'sandbox_errors': 0}
    try:
        for task_id in sorted(manifest['tasks']):
            task = manifest['tasks'][task_id]
            language = str(task['language'])
            entrypoint = str(task['entrypoint'])
            image_ref, image_digest = _runtime(manifest, language)
            test_path = root / 'evals/fixtures/coding-v1' / task['test_path']
            cases_path = root / 'evals/fixtures/coding-v1' / task['fixture_path']
            test_source = test_path.read_text(encoding='utf-8')
            cases_json = cases_path.read_text(encoding='utf-8')
            for trial in range(1, 4):
                started = _utcnow()
                result = generate_fn(
                    base_url,
                    _request(model_id, task_id, root),
                    timeout_s=timeout_s,
                )
                raw_hash = _sha(result.text)
                candidate = None
                static_result = {'passed': False, 'failure_kind': 'EXTRACTION_ERROR', 'detail': ''}
                worker_result = None
                sandbox_status = 'NOT_RUN'
                try:
                    candidate = extract_code(
                        result.text, language=language,
                        done_reason=result.done_reason, policy=extraction_policy,
                    )
                    static_result = static_check(
                        candidate.source, language=language,
                        entrypoint=entrypoint, rules=list(task['static_rules']),
                    )
                    if static_result['passed']:
                        worker_result = sandbox_runner(
                            SandboxRequest(
                                language=language, entrypoint=entrypoint,
                                candidate_source=candidate.source,
                                test_source=test_source, cases_json=cases_json,
                                image_ref=f'{image_ref}@{image_digest}',
                                image_digest=image_digest, policy=resource_policy,
                            )
                        )
                        sandbox_status = str(worker_result.get('status', 'UNKNOWN'))
                    else:
                        sandbox_status = 'STATIC_FAILURE'
                        counts['static_failures'] += 1
                except Exception as exc:
                    sandbox_status = 'EXTRACTION_ERROR'
                    static_result = {
                        'passed': False, 'failure_kind': type(exc).__name__,
                        'detail': str(exc),
                    }
                    if isinstance(exc, SandboxUnavailable):
                        counts['sandbox_errors'] += 1
                passed = bool(
                    static_result.get('passed')
                    and worker_result
                    and worker_result.get('passed')
                )
                record_coding_run(
                    conn, execution_id=execution_id, model_config_id=model_id,
                    task_id=task_id, trial=trial, language=language,
                    entrypoint=entrypoint, raw_output_sha256=raw_hash,
                    candidate_sha256=candidate.source_sha256 if candidate else None,
                    extraction_method=candidate.extraction_method if candidate else None,
                    static_result=static_result, sandbox_status=sandbox_status,
                    worker_result=worker_result, started_at_utc=started,
                    ended_at_utc=_utcnow(),
                )
                counts['rows'] += 1
                counts['passes'] += int(passed)
        return {'execution_id': execution_id, 'model_config_id': model_id, **counts}
    finally:
        conn.close()


def _task_prompt(root: Path, task_id: str) -> str:
    with open(root / 'evals/datasets/eval-v1/executable-v1.jsonl', encoding='utf-8') as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                if str(row['id']) == task_id:
                    return str(row['prompt'])
    raise CodingRunError(f'missing source prompt: {task_id}')


def _request(model_id: str, task_id: str, root: Path) -> GenerationRequest:
    config = get_model_config(model_id)
    return GenerationRequest(
        model=config.ollama_identifier,
        prompt=render_prompt(config, _task_prompt(root, task_id)),
        temperature=0.0,
        num_ctx=config.num_ctx,
        num_predict=2048,
        stop=config.stop_tokens,
        raw=config.mode == 'raw',
        think=config.think,
        num_gpu=config.num_gpu,
    )


def main(argv: list[str] | None = None) -> int:
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description='Run Coding v1')
    parser.add_argument('--model', required=True)
    parser.add_argument('--db', required=True)
    parser.add_argument('--base-url', default='http://127.0.0.1:11434')
    args = parser.parse_args(argv)
    try:
        result = run_coding_model(
            root, model_id=args.model, db_path=Path(args.db), base_url=args.base_url
        )
    except (CodingRunError, SandboxUnavailable, OSError, ValueError) as exc:
        print(f'CODING SUITE REFUSED: {exc}', flush=True)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
