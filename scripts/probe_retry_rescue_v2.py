"""Run the pre-registered non-recording numeric representability probes."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis.retry_population import load_grading_spec, load_task_meta
from analysis.retry_rescue_v2 import (
    NUMERIC_CONSTRAINT,
    PROBE_COHORTS,
    constraint_sha256,
    freeze_matrix,
    is_bare_json_number,
)
from evals.graders.engine import grade_output
from inference.adapters import get_model_config, render_prompt
from inference.eligibility import (
    effective_options_for,
    run_canonical_eligibility,
    run_operational_canary,
)
from inference.ollama_client import GenerationRequest, OllamaClientError, fetch_ps, generate
from storage.sweep import (
    disk_free_bytes,
    disk_reclaimed_ok,
    ollama_model_present,
    ollama_pull,
    ollama_remove,
    ollama_stop,
    resolve_ollama_bin,
    wait_until_unloaded,
)


ROOT = Path(__file__).resolve().parent.parent
MATRIX_PATH = ROOT / 'results/reports/retry-rescue-v2-capability-matrix.json'
EVIDENCE_PATH = ROOT / 'results/reports/retry-rescue-v2-probe-evidence.json'
MODEL_CONFIG_ID = 'qwen3-4b-q4'
WARMUP_PROMPTS = ('Return only the word OK.', 'Return only the digit 7.')


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def _sha256_bytes(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_head() -> str:
    import subprocess

    return subprocess.run(
        ['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True,
        timeout=15, check=True,
    ).stdout.strip()


def _render(config: Any, prompt: str) -> str:
    return render_prompt(config, prompt) if config.mode == 'raw' else prompt


def _teardown(
    base_url: str, ollama_bin: str, identifier: str, free_before: int
) -> str | None:
    ollama_stop(ollama_bin, identifier)
    unloaded = wait_until_unloaded(
        lambda: not _model_loaded(base_url, identifier), timeout_s=180.0,
    )
    if not unloaded:
        return 'model remained loaded after stop'
    removed, detail = ollama_remove(ollama_bin, identifier)
    if not removed or ollama_model_present(ollama_bin, identifier):
        return f'weight deletion failed: {detail}'
    if not disk_reclaimed_ok(free_before, disk_free_bytes(ROOT)):
        return 'weight deletion did not reclaim expected disk space'
    return None


def _model_loaded(base_url: str, identifier: str) -> bool:
    try:
        entries = fetch_ps(base_url).get('models', [])
    except OllamaClientError:
        return True
    return any(
        isinstance(entry, dict)
        and (entry.get('model') == identifier or entry.get('name') == identifier)
        for entry in entries
    )


def run_probes(args: argparse.Namespace) -> int:
    if EVIDENCE_PATH.exists():
        print(f'refusing to overwrite existing probe evidence: {EVIDENCE_PATH}')
        return 2
    matrix = json.loads(MATRIX_PATH.read_text(encoding='utf-8'))
    if matrix.get('matrix_status') != 'DRAFT_STATIC':
        print('matrix is not DRAFT_STATIC; refusing to probe')
        return 2
    config = get_model_config(MODEL_CONFIG_ID)
    ollama_bin = resolve_ollama_bin(args.ollama_bin)
    free_before = disk_free_bytes(ROOT)
    pulled = False
    try:
        ok, detail = ollama_pull(ollama_bin, config.ollama_identifier)
        if not ok or not ollama_model_present(ollama_bin, config.ollama_identifier):
            print(f'pull failed: {detail}')
            return 3
        pulled = True
        ollama_stop(ollama_bin, config.ollama_identifier)
        if not wait_until_unloaded(
            lambda: not _model_loaded(args.base_url, config.ollama_identifier), timeout_s=180.0,
        ):
            print('could not establish a cold model state')
            return 3
        options = effective_options_for(
            mode=config.mode,
            num_ctx=config.num_ctx,
            num_gpu=config.num_gpu,
            temperature=0.0,
            template_sha256=config.template_sha256,
        )
        canonical = run_canonical_eligibility(
            base_url=args.base_url,
            model_identifier=config.ollama_identifier,
            expected_digest=config.ollama_model_digest,
            effective_options=options,
            render_prompt=lambda prompt: _render(config, prompt),
            timeout_s=args.timeout_s,
        )
        if not canonical.result.eligible:
            print(f'eligibility failed: {canonical.result.status}')
            return 3
        canary = run_operational_canary(
            base_url=args.base_url,
            model_identifier=config.ollama_identifier,
            effective_options=canonical.effective_options,
            render_prompt=lambda prompt: _render(config, prompt),
            timeout_s=args.timeout_s,
        )
        if not canary.passed:
            print(f'canary failed: {canary.failure_kind} ({canary.detail})')
            return 3
        for prompt in WARMUP_PROMPTS:
            generate(
                args.base_url,
                GenerationRequest(
                    model=config.ollama_identifier, prompt=_render(config, prompt),
                    temperature=0.0, num_ctx=config.num_ctx,
                    num_predict=config.num_predict_default, stop=config.stop_tokens,
                    raw=(config.mode == 'raw'), think=config.think, num_gpu=config.num_gpu,
                ),
                timeout_s=args.timeout_s,
            )
        spec = load_grading_spec(str(ROOT / 'evals/specs/eval-v1-grading.yaml'))
        task_meta = load_task_meta(str(ROOT / 'evals/datasets/eval-v1/executable-v1.jsonl'))
        records: list[dict[str, Any]] = []
        for task_id, cohort in PROBE_COHORTS.items():
            record: dict[str, Any] = {
                'task_id': task_id,
                'cohort': cohort,
                'constraint': NUMERIC_CONSTRAINT,
                'constraint_sha256': constraint_sha256(NUMERIC_CONSTRAINT),
                'raw_output': None,
                'done_reason': None,
                'transport_completed': False,
                'representation_compatible': False,
                'frozen_grader_passed': None,
            }
            try:
                result = generate(
                    args.base_url,
                    GenerationRequest(
                        model=config.ollama_identifier,
                        prompt=_render(config, str(task_meta[task_id]['prompt'])),
                        temperature=0.0, num_ctx=config.num_ctx,
                        num_predict=config.num_predict_default, stop=config.stop_tokens,
                        raw=(config.mode == 'raw'), think=config.think, num_gpu=config.num_gpu,
                        format=NUMERIC_CONSTRAINT,
                    ),
                    timeout_s=args.timeout_s,
                )
                record['raw_output'] = result.text
                record['done_reason'] = result.done_reason
                record['transport_completed'] = True
                record['representation_compatible'] = is_bare_json_number(result.text)
                record['frozen_grader_passed'] = all(
                    detail.passed for detail in grade_output(
                        result.text, spec['tasks'][task_id]['graders']
                    )
                )
            except OllamaClientError as error:
                record['error'] = f'{type(error).__name__}: {error}'
            records.append(record)
        evidence = {
            'experiment_id': 'retry-rescue-v2-representability-probes',
            'created_at_utc': _utcnow(),
            'model_config_id': config.config_id,
            'model_identifier': config.ollama_identifier,
            'model_digest': config.ollama_model_digest,
            'adapter_template_sha256': config.template_sha256,
            'constraint': NUMERIC_CONSTRAINT,
            'constraint_sha256': constraint_sha256(NUMERIC_CONSTRAINT),
            'warmup_count': len(WARMUP_PROMPTS),
            'canary_eval_count': canary.eval_count,
            'records': records,
            'provenance': {
                'matrix_draft_sha256': _sha256_bytes(MATRIX_PATH),
                'probe_code_git_commit': _git_head(),
            },
        }
        EVIDENCE_PATH.write_text(json.dumps(evidence, indent=2) + '\n', encoding='utf-8')
        if not all(record['transport_completed'] for record in records):
            print('one or more probes did not complete; matrix remains DRAFT_STATIC')
            return 4
        frozen = freeze_matrix(
            matrix,
            records,
            evidence_path='results/reports/retry-rescue-v2-probe-evidence.json',
            evidence_sha256=_sha256_bytes(EVIDENCE_PATH),
            probe_code_git_commit=_git_head(),
        )
        MATRIX_PATH.write_text(json.dumps(frozen, indent=2) + '\n', encoding='utf-8')
        print('probe evidence written and capability matrix frozen')
        return 0
    finally:
        if pulled:
            cleanup_error = _teardown(
                args.base_url, ollama_bin, config.ollama_identifier, free_before
            )
            if cleanup_error:
                raise RuntimeError(f'CLEANUP FAILED: {cleanup_error}')


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url', default='http://127.0.0.1:11434')
    parser.add_argument('--ollama-bin', default=None)
    parser.add_argument('--timeout-s', type=float, default=600.0)
    return run_probes(parser.parse_args(argv))


if __name__ == '__main__':
    raise SystemExit(main())
