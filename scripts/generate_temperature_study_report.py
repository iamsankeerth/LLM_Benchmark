"""Fail-closed paired report for the frozen Qwen Q4 temperature study."""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis.comparison import ComparisonTrial, compare_executions, comparison_to_json
from analysis.temperature_study import (
    TemperatureStudyError,
    load_temperature_study_contract,
    validate_temperature_study_run,
)
from inference.adapters import get_model_config


def _task_meta(root: Path) -> dict[str, dict[str, str]]:
    spec = yaml.safe_load(
        (root / 'evals/specs/eval-v1-grading.yaml').read_text(encoding='utf-8')
    )
    meta = {
        str(task_id): {
            'grading_status': str(entry['grading_status']),
            'primary_class': str(entry['primary_class']),
        }
        for task_id, entry in spec['tasks'].items()
    }
    with open(root / 'evals/datasets/eval-v1/executable-v1.jsonl', encoding='utf-8') as handle:
        for line in handle:
            row = json.loads(line)
            task_id = str(row['id'])
            if task_id in meta:
                meta[task_id]['suite'] = str(row['suite'])
                meta[task_id]['difficulty'] = str(row['difficulty'])
    return meta


def _validate_arm_config(
    root: Path, path: Path, *, arm: str, statuses: dict[str, str]
) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding='utf-8'))
    if not isinstance(config, dict):
        raise TemperatureStudyError(f'arm config is not a mapping: {path}')
    declaration = config.get('temperature_study')
    if not isinstance(declaration, dict) or declaration.get('arm') != arm:
        raise TemperatureStudyError(f'wrong arm config supplied for {arm}: {path}')
    validate_temperature_study_run(
        config, root=root, model_config_id='qwen3-4b-q4',
        task_ids=[str(item) for item in config.get('task_ids', [])],
        temperature=float(config.get('temperature', 0.0)),
        trials=int(config.get('trials', 1)),
        num_predict=int(config.get('num_predict', 512)), grading_statuses=statuses,
    )
    return config


def _load_arm(
    db_path: str,
    *,
    execution_id: str,
    temperature: float,
    expected_pairs: set[tuple[str, int]],
    expected_digest: str,
    meta: dict[str, dict[str, str]],
) -> tuple[dict[str, list[ComparisonTrial]], dict[str, Any]]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        provenance = conn.execute(
            'SELECT experiment_spec_id, model_config_id, model_artifact_digest'
            ' FROM execution_provenance WHERE execution_id=?',
            (execution_id,),
        ).fetchone()
        if provenance is None:
            raise TemperatureStudyError(f'{execution_id}: missing execution provenance')
        expected_spec_id = execution_id.split('__', 1)[0]
        if str(provenance['experiment_spec_id']) != expected_spec_id:
            raise TemperatureStudyError(f'{execution_id}: wrong experiment spec in provenance')
        if str(provenance['model_config_id']) != 'qwen3-4b-q4':
            raise TemperatureStudyError(f'{execution_id}: wrong model in provenance')
        if str(provenance['model_artifact_digest'] or '') != expected_digest:
            raise TemperatureStudyError(f'{execution_id}: model digest drift in provenance')
        rows = conn.execute(
            'SELECT task_id, trial, run_kind, status, temperature, model_digest,'
            ' rendered_prompt_sha256, template_sha256, num_predict, num_ctx, num_gpu,'
            ' stop_tokens_json, think, run_config_hash, grader_verdict, raw_output,'
            ' decode_tok_s, ttft_ms, eval_count FROM runs'
            ' WHERE experiment_id=? AND is_warmup=0 ORDER BY task_id, trial',
            (execution_id,),
        ).fetchall()
    finally:
        conn.close()
    pairs = [(str(row['task_id']), int(row['trial'])) for row in rows]
    if len(pairs) != len(set(pairs)) or set(pairs) != expected_pairs:
        raise TemperatureStudyError(
            f'{execution_id}: measured task/trial population is not the frozen 75 rows'
        )
    if any(
        str(row['run_kind']) != 'TEMPERATURE'
        or str(row['status']) != 'COMPLETE'
        or float(row['temperature']) != temperature
        or str(row['model_digest'] or '') != expected_digest
        for row in rows
    ):
        raise TemperatureStudyError(f'{execution_id}: measured-row identity drift')
    grouped: dict[str, list[ComparisonTrial]] = {}
    evidence_rows: dict[tuple[str, int], dict[str, Any]] = {}
    for row in rows:
        task_id = str(row['task_id'])
        task = meta.get(task_id)
        if task is None or task['grading_status'] != 'READY_DETERMINISTIC':
            raise TemperatureStudyError(f'{execution_id}: non-deterministic row {task_id}')
        grouped.setdefault(task_id, []).append(ComparisonTrial(
            task_id=task_id, trial=int(row['trial']), verdict=str(row['grader_verdict']),
            raw_output=str(row['raw_output'] or ''), decode_tok_s=row['decode_tok_s'],
            ttft_ms=row['ttft_ms'], eval_count=row['eval_count'],
            primary_class=task['primary_class'], suite=task['suite'],
            difficulty=task['difficulty'],
        ))
        evidence_rows[(task_id, int(row['trial']))] = {
            key: row[key] for key in (
                'rendered_prompt_sha256', 'template_sha256', 'num_predict', 'num_ctx',
                'num_gpu', 'stop_tokens_json', 'think', 'run_config_hash',
            )
        }
    return grouped, {'rows': evidence_rows, 'run_config_hashes': sorted({
        str(row['run_config_hash']) for row in rows
    })}


def generate_temperature_study_report(
    *,
    root: Path,
    db_t0: str,
    db_t07: str,
    config_t0: Path,
    config_t07: Path,
) -> dict[str, Any]:
    """Load, validate, and compare exactly the two frozen temperature arms."""
    contract = load_temperature_study_contract(root, 'temperature-study-v1')
    meta = _task_meta(root)
    statuses = {task_id: item['grading_status'] for task_id, item in meta.items()}
    _validate_arm_config(root, config_t0, arm='t0', statuses=statuses)
    _validate_arm_config(root, config_t07, arm='t07', statuses=statuses)
    task_ids = [str(task_id) for task_id in contract['task_ids']]
    expected_pairs = {(task_id, trial) for task_id in task_ids for trial in range(1, 6)}
    expected_digest = get_model_config('qwen3-4b-q4').ollama_model_digest
    base, base_evidence = _load_arm(
        db_t0, execution_id='temperature-study-v1-t0__qwen3-4b-q4',
        temperature=0.0, expected_pairs=expected_pairs, expected_digest=expected_digest,
        meta=meta,
    )
    candidate, candidate_evidence = _load_arm(
        db_t07, execution_id='temperature-study-v1-t07__qwen3-4b-q4',
        temperature=0.7, expected_pairs=expected_pairs, expected_digest=expected_digest,
        meta=meta,
    )
    shared_keys = (
        'rendered_prompt_sha256', 'template_sha256', 'num_predict', 'num_ctx',
        'num_gpu', 'stop_tokens_json', 'think',
    )
    for pair in expected_pairs:
        if any(base_evidence['rows'][pair][key] != candidate_evidence['rows'][pair][key]
               for key in shared_keys):
            raise TemperatureStudyError(f'{pair}: non-temperature configuration drift')
    if base_evidence['run_config_hashes'] == candidate_evidence['run_config_hashes']:
        raise TemperatureStudyError('arms have identical run-config hashes; temperature was not isolated')
    report = compare_executions(
        experiment_spec_id='temperature-study-v1',
        base_execution_id='temperature-study-v1-t0__qwen3-4b-q4',
        candidate_execution_id='temperature-study-v1-t07__qwen3-4b-q4',
        base_grouped=base, cand_grouped=candidate,
        base_artifact_digest=expected_digest, cand_artifact_digest=expected_digest,
    )
    trial_transitions: dict[str, list[str]] = {
        'pass_to_pass': [], 'fail_to_fail': [], 'fail_to_pass': [], 'pass_to_fail': [],
    }
    for task_id, trial in sorted(expected_pairs):
        before = next(item for item in base[task_id] if item.trial == trial).verdict == 'PASS'
        after = next(item for item in candidate[task_id] if item.trial == trial).verdict == 'PASS'
        label = f'{task_id}#{trial}'
        trial_transitions[
            'pass_to_pass' if before and after else 'fail_to_fail' if not before and not after
            else 'fail_to_pass' if after else 'pass_to_fail'
        ].append(label)
    return {
        'study': 'temperature-study-v1',
        'population': {'task_ids': task_ids, 'trials_per_task': 5, 'measured_rows_per_arm': 75},
        'arms': {'t0_temperature': 0.0, 't07_temperature': 0.7},
        'comparison': json.loads(comparison_to_json(report)),
        'trial_transitions': trial_transitions,
        'latency_note': 'Latency metrics are descriptive only, not causal accuracy evidence.',
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Report the frozen temperature study')
    parser.add_argument('--db-t0', required=True)
    parser.add_argument('--db-t07', required=True)
    parser.add_argument('--config-t0', default='configs/temperature-study-v1-t0.yaml')
    parser.add_argument('--config-t07', default='configs/temperature-study-v1-t07.yaml')
    parser.add_argument('--out', default='results/reports/temperature-study-v1.json')
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parent.parent
    try:
        report = generate_temperature_study_report(
            root=root, db_t0=args.db_t0, db_t07=args.db_t07,
            config_t0=root / args.config_t0, config_t07=root / args.config_t07,
        )
    except (OSError, sqlite3.Error, TemperatureStudyError, ValueError) as exc:
        print(f'TEMPERATURE STUDY REFUSED: {exc}', flush=True)
        return 2
    out = root / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    print(f'report: {out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
