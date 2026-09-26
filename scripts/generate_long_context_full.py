"""Validate and freeze the full long-context-v1 model cohort."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis.long_context import load_long_context_tasks


class LongContextFullError(ValueError):
    """The full long-context cohort is incomplete or contaminated."""


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(value, dict):
        raise LongContextFullError(f'{path}: expected a JSON object')
    return value


def _validate_db(path: Path, model_id: str, task_ids: set[str]) -> None:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    try:
        execution_id = f'long-context-v1__{model_id}'
        warmups = int(conn.execute(
            'SELECT COUNT(*) FROM runs WHERE experiment_id=? AND run_kind=\'WARMUP\'',
            (execution_id,),
        ).fetchone()[0])
        rows = conn.execute(
            'SELECT task_id,trial,status,grader_verdict,model_digest '
            'FROM runs WHERE experiment_id=? AND is_warmup=0',
            (execution_id,),
        ).fetchall()
    finally:
        conn.close()
    expected = {(task_id, trial) for task_id in task_ids for trial in (1, 2, 3)}
    observed = {(str(row['task_id']), int(row['trial'])) for row in rows}
    if warmups != 2 or len(rows) != 63 or observed != expected:
        raise LongContextFullError(
            f'{model_id}: expected 2 warmups and 63 exact measured rows'
        )
    if any(row['status'] != 'COMPLETE' or row['grader_verdict'] == 'ERROR' for row in rows):
        raise LongContextFullError(f'{model_id}: error rows present')


def build_full(
    root: Path,
    *,
    report_paths: dict[str, Path],
    db_paths: dict[str, Path],
    cleanup_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    config_path = root / 'configs/long-context-v1.yaml'
    config_document = yaml.safe_load(config_path.read_text(encoding='utf-8'))
    full_models = [str(value) for value in config_document['full_models']]
    pilot_models = {str(value) for value in config_document['pilot_models']}
    if len(full_models) != 11 or len(set(full_models)) != 11:
        raise LongContextFullError('full model cohort must contain 11 unique models')
    if set(report_paths) != set(full_models) or set(db_paths) != set(full_models):
        raise LongContextFullError('full model cohort is incomplete')
    tasks = load_long_context_tasks(root)
    task_ids = {task.task_id for task in tasks}
    if len(tasks) != 21:
        raise LongContextFullError('frozen task population is not 21 tasks')
    dataset_path = root / 'evals/datasets/long-context-v1/tasks.jsonl'
    grading_path = root / 'evals/specs/long-context-v1-grading.yaml'
    dataset_sha256 = _hash(dataset_path)
    grading_sha256 = _hash(grading_path)
    pilot_freeze_path = root / 'results/reports/long-context-v1-pilot-freeze.json'
    pilot_freeze = _load_json(pilot_freeze_path)
    cleanup = _load_json(cleanup_path)
    if cleanup.get('status') != 'VERIFIED':
        raise LongContextFullError('cleanup report is not VERIFIED')
    model_rows: list[dict[str, Any]] = []
    freeze_models: dict[str, Any] = {}
    for model_id in full_models:
        report_path = report_paths[model_id]
        db_path = db_paths[model_id]
        report = _load_json(report_path)
        if report.get('model_config_id') != model_id:
            raise LongContextFullError(f'{model_id}: report model mismatch')
        if report.get('task_count') != 21 or report.get('trial_count') != 63:
            raise LongContextFullError(f'{model_id}: report population mismatch')
        if report.get('dataset_sha256') != dataset_sha256:
            raise LongContextFullError(f'{model_id}: dataset hash mismatch')
        if report.get('grading_spec_sha256') != grading_sha256:
            raise LongContextFullError(f'{model_id}: grading hash mismatch')
        if report.get('request_accounting', {}).get('total') != 87:
            raise LongContextFullError(f'{model_id}: request accounting mismatch')
        if not report.get('task_id_coverage', False):
            raise LongContextFullError(f'{model_id}: task coverage incomplete')
        _validate_db(db_path, model_id, task_ids)
        reused_pilot = model_id in pilot_models
        if reused_pilot:
            frozen = pilot_freeze.get('models', {}).get(model_id, {})
            if frozen.get('report_sha256') != _hash(report_path):
                raise LongContextFullError(f'{model_id}: pilot report hash drift')
            if frozen.get('database_sha256') != _hash(db_path):
                raise LongContextFullError(f'{model_id}: pilot database hash drift')
        model_rows.append({
            'model_config_id': model_id,
            'passes': int(report['passes']),
            'trial_accuracy': float(report['trial_accuracy']),
            'by_length': report['by_length'],
            'reused_sealed_pilot': reused_pilot,
        })
        freeze_models[model_id] = {
            'database_sha256': _hash(db_path),
            'report_sha256': _hash(report_path),
            'reused_sealed_pilot': reused_pilot,
        }
    report = {
        'schema_version': 'long-context-full-v1',
        'suite': 'long-context-v1',
        'status': 'FULL_COMPLETE',
        'model_count': len(model_rows),
        'task_count': len(tasks),
        'measured_trials': 63 * len(model_rows),
        'total_passes': sum(int(row['passes']) for row in model_rows),
        'request_accounting': {
            'requests_per_model': 87,
            'total_requests_including_sealed_pilot': 87 * len(model_rows),
            'new_requests_for_remaining_models': 87 * (len(model_rows) - len(pilot_models)),
        },
        'pilot_models_reused': sorted(pilot_models),
        'models': model_rows,
        'claim_scope': (
            'Configured 4096-token full-context window across 11 V2 configurations; '
            'no native 8K/32K claim; no RAG.'
        ),
    }
    freeze = {
        'schema_version': 'long-context-full-freeze-v1',
        'suite': 'long-context-v1',
        'status': 'FULL_SEALED',
        'runner_code_sha256': _hash(root / 'scripts/run_long_context.py'),
        'aggregator_code_sha256': _hash(root / 'scripts/generate_long_context_full.py'),
        'config_sha256': _hash(config_path),
        'dataset_sha256': dataset_sha256,
        'grading_spec_sha256': grading_sha256,
        'models': freeze_models,
        'full_report_sha256': hashlib.sha256((
            json.dumps(report, indent=2, sort_keys=True) + '\n'
        ).encode('utf-8')).hexdigest(),
        'cleanup_sha256': _hash(cleanup_path),
        'pilot_freeze_sha256': _hash(pilot_freeze_path),
        'request_accounting': report['request_accounting'],
        'measured_trials_per_model': 63,
    }
    return report, freeze


def main(argv: list[str] | None = None) -> int:
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description='Freeze full long-context-v1 cohort')
    parser.add_argument('--reports-dir', default='results/reports')
    parser.add_argument('--db-dir', default='results/local')
    parser.add_argument('--cleanup-report', required=True)
    parser.add_argument('--out', default='results/reports/long-context-v1-full.json')
    parser.add_argument('--freeze-out', default='results/reports/long-context-v1-full-freeze.json')
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args(argv)
    reports_dir = root / args.reports_dir
    db_dir = root / args.db_dir
    config_document = yaml.safe_load(
        (root / 'configs/long-context-v1.yaml').read_text(encoding='utf-8')
    )
    report_paths = {
        str(model_id): reports_dir / f'long-context-v1__{model_id}.json'
        for model_id in config_document['full_models']
    }
    db_paths = {
        str(model_id): db_dir / f'long-context-v1__{model_id}.db'
        for model_id in config_document['full_models']
    }
    try:
        report, freeze = build_full(
            root,
            report_paths=report_paths,
            db_paths=db_paths,
            cleanup_path=root / args.cleanup_report,
        )
    except (OSError, sqlite3.Error, ValueError, KeyError, LongContextFullError) as exc:
        print(f'LONG-CONTEXT FULL REFUSED: {exc}', flush=True)
        return 2
    for path in (root / args.out, root / args.freeze_out):
        if path.exists() and not args.force:
            print(f'LONG-CONTEXT FULL REFUSED: exists {path}', flush=True)
            return 2
    (root / args.out).write_text(
        json.dumps(report, indent=2, sort_keys=True) + '\n', encoding='utf-8'
    )
    (root / args.freeze_out).write_text(
        json.dumps(freeze, sort_keys=True) + '\n', encoding='utf-8'
    )
    print('long-context full cohort: 11 models, 693 trials, FULL_SEALED')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
