"""Validate and freeze the full live coding-v1 model cohort."""

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

from analysis.coding import load_fixture_manifest


class CodingFullError(ValueError):
    """The full coding cohort is incomplete or contaminated."""


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(value, dict):
        raise CodingFullError(f'{path}: expected a JSON object')
    return value


def _models(root: Path) -> list[str]:
    registry = yaml.safe_load(
        (root / 'configs/models-v2.yaml').read_text(encoding='utf-8')
    )
    model_ids = [
        str(entry['model_config_id'])
        for entry in registry['models']
        if not str(entry['model_config_id']).startswith('phi-')
    ]
    if len(model_ids) != 11 or len(set(model_ids)) != 11:
        raise CodingFullError('eligible coding cohort must contain 11 unique models')
    return model_ids


def _validate_db(path: Path, model_id: str, task_ids: set[str]) -> dict[str, Any]:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    try:
        execution_id = f'coding-v1__{model_id}'
        rows = conn.execute(
            'SELECT task_id,trial,static_passed,functional_passed,sandbox_status '
            'FROM coding_runs WHERE execution_id=? ORDER BY task_id,trial',
            (execution_id,),
        ).fetchall()
    finally:
        conn.close()
    expected = {(task_id, trial) for task_id in task_ids for trial in (1, 2, 3)}
    observed = {(str(row['task_id']), int(row['trial'])) for row in rows}
    if len(rows) != 24 or observed != expected:
        raise CodingFullError(f'{model_id}: expected 24 exact task/trial rows')
    if any(row['sandbox_status'] not in {'PASS', 'FAIL', 'STATIC_FAILURE'} for row in rows):
        raise CodingFullError(f'{model_id}: sandbox or infrastructure error present')
    return {
        'passes': sum(
            bool(row['static_passed']) and bool(row['functional_passed'])
            for row in rows
        ),
        'static_failures': sum(not bool(row['static_passed']) for row in rows),
        'sandbox_errors': 0,
    }


def build_full(
    root: Path,
    *,
    report_paths: dict[str, Path],
    db_paths: dict[str, Path],
    cleanup_path: Path,
    classification_repair_path: Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    model_ids = _models(root)
    if set(report_paths) != set(model_ids) or set(db_paths) != set(model_ids):
        raise CodingFullError('coding model cohort is incomplete')
    manifest_path = root / 'evals/fixtures/coding-v1/fixture-manifest.json'
    manifest = load_fixture_manifest(manifest_path)
    task_ids = set(manifest['tasks'])
    if len(task_ids) != 8:
        raise CodingFullError('coding task population is not Q027..Q034')
    cleanup = _load_json(cleanup_path)
    if cleanup.get('status') != 'VERIFIED':
        raise CodingFullError('cleanup report is not VERIFIED')
    if classification_repair_path is not None:
        classification_repair = _load_json(classification_repair_path)
        if classification_repair.get('status') != 'VERIFIED':
            raise CodingFullError('classification repair report is not VERIFIED')
    rows: list[dict[str, Any]] = []
    freeze_models: dict[str, Any] = {}
    for model_id in model_ids:
        report = _load_json(report_paths[model_id])
        if report.get('model_config_id') != model_id:
            raise CodingFullError(f'{model_id}: report model mismatch')
        if report.get('task_count') != 8 or report.get('trial_count') != 24:
            raise CodingFullError(f'{model_id}: report population mismatch')
        db_summary = _validate_db(db_paths[model_id], model_id, task_ids)
        rows.append({'model_config_id': model_id, **db_summary})
        freeze_models[model_id] = {
            'database_sha256': _hash(db_paths[model_id]),
            'report_sha256': _hash(report_paths[model_id]),
        }
    total_rows = 24 * len(rows)
    report = {
        'schema_version': 'coding-full-v1',
        'suite': 'coding-v1',
        'status': 'FULL_COMPLETE',
        'model_count': len(rows),
        'task_count': len(task_ids),
        'trial_count': total_rows,
        'total_passes': sum(int(row['passes']) for row in rows),
        'trial_accuracy': sum(int(row['passes']) for row in rows) / total_rows,
        'static_failures': sum(int(row['static_failures']) for row in rows),
        'sandbox_errors': 0,
        'models': rows,
        'runtime_digests': {
            key: value['digest']
            for key, value in manifest['runtimes'].items()
        },
        'partial_scores_are_diagnostic_only': True,
        'claim_scope': 'Binary task success in pinned isolated OCI workers; no host execution.',
    }
    freeze = {
        'schema_version': 'coding-full-freeze-v1',
        'suite': 'coding-v1',
        'status': 'LIVE_SEALED',
        'spec_sha256': _hash(root / 'evals/specs/coding-v1-grading.yaml'),
        'fixture_manifest_sha256': _hash(manifest_path),
        'fixture_bundle_sha256': manifest['bundle_sha256'],
        'extraction_policy_sha256': _hash(root / 'evals/fixtures/coding-v1/extraction-policy.json'),
        'resource_policy_sha256': _hash(root / 'evals/fixtures/coding-v1/resource-policy.json'),
        'runner_code_sha256': _hash(root / 'scripts/run_coding_suite.py'),
        'sandbox_code_sha256': _hash(root / 'inference/sandbox.py'),
        'aggregator_code_sha256': _hash(root / 'scripts/generate_coding_full.py'),
        'runtime_digests': report['runtime_digests'],
        'models': freeze_models,
        'full_report_sha256': hashlib.sha256((
            json.dumps(report, indent=2, sort_keys=True) + '\n'
        ).encode('utf-8')).hexdigest(),
        'cleanup_sha256': _hash(cleanup_path),
        'rows_per_model': 24,
    }
    if classification_repair_path is not None:
        freeze['classification_repair_sha256'] = _hash(classification_repair_path)
    return report, freeze


def main(argv: list[str] | None = None) -> int:
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description='Freeze full coding-v1 cohort')
    parser.add_argument('--reports-dir', default='results/reports')
    parser.add_argument('--db-dir', default='results/local')
    parser.add_argument('--cleanup-report', required=True)
    parser.add_argument(
        '--classification-repair',
        default='results/reports/coding-v1-classification-repair.json',
    )
    parser.add_argument('--out', default='results/reports/coding-v1-full.json')
    parser.add_argument('--freeze-out', default='results/reports/coding-v1-full-freeze.json')
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args(argv)
    model_ids = _models(root)
    report_paths = {
        model_id: root / args.reports_dir / f'coding-v1__{model_id}.json'
        for model_id in model_ids
    }
    db_paths = {
        model_id: root / args.db_dir / f'coding-v1__{model_id}.db'
        for model_id in model_ids
    }
    try:
        report, freeze = build_full(
            root,
            report_paths=report_paths,
            db_paths=db_paths,
            cleanup_path=root / args.cleanup_report,
            classification_repair_path=root / args.classification_repair,
        )
    except (OSError, sqlite3.Error, ValueError, KeyError, CodingFullError) as exc:
        print(f'CODING FULL REFUSED: {exc}', flush=True)
        return 2
    for path in (root / args.out, root / args.freeze_out):
        if path.exists() and not args.force:
            print(f'CODING FULL REFUSED: exists {path}', flush=True)
            return 2
    (root / args.out).write_text(
        json.dumps(report, indent=2, sort_keys=True) + '\n', encoding='utf-8'
    )
    (root / args.freeze_out).write_text(
        json.dumps(freeze, sort_keys=True) + '\n', encoding='utf-8'
    )
    print('coding full cohort: 11 models, 264 rows, LIVE_SEALED')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
