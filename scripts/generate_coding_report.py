"""Generate a Coding v1 report from persisted candidate results."""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class CodingReportError(ValueError):
    """Persisted coding evidence is incomplete."""


def build_report(db_path: Path, model_id: str) -> dict[str, Any]:
    execution_id = f'coding-v1__{model_id}'
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            'SELECT task_id, trial, language, static_passed, sandbox_status,'
            ' functional_passed, static_passed_tests, total_tests, failure_kind'
            ' FROM coding_runs WHERE execution_id=? ORDER BY task_id, trial',
            (execution_id,),
        ).fetchall()
    finally:
        conn.close()
    expected = {(f'Q{index:03d}', trial) for index in range(27, 35) for trial in (1, 2, 3)}
    observed = {(str(row['task_id']), int(row['trial'])) for row in rows}
    if len(rows) != 24 or observed != expected:
        raise CodingReportError(f'{execution_id}: expected 24 task/trial rows, got {len(rows)}')
    passes = sum(
        bool(row['static_passed']) and bool(row['functional_passed'])
        for row in rows
    )
    return {
        'schema_version': 'coding-report-v1',
        'suite': 'coding-v1',
        'execution_id': execution_id,
        'model_config_id': model_id,
        'task_count': 8,
        'trial_count': 24,
        'passes': passes,
        'trial_accuracy': passes / 24,
        'static_failures': sum(not bool(row['static_passed']) for row in rows),
        'sandbox_errors': sum(row['sandbox_status'] not in {'PASS', 'FAIL', 'STATIC_FAILURE'} for row in rows),
        'rows': [dict(row) for row in rows],
        'partial_scores_are_diagnostic_only': True,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Generate Coding v1 report')
    parser.add_argument('--db', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args(argv)
    try:
        report = build_report(Path(args.db), args.model)
    except (OSError, sqlite3.Error, CodingReportError) as exc:
        print(f'CODING REPORT REFUSED: {exc}', flush=True)
        return 2
    output = Path(args.out)
    if output.exists() and not args.force:
        print(f'CODING REPORT REFUSED: exists {output}', flush=True)
        return 2
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    print(f'coding report: {report["passes"]}/{report["trial_count"]}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
