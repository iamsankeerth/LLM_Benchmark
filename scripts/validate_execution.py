"""Execution validator: explicit ID, hard errors on empty/mismatch.

A typo must never masquerade as an empty experiment. Zero measured rows,
a wrong experiment_spec_id, or a wrong model_config_id all fail closed.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis.reliability import load_spec_statuses


def validate_execution(
    db_path: str, execution_id: str, *, spec_expect: str, model_expect: str
) -> dict[str, object]:
    """Validate one execution; raise ValueError with the reason on failure."""
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            'SELECT task_id, trial, grader_verdict, status FROM runs'
            ' WHERE experiment_id=? AND is_warmup=0',
            (execution_id,),
        ).fetchall()
    finally:
        conn.close()
    if not rows:
        raise ValueError(f'0 measured rows for execution {execution_id!r}')
    parts = execution_id.split('__')
    if len(parts) != 2:
        raise ValueError(
            f'execution id {execution_id!r} is not <spec>__<model>'
        )
    spec_id, model_id = parts
    if spec_id != spec_expect:
        raise ValueError(
            f'wrong experiment_spec_id: {spec_id!r} != expected {spec_expect!r}'
        )
    if model_id != model_expect:
        raise ValueError(
            f'wrong model_config_id: {model_id!r} != expected {model_expect!r}'
        )
    return {
        'execution_id': execution_id,
        'measured_rows': len(rows),
        'spec_id': spec_id,
        'model_id': model_id,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Validate one execution')
    parser.add_argument('--db', required=True)
    parser.add_argument('--execution-id', required=True)
    parser.add_argument('--spec-expect', required=True)
    parser.add_argument('--model-expect', required=True)
    parser.add_argument('--matrix', action='store_true',
                        help='also print the 204/12/0/0/0 status matrix')
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parent.parent
    try:
        result = validate_execution(
            args.db, args.execution_id, spec_expect=args.spec_expect,
            model_expect=args.model_expect,
        )
    except ValueError as exc:
        print(f'VALIDATION HARD ERROR: {exc}', flush=True)
        return 1
    except sqlite3.Error as exc:
        print(f'VALIDATION HARD ERROR: unreadable db: {exc}', flush=True)
        return 1
    print(f"execution {result['execution_id']}: "
          f"{result['measured_rows']} measured rows", flush=True)
    if args.matrix:
        statuses = load_spec_statuses(
            str(root / 'evals/specs/eval-v1-grading.yaml')
        )
        conn = sqlite3.connect(args.db)
        try:
            rows = conn.execute(
                'SELECT task_id, trial, grader_verdict, status FROM runs'
                ' WHERE experiment_id=? AND is_warmup=0',
                (args.execution_id,),
            ).fetchall()
        finally:
            conn.close()
        det = [r for r in rows if statuses.get(r[0]) == 'READY_DETERMINISTIC']
        judge = [r for r in rows if statuses.get(r[0]) == 'READY_JUDGE']
        unknown = [r for r in rows if r[0] not in statuses]
        idents = [(r[0], r[1]) for r in rows]
        errors = [r for r in rows if r[3] == 'ERROR' or r[2] == 'ERROR']
        matrix = {
            'measured': len(rows), 'deterministic': len(det),
            'judge': len(judge), 'unknown': len(unknown),
            'duplicates': len(idents) - len(set(idents)),
            'errors': len(errors),
        }
        print(f'matrix: {matrix}', flush=True)
        if unknown or len(idents) != len(set(idents)) or errors:
            print('VALIDATION HARD ERROR: matrix contamination', flush=True)
            return 1
    print('VALIDATION PASS', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
