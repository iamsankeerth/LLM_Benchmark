"""Append-only Eval-v1.1 regrade overlay generation."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any

from evals.contract import EvalContract, file_sha256
from evals.graders.engine import grade_output
from evals.verdicts import reduce_verdict

OLD_SPEC_HASH = 'd1cff373d5216f9d1d86d863bd4d86468d05d1f6a43a4e2cab839e257947718f'


class RegradeError(ValueError):
    """The source population or its persisted evidence is not regradeable."""


def _open_readonly(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(f'file:{path.resolve().as_posix()}?mode=ro', uri=True)


def _sha_text(value: str) -> str:
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


def _result_document(results: list[Any]) -> list[dict[str, Any]]:
    return [
        {
            'grader_type': result.grader_type,
            'passed': result.passed,
            'detail': result.detail,
            'violations': result.violations,
        }
        for result in results
    ]


def regrade_execution(
    db_path: Path,
    *,
    execution_id: str,
    contract: EvalContract,
    old_spec_hash: str = OLD_SPEC_HASH,
) -> list[dict[str, Any]]:
    """Regrade one immutable execution without modifying its database."""
    conn = _open_readonly(db_path)
    try:
        rows = conn.execute(
            'SELECT task_id, trial, status, run_kind, raw_output, grader_verdict,'
            ' grader_details_json FROM runs WHERE experiment_id=? AND is_warmup=0'
            ' ORDER BY task_id, trial',
            (execution_id,),
        ).fetchall()
    finally:
        conn.close()
    pairs = [(str(row[0]), int(row[1])) for row in rows]
    expected = {
        (str(task_id), trial)
        for task_id, entry in contract.grader_entries().items()
        if entry.get('grading_status') in ('READY_DETERMINISTIC', 'READY_JUDGE')
        for trial in range(1, 4)
    }
    if len(pairs) != len(set(pairs)) or set(pairs) != expected:
        raise RegradeError(f'{execution_id}: source population is not the expected 72-task V2 set')
    output: list[dict[str, Any]] = []
    for row in rows:
        task_id, trial, status, run_kind, raw_output, old_verdict, old_details_json = row
        if status != 'COMPLETE' or run_kind != 'BASELINE':
            raise RegradeError(f'{execution_id} {task_id}#{trial}: non-complete source row')
        entry = contract.grader_entries()[str(task_id)]
        results = grade_output(str(raw_output or ''), list(entry['graders']))
        new_verdict = reduce_verdict(str(entry['grading_status']), results)
        try:
            old_details = json.loads(old_details_json or '[]')
        except json.JSONDecodeError as exc:
            raise RegradeError(f'{execution_id} {task_id}#{trial}: invalid old details') from exc
        output.append({
            'execution_id': execution_id,
            'task_id': str(task_id),
            'trial': int(trial),
            'source_db_sha256': file_sha256(db_path),
            'raw_output_sha256': _sha_text(str(raw_output or '')),
            'old_spec_sha256': old_spec_hash,
            'new_spec_sha256': contract.hashes['spec_sha256'],
            'grader_engine_sha256': file_sha256(Path(__file__).resolve().parents[1] / 'evals/graders/engine.py'),
            'old_verdict': str(old_verdict),
            'old_details': old_details,
            'new_verdict': new_verdict,
            'new_details': _result_document(results),
        })
    return output


def build_overlay(
    *,
    root: Path,
    contract: EvalContract,
    completed_ids: list[str],
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for model_id in sorted(completed_ids):
        execution_id = f'full-baseline-v2__{model_id}'
        db_path = root / 'results/local' / f'{execution_id}.db'
        if not db_path.is_file():
            raise RegradeError(f'missing source database: {db_path}')
        records.extend(regrade_execution(
            db_path, execution_id=execution_id, contract=contract,
        ))
    return {
        'schema_version': 'eval-v1.1-regrade-overlay-v1',
        'source_spec_sha256': OLD_SPEC_HASH,
        'target_spec_sha256': contract.hashes['spec_sha256'],
        'target_spec_version': contract.spec_version,
        'source_db_count': len(completed_ids),
        'record_count': len(records),
        'records': records,
    }
