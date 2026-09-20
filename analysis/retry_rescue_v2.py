"""Pure eligibility resolution for retry-rescue-v2 constrained decoding."""

from __future__ import annotations

import hashlib
import json
from typing import Any


NUMERIC_CONSTRAINT = {'type': 'number'}
PROBE_COHORTS = {
    'Q019': 'primary',
    'Q020': 'primary',
    'Q021': 'primary',
    'Q022': 'primary',
    'Q024': 'primary',
    'Q070': 'primary',
    'Q026': 'control',
}


def constraint_sha256(constraint: dict[str, Any]) -> str:
    encoded = json.dumps(constraint, sort_keys=True, separators=(',', ':')).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()


def is_bare_json_number(output: str) -> bool:
    """Accept only a complete standard-JSON numeric value, never a wrapper."""
    try:
        value = json.loads(
            output.strip(),
            parse_constant=lambda constant: (_ for _ in ()).throw(ValueError(constant)),
        )
    except (json.JSONDecodeError, ValueError):
        return False
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def validate_probe_records(records: list[dict[str, Any]]) -> None:
    expected_tasks = set(PROBE_COHORTS)
    task_ids = [record.get('task_id') for record in records]
    if len(records) != len(expected_tasks) or set(task_ids) != expected_tasks:
        raise ValueError(f'probe task set must be exactly {sorted(expected_tasks)}')
    if len(set(task_ids)) != len(task_ids):
        raise ValueError('each pre-registered task must appear exactly once')
    digest = constraint_sha256(NUMERIC_CONSTRAINT)
    for record in records:
        task_id = str(record['task_id'])
        if record.get('cohort') != PROBE_COHORTS[task_id]:
            raise ValueError(f'{task_id}: wrong cohort')
        if record.get('constraint') != NUMERIC_CONSTRAINT:
            raise ValueError(f'{task_id}: alternate constraint is not permitted')
        if record.get('constraint_sha256') != digest:
            raise ValueError(f'{task_id}: constraint digest mismatch')
        if record.get('transport_completed') is not True:
            raise ValueError(f'{task_id}: transport did not complete')


def freeze_matrix(
    matrix: dict[str, Any],
    records: list[dict[str, Any],],
    *,
    evidence_path: str,
    evidence_sha256: str,
    probe_code_git_commit: str,
) -> dict[str, Any]:
    """Resolve all preregistered rows using representation, never correctness."""
    if matrix.get('matrix_status') != 'DRAFT_STATIC':
        raise ValueError('primary matrix is not a draft')
    control = matrix.get('control_population')
    if not isinstance(control, dict) or control.get('matrix_status') != 'DRAFT_STATIC':
        raise ValueError('control matrix is not a draft')
    validate_probe_records(records)
    compatible = {
        str(record['task_id']): bool(record['representation_compatible'])
        for record in records
    }
    _resolve_rows(matrix['row_eligibility'], compatible)
    _resolve_rows(control['row_eligibility'], compatible)
    _update_counts(matrix, 'primary')
    _update_counts(control, 'control')
    matrix['matrix_status'] = 'FROZEN'
    control['matrix_status'] = 'FROZEN'
    provenance = matrix.setdefault('provenance', {})
    provenance.update({
        'probe_evidence_path': evidence_path,
        'probe_evidence_sha256': evidence_sha256,
        'probe_constraint_sha256': constraint_sha256(NUMERIC_CONSTRAINT),
        'probe_code_git_commit': probe_code_git_commit,
    })
    return matrix


def _resolve_rows(rows: Any, compatible: dict[str, bool]) -> None:
    if not isinstance(rows, list):
        raise ValueError('matrix row eligibility must be a list')
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError('matrix row eligibility must contain objects')
        task_id = str(row.get('task_id'))
        if row.get('eligible') == 'REQUIRES_LIVE_VERIFICATION':
            if task_id not in compatible:
                raise ValueError(f'{task_id}: missing probe result')
            row['eligible'] = compatible[task_id]


def _update_counts(population: dict[str, Any], cohort: str) -> None:
    rows = population.get('row_eligibility')
    if not isinstance(rows, list):
        raise ValueError(f'{cohort}: row eligibility missing')
    eligible = sum(row.get('eligible') is True for row in rows if isinstance(row, dict))
    ineligible = sum(row.get('eligible') is False for row in rows if isinstance(row, dict))
    unresolved = len(rows) - eligible - ineligible
    if unresolved:
        raise ValueError(f'{cohort}: unresolved eligibility rows remain')
    if eligible % 3 != 0:
        raise ValueError(f'{cohort}: eligible rows must be whole task triplets')
    if cohort == 'primary' and not 0 <= eligible <= 18:
        raise ValueError('primary eligible rows must be within 0..18')
    if cohort == 'control' and eligible not in (3, 6):
        raise ValueError('control eligible rows must be 3 or 6')
    if eligible + ineligible != population.get('source_population_rows', population.get('source_rows')):
        raise ValueError(f'{cohort}: row counts do not reconcile')
    population['eligible_rows'] = eligible
    population['ineligible_rows'] = ineligible
    population['requires_live_verification_rows'] = 0
