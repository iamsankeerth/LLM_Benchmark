"""Pure eligibility resolution for retry-rescue-v2 constrained decoding."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


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

V1_SUFFIX_MARKERS = (
    'Return only the final numeric value required by the task.',
    'Return only the exact requested text or token.',
    'Return only valid JSON matching the required structure for this task.',
)


@dataclass(frozen=True)
class V2Contract:
    primary_identities: tuple[tuple[str, int], ...]
    control_identities: tuple[tuple[str, int], ...]
    formats: dict[str, tuple[str, dict[str, Any]]]
    population_sha256: str
    format_mapping_sha256: str

    def format_for(self, task_id: str) -> dict[str, Any]:
        return self.formats[task_id][1]

    def format_kind_for(self, task_id: str) -> str:
        return self.formats[task_id][0]


def canonical_format_bytes(response_format: dict[str, Any]) -> bytes:
    return json.dumps(
        response_format, sort_keys=True, separators=(',', ':'), ensure_ascii=False,
    ).encode('utf-8')


def canonical_format_mapping(
    formats: dict[str, tuple[str, dict[str, Any]]],
) -> bytes:
    document = [
        {'task_id': task_id, 'kind': kind, 'format': response_format}
        for task_id, (kind, response_format) in sorted(formats.items())
    ]
    return json.dumps(
        document, sort_keys=True, separators=(',', ':'), ensure_ascii=False,
    ).encode('utf-8')


def format_mapping_sha256(mapping: bytes | dict[str, tuple[str, dict[str, Any]]]) -> str:
    raw = canonical_format_mapping(mapping) if isinstance(mapping, dict) else mapping
    return hashlib.sha256(raw).hexdigest()


def reject_retry_suffix(prompt: str) -> None:
    if any(marker in prompt for marker in V1_SUFFIX_MARKERS):
        raise ValueError('retry suffix is forbidden in constrained decoding prompt')


def v2_run_config_hash(
    *,
    model_config_id: str,
    model_config_sha256: str,
    temperature: float,
    num_ctx: int,
    num_predict: int,
    template_sha256: str,
    grader_spec_sha256: str,
    population_sha256: str,
    format_mapping_sha256: str,
) -> str:
    """Hash the complete V2 intervention identity, not a per-row format."""
    document = {
        'model_config_id': model_config_id,
        'model_config_sha256': model_config_sha256,
        'temperature': temperature,
        'num_ctx': num_ctx,
        'num_predict': num_predict,
        'template_sha256': template_sha256,
        'grader_spec_sha256': grader_spec_sha256,
        'population_sha256': population_sha256,
        'format_mapping_sha256': format_mapping_sha256,
        'intervention': 'CONSTRAINED_DECODING',
        'retry_budget': 1,
    }
    return hashlib.sha256(
        json.dumps(document, sort_keys=True, separators=(',', ':')).encode('utf-8')
    ).hexdigest()


def validate_c_resume(
    conn: Any,
    execution_id: str,
    expected_identities: set[tuple[str, int]],
    expected_run_config_hash: str,
    formats: dict[str, tuple[str, dict[str, Any]]],
) -> set[tuple[str, int]]:
    """Fail closed if a C database has rows from another frozen contract."""
    rows = conn.execute(
        'SELECT task_id, trial, run_config_hash, response_format_kind, '
        'response_format_sha256, status FROM runs '
        'WHERE experiment_id=? AND is_warmup=0', (execution_id,),
    ).fetchall()
    completed: set[tuple[str, int]] = set()
    for task_id, trial, config_hash, kind, format_hash, status in rows:
        identity = (str(task_id), int(trial))
        if identity not in expected_identities:
            raise ValueError('unexpected C identity')
        if config_hash != expected_run_config_hash:
            raise ValueError('run config hash mismatch')
        expected_kind, response_format = formats[identity[0]]
        if kind != expected_kind:
            raise ValueError('response format kind mismatch')
        if format_hash != constraint_sha256(response_format):
            raise ValueError('response format hash mismatch')
        if status == 'COMPLETE':
            completed.add(identity)
        else:
            raise ValueError('incomplete C row prevents resume')
    return completed


def _identities(rows: Any) -> tuple[tuple[str, int], ...]:
    if not isinstance(rows, list):
        raise ValueError('row eligibility must be a list')
    selected = [
        (str(row['task_id']), int(row['trial']))
        for row in rows
        if isinstance(row, dict) and row.get('eligible') is True
    ]
    return tuple(sorted(selected))


def load_v2_contract(config_path: str, matrix_path: str) -> V2Contract:
    """Load the sole V2 mapping authority and verify its frozen population."""
    config_bytes = Path(config_path).read_bytes()
    config = yaml.safe_load(config_bytes)
    if not isinstance(config, dict):
        raise ValueError('V2 config must be a mapping')
    matrix_bytes = Path(matrix_path).read_bytes()
    matrix = json.loads(matrix_bytes)
    if not isinstance(matrix, dict) or matrix.get('matrix_status') != 'FROZEN':
        raise ValueError('V2 capability matrix must be FROZEN')
    population_hash = hashlib.sha256(matrix_bytes).hexdigest()
    if config.get('population_sha256') != population_hash:
        raise ValueError('V2 population hash mismatch')
    primary = _identities(matrix.get('row_eligibility'))
    control_doc = matrix.get('control_population')
    if not isinstance(control_doc, dict):
        raise ValueError('V2 control population missing')
    control = _identities(control_doc.get('row_eligibility'))
    if len(primary) != 18 or len(control) != 6:
        raise ValueError('V2 frozen population must resolve to 18 primary and 6 control rows')
    mapping_doc = config.get('response_formats')
    if not isinstance(mapping_doc, dict):
        raise ValueError('V2 response_formats missing')
    formats: dict[str, tuple[str, dict[str, Any]]] = {}
    for task_id, item in mapping_doc.items():
        if not isinstance(task_id, str) or not isinstance(item, dict):
            raise ValueError('V2 response format entries must be mappings')
        kind = item.get('kind')
        response_format = item.get('format')
        if kind not in ('NUMBER', 'STRUCTURED_JSON') or not isinstance(response_format, dict):
            raise ValueError(f'{task_id}: invalid response format entry')
        formats[task_id] = (kind, response_format)
    selected_task_ids = {task_id for task_id, _ in primary + control}
    if set(formats) != selected_task_ids:
        raise ValueError('V2 response format mapping must cover exactly selected tasks')
    mapping_hash = format_mapping_sha256(formats)
    if config.get('format_mapping_sha256') != mapping_hash:
        raise ValueError('V2 format mapping hash mismatch')
    return V2Contract(primary, control, formats, population_hash, mapping_hash)


def constraint_sha256(constraint: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_format_bytes(constraint)).hexdigest()


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
