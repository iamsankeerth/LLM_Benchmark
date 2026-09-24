"""Frozen-contract validation shared by the temperature-study runner and report."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import yaml


class TemperatureStudyError(ValueError):
    """The requested temperature arm does not satisfy the frozen contract."""


def load_temperature_study_contract(root: Path, contract_id: str) -> dict[str, Any]:
    """Load one named frozen study contract from the repository configuration."""
    path = root / 'configs' / f'{contract_id}.yaml'
    if not path.is_file():
        raise TemperatureStudyError(f'missing temperature-study contract: {path}')
    document = yaml.safe_load(path.read_text(encoding='utf-8'))
    if not isinstance(document, dict) or document.get('study') != contract_id:
        raise TemperatureStudyError(f'invalid temperature-study contract: {path}')
    return document


def validate_temperature_study_run(
    run_config: Mapping[str, Any],
    *,
    root: Path,
    model_config_id: str,
    task_ids: list[str],
    temperature: float,
    trials: int,
    num_predict: int,
    grading_statuses: Mapping[str, str],
) -> dict[str, Any] | None:
    """Validate a scalar arm before it can create a database or model request.

    Non-study configs return ``None`` so the general runner remains unchanged.
    """
    declaration = run_config.get('temperature_study')
    if declaration is None:
        return None
    if not isinstance(declaration, dict):
        raise TemperatureStudyError('temperature_study must be a mapping')
    contract_id = declaration.get('contract')
    arm_id = declaration.get('arm')
    if not isinstance(contract_id, str) or not isinstance(arm_id, str):
        raise TemperatureStudyError('temperature_study requires string contract and arm')
    contract = load_temperature_study_contract(root, contract_id)
    arms = contract.get('arms')
    if not isinstance(arms, dict) or not isinstance(arms.get(arm_id), dict):
        raise TemperatureStudyError(f'unknown temperature-study arm: {arm_id!r}')
    arm = arms[arm_id]
    expected = {
        'experiment': arm.get('experiment'),
        'run_kind': contract.get('run_kind'),
        'model_config_id': contract.get('model_config_id'),
        'temperature': arm.get('temperature'),
        'trials': contract.get('trials'),
        'num_predict': contract.get('num_predict'),
        'task_ids': contract.get('task_ids'),
    }
    observed = {
        'experiment': run_config.get('experiment'),
        'run_kind': run_config.get('run_kind', 'BASELINE'),
        'model_config_id': model_config_id,
        'temperature': temperature,
        'trials': trials,
        'num_predict': num_predict,
        'task_ids': task_ids,
    }
    mismatches = sorted(key for key in expected if observed[key] != expected[key])
    if mismatches:
        raise TemperatureStudyError(
            f'temperature-study arm {arm_id!r} drifted on {mismatches}'
        )
    if len(task_ids) != len(set(task_ids)):
        raise TemperatureStudyError('temperature-study task_ids must be unique')
    non_deterministic = [
        task_id for task_id in task_ids
        if grading_statuses.get(task_id) != 'READY_DETERMINISTIC'
    ]
    if non_deterministic:
        raise TemperatureStudyError(
            f'temperature-study includes non-deterministic tasks: {non_deterministic}'
        )
    return contract
