"""Versioned evaluation-contract loading and byte provenance."""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml


class EvalContractError(ValueError):
    """The selected evaluation contract is missing, inconsistent, or drifted."""


@dataclass(frozen=True)
class EvalContract:
    spec: dict[str, Any]
    spec_version: str
    spec_path: Path
    dataset_path: Path
    hashes: Mapping[str, str]
    statuses: Mapping[str, str]

    def grader_entries(self) -> dict[str, dict[str, Any]]:
        tasks = self.spec.get('tasks')
        if not isinstance(tasks, dict):
            raise EvalContractError('grading contract has no tasks mapping')
        return {str(task_id): dict(entry) for task_id, entry in tasks.items()}


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _resolve(root: Path, value: Path | str) -> Path:
    path = value if isinstance(value, Path) else Path(value)
    return path if path.is_absolute() else root / path


def _load_yaml(path: Path) -> tuple[dict[str, Any], str]:
    if not path.is_file():
        raise EvalContractError(f'missing grading contract: {path}')
    raw = path.read_bytes()
    try:
        document = yaml.safe_load(raw.decode('utf-8'))
    except (UnicodeDecodeError, yaml.YAMLError) as exc:
        raise EvalContractError(f'invalid grading contract {path}: {exc}') from exc
    if not isinstance(document, dict):
        raise EvalContractError(f'grading contract is not a mapping: {path}')
    return document, hashlib.sha256(raw).hexdigest()


def resolve_grading_spec(path: Path) -> tuple[dict[str, Any], dict[str, str]]:
    """Resolve a full spec or a versioned overlay against its declared base."""
    document, overlay_hash = _load_yaml(path)
    if 'overrides' not in document:
        if not isinstance(document.get('tasks'), dict):
            raise EvalContractError(f'full grading contract has no tasks: {path}')
        return document, {'spec_sha256': overlay_hash, 'spec_path': str(path)}

    base_id = document.get('base_spec')
    if not isinstance(base_id, str) or not base_id:
        raise EvalContractError(f'overlay has no base_spec: {path}')
    base_path = path.parent / f'{base_id}-grading.yaml'
    base, base_hash = _load_yaml(base_path)
    declared_base = document.get('provenance', {}).get('base_spec_sha256')
    if declared_base != base_hash:
        raise EvalContractError(
            f'overlay base hash mismatch: declared {declared_base!r}, '
            f'actual {base_hash!r}'
        )
    overrides = document.get('overrides')
    if not isinstance(overrides, dict) or not overrides:
        raise EvalContractError(f'overlay has no task overrides: {path}')
    effective = copy.deepcopy(base)
    effective['spec_version'] = document['spec_version']
    effective['status'] = document['status']
    tasks = effective.get('tasks')
    if not isinstance(tasks, dict):
        raise EvalContractError(f'base grading contract has no tasks: {base_path}')
    for task_id, entry in overrides.items():
        if task_id not in tasks:
            raise EvalContractError(f'overlay task is absent from base: {task_id}')
        if not isinstance(entry, dict):
            raise EvalContractError(f'overlay task is not a mapping: {task_id}')
        tasks[task_id] = copy.deepcopy(entry)
    return effective, {
        'spec_sha256': overlay_hash,
        'base_spec_sha256': base_hash,
        'spec_path': str(path),
        'base_spec_path': str(base_path),
    }


def load_eval_contract(
    root: Path,
    spec_path: Path | str,
    *,
    freeze_path: Path | str | None = None,
    dataset_path: Path | str | None = None,
) -> EvalContract:
    """Load one effective contract and verify any supplied freeze hashes."""
    selected_spec = _resolve(root, spec_path)
    spec, hashes = resolve_grading_spec(selected_spec)
    selected_dataset = _resolve(
        root,
        dataset_path or 'evals/datasets/eval-v1/executable-v1.jsonl',
    )
    if not selected_dataset.is_file():
        raise EvalContractError(f'missing executable dataset: {selected_dataset}')
    dataset_hash = file_sha256(selected_dataset)
    hashes['dataset_sha256'] = dataset_hash
    if freeze_path is not None:
        selected_freeze = _resolve(root, freeze_path)
        if not selected_freeze.is_file():
            raise EvalContractError(f'missing contract freeze: {selected_freeze}')
        freeze = json.loads(selected_freeze.read_text(encoding='utf-8'))
        artifacts = freeze.get('artifacts')
        if not isinstance(artifacts, dict):
            raise EvalContractError(f'freeze has no artifacts mapping: {selected_freeze}')
        expected = {
            'spec_sha256': hashes['spec_sha256'],
            'base_spec_sha256': hashes.get('base_spec_sha256'),
            'dataset_sha256': dataset_hash,
        }
        key_by_hash = {
            'spec_sha256': selected_spec.name,
            'base_spec_sha256': Path(hashes.get('base_spec_path', '')).name,
            'dataset_sha256': selected_dataset.name,
        }
        for field, value in expected.items():
            if value is None:
                continue
            artifact_key = key_by_hash[field]
            if artifact_key not in artifacts:
                raise EvalContractError(
                    f'freeze missing {artifact_key} for {field}'
                )
            if artifacts[artifact_key] != value:
                raise EvalContractError(
                    f'{selected_freeze.name} hash mismatch for {artifact_key}'
                )
        extra_paths = {
            'eval-v1.1-grading.schema.json': root / 'evals/specs/eval-v1.1-grading.schema.json',
            'grading-spec.schema.json': root / 'evals/specs/grading-spec.schema.json',
            'grading-spec-v1.1.schema.json': root / 'evals/specs/grading-spec-v1.1.schema.json',
            'contract.py': root / 'evals/contract.py',
            'engine.py': root / 'evals/graders/engine.py',
            'verdicts.py': root / 'evals/verdicts.py',
            'manifest.json': root / 'evals/datasets/eval-v1/manifest.json',
            'grading-review.json': root / 'evals/datasets/eval-v1/grading-review.json',
            'LocalLLM_Eval_Questions.xlsx': root / 'LocalLLM_Eval_Questions.xlsx',
        }
        for artifact_key, path in extra_paths.items():
            if artifact_key not in artifacts:
                continue
            if not path.is_file():
                raise EvalContractError(f'freeze artifact missing: {path}')
            if artifacts[artifact_key] != file_sha256(path):
                raise EvalContractError(
                    f'{selected_freeze.name} hash mismatch for {artifact_key}'
                )
        hashes['freeze_sha256'] = file_sha256(selected_freeze)
    statuses = {
        str(task_id): str(entry.get('grading_status', ''))
        for task_id, entry in spec.get('tasks', {}).items()
        if isinstance(entry, dict)
    }
    return EvalContract(
        spec=spec,
        spec_version=str(spec['spec_version']),
        spec_path=selected_spec,
        dataset_path=selected_dataset,
        hashes=hashes,
        statuses=statuses,
    )
