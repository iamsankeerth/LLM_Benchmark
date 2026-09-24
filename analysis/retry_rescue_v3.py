"""Pure deterministic renderer and frozen-contract helpers for retry-rescue-v3."""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path

import yaml


RENDERED = 'RENDERED'
UNRENDERABLE = 'UNRENDERABLE'
FROZEN_IDENTITIES = tuple(
    (task_id, trial)
    for task_id in ('Q019', 'Q020', 'Q021', 'Q022', 'Q024', 'Q070')
    for trial in (1, 2, 3)
)
_NUMBER = r'-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?'
_FINAL = re.compile(
    rf'^\s*(?:final\s+answer|answer)(?:\s+is)?\s*[:\-]?\s*({_NUMBER})\s*$', re.I,
)
_TERMINAL = re.compile(rf'^\s*({_NUMBER})\s*$')


@dataclass(frozen=True)
class RenderResult:
    status: str
    output: str | None


@dataclass(frozen=True)
class V3Contract:
    identities: tuple[tuple[str, int], ...]
    baseline_db: Path
    baseline_execution: str
    baseline_db_sha256: str
    grading_spec: Path
    grading_spec_sha256: str
    population_matrix: Path
    population_sha256: str
    renderer_code_sha256: str
    config_sha256: str


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def open_readonly_source(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(f'file:{path.resolve().as_posix()}?mode=ro', uri=True)


def _canonical_number(candidate: str) -> str | None:
    try:
        value = Decimal(candidate)
    except InvalidOperation:
        return None
    if not value.is_finite():
        return None
    if value.is_zero():
        return '0'
    normalized = format(value.normalize(), 'f')
    if '.' in normalized:
        normalized = normalized.rstrip('0').rstrip('.')
    return normalized if len(normalized) <= 128 else None


def render_numeric(raw_output: str) -> RenderResult:
    """Render one unambiguous asserted numeric value without grader inputs."""
    candidates: set[str] = set()
    for line in raw_output.splitlines():
        marker = _FINAL.fullmatch(line)
        if marker is not None:
            candidate = _canonical_number(marker.group(1))
            if candidate is not None:
                candidates.add(candidate)
    non_empty = [line for line in raw_output.splitlines() if line.strip()]
    if non_empty:
        terminal = _TERMINAL.fullmatch(non_empty[-1])
        if terminal is not None:
            candidate = _canonical_number(terminal.group(1))
            if candidate is not None:
                candidates.add(candidate)
    if len(candidates) != 1:
        return RenderResult(UNRENDERABLE, None)
    return RenderResult(RENDERED, next(iter(candidates)))


def load_v3_contract(config_path: str) -> V3Contract:
    config_file = Path(config_path)
    config_bytes = config_file.read_bytes()
    config = yaml.safe_load(config_bytes)
    if not isinstance(config, dict):
        raise ValueError('V3 config must be a mapping')
    source = config.get('source')
    if not isinstance(source, dict):
        raise ValueError('V3 source missing')
    identities_doc = config.get('identities')
    if not isinstance(identities_doc, list):
        raise ValueError('V3 identities missing')
    identities = tuple(sorted(
        (str(item['task_id']), int(item['trial']))
        for item in identities_doc
        if isinstance(item, dict)
    ))
    if identities != FROZEN_IDENTITIES:
        raise ValueError('V3 identities must equal frozen numeric primary X')
    root = config_file.parent.parent
    baseline_db = root / str(source.get('baseline_db', ''))
    grading_spec = root / str(source.get('grading_spec', ''))
    population_matrix = root / str(source.get('population_matrix', ''))
    baseline_hash = sha256_file(baseline_db)
    grading_hash = sha256_file(grading_spec)
    population_hash = sha256_file(population_matrix)
    renderer_hash = sha256_file(Path(__file__))
    if config.get('baseline_db_sha256') != baseline_hash:
        raise ValueError('V3 baseline source hash mismatch')
    if config.get('grading_spec_sha256') != grading_hash:
        raise ValueError('V3 grading spec hash mismatch')
    if config.get('renderer_code_sha256') != renderer_hash:
        raise ValueError('V3 renderer code hash mismatch')
    if config.get('population_sha256') != population_hash:
        raise ValueError('V3 population hash mismatch')
    return V3Contract(
        identities=identities,
        baseline_db=baseline_db,
        baseline_execution=str(source.get('baseline_execution')),
        baseline_db_sha256=baseline_hash,
        grading_spec=grading_spec,
        grading_spec_sha256=grading_hash,
        population_matrix=population_matrix,
        population_sha256=population_hash,
        renderer_code_sha256=renderer_hash,
        config_sha256=hashlib.sha256(config_bytes).hexdigest(),
    )


def load_source_rows(contract: V3Contract) -> dict[tuple[str, int], str]:
    expected = set(contract.identities)
    conn = open_readonly_source(contract.baseline_db)
    try:
        rows = conn.execute(
            'SELECT task_id, trial, raw_output FROM runs '
            'WHERE experiment_id=? AND is_warmup=0',
            (contract.baseline_execution,),
        ).fetchall()
    finally:
        conn.close()
    selected_rows = [
        ((str(task_id), int(trial)), str(raw_output))
        for task_id, trial, raw_output in rows
        if (str(task_id), int(trial)) in expected
    ]
    if len(selected_rows) != len(expected) or {identity for identity, _ in selected_rows} != expected:
        raise ValueError('V3 source identity mismatch')
    return dict(selected_rows)


def canonical_json(document: object) -> str:
    return json.dumps(document, indent=2, sort_keys=True) + '\n'
