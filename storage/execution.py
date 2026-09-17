"""Execution identity: spec x model derivation, collision rules, legacy fence.

- execution_id absent               -> CREATE
- execution_id exists + triple match -> RESUME
- execution_id exists + any mismatch -> MISMATCH (hard error, no writes)
- legacy execution (rows but no provenance triple) -> LEGACY_READ_ONLY:
  analysis/comparison only, never new generations or resume writes.

The triple is (experiment_config_hash, model_config_hash,
model_artifact_digest): frozen contract x declared config x observed bytes.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

CREATE = 'CREATE'
RESUME = 'RESUME'
MISMATCH = 'MISMATCH'
LEGACY_READ_ONLY = 'LEGACY_READ_ONLY'

PROVENANCE_SQL = """
CREATE TABLE IF NOT EXISTS execution_provenance(
  execution_id TEXT PRIMARY KEY,
  experiment_spec_id TEXT NOT NULL,
  model_config_id TEXT NOT NULL,
  experiment_config_hash TEXT NOT NULL,
  model_config_hash TEXT NOT NULL,
  model_artifact_digest TEXT,
  created_at_utc TEXT NOT NULL
);
"""


@dataclass(frozen=True)
class ExecutionResolution:
    decision: str
    detail: str


def ensure_provenance_table(conn: sqlite3.Connection) -> None:
    conn.executescript(PROVENANCE_SQL)
    conn.commit()


def record_execution_provenance(
    conn: sqlite3.Connection,
    *,
    execution_id: str,
    experiment_spec_id: str,
    model_config_id: str,
    experiment_config_hash: str,
    model_config_hash: str,
    model_artifact_digest: str | None,
    created_at_utc: str,
) -> None:
    conn.execute(
        'INSERT OR IGNORE INTO execution_provenance(execution_id,'
        ' experiment_spec_id, model_config_id, experiment_config_hash,'
        ' model_config_hash, model_artifact_digest, created_at_utc)'
        ' VALUES(?,?,?,?,?,?,?)',
        (
            execution_id, experiment_spec_id, model_config_id,
            experiment_config_hash, model_config_hash, model_artifact_digest,
            created_at_utc,
        ),
    )
    conn.commit()


def derive_execution_id(experiment_spec_id: str, model_config_id: str) -> str:
    return f'{experiment_spec_id}__{model_config_id}'


def resolve_execution(
    conn: sqlite3.Connection,
    execution_id: str,
    *,
    experiment_spec_id: str,
    model_config_id: str,
    experiment_config_hash: str,
    model_config_hash: str,
    model_artifact_digest: str | None,
) -> ExecutionResolution:
    """Decide CREATE / RESUME / MISMATCH / LEGACY_READ_ONLY (pure, no writes)."""
    experiment_row = conn.execute(
        'SELECT experiment_id FROM experiments WHERE experiment_id=?',
        (execution_id,),
    ).fetchone()
    provenance_row = conn.execute(
        'SELECT experiment_spec_id, model_config_id, experiment_config_hash,'
        ' model_config_hash, model_artifact_digest FROM execution_provenance'
        ' WHERE execution_id=?',
        (execution_id,),
    ).fetchone()
    if experiment_row is None and provenance_row is None:
        return ExecutionResolution(CREATE, f'new execution {execution_id}')
    if provenance_row is None:
        return ExecutionResolution(
            LEGACY_READ_ONLY,
            f'{execution_id} predates provenance triples: analysis only, no writes',
        )
    stored = {
        'experiment_spec_id': provenance_row[0],
        'model_config_id': provenance_row[1],
        'experiment_config_hash': provenance_row[2],
        'model_config_hash': provenance_row[3],
        'model_artifact_digest': provenance_row[4],
    }
    incoming = {
        'experiment_spec_id': experiment_spec_id,
        'model_config_id': model_config_id,
        'experiment_config_hash': experiment_config_hash,
        'model_config_hash': model_config_hash,
        'model_artifact_digest': model_artifact_digest,
    }
    mismatched = sorted(k for k in stored if stored[k] != incoming[k])
    if mismatched:
        return ExecutionResolution(
            MISMATCH,
            f'{execution_id} provenance mismatch on {mismatched}: refusing writes'
            ' (use an explicit __rerun-NN execution id for a fresh run)',
        )
    return ExecutionResolution(RESUME, f'resuming {execution_id}')
