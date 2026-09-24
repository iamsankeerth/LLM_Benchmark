"""Append-only SQLite storage for Judge Suite V1."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from analysis.judge import JudgeProtocol, JudgeSourceItem

JUDGE_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS judge_protocols(
  protocol_id TEXT PRIMARY KEY,
  status TEXT NOT NULL,
  eval_spec_hash TEXT NOT NULL,
  dataset_hash TEXT NOT NULL,
  rubric_hash TEXT NOT NULL,
  source_population_hash TEXT NOT NULL,
  created_at_utc TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS judge_source_items(
  source_item_id TEXT PRIMARY KEY,
  protocol_id TEXT NOT NULL,
  execution_id TEXT NOT NULL,
  model_config_id TEXT NOT NULL,
  task_id TEXT NOT NULL,
  trial INTEGER NOT NULL,
  raw_output_sha256 TEXT NOT NULL,
  precheck_status TEXT NOT NULL,
  precheck_json TEXT NOT NULL,
  UNIQUE(protocol_id, execution_id, task_id, trial)
);
CREATE TABLE IF NOT EXISTS judge_calls(
  call_id TEXT PRIMARY KEY,
  protocol_id TEXT NOT NULL,
  item_id TEXT,
  pair_id TEXT,
  judge_type TEXT NOT NULL,
  judge_model_config_id TEXT,
  judge_artifact_digest TEXT,
  orientation TEXT NOT NULL,
  request_sha256 TEXT NOT NULL,
  response_sha256 TEXT,
  status TEXT NOT NULL,
  parse_status TEXT NOT NULL,
  error_json TEXT,
  started_at_utc TEXT NOT NULL,
  ended_at_utc TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS judge_rubric_results(
  call_id TEXT NOT NULL,
  criterion_id TEXT NOT NULL,
  observed INTEGER NOT NULL,
  desired INTEGER NOT NULL,
  passed INTEGER NOT NULL,
  confidence TEXT NOT NULL,
  evidence_json TEXT NOT NULL,
  PRIMARY KEY(call_id, criterion_id)
);
CREATE TABLE IF NOT EXISTS judge_pairs(
  pair_id TEXT PRIMARY KEY,
  protocol_id TEXT NOT NULL,
  task_id TEXT NOT NULL,
  trial INTEGER NOT NULL,
  left_source_item_id TEXT NOT NULL,
  right_source_item_id TEXT NOT NULL,
  population_hash TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS judge_pair_results(
  call_id TEXT PRIMARY KEY,
  pair_id TEXT NOT NULL,
  orientation TEXT NOT NULL,
  decision TEXT NOT NULL,
  confidence TEXT NOT NULL,
  evidence_json TEXT NOT NULL,
  parse_status TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS judge_adjudications(
  adjudication_id TEXT PRIMARY KEY,
  protocol_id TEXT NOT NULL,
  item_or_pair_id TEXT NOT NULL,
  adjudicator_id TEXT NOT NULL,
  decision_json TEXT NOT NULL,
  rationale_sha256 TEXT,
  created_at_utc TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS judge_failures(
  failure_id TEXT PRIMARY KEY,
  protocol_id TEXT NOT NULL,
  item_or_pair_id TEXT NOT NULL,
  attempt INTEGER NOT NULL,
  error_type TEXT NOT NULL,
  action TEXT NOT NULL,
  retryable INTEGER NOT NULL,
  detail_sha256 TEXT,
  created_at_utc TEXT NOT NULL
);
"""


def init_judge_db(conn: sqlite3.Connection) -> None:
    conn.executescript(JUDGE_SCHEMA_SQL)
    conn.commit()


def source_population_hash(items: list[JudgeSourceItem]) -> str:
    import hashlib
    payload = '\n'.join(
        f'{item.source_item_id}|{item.raw_output_sha256}|{item.precheck_status}'
        for item in sorted(items, key=lambda item: item.source_item_id)
    )
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


def record_protocol(
    conn: sqlite3.Connection,
    *,
    protocol: JudgeProtocol,
    population_hash: str,
    rubric_hash: str,
    created_at_utc: str,
) -> None:
    document = protocol.document
    conn.execute(
        'INSERT OR REPLACE INTO judge_protocols'
        '(protocol_id,status,eval_spec_hash,dataset_hash,rubric_hash,'
        'source_population_hash,created_at_utc) VALUES(?,?,?,?,?,?,?)',
        (
            str(document['study']), str(document['status']),
            protocol.contract.hashes['spec_sha256'],
            protocol.contract.hashes['dataset_sha256'], rubric_hash,
            population_hash, created_at_utc,
        ),
    )
    conn.commit()


def record_source_item(
    conn: sqlite3.Connection,
    *,
    protocol_id: str,
    item: JudgeSourceItem,
) -> None:
    conn.execute(
        'INSERT OR REPLACE INTO judge_source_items'
        '(source_item_id,protocol_id,execution_id,model_config_id,task_id,trial,'
        'raw_output_sha256,precheck_status,precheck_json) VALUES(?,?,?,?,?,?,?,?,?)',
        (
            item.source_item_id, protocol_id, item.execution_id,
            item.model_config_id, item.task_id, item.trial,
            item.raw_output_sha256, item.precheck_status,
            json.dumps(item.precheck_details, sort_keys=True),
        ),
    )
    conn.commit()


def open_judge_db(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    init_judge_db(conn)
    return conn
