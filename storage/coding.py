"""Append-only persistence for Coding v1 candidate and sandbox results."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

CODING_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS coding_runs(
  execution_id TEXT NOT NULL,
  model_config_id TEXT NOT NULL,
  task_id TEXT NOT NULL,
  trial INTEGER NOT NULL,
  language TEXT NOT NULL,
  entrypoint TEXT NOT NULL,
  raw_output_sha256 TEXT NOT NULL,
  candidate_sha256 TEXT,
  extraction_method TEXT,
  static_passed INTEGER,
  static_failure_kind TEXT,
  sandbox_status TEXT NOT NULL,
  functional_passed INTEGER,
  static_passed_tests INTEGER,
  total_tests INTEGER,
  failure_kind TEXT,
  worker_result_json TEXT,
  started_at_utc TEXT NOT NULL,
  ended_at_utc TEXT NOT NULL,
  PRIMARY KEY(execution_id, task_id, trial)
);
"""


def open_coding_db(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.executescript(CODING_SCHEMA_SQL)
    conn.commit()
    return conn


def record_coding_run(
    conn: sqlite3.Connection,
    *,
    execution_id: str,
    model_config_id: str,
    task_id: str,
    trial: int,
    language: str,
    entrypoint: str,
    raw_output_sha256: str,
    candidate_sha256: str | None,
    extraction_method: str | None,
    static_result: dict[str, Any],
    sandbox_status: str,
    worker_result: dict[str, Any] | None,
    started_at_utc: str,
    ended_at_utc: str,
) -> None:
    tests = (worker_result or {}).get('tests', {})
    conn.execute(
        'INSERT OR REPLACE INTO coding_runs'
        '(execution_id,model_config_id,task_id,trial,language,entrypoint,'
        'raw_output_sha256,candidate_sha256,extraction_method,static_passed,'
        'static_failure_kind,sandbox_status,functional_passed,static_passed_tests,'
        'total_tests,failure_kind,worker_result_json,started_at_utc,ended_at_utc)'
        ' VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)',
        (
            execution_id, model_config_id, task_id, trial, language, entrypoint,
            raw_output_sha256, candidate_sha256, extraction_method,
            int(bool(static_result.get('passed'))), static_result.get('failure_kind'),
            sandbox_status, int(bool((worker_result or {}).get('passed'))),
            tests.get('static_passed'), tests.get('total'),
            (worker_result or {}).get('failure_kind'),
            json.dumps(worker_result, sort_keys=True) if worker_result else None,
            started_at_utc, ended_at_utc,
        ),
    )
    conn.commit()
