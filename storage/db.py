"""SQLite persistence: experiments + per-generation run rows.

Execution identity (resume-safe)::

    (experiment_id, model_config_id, task_id, trial, run_kind, run_config_hash)

``run_config_hash`` canonicalizes the effective generation configuration,
so the same task at temperature 0 and 0.7 (or different ctx/templates)
can coexist without collision and resume never skips the wrong row.
Warm-ups are rows with ``run_kind='WARMUP'`` / ``is_warmup=1``; every
measured aggregate filters them out.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass, field
from typing import Any, Mapping

RUN_KINDS = frozenset(
    {'WARMUP', 'BASELINE', 'TEMPERATURE', 'RELIABILITY', 'LONG_CONTEXT', 'JUDGE'}
)

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS experiments(
  experiment_id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  config_hash TEXT NOT NULL,
  config_yaml TEXT NOT NULL,
  created_at_utc TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS runs(
  run_id INTEGER PRIMARY KEY AUTOINCREMENT,
  experiment_id TEXT NOT NULL REFERENCES experiments(experiment_id),
  model_config_id TEXT NOT NULL,
  task_id TEXT NOT NULL,
  trial INTEGER NOT NULL,
  run_kind TEXT NOT NULL,
  run_config_hash TEXT NOT NULL,
  is_warmup INTEGER NOT NULL,
  prompt TEXT NOT NULL,
  rendered_prompt_sha256 TEXT NOT NULL,
  raw_output TEXT NOT NULL DEFAULT '',
  thinking_output TEXT NOT NULL DEFAULT '',
  done_reason TEXT,
  temperature REAL NOT NULL,
  num_ctx INTEGER NOT NULL,
  num_predict INTEGER NOT NULL,
  num_gpu INTEGER,
  stop_tokens_json TEXT NOT NULL DEFAULT '[]',
  think TEXT NOT NULL DEFAULT 'default',
  template_sha256 TEXT NOT NULL,
  ttft_ms REAL,
  client_e2e_ms REAL,
  server_total_duration_ms REAL,
  server_load_duration_ms REAL,
  prompt_eval_count INTEGER,
  prompt_eval_cached_count INTEGER,
  prompt_eval_uncached_count INTEGER,
  prompt_cache_ratio REAL,
  prompt_eval_duration_ms REAL,
  eval_count INTEGER,
  eval_duration_ms REAL,
  prefill_compute_tok_s REAL,
  prefill_cache_state TEXT,
  decode_tok_s REAL,
  decode_ms_per_token REAL,
  client_overhead_ms REAL,
  ram_baseline_mb REAL,
  ram_peak_mb REAL,
  vram_baseline_mib REAL,
  vram_peak_mib REAL,
  vram_post_mib REAL,
  vram_total_mib REAL,
  grader_verdict TEXT NOT NULL,
  grader_details_json TEXT NOT NULL DEFAULT '[]',
  status TEXT NOT NULL,
  error TEXT,
  eligibility_status TEXT,
  gpu_residency_ratio REAL,
  eligibility_evidence_json TEXT,
  started_at_utc TEXT NOT NULL,
  ended_at_utc TEXT NOT NULL,
  ollama_version TEXT,
  model_digest TEXT,
  UNIQUE(experiment_id, model_config_id, task_id, trial, run_kind, run_config_hash)
);
CREATE INDEX IF NOT EXISTS idx_runs_experiment
  ON runs(experiment_id, run_kind, is_warmup);
"""


def run_config_hash(effective: Mapping[str, Any]) -> str:
    """Canonical sha256 of the effective generation configuration."""
    canonical = json.dumps(
        dict(effective), sort_keys=True, separators=(',', ':'), ensure_ascii=False
    )
    return hashlib.sha256(canonical.encode('utf-8')).hexdigest()


def effective_generation_config(
    *,
    ollama_identifier: str,
    quantization: str,
    mode: str,
    temperature: float,
    num_ctx: int,
    num_predict: int,
    num_gpu: int | None,
    stop_tokens: tuple[str, ...] | list[str],
    think: bool | None,
    template_sha256: str,
) -> dict[str, Any]:
    """Build the canonical effective-config mapping (task prompt excluded;
    task identity travels in task_id, not in the config hash)."""
    return {
        'ollama_identifier': ollama_identifier,
        'quantization': quantization,
        'mode': mode,
        'temperature': temperature,
        'num_ctx': num_ctx,
        'num_predict': num_predict,
        'num_gpu': num_gpu,
        'stop_tokens': list(stop_tokens),
        'think': think,
        'template_sha256': template_sha256,
    }


@dataclass
class RunRecord:
    experiment_id: str
    model_config_id: str
    task_id: str
    trial: int
    run_kind: str
    run_config_hash: str
    is_warmup: bool
    prompt: str
    rendered_prompt_sha256: str
    temperature: float
    num_ctx: int
    num_predict: int
    template_sha256: str
    grader_verdict: str
    status: str
    started_at_utc: str
    ended_at_utc: str
    raw_output: str = ''
    thinking_output: str = ''
    done_reason: str | None = None
    num_gpu: int | None = None
    stop_tokens: list[str] = field(default_factory=list)
    think: str = 'default'
    ttft_ms: float | None = None
    client_e2e_ms: float | None = None
    server_total_duration_ms: float | None = None
    server_load_duration_ms: float | None = None
    prompt_eval_count: int | None = None
    prompt_eval_cached_count: int | None = None
    prompt_eval_uncached_count: int | None = None
    prompt_cache_ratio: float | None = None
    prompt_eval_duration_ms: float | None = None
    eval_count: int | None = None
    eval_duration_ms: float | None = None
    prefill_compute_tok_s: float | None = None
    prefill_cache_state: str | None = None
    decode_tok_s: float | None = None
    decode_ms_per_token: float | None = None
    client_overhead_ms: float | None = None
    ram_baseline_mb: float | None = None
    ram_peak_mb: float | None = None
    vram_baseline_mib: float | None = None
    vram_peak_mib: float | None = None
    vram_post_mib: float | None = None
    vram_total_mib: float | None = None
    grader_details_json: str = '[]'
    error: str | None = None
    eligibility_status: str | None = None
    gpu_residency_ratio: float | None = None
    eligibility_evidence_json: str | None = None
    ollama_version: str | None = None
    model_digest: str | None = None


_RUN_COLUMNS = (
    'experiment_id model_config_id task_id trial run_kind run_config_hash '
    'is_warmup prompt rendered_prompt_sha256 raw_output thinking_output done_reason '
    'temperature num_ctx num_predict num_gpu stop_tokens_json think template_sha256 '
    'ttft_ms client_e2e_ms server_total_duration_ms server_load_duration_ms '
    'prompt_eval_count prompt_eval_cached_count prompt_eval_uncached_count '
    'prompt_cache_ratio prompt_eval_duration_ms eval_count eval_duration_ms '
    'prefill_compute_tok_s prefill_cache_state decode_tok_s decode_ms_per_token '
    'client_overhead_ms ram_baseline_mb ram_peak_mb vram_baseline_mib vram_peak_mib '
    'vram_post_mib vram_total_mib grader_verdict grader_details_json status error '
    'eligibility_status gpu_residency_ratio eligibility_evidence_json '
    'started_at_utc ended_at_utc ollama_version model_digest'
).split()


def connect(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.execute('PRAGMA foreign_keys = ON')
    return conn


def init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA_SQL)
    conn.commit()


def create_experiment(
    conn: sqlite3.Connection,
    *,
    experiment_id: str,
    name: str,
    config_hash: str,
    config_yaml: str,
    created_at_utc: str,
) -> None:
    conn.execute(
        'INSERT INTO experiments(experiment_id, name, config_hash, config_yaml,'
        ' created_at_utc) VALUES(?,?,?,?,?)',
        (experiment_id, name, config_hash, config_yaml, created_at_utc),
    )
    conn.commit()


def insert_run(conn: sqlite3.Connection, record: RunRecord) -> int:
    """Insert one run row; caller commits (checkpoint per generation)."""
    if record.run_kind not in RUN_KINDS:
        raise ValueError(f'unknown run_kind {record.run_kind!r}')
    values = (
        record.experiment_id, record.model_config_id, record.task_id, record.trial,
        record.run_kind, record.run_config_hash, 1 if record.is_warmup else 0,
        record.prompt, record.rendered_prompt_sha256, record.raw_output,
        record.thinking_output, record.done_reason, record.temperature,
        record.num_ctx, record.num_predict, record.num_gpu,
        json.dumps(record.stop_tokens), record.think, record.template_sha256,
        record.ttft_ms, record.client_e2e_ms, record.server_total_duration_ms,
        record.server_load_duration_ms, record.prompt_eval_count,
        record.prompt_eval_cached_count, record.prompt_eval_uncached_count,
        record.prompt_cache_ratio, record.prompt_eval_duration_ms,
        record.eval_count, record.eval_duration_ms, record.prefill_compute_tok_s,
        record.prefill_cache_state, record.decode_tok_s, record.decode_ms_per_token,
        record.client_overhead_ms, record.ram_baseline_mb, record.ram_peak_mb,
        record.vram_baseline_mib, record.vram_peak_mib, record.vram_post_mib,
        record.vram_total_mib, record.grader_verdict, record.grader_details_json,
        record.status, record.error, record.eligibility_status,
        record.gpu_residency_ratio, record.eligibility_evidence_json,
        record.started_at_utc, record.ended_at_utc, record.ollama_version,
        record.model_digest,
    )
    placeholders = ','.join(['?'] * len(_RUN_COLUMNS))
    cursor = conn.execute(
        f'INSERT INTO runs({",".join(_RUN_COLUMNS)}) VALUES({placeholders})', values
    )
    return int(cursor.lastrowid or 0)


def completed_identities(
    conn: sqlite3.Connection, experiment_id: str, run_kind: str
) -> set[tuple[str, str, int, str]]:
    """Resume set: (model_config_id, task_id, trial, run_config_hash) for
    COMPLETE rows. Resume skips exactly these tuples."""
    rows = conn.execute(
        'SELECT model_config_id, task_id, trial, run_config_hash FROM runs'
        " WHERE experiment_id=? AND run_kind=? AND status='COMPLETE'",
        (experiment_id, run_kind),
    ).fetchall()
    return {(str(m), str(t), int(n), str(h)) for m, t, n, h in rows}


def fetch_measured(conn: sqlite3.Connection, experiment_id: str) -> list[sqlite3.Row]:
    """Measured rows only: warm-ups excluded from every aggregate."""
    conn.row_factory = sqlite3.Row
    try:
        return conn.execute(
            'SELECT * FROM runs WHERE experiment_id=? AND is_warmup=0 ORDER BY run_id',
            (experiment_id,),
        ).fetchall()
    finally:
        conn.row_factory = None
