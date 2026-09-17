"""Execution identity tests: derivation, collision rules, legacy fence."""

from __future__ import annotations

import sqlite3
import unittest

from inference.adapters import get_model_config, model_config_hash
from storage.db import connect, create_experiment, init_schema
from storage.execution import (
    CREATE,
    LEGACY_READ_ONLY,
    MISMATCH,
    RESUME,
    derive_execution_id,
    ensure_provenance_table,
    record_execution_provenance,
    resolve_execution,
)

SPEC = 'full-baseline-v1'
MODEL = 'qwen3-4b-q5'
EXE = 'full-baseline-v1__qwen3-4b-q5'
TRIPLE = {
    'experiment_spec_id': SPEC,
    'model_config_id': MODEL,
    'experiment_config_hash': 'e' * 64,
    'model_config_hash': 'm' * 64,
    'model_artifact_digest': 'd' * 64,
}


def _db() -> sqlite3.Connection:
    conn = connect(':memory:')
    init_schema(conn)
    ensure_provenance_table(conn)
    return conn


class ExecutionIdentityTests(unittest.TestCase):
    def test_derivation(self) -> None:
        self.assertEqual(
            derive_execution_id('full-baseline-v1', 'qwen3-4b-q5'),
            'full-baseline-v1__qwen3-4b-q5',
        )

    def test_absent_creates(self) -> None:
        conn = _db()
        resolution = resolve_execution(conn, EXE, **TRIPLE)
        self.assertEqual(resolution.decision, CREATE)
        conn.close()

    def test_matching_triple_resumes(self) -> None:
        conn = _db()
        create_experiment(
            conn, experiment_id=EXE, name=EXE, config_hash='c' * 64,
            config_yaml='x', created_at_utc='2026-09-18T00:00:00Z',
        )
        record_execution_provenance(
            conn, execution_id=EXE, created_at_utc='2026-09-18T00:00:00Z',
            **TRIPLE,
        )
        resolution = resolve_execution(conn, EXE, **TRIPLE)
        self.assertEqual(resolution.decision, RESUME)
        conn.close()

    def test_any_mismatch_is_hard_error(self) -> None:
        for key, bad in (
            ('experiment_config_hash', 'f' * 64),
            ('model_config_hash', 'n' * 64),
            ('model_artifact_digest', 'e' * 64),
            ('model_config_id', 'qwen3-4b-q4'),
        ):
            conn = _db()
            create_experiment(
                conn, experiment_id=EXE, name=EXE, config_hash='c' * 64,
                config_yaml='x', created_at_utc='2026-09-18T00:00:00Z',
            )
            record_execution_provenance(
                conn, execution_id=EXE, created_at_utc='2026-09-18T00:00:00Z',
                **TRIPLE,
            )
            incoming = dict(TRIPLE)
            incoming[key] = bad
            resolution = resolve_execution(conn, EXE, **incoming)
            self.assertEqual(resolution.decision, MISMATCH, key)
            conn.close()

    def test_legacy_execution_is_read_only(self) -> None:
        # Pre-triple rows (all Q4-era experiments): resolvable for analysis,
        # refused for writes.
        conn = _db()
        create_experiment(
            conn, experiment_id='creator-baseline-v1', name='creator-baseline-v1',
            config_hash='c' * 64, config_yaml='x',
            created_at_utc='2026-09-17T00:00:00Z',
        )
        resolution = resolve_execution(conn, 'creator-baseline-v1', **TRIPLE)
        self.assertEqual(resolution.decision, LEGACY_READ_ONLY)
        conn.close()


class ModelConfigHashTests(unittest.TestCase):
    def test_stable_and_config_sensitive(self) -> None:
        q5 = get_model_config('qwen3-4b-q5')
        again = get_model_config('qwen3-4b-q5')
        self.assertEqual(model_config_hash(q5), model_config_hash(again))
        self.assertEqual(len(model_config_hash(q5)), 64)
        q4 = get_model_config('qwen3-4b-q4')
        self.assertNotEqual(model_config_hash(q4), model_config_hash(q5))

    def test_live_digest_excluded_by_construction(self) -> None:
        # A re-pulled weights blob under the same tag changes the observed
        # artifact digest but must NOT change the declared config hash.
        import dataclasses

        q5 = get_model_config('qwen3-4b-q5')
        repulled = dataclasses.replace(q5, ollama_model_digest='f' * 64)
        self.assertEqual(model_config_hash(q5), model_config_hash(repulled))


if __name__ == '__main__':
    unittest.main()
