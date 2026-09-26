"""Offline tests for the full live coding cohort freeze."""

from __future__ import annotations

import json
import sqlite3
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from analysis.coding import load_fixture_manifest
from scripts.generate_coding_full import build_full


ROOT = Path(__file__).resolve().parents[1]


class CodingFullTests(unittest.TestCase):
    def test_full_cohort_validates_exact_population(self) -> None:
        manifest = load_fixture_manifest(
            ROOT / 'evals/fixtures/coding-v1/fixture-manifest.json'
        )
        task_ids = set(manifest['tasks'])
        report_paths: dict[str, Path] = {}
        db_paths: dict[str, Path] = {}
        with TemporaryDirectory() as tmp:
            temp = Path(tmp)
            for index in range(11):
                model_id = (
                    'qwen3-4b-q4', 'qwen3-4b-q5', 'llama3.2-3b-q4',
                    'llama3.2-3b-q5', 'llama3.2-3b-q6', 'gemma-3n-e2b-q4',
                    'gemma-3n-e2b-q5', 'smolm2-1.7b-q4', 'smolm2-1.7b-q5',
                    'smolm2-1.7b-q6', 'smolm2-1.7b-q8',
                )[index]
                db_path = temp / f'{model_id}.db'
                conn = sqlite3.connect(db_path)
                conn.execute(
                    'CREATE TABLE coding_runs(execution_id TEXT,task_id TEXT,trial INTEGER,'
                    'static_passed INTEGER,functional_passed INTEGER,sandbox_status TEXT)'
                )
                conn.executemany(
                    'INSERT INTO coding_runs VALUES(?,?,?,?,?,?)',
                    [
                        (f'coding-v1__{model_id}', task_id, trial, 1, 1, 'PASS')
                        for task_id in sorted(task_ids)
                        for trial in (1, 2, 3)
                    ],
                )
                conn.commit()
                conn.close()
                report_path = temp / f'{model_id}.json'
                report_path.write_text(json.dumps({
                    'model_config_id': model_id,
                    'task_count': 8,
                    'trial_count': 24,
                }) + '\n', encoding='utf-8')
                report_paths[model_id] = report_path
                db_paths[model_id] = db_path
            cleanup = temp / 'cleanup.json'
            cleanup.write_text(json.dumps({'status': 'VERIFIED'}) + '\n', encoding='utf-8')
            report, freeze = build_full(
                ROOT,
                report_paths=report_paths,
                db_paths=db_paths,
                cleanup_path=cleanup,
            )
        self.assertEqual(report['model_count'], 11)
        self.assertEqual(report['trial_count'], 264)
        self.assertEqual(report['sandbox_errors'], 0)
        self.assertEqual(freeze['status'], 'LIVE_SEALED')


if __name__ == '__main__':
    unittest.main()
