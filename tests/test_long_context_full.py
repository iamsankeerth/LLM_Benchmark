"""Offline tests for the full long-context cohort freeze."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import yaml

from analysis.long_context import load_long_context_tasks
from scripts.generate_long_context_full import build_full


ROOT = Path(__file__).resolve().parents[1]


class LongContextFullTests(unittest.TestCase):
    def test_full_cohort_validates_exact_population(self) -> None:
        config = yaml.safe_load(
            (ROOT / 'configs/long-context-v1.yaml').read_text(encoding='utf-8')
        )
        full_models = [str(value) for value in config['full_models']]
        pilot_models = {str(value) for value in config['pilot_models']}
        dataset_sha256 = hashlib.sha256(
            (ROOT / 'evals/datasets/long-context-v1/tasks.jsonl').read_bytes()
        ).hexdigest()
        grading_sha256 = hashlib.sha256(
            (ROOT / 'evals/specs/long-context-v1-grading.yaml').read_bytes()
        ).hexdigest()
        task_ids = [task.task_id for task in load_long_context_tasks(ROOT)]
        report_paths: dict[str, Path] = {}
        db_paths: dict[str, Path] = {}
        with TemporaryDirectory() as tmp:
            temp = Path(tmp)
            for model_id in full_models:
                if model_id in pilot_models:
                    report_paths[model_id] = (
                        ROOT / f'results/reports/long-context-v1__{model_id}.json'
                    )
                    db_paths[model_id] = (
                        ROOT / f'results/local/long-context-v1__{model_id}.db'
                    )
                    continue
                db_path = temp / f'{model_id}.db'
                conn = sqlite3.connect(db_path)
                conn.execute(
                    'CREATE TABLE runs(experiment_id TEXT,task_id TEXT,trial INTEGER,'
                    'run_kind TEXT,is_warmup INTEGER,status TEXT,grader_verdict TEXT,'
                    'model_digest TEXT)'
                )
                execution_id = f'long-context-v1__{model_id}'
                conn.executemany(
                    'INSERT INTO runs VALUES(?,?,?,?,?,?,?,?)',
                    [
                        (execution_id, f'WARMUP-{trial}', trial, 'WARMUP', 1,
                         'COMPLETE', 'WARMUP', 'sha256:test')
                        for trial in (1, 2)
                    ],
                )
                conn.executemany(
                    'INSERT INTO runs VALUES(?,?,?,?,?,?,?,?)',
                    [
                        (execution_id, task_id, trial, 'LONG_CONTEXT', 0,
                         'COMPLETE', 'PASS', 'sha256:test')
                        for task_id in task_ids
                        for trial in (1, 2, 3)
                    ],
                )
                conn.commit()
                conn.close()
                report_path = temp / f'{model_id}.json'
                report_path.write_text(json.dumps({
                    'model_config_id': model_id,
                    'task_count': 21,
                    'trial_count': 63,
                    'passes': 63,
                    'trial_accuracy': 1.0,
                    'by_length': {},
                    'dataset_sha256': dataset_sha256,
                    'grading_spec_sha256': grading_sha256,
                    'request_accounting': {'total': 87},
                    'task_id_coverage': True,
                }, sort_keys=True) + '\n', encoding='utf-8')
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
        self.assertEqual(report['measured_trials'], 693)
        self.assertEqual(report['request_accounting']['total_requests_including_sealed_pilot'], 957)
        self.assertEqual(freeze['status'], 'FULL_SEALED')
        self.assertEqual(len(freeze['models']), 11)


if __name__ == '__main__':
    unittest.main()
