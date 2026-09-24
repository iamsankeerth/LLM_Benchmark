"""Append-only Eval-v1.1 regrade tests."""

from __future__ import annotations

import sqlite3
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from analysis.regrade import RegradeError, regrade_execution
from evals.contract import load_eval_contract
from inference.adapters import get_model_config
from storage.db import RunRecord, init_schema, insert_run


ROOT = Path(__file__).resolve().parents[1]


class RegradeTests(unittest.TestCase):
    def test_regrade_is_read_only_and_complete(self) -> None:
        contract = load_eval_contract(
            ROOT,
            'evals/specs/eval-v1.1-grading.yaml',
            freeze_path='evals/specs/eval-v1.1-grading.freeze.json',
        )
        entries = contract.grader_entries()
        model = get_model_config('qwen3-4b-q4')
        execution_id = 'full-baseline-v2__qwen3-4b-q4'
        with TemporaryDirectory() as tmp:
            db_path = Path(tmp) / 'source.db'
            conn = sqlite3.connect(db_path)
            init_schema(conn)
            for task_id, entry in entries.items():
                if entry['grading_status'] not in ('READY_DETERMINISTIC', 'READY_JUDGE'):
                    continue
                for trial in range(1, 4):
                    insert_run(conn, RunRecord(
                        experiment_id=execution_id,
                        model_config_id='qwen3-4b-q4',
                        task_id=task_id,
                        trial=trial,
                        run_kind='BASELINE',
                        run_config_hash='source',
                        is_warmup=False,
                        prompt='prompt',
                        rendered_prompt_sha256='rendered',
                        temperature=0.0,
                        num_ctx=model.num_ctx,
                        num_predict=2048,
                        template_sha256=model.template_sha256,
                        grader_verdict='FAIL',
                        status='COMPLETE',
                        started_at_utc='2026-09-24T00:00:00Z',
                        ended_at_utc='2026-09-24T00:00:01Z',
                        num_gpu=model.num_gpu,
                        stop_tokens=list(model.stop_tokens),
                        think=repr(model.think),
                        raw_output='answer',
                        grader_details_json='[]',
                        model_digest=model.ollama_model_digest,
                    ))
            conn.commit()
            conn.close()
            before = db_path.read_bytes()
            records = regrade_execution(
                db_path, execution_id=execution_id, contract=contract
            )
            self.assertEqual(len(records), 216)
            self.assertEqual(db_path.read_bytes(), before)
            self.assertTrue(all('old_verdict' in row for row in records))
            self.assertTrue(all('new_verdict' in row for row in records))
            self.assertTrue(all(row['new_spec_sha256'] == contract.hashes['spec_sha256'] for row in records))

    def test_missing_source_identity_is_refused(self) -> None:
        contract = load_eval_contract(ROOT, 'evals/specs/eval-v1.1-grading.yaml')
        with TemporaryDirectory() as tmp:
            db_path = Path(tmp) / 'empty.db'
            conn = sqlite3.connect(db_path)
            init_schema(conn)
            conn.close()
            with self.assertRaises(RegradeError):
                regrade_execution(
                    db_path, execution_id='full-baseline-v2__qwen3-4b-q4',
                    contract=contract,
                )


if __name__ == '__main__':
    unittest.main()
