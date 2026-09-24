"""Offline long-context-v1 dataset and runner tests."""

from __future__ import annotations

import sqlite3
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from analysis.long_context import load_long_context_tasks
from inference.ollama_client import GenerationResult
from scripts.run_long_context import run_model
from scripts.audit_long_context import audit


ROOT = Path(__file__).resolve().parents[1]


def _generation(prompt: str) -> GenerationResult:
    if 'Return only OK.' in prompt:
        text = 'OK'
    elif 'Return only 7.' in prompt:
        text = '7'
    elif 'Return only JSON with alpha' in prompt:
        text = '{"alpha":"A7","beta":"B3","gamma":"C9"}'
    elif 'largest VALUE among all METRIC' in prompt:
        text = '17'
    elif 'INSUFFICIENT_INFORMATION' in prompt:
        text = 'INSUFFICIENT_INFORMATION'
    else:
        text = 'BLUE-17'
    return GenerationResult(
        text=text, thinking='', done_reason='stop',
        total_duration_ns=200_000_000, load_duration_ns=10_000_000,
        prompt_eval_count=100, prompt_eval_cached_count=0,
        prompt_eval_duration_ns=90_000_000, eval_count=2,
        eval_duration_ns=60_000_000, request_start_ns=0,
        first_token_ns=100_000_000, request_end_ns=200_000_000,
    )


class LongContextTests(unittest.TestCase):
    def test_dataset_audit_and_task_count(self) -> None:
        result = audit()
        self.assertTrue(result['audit_pass'], result)
        self.assertEqual(len(load_long_context_tasks(ROOT)), 21)

    def test_fake_runner_persists_exact_population(self) -> None:
        calls: list[str] = []

        def fake_generate(_base_url: str, request: object, **_kwargs: object) -> GenerationResult:
            prompt = getattr(request, 'prompt')
            calls.append(prompt)
            return _generation(prompt)

        with TemporaryDirectory() as tmp:
            db = Path(tmp) / 'long.db'
            with patch('scripts.run_long_context._wait_model_absent', return_value=True), patch(
                'scripts.run_long_context.ollama_stop', return_value=(True, 'stopped')
            ), patch(
                'scripts.run_long_context.wait_until_unloaded', return_value=True
            ):
                result = run_model(
                    ROOT, model_id='qwen3-4b-q4', db_path=db,
                    generate_fn=fake_generate,
                )
            self.assertEqual(result['trial_count'], 63)
            self.assertEqual(result['request_accounting']['total'], 87)
            self.assertEqual(len(calls), 87)
            conn = sqlite3.connect(db)
            self.assertEqual(conn.execute(
                "SELECT COUNT(*) FROM runs WHERE run_kind='WARMUP'"
            ).fetchone()[0], 2)
            self.assertEqual(conn.execute(
                "SELECT COUNT(*) FROM runs WHERE run_kind='LONG_CONTEXT'"
            ).fetchone()[0], 63)
            conn.close()


if __name__ == '__main__':
    unittest.main()
