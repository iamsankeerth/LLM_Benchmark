"""Runner warmup-session tests: fresh counter + mid-run re-warm (offline).

The live model is replaced by a scripted generate(); grading, storage and
profiling run for real against a temp DB.
"""

from __future__ import annotations

import argparse
import sqlite3
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from inference.eligibility import EligibilityResult
from inference.ollama_client import GenerationResult
from scripts import run_benchmark


def _generation(
    text: str = 'ok', *, load_ms: float = 50.0, eval_count: int = 2
) -> GenerationResult:
    load_ns = int(load_ms * 1_000_000)
    return GenerationResult(
        text=text, thinking='', done_reason='stop',
        total_duration_ns=load_ns + 100_000_000,
        load_duration_ns=load_ns,
        prompt_eval_count=10, prompt_eval_cached_count=0,
        prompt_eval_duration_ns=90_000_000,
        eval_count=eval_count, eval_duration_ns=60_000_000,
        request_start_ns=0, first_token_ns=100_000_000,
        request_end_ns=200_000_000,
    )


def _eligible() -> EligibilityResult:
    return EligibilityResult(
        eligible=True, status='ELIGIBLE_GPU', model_size_bytes=100,
        size_vram_bytes=100, gpu_residency_ratio=1.0,
        observed_digest='d' * 64, eligibility_evidence_json='{}',
    )


def _namespace(db: str, **overrides: object) -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    base: dict[str, object] = {
        'config': str(root / 'configs/smoke-3.yaml'),
        'model': 'qwen3-4b-q4', 'db': db, 'base_url': 'http://127.0.0.1:9',
        'resume': False, 'execution_id': None, 'num_predict': None,
        'timeout_s': 30.0, 'max_tasks': 1,
    }
    base.update(overrides)
    return argparse.Namespace(**base)


class WarmupSessionTests(unittest.TestCase):
    def _run_scripted(self, db: str, script: list[GenerationResult]) -> int:
        calls = {'n': 0}

        def fake_generate(base_url: str, request: object, **kwargs: object) -> GenerationResult:
            result = script[min(calls['n'], len(script) - 1)]
            calls['n'] += 1
            return result

        with patch.object(run_benchmark, 'generate', side_effect=fake_generate):
            with patch.object(
                run_benchmark, 'check_eligibility', return_value=_eligible()
            ):
                return run_benchmark.run_experiment(_namespace(db))

    def test_fresh_warmups_then_measured(self) -> None:
        with TemporaryDirectory() as tmp:
            db = str(Path(tmp) / 'bench.db')
            code = self._run_scripted(db, [
                _generation('probe'),      # load probe (unrecorded)
                _generation('OK'),         # warmup 1
                _generation('7'),          # warmup 2
                _generation('answer 00:55'),  # measured Q071
            ])
            self.assertEqual(code, 0)
            conn = sqlite3.connect(db)
            warmups = conn.execute(
                "SELECT trial FROM runs WHERE run_kind='WARMUP' ORDER BY trial"
            ).fetchall()
            measured = conn.execute(
                'SELECT task_id, trial, status FROM runs WHERE is_warmup=0'
            ).fetchall()
            conn.close()
            self.assertEqual([r[0] for r in warmups], [1, 2])
            self.assertEqual(len(measured), 1)
            self.assertEqual(measured[0][2], 'COMPLETE')

    def test_reload_spike_triggers_rewarm(self) -> None:
        with TemporaryDirectory() as tmp:
            db = str(Path(tmp) / 'bench.db')
            code = self._run_scripted(db, [
                _generation('probe'),
                _generation('OK'),
                _generation('7'),
                # Measured generation with reload evidence (2.5 s load).
                _generation('answer 00:55', load_ms=2500.0),
            ])
            self.assertEqual(code, 0)
            conn = sqlite3.connect(db)
            warmups = conn.execute(
                "SELECT trial FROM runs WHERE run_kind='WARMUP' ORDER BY trial"
            ).fetchall()
            measured = conn.execute(
                'SELECT COUNT(*) FROM runs WHERE is_warmup=0'
            ).fetchone()[0]
            conn.close()
            # 2 initial + 2 automatic re-warm trials, continuing numbering.
            self.assertEqual([r[0] for r in warmups], [1, 2, 3, 4])
            self.assertEqual(measured, 1)

    def test_no_spike_no_rewarm(self) -> None:
        with TemporaryDirectory() as tmp:
            db = str(Path(tmp) / 'bench.db')
            self._run_scripted(db, [
                _generation('probe'),
                _generation('OK', load_ms=9000.0),  # warmup load spike: ignored
                _generation('7'),
                _generation('answer 00:55', load_ms=999.0),  # under threshold
            ])
            conn = sqlite3.connect(db)
            warmups = conn.execute(
                "SELECT COUNT(*) FROM runs WHERE run_kind='WARMUP'"
            ).fetchone()[0]
            conn.close()
            self.assertEqual(warmups, 2)


if __name__ == '__main__':
    unittest.main()
