"""Offline fixed-budget temperature-runner tests."""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import unittest
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Callable, Iterator
from unittest.mock import MagicMock, patch

from inference.eligibility import EligibilityResult

from inference.ollama_client import GenerationRequest, GenerationResult, OllamaTimeoutError
from scripts import run_benchmark
from storage.db import create_experiment, init_schema
from storage.execution import ensure_provenance_table, record_execution_provenance


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'configs/temperature-study-v1-t07.yaml'


@contextmanager
def _isolated_results(tmp: str) -> Iterator[None]:
    with patch.dict(os.environ, {'LLM_BENCH_RESULTS_ROOT': tmp}):
        yield


def _generation(*, load_ms: float = 10.0) -> GenerationResult:
    load_ns = int(load_ms * 1_000_000)
    return GenerationResult(
        text='answer', thinking='', done_reason='stop',
        total_duration_ns=200_000_000, load_duration_ns=load_ns,
        prompt_eval_count=10, prompt_eval_cached_count=0,
        prompt_eval_duration_ns=90_000_000, eval_count=2,
        eval_duration_ns=60_000_000, request_start_ns=0,
        first_token_ns=100_000_000, request_end_ns=200_000_000,
    )


def _eligible() -> EligibilityResult:
    return EligibilityResult(
        eligible=True, status='ELIGIBLE_GPU', model_size_bytes=100,
        size_vram_bytes=100, gpu_residency_ratio=1.0,
        observed_digest='d' * 64, eligibility_evidence_json='{}',
    )


def _args(db: str, **overrides: object) -> argparse.Namespace:
    values: dict[str, object] = {
        'config': str(CONFIG), 'model': 'qwen3-4b-q4', 'db': db,
        'base_url': 'http://127.0.0.1:9', 'resume': False,
        'execution_id': None, 'num_predict': None, 'timeout_s': 30.0,
        'max_tasks': None, 'ollama_bin': 'ollama-test',
    }
    values.update(overrides)
    return argparse.Namespace(**values)


class FixedBudgetRunnerTests(unittest.TestCase):
    def _run(
        self,
        db: str,
        generate: Callable[..., GenerationResult],
        *,
        overrides: dict[str, object] | None = None,
    ) -> tuple[int, MagicMock]:
        with patch.object(run_benchmark, 'generate', side_effect=generate), patch.object(
            run_benchmark, 'check_eligibility', return_value=_eligible()
        ), patch.object(
            run_benchmark, 'ollama_stop', return_value=(True, 'stopped')
        ) as stop, patch.object(
            run_benchmark, 'wait_until_unloaded', return_value=True
        ), patch.object(
            run_benchmark, 'ollama_version', return_value='0.34.4'
        ), _isolated_results(str(Path(db).parent)):
            code = run_benchmark.run_experiment(
                _args(db, **(overrides or {}))
            )
        return code, stop

    def test_success_has_exact_request_and_row_budget(self) -> None:
        with TemporaryDirectory() as tmp:
            db = str(Path(tmp) / 'arm.db')
            requests: list[GenerationRequest] = []

            def generate(_base_url: str, request: GenerationRequest, **_kwargs: object) -> GenerationResult:
                requests.append(request)
                return _generation()

            code, _ = self._run(db, generate)
            self.assertEqual(code, 0)
            self.assertEqual(len(requests), 78)
            self.assertEqual(requests[0].temperature, 0.7)
            self.assertEqual(requests[0].num_predict, 1)
            self.assertEqual(requests[0].num_ctx, 4096)
            self.assertEqual(requests[0].num_gpu, 99)
            self.assertTrue(requests[0].raw)
            self.assertFalse(requests[0].think)
            self.assertEqual(requests[0].stop, ('<|im_start|>', '<|im_end|>'))
            conn = sqlite3.connect(db)
            self.assertEqual(conn.execute(
                "SELECT COUNT(*) FROM runs WHERE run_kind='WARMUP'"
            ).fetchone()[0], 2)
            self.assertEqual(conn.execute(
                'SELECT COUNT(*) FROM runs WHERE is_warmup=0'
            ).fetchone()[0], 75)
            conn.close()
            summary = json.loads(
                (Path(tmp) / 'summaries/temperature-study-v1-t07__qwen3-4b-q4.json').read_text(encoding='utf-8')
            )
            self.assertEqual(summary['request_budget']['total_calls'], 78)

    def test_timeout_stops_without_retry_or_measured_row(self) -> None:
        with TemporaryDirectory() as tmp:
            db = str(Path(tmp) / 'arm.db')
            calls = {'n': 0}

            def generate(_base_url: str, _request: GenerationRequest, **_kwargs: object) -> GenerationResult:
                calls['n'] += 1
                if calls['n'] == 4:
                    raise OllamaTimeoutError('fixed budget timeout')
                return _generation()

            code, stop = self._run(db, generate)
            self.assertEqual(code, run_benchmark.MEASUREMENT_FAILED_EXIT)
            self.assertEqual(calls['n'], 4)
            self.assertGreaterEqual(stop.call_count, 2)
            conn = sqlite3.connect(db)
            self.assertEqual(conn.execute(
                "SELECT COUNT(*) FROM runs WHERE run_kind='WARMUP'"
            ).fetchone()[0], 2)
            self.assertEqual(conn.execute(
                'SELECT COUNT(*) FROM runs WHERE is_warmup=0'
            ).fetchone()[0], 0)
            conn.close()
            self.assertTrue((Path(tmp) / 'summaries/temperature-study-v1-t07__qwen3-4b-q4-fixed-budget-failure.json').exists())

    def test_reload_evidence_stops_before_persistence(self) -> None:
        with TemporaryDirectory() as tmp:
            db = str(Path(tmp) / 'arm.db')
            calls = {'n': 0}

            def generate(_base_url: str, _request: GenerationRequest, **_kwargs: object) -> GenerationResult:
                calls['n'] += 1
                return _generation(load_ms=1001.0 if calls['n'] == 4 else 10.0)

            code, _ = self._run(db, generate)
            self.assertEqual(code, run_benchmark.MEASUREMENT_FAILED_EXIT)
            self.assertEqual(calls['n'], 4)
            conn = sqlite3.connect(db)
            self.assertEqual(conn.execute(
                'SELECT COUNT(*) FROM runs WHERE is_warmup=0'
            ).fetchone()[0], 0)
            conn.close()

    def test_existing_execution_refuses_before_generation(self) -> None:
        with TemporaryDirectory() as tmp:
            db = Path(tmp) / 'arm.db'
            conn = sqlite3.connect(db)
            init_schema(conn)
            ensure_provenance_table(conn)
            execution_id = 'temperature-study-v1-t07__qwen3-4b-q4'
            create_experiment(
                conn, experiment_id=execution_id, name=execution_id,
                config_hash='c' * 64, config_yaml='old',
                created_at_utc='2026-09-24T00:00:00Z',
            )
            record_execution_provenance(
                conn, execution_id=execution_id,
                experiment_spec_id='temperature-study-v1-t07',
                model_config_id='qwen3-4b-q4', experiment_config_hash='e' * 64,
                model_config_hash='m' * 64, model_artifact_digest='d' * 64,
                created_at_utc='2026-09-24T00:00:00Z',
            )
            conn.commit()
            conn.close()
            with patch.object(run_benchmark, 'generate', side_effect=AssertionError('must not call')) as generate:
                code = run_benchmark.run_experiment(_args(str(db)))
            self.assertEqual(code, 2)
            generate.assert_not_called()

    def test_static_cli_overrides_are_refused(self) -> None:
        with TemporaryDirectory() as tmp:
            for override in ({'resume': True}, {'max_tasks': 1}, {'num_predict': 2048}):
                with self.subTest(override=override):
                    with patch.object(run_benchmark, 'generate', side_effect=AssertionError('must not call')) as generate:
                        code = run_benchmark.run_experiment(
                            _args(str(Path(tmp) / 'arm.db'), **override)
                        )
                    self.assertEqual(code, 2)
                    generate.assert_not_called()


if __name__ == '__main__':
    unittest.main()
