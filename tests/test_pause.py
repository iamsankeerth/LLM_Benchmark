"""Pause/resume tests: sentinel, flag-only SIGINT, boundaries (all offline)."""

from __future__ import annotations

import argparse
import signal
import sqlite3
import unittest
from typing import Any
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from inference.ollama_client import GenerationResult
from scripts import run_benchmark, run_model_sweep
from storage.pause import (
    PauseFlag,
    cancel_pause,
    consume_pause_request,
    pause_record_from_json,
    pause_record_to_json,
    pause_requested,
    request_pause,
)
from storage.pause import PauseRecord
from storage.sweep import (
    load_sweep_state,
    new_sweep_state,
    save_sweep_state,
)


def _generation(text: str = 'ok') -> GenerationResult:
    return GenerationResult(
        text=text, thinking='', done_reason='stop',
        total_duration_ns=100_000_000, load_duration_ns=50_000_000,
        prompt_eval_count=10, prompt_eval_cached_count=0,
        prompt_eval_duration_ns=90_000_000,
        eval_count=2, eval_duration_ns=60_000_000,
        request_start_ns=0, first_token_ns=100_000_000,
        request_end_ns=200_000_000,
    )


def _namespace(db: str, **overrides: object) -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    base: dict[str, object] = {
        'config': str(root / 'configs/smoke-3.yaml'),
        'model': 'qwen3-4b-q4', 'db': db, 'base_url': 'http://127.0.0.1:9',
        'resume': False, 'execution_id': None, 'num_predict': None,
        'timeout_s': 30.0, 'max_tasks': None, 'ollama_bin': 'ollama',
    }
    base.update(overrides)
    return argparse.Namespace(**base)


def _offline_patches(test: unittest.TestCase) -> None:
    from inference.eligibility import EligibilityResult

    def eligible(*args: object, **kwargs: object) -> EligibilityResult:
        return EligibilityResult(
            eligible=True, status='ELIGIBLE_GPU', model_size_bytes=100,
            size_vram_bytes=100, gpu_residency_ratio=1.0,
            observed_digest='d' * 64, eligibility_evidence_json='{}',
        )

    test.addCleanup(patch.object(run_benchmark, 'generate').stop)
    test.addCleanup(patch.object(run_benchmark, 'check_eligibility').stop)
    test.addCleanup(patch.object(run_benchmark, 'ollama_stop').stop)
    test.addCleanup(patch.object(run_benchmark, 'wait_until_unloaded').stop)
    patch.object(run_benchmark, 'generate').start()
    patch.object(run_benchmark, 'check_eligibility', return_value=eligible()).start()
    patch.object(run_benchmark, 'ollama_stop', return_value=(True, 's')).start()
    patch.object(run_benchmark, 'wait_until_unloaded', return_value=True).start()


class SentinelTests(unittest.TestCase):
    def test_pause_record_round_trip(self) -> None:
        record = PauseRecord(
            paused_from='BENCHMARKING', resume_stage='benchmarked',
            model_config_id='m',
        )
        restored = pause_record_from_json(pause_record_to_json(record))
        self.assertEqual(restored, record)
        with self.assertRaises(ValueError):
            pause_record_from_json('[]')

    def test_request_consume_cancel_round_trip(self) -> None:
        with TemporaryDirectory() as tmp:
            self.assertFalse(pause_requested(tmp, 'exp'))
            request_pause(tmp, 'exp')
            self.assertTrue(pause_requested(tmp, 'exp'))
            self.assertTrue(cancel_pause(tmp, 'exp'))
            self.assertFalse(pause_requested(tmp, 'exp'))
            self.assertFalse(cancel_pause(tmp, 'exp'))
            request_pause(tmp, 'exp')
            consume_pause_request(tmp, 'exp')
            self.assertFalse(pause_requested(tmp, 'exp'))


class SigintDisciplineTests(unittest.TestCase):
    def test_first_interrupt_is_flag_only(self) -> None:
        previous = signal.getsignal(signal.SIGINT)
        flag = PauseFlag()
        try:
            flag.install_sigint_handler()
            with TemporaryDirectory() as tmp:
                marker = Path(tmp) / 'touched'
                # Simulate the OS delivering SIGINT: handler must not do IO.
                flag._sigint_handler(signal.SIGINT, None)
                self.assertTrue(flag.requested)
                self.assertFalse(marker.exists())
                # Second interrupt restores hard abort.
                with self.assertRaises(KeyboardInterrupt):
                    flag._sigint_handler(signal.SIGINT, None)
                self.assertIs(
                    signal.getsignal(signal.SIGINT), signal.default_int_handler
                )
        finally:
            signal.signal(signal.SIGINT, previous)


class SweepPauseTests(unittest.TestCase):
    def test_state_paused_round_trip(self) -> None:
        with TemporaryDirectory() as tmp:
            path = str(Path(tmp) / 'sweep.json')
            state = new_sweep_state('s', ['a'])
            state.status = 'PAUSED'
            state.pause = {
                'paused_from': 'BENCHMARKING', 'resume_stage': 'benchmarked',
                'model_config_id': 'a', 'pause_reason': 'USER_REQUESTED',
                'weights_retained': True,
            }
            save_sweep_state(path, state)
            loaded = load_sweep_state(path)
            assert loaded is not None
            assert loaded.pause is not None
            self.assertEqual(loaded.status, 'PAUSED')
            self.assertEqual(loaded.pause['resume_stage'], 'benchmarked')
            # Parking bay: registry position never advances on pause.
            self.assertEqual(loaded.next_model, 'a')

    def test_graceful_pause_unloads_and_banners(self) -> None:
        import io
        from contextlib import redirect_stdout

        with TemporaryDirectory() as tmp:
            path = str(Path(tmp) / 'sweep.json')
            state = new_sweep_state('s', ['a'])
            save_sweep_state(path, state)
            request_pause(tmp, 'exp')
            buffer = io.StringIO()
            with patch.object(
                run_model_sweep, 'ollama_stop', return_value=(True, 's')
            ):
                with patch.object(
                    run_model_sweep, 'wait_until_unloaded', return_value=True
                ):
                    with redirect_stdout(buffer):
                        outcome = run_model_sweep.graceful_pause(
                            state=state, state_path=Path(path),
                            checkpoints_dir=Path(tmp),
                            experiment_spec='exp', model_config_id='a',
                            paused_from='BENCHMARKING',
                            resume_stage='benchmarked',
                            ollama_bin='o', base_url='u', identifier='m',
                        )
            out = buffer.getvalue()
            self.assertEqual(outcome, 'PAUSED')
            self.assertIn('SAFE TO SHUT DOWN', out)
            self.assertIn('weights retained: yes', out)
            self.assertFalse(pause_requested(tmp, 'exp'))
            reloaded = load_sweep_state(path)
            assert reloaded is not None
            self.assertEqual(reloaded.status, 'PAUSED')
            self.assertEqual(reloaded.next_model, 'a')


class RunnerPauseTests(unittest.TestCase):
    def _start_scripted(self, items: list[object]) -> Any:
        calls = {'n': 0}

        def fake_generate(
            base_url: str, request: object, **kwargs: object
        ) -> GenerationResult:
            item = items[min(calls['n'], len(items) - 1)]
            calls['n'] += 1
            if item == 'CREATE_SENTINEL':
                root = Path(__file__).resolve().parents[1]
                request_pause(root / 'results/checkpoints', 'smoke-3')
                return _generation('r')
            assert isinstance(item, GenerationResult)
            return item

        patcher = patch.object(
            run_benchmark, 'generate', side_effect=fake_generate
        )
        patcher.start()
        self.addCleanup(patcher.stop)
        return patcher

    def test_pause_after_measured_row(self) -> None:
        # Sentinel appears during row 2: row commits, row 3 never starts,
        # exit 7, sentinel consumed, weights "retained" (mocked unload).
        with TemporaryDirectory() as tmp:
            db = str(Path(tmp) / 'bench.db')
            root = Path(__file__).resolve().parents[1]
            checkpoints = root / 'results/checkpoints'
            sentinel = checkpoints / 'pause-smoke-3.request'
            if sentinel.exists():
                sentinel.unlink()
            _offline_common(self)
            self._start_scripted([
                _generation('probe'), _generation('OK'), _generation('7'),
                _generation('r1'), 'CREATE_SENTINEL', _generation('r2'),
                _generation('r3'),
            ])
            try:
                code = run_benchmark.run_experiment(_namespace(db))
                self.assertEqual(code, run_benchmark.PAUSED_EXIT)
                conn = sqlite3.connect(db)
                measured = conn.execute(
                    'SELECT task_id FROM runs WHERE is_warmup=0 ORDER BY run_id'
                ).fetchall()
                conn.close()
                self.assertEqual(len(measured), 2)
                self.assertFalse(sentinel.exists())
            finally:
                if sentinel.exists():
                    sentinel.unlink()

    def test_max_tasks_change_is_identity_mismatch(self) -> None:
        # Same execution id but a different task set (max_tasks) changes the
        # config triple -> MISMATCH exit, never silent resume.
        with TemporaryDirectory() as tmp:
            db = str(Path(tmp) / 'bench.db')
            _offline_common(self)
            patcher = self._start_scripted([
                _generation('probe'), _generation('OK'), _generation('7'),
                _generation('r1'),
            ])
            try:
                code1 = run_benchmark.run_experiment(
                    _namespace(db, max_tasks=1)
                )
                self.assertEqual(code1, 0)
                patcher.stop()
                patcher = self._start_scripted([
                    _generation('probe'), _generation('OK'), _generation('7'),
                    _generation('r2'),
                ])
                code2 = run_benchmark.run_experiment(
                    _namespace(db, resume=True)
                )
                self.assertEqual(code2, 4)
            finally:
                patcher.stop()

    def test_overnight_pause_resume(self) -> None:
        # 2 rows -> PAUSED (exit 7) -> simulated restart -> warmups ->
        # remaining row -> COMPLETE. Final matrix intact, zero duplicates.
        with TemporaryDirectory() as tmp:
            db = str(Path(tmp) / 'bench.db')
            root = Path(__file__).resolve().parents[1]
            checkpoints = root / 'results/checkpoints'
            sentinel = checkpoints / 'pause-smoke-3.request'
            if sentinel.exists():
                sentinel.unlink()
            _offline_common(self)
            try:
                patcher = self._start_scripted([
                    _generation('probe'), _generation('OK'), _generation('7'),
                    _generation('r1'), 'CREATE_SENTINEL', _generation('r2'),
                ])
                code1 = run_benchmark.run_experiment(
                    _namespace(db)
                )
                self.assertEqual(code1, run_benchmark.PAUSED_EXIT)
                patcher.stop()
                patcher = self._start_scripted([
                    _generation('probe'), _generation('OK'), _generation('7'),
                    _generation('r3'),
                ])
                code2 = run_benchmark.run_experiment(
                    _namespace(db, resume=True)
                )
                self.assertEqual(code2, 0)
                conn = sqlite3.connect(db)
                measured = conn.execute(
                    'SELECT task_id, trial FROM runs WHERE is_warmup=0 ORDER BY run_id'
                ).fetchall()
                warmups = conn.execute(
                    "SELECT COUNT(*) FROM runs WHERE run_kind='WARMUP'"
                ).fetchone()[0]
                conn.close()
                self.assertEqual(len(measured), 3)
                self.assertEqual(len({(r[0], r[1]) for r in measured}), 3)
                self.assertGreaterEqual(warmups, 4)
            finally:
                if sentinel.exists():
                    sentinel.unlink()


def _offline_common(test: unittest.TestCase) -> None:
    from inference.eligibility import EligibilityResult

    def eligible(*args: object, **kwargs: object) -> EligibilityResult:
        return EligibilityResult(
            eligible=True, status='ELIGIBLE_GPU', model_size_bytes=100,
            size_vram_bytes=100, gpu_residency_ratio=1.0,
            observed_digest='d' * 64, eligibility_evidence_json='{}',
        )

    test.addCleanup(patch.object(run_benchmark, 'check_eligibility').stop)
    test.addCleanup(patch.object(run_benchmark, 'ollama_stop').stop)
    test.addCleanup(patch.object(run_benchmark, 'wait_until_unloaded').stop)
    patch.object(run_benchmark, 'check_eligibility', return_value=eligible()).start()
    patch.object(run_benchmark, 'ollama_stop', return_value=(True, 's')).start()
    patch.object(run_benchmark, 'wait_until_unloaded', return_value=True).start()


if __name__ == '__main__':
    unittest.main()
