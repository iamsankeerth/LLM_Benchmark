"""Sweep lifecycle tests: state, verify gate, deletion refusal (all offline)."""

from __future__ import annotations

import subprocess
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from storage.db import RunRecord, connect, create_experiment, init_schema, insert_run
from storage.sweep import (
    BENCHMARK_ERROR,
    BENCHMARKED,
    BENCHMARKING,
    COMPLETE,
    COMPLETE_INELIGIBLE,
    DOWNLOAD_FAILED,
    PENDING,
    RESTART_STAGE,
    SMOKED,
    STAGES,
    STOPPED_STATE,
    WARMING_UP,
    disk_reclaimed_ok,
    guard_deletion,
    load_sweep_state,
    new_sweep_state,
    ollama_model_present,
    ollama_pull,
    ollama_remove,
    ollama_stop,
    save_sweep_state,
    set_lifecycle,
    verify_execution_persisted,
    wait_until_unloaded,
)

STATUSES = {'QA': 'READY_DETERMINISTIC', 'QB': 'READY_DETERMINISTIC',
            'QJ': 'READY_JUDGE'}


def _record(experiment: str, task: str, trial: int, verdict: str = 'PASS') -> RunRecord:
    return RunRecord(
        experiment_id=experiment, model_config_id='m', task_id=task,
        trial=trial, run_kind='BASELINE', run_config_hash='h' * 64,
        is_warmup=False, prompt='p', rendered_prompt_sha256='s' * 64,
        temperature=0.0, num_ctx=4096, num_predict=2048,
        template_sha256='t' * 64, grader_verdict=verdict, status='COMPLETE',
        started_at_utc='2026-09-18T00:00:00Z', ended_at_utc='2026-09-18T00:00:01Z',
    )


def _ok_process(output: str = 'ok') -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(args=[], returncode=0, stdout=output, stderr='')


def _fail_process() -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(args=[], returncode=1, stdout='', stderr='boom')


class SweepStateTests(unittest.TestCase):
    def test_stopped_states_cover_every_stage(self) -> None:
        for stage in STAGES:
            self.assertIn(stage, STOPPED_STATE, stage)

    def test_restart_mapping_covers_lifecycles(self) -> None:
        for lifecycle in (
            PENDING, DOWNLOAD_FAILED, 'PULLED', 'DERIVING', 'DERIVED',
            'ELIGIBLE', 'PREFLIGHTED', 'SMOKED', WARMING_UP, BENCHMARKING,
            BENCHMARK_ERROR, BENCHMARKED, 'VALIDATED', 'SUMMARIZED',
            'VERIFYING',
        ):
            self.assertIn(lifecycle, RESTART_STAGE, lifecycle)
        # Warmup-adjacent states resume into the benchmark (fresh re-warm).
        self.assertEqual(RESTART_STAGE[SMOKED], 'benchmarked')
        self.assertEqual(RESTART_STAGE[WARMING_UP], 'benchmarked')
        self.assertEqual(RESTART_STAGE[BENCHMARKED], 'validated')

    def test_round_trip_and_progression(self) -> None:
        with TemporaryDirectory() as tmp:
            path = str(Path(tmp) / 'sweep.json')
            state = new_sweep_state('sweep', ['a', 'b'])
            self.assertEqual(state.next_model, 'a')
            set_lifecycle(state, 'a', COMPLETE)
            save_sweep_state(path, state)
            loaded = load_sweep_state(path)
            assert loaded is not None
            self.assertEqual(loaded.next_model, 'b')
            self.assertEqual(loaded.completed, ['a'])
            set_lifecycle(state, 'b', COMPLETE_INELIGIBLE)
            self.assertEqual(state.next_model, None)
            self.assertEqual(state.ineligible, ['b'])

    def test_failed_tracked(self) -> None:
        state = new_sweep_state('sweep', ['a'])
        set_lifecycle(state, 'a', DOWNLOAD_FAILED, 'net down')
        self.assertEqual(state.failed, ['a'])
        self.assertEqual(state.next_model, 'a')

    def test_missing_state_is_none(self) -> None:
        with TemporaryDirectory() as tmp:
            self.assertIsNone(load_sweep_state(str(Path(tmp) / 'nope.json')))


class VerifyGateTests(unittest.TestCase):
    def _db(self, tmp: str, execution: str, *, errors: bool = False) -> str:
        path = str(Path(tmp) / 'bench.db')
        conn = connect(path)
        init_schema(conn)
        create_experiment(
            conn, experiment_id=execution, name=execution, config_hash='c' * 64,
            config_yaml='x', created_at_utc='2026-09-18T00:00:00Z',
        )
        for task in ('QA', 'QB'):
            for trial in (1, 2, 3):
                insert_run(conn, _record(execution, task, trial))
        for trial in (1, 2, 3):
            insert_run(conn, _record(execution, 'QJ', trial, verdict='NEEDS_JUDGE'))
        if errors:
            insert_run(conn, _record(execution, 'QA', 4, verdict='ERROR'))
            conn.execute(
                "UPDATE runs SET status='ERROR' WHERE task_id='QA' AND trial=4"
            )
        conn.commit()
        conn.close()
        return path

    def _files(self, tmp: str) -> dict[str, str]:
        paths = {}
        for name in ('manifest', 'capability', 'performance', 'preflight',
                     'failure_modes', 'summary'):
            file = Path(tmp) / f'{name}.json'
            file.write_text('{"ok": true}', encoding='utf-8')
            paths[name] = str(file)
        return paths

    def test_verify_pass(self) -> None:
        with TemporaryDirectory() as tmp:
            db = self._db(tmp, 'exe')
            verdict = verify_execution_persisted(
                db, 'exe', STATUSES, expected_det_tasks=2,
                expected_judge_tasks=1, trials_per_task=3,
                required_files=self._files(tmp),
            )
            self.assertTrue(verdict.ok, verdict.detail)

    def test_verify_fails_on_error_rows(self) -> None:
        with TemporaryDirectory() as tmp:
            db = self._db(tmp, 'exe', errors=True)
            verdict = verify_execution_persisted(
                db, 'exe', STATUSES, expected_det_tasks=2,
                expected_judge_tasks=1, trials_per_task=3,
                required_files=self._files(tmp),
            )
            self.assertFalse(verdict.ok)
            self.assertFalse(verdict.checks['no_error_rows'])

    def test_verify_fails_on_missing_file(self) -> None:
        with TemporaryDirectory() as tmp:
            db = self._db(tmp, 'exe')
            files = self._files(tmp)
            del files['manifest']
            verdict = verify_execution_persisted(
                db, 'exe', STATUSES, expected_det_tasks=2,
                expected_judge_tasks=1, trials_per_task=3,
                required_files=files,
            )
            # Missing required file is not checked... must still pass other gates.
            self.assertTrue(verdict.ok)
            files['manifest'] = str(Path(tmp) / 'absent.json')
            verdict = verify_execution_persisted(
                db, 'exe', STATUSES, expected_det_tasks=2,
                expected_judge_tasks=1, trials_per_task=3,
                required_files=files,
            )
            self.assertFalse(verdict.ok)

    def test_verify_fails_on_wrong_counts(self) -> None:
        with TemporaryDirectory() as tmp:
            db = self._db(tmp, 'exe')
            verdict = verify_execution_persisted(
                db, 'exe', STATUSES, expected_det_tasks=68,
                expected_judge_tasks=4, trials_per_task=3,
                required_files=self._files(tmp),
            )
            self.assertFalse(verdict.ok)
            self.assertFalse(verdict.checks['det_rows'])

    def test_deletion_gate_refuses(self) -> None:
        with self.assertRaises(RuntimeError):
            guard_deletion(False)
        guard_deletion(True)  # must not raise


class OllamaHelperTests(unittest.TestCase):
    def test_pull_ok_and_fail(self) -> None:
        ok, _ = ollama_pull('ollama', 'm', run=lambda *a, **k: _ok_process())
        self.assertTrue(ok)
        ok, detail = ollama_pull('ollama', 'm', run=lambda *a, **k: _fail_process())
        self.assertFalse(ok)
        self.assertIn('DOWNLOAD_FAILED', detail)

    def test_stop_remove(self) -> None:
        ok, _ = ollama_stop('ollama', 'm', run=lambda *a, **k: _ok_process())
        self.assertTrue(ok)
        ok, _ = ollama_remove('ollama', 'm', run=lambda *a, **k: _ok_process())
        self.assertTrue(ok)
        ok, _ = ollama_remove('ollama', 'm', run=lambda *a, **k: _fail_process())
        self.assertFalse(ok)

    def test_present_fail_closed(self) -> None:
        self.assertTrue(
            ollama_model_present('o', 'm', run=lambda *a, **k: _fail_process())
        )
        self.assertFalse(
            ollama_model_present(
                'o', 'm', run=lambda *a, **k: _ok_process('NAME ID')
            )
        )
        self.assertTrue(
            ollama_model_present(
                'o', 'hf.co/x:m', run=lambda *a, **k: _ok_process('hf.co/x:m 123')
            )
        )

    def test_disk_math(self) -> None:
        free_before = 30_000_000_000
        self.assertTrue(disk_reclaimed_ok(free_before, free_before))
        self.assertTrue(disk_reclaimed_ok(free_before, free_before - 500_000_000))
        self.assertFalse(disk_reclaimed_ok(free_before, free_before - 2_000_000_000))

    def test_wait_until_unloaded(self) -> None:
        calls = {'n': 0}

        def absent() -> bool:
            calls['n'] += 1
            return calls['n'] >= 3

        self.assertTrue(wait_until_unloaded(absent, timeout_s=30.0, poll_s=0.01))
        self.assertFalse(wait_until_unloaded(lambda: False, timeout_s=0.05, poll_s=0.01))


if __name__ == '__main__':
    unittest.main()
