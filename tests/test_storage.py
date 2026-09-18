"""Storage + verdict tests: identity, resume, warm-up exclusion (temp DB)."""

from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from typing import Any

from evals.graders.engine import GraderResult
from evals.verdicts import (
    ERROR,
    FAIL,
    NEEDS_JUDGE,
    PASS,
    REFUSED_NOT_EXECUTABLE,
    reduce_verdict,
)
from storage.db import (
    RunRecord,
    completed_identities,
    connect,
    create_experiment,
    effective_generation_config,
    fetch_measured,
    init_schema,
    insert_run,
    run_config_hash,
)
from storage.manifest import (
    ExperimentManifest,
    LiveEnvironment,
    build_manifest,
    hash_experiment_config,
)


def _record(**overrides: object) -> RunRecord:
    base: dict[str, object] = {
        'experiment_id': 'exp1',
        'model_config_id': 'qwen3-4b-q4',
        'task_id': 'Q001',
        'trial': 1,
        'run_kind': 'BASELINE',
        'run_config_hash': 'h' * 64,
        'is_warmup': False,
        'prompt': 'p',
        'rendered_prompt_sha256': 's' * 64,
        'temperature': 0.0,
        'num_ctx': 4096,
        'num_predict': 512,
        'template_sha256': 't' * 64,
        'grader_verdict': PASS,
        'status': 'COMPLETE',
        'started_at_utc': '2026-09-17T00:00:00Z',
        'ended_at_utc': '2026-09-17T00:00:01Z',
    }
    base.update(overrides)
    return RunRecord(
        experiment_id=str(base['experiment_id']),
        model_config_id=str(base['model_config_id']),
        task_id=str(base['task_id']),
        trial=int(str(base['trial'])),
        run_kind=str(base['run_kind']),
        run_config_hash=str(base['run_config_hash']),
        is_warmup=bool(base['is_warmup']),
        prompt=str(base['prompt']),
        rendered_prompt_sha256=str(base['rendered_prompt_sha256']),
        temperature=float(str(base['temperature'])),
        num_ctx=int(str(base['num_ctx'])),
        num_predict=int(str(base['num_predict'])),
        template_sha256=str(base['template_sha256']),
        grader_verdict=str(base['grader_verdict']),
        status=str(base['status']),
        started_at_utc=str(base['started_at_utc']),
        ended_at_utc=str(base['ended_at_utc']),
    )


class VerdictReducerTests(unittest.TestCase):
    def test_pending_refused(self) -> None:
        self.assertEqual(reduce_verdict('PENDING_SPECIFICATION', []), REFUSED_NOT_EXECUTABLE)
        self.assertEqual(reduce_verdict('PENDING_REVIEW', []), REFUSED_NOT_EXECUTABLE)

    def test_deterministic_fail(self) -> None:
        results = [
            GraderResult('exact', True, 'ok'),
            GraderResult('constraints', False, 'bad', ['too long']),
        ]
        self.assertEqual(reduce_verdict('READY_DETERMINISTIC', results), FAIL)

    def test_deterministic_pass(self) -> None:
        results = [GraderResult('structured', True, 'ok')]
        self.assertEqual(reduce_verdict('READY_DETERMINISTIC', results), PASS)

    def test_judge_only_defers(self) -> None:
        # Q014/Q051/Q059/Q060 shape: rubric_judge passed=True is a deferral.
        results = [GraderResult('rubric_judge', True, 'deferrable')]
        self.assertEqual(reduce_verdict('READY_JUDGE', results), NEEDS_JUDGE)

    def test_mixed_deterministic_plus_judge_defers(self) -> None:
        results = [
            GraderResult('exact', True, 'ok'),
            GraderResult('rubric_judge', True, 'deferrable'),
        ]
        self.assertEqual(reduce_verdict('READY_DETERMINISTIC', results), NEEDS_JUDGE)

    def test_mixed_with_deterministic_failure_is_fail(self) -> None:
        results = [
            GraderResult('exact', False, 'wrong'),
            GraderResult('rubric_judge', True, 'deferrable'),
        ]
        self.assertEqual(reduce_verdict('READY_DETERMINISTIC', results), FAIL)

    def test_empty_results_raise_loudly(self) -> None:
        with self.assertRaises(ValueError):
            reduce_verdict('READY_DETERMINISTIC', [])

    def test_error_constant_available(self) -> None:
        self.assertEqual(ERROR, 'ERROR')


class StorageTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.conn = connect(str(Path(self._tmp.name) / 'bench.db'))
        self.addCleanup(self.conn.close)
        init_schema(self.conn)
        create_experiment(
            self.conn,
            experiment_id='exp1',
            name='test',
            config_hash='c' * 64,
            config_yaml='kind: test',
            created_at_utc='2026-09-17T00:00:00Z',
        )

    def test_insert_and_resume_identity(self) -> None:
        insert_run(self.conn, _record())
        self.conn.commit()
        done = completed_identities(self.conn, 'exp1', 'BASELINE')
        self.assertEqual(done, {('qwen3-4b-q4', 'Q001', 1, 'h' * 64)})

    def test_duplicate_identity_rejected(self) -> None:
        insert_run(self.conn, _record())
        self.conn.commit()
        with self.assertRaises(sqlite3.IntegrityError):
            insert_run(self.conn, _record())

    def test_config_change_creates_new_identity(self) -> None:
        insert_run(self.conn, _record())
        other = _record(run_config_hash='a' * 64)
        insert_run(self.conn, other)
        self.conn.commit()
        done = completed_identities(self.conn, 'exp1', 'BASELINE')
        self.assertEqual(len(done), 2)

    def test_trial_change_creates_new_identity(self) -> None:
        insert_run(self.conn, _record())
        insert_run(self.conn, _record(trial=2))
        self.conn.commit()
        done = completed_identities(self.conn, 'exp1', 'BASELINE')
        self.assertEqual(len(done), 2)

    def test_warmups_excluded_from_measured(self) -> None:
        insert_run(self.conn, _record())
        insert_run(
            self.conn,
            _record(
                task_id='Q002', run_kind='WARMUP', is_warmup=True,
                run_config_hash='w' * 64, grader_verdict=PASS,
            ),
        )
        self.conn.commit()
        measured = fetch_measured(self.conn, 'exp1')
        self.assertEqual([row['task_id'] for row in measured], ['Q001'])

    def test_error_rows_not_resumed(self) -> None:
        insert_run(self.conn, _record(status='ERROR', grader_verdict=ERROR))
        self.conn.commit()
        self.assertEqual(completed_identities(self.conn, 'exp1', 'BASELINE'), set())

    def test_unknown_run_kind_rejected(self) -> None:
        with self.assertRaises(ValueError):
            insert_run(self.conn, _record(run_kind='SOMEDAY'))

    def test_run_config_hash_stable_and_sensitive(self) -> None:
        cfg = effective_generation_config(
            ollama_identifier='m', quantization='Q4_K_M', mode='raw',
            temperature=0.0, num_ctx=4096, num_predict=512, num_gpu=99,
            stop_tokens=('</s>',), think=False, template_sha256='t' * 64,
        )
        same = effective_generation_config(
            ollama_identifier='m', quantization='Q4_K_M', mode='raw',
            temperature=0.0, num_ctx=4096, num_predict=512, num_gpu=99,
            stop_tokens=('</s>',), think=False, template_sha256='t' * 64,
        )
        hotter = dict(cfg)
        hotter['temperature'] = 0.7
        self.assertEqual(run_config_hash(cfg), run_config_hash(same))
        self.assertEqual(len(run_config_hash(cfg)), 64)
        self.assertNotEqual(run_config_hash(cfg), run_config_hash(hotter))


class ManifestTests(unittest.TestCase):
    def _live(self) -> LiveEnvironment:
        return LiveEnvironment(
            ollama_version='0.34.1',
            python_version='3.13.5',
            hardware_id='Windows-AMD64-TestCPU',
            power_mode=None,
            started_at_utc='2026-09-17T00:00:00Z',
        )

    def _manifest(self, **overrides: Any) -> ExperimentManifest:
        kwargs: dict[str, Any] = {
            'app_git_commit': '4a22cc5',
            'eval_freeze_hash': 'f' * 64,
            'grading_spec_hash': 'g' * 64,
            'dataset_hash': 'd' * 64,
            'experiment_spec_id': 'full-baseline-v1',
            'execution_id': 'full-baseline-v1__qwen3-4b-q4',
            'model_config_id': 'qwen3-4b-q4',
            'experiment_config_hash': 'e' * 64,
            'model_config_hash': 'm' * 64,
            'model_artifact_digest': '0' * 64,
            'model_artifact_size_bytes': 3178149969,
            'model_identifier': 'm',
            'quantization': 'Q4_K_M',
            'template_sha256': 't' * 64,
            'temperature': 0.0,
            'num_ctx': 4096,
            'num_predict': 512,
            'num_gpu': 99,
            'stop_tokens': ('a', 'b'),
            'think': False,
            'thinking_source': 'explicit_config',
            'reload_evidence_load_duration_ms': 1000.0,
            'live': self._live(),
        }
        kwargs.update(overrides)
        return build_manifest(**kwargs)

    def test_manifest_complete_and_serializable(self) -> None:
        manifest = self._manifest()
        document = json.loads(manifest.to_json())
        required = {
            'app_git_commit', 'eval_freeze_hash', 'grading_spec_hash',
            'dataset_hash', 'experiment_spec_id', 'execution_id',
            'model_config_id', 'experiment_config_hash', 'model_config_hash',
            'model_artifact_digest', 'model_artifact_size_bytes',
            'model_identifier',
            'quantization', 'template_sha256', 'ollama_version',
            'python_version', 'hardware_id', 'power_mode', 'temperature',
            'num_ctx', 'num_predict', 'num_gpu', 'stop_tokens', 'think',
            'thinking_source', 'started_at_utc', 'reload_evidence_load_duration_ms',
        }
        self.assertTrue(required.issubset(document.keys()), required - set(document.keys()))

    def test_config_hash_stability(self) -> None:
        first = hash_experiment_config({'b': 1, 'a': [1, 2]})
        second = hash_experiment_config({'a': [1, 2], 'b': 1})
        self.assertEqual(first, second)
        self.assertNotEqual(first, hash_experiment_config({'a': [1, 2], 'b': 2}))


if __name__ == '__main__':
    unittest.main()
