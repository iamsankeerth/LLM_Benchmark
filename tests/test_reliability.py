"""Offline reliability tests: synthetic trials, no live model."""

from __future__ import annotations

import json
import subprocess
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from analysis.reliability import (
    READY_DETERMINISTIC,
    READY_JUDGE,
    DirtyWorktreeError,
    TrialRecord,
    UnknownStatusError,
    analyze_experiment,
    analyze_grouped,
    analyze_task,
    collect_provenance,
)

from evals.contract import file_sha256


def _trial(
    task_id: str,
    trial: int,
    verdict: str,
    raw: str,
    *,
    details: list[dict[str, object]] | None = None,
    decode: float | None = 37.5,
    prefill: float | None = 190.0,
    cache: str | None = 'UNCACHED',
    uncached: int | None = 19,
) -> TrialRecord:
    if details is None:
        details = (
            [{'grader_type': 'exact', 'passed': True, 'detail': 'ok', 'violations': []}]
            if verdict == 'PASS'
            else [
                {
                    'grader_type': 'exact',
                    'passed': False,
                    'detail': 'mismatch',
                    'violations': ['wrong value'],
                }
            ]
        )
    return TrialRecord(
        task_id=task_id,
        trial=trial,
        verdict=verdict,
        raw_output=raw,
        grader_details=[dict(d) for d in details],
        decode_tok_s=decode,
        ttft_ms=100.0,
        client_e2e_ms=2000.0,
        prefill_compute_tok_s=prefill,
        prefill_cache_state=cache,
        prompt_eval_uncached_count=uncached,
        ram_peak_mb=100.0,
        vram_peak_mib=3200.0,
    )


def _fails(task_id: str, raws: list[str]) -> list[TrialRecord]:
    return [
        _trial(task_id, i + 1, 'FAIL', raw) for i, raw in enumerate(raws)
    ]


class ReliabilityTaskTests(unittest.TestCase):
    def test_all_pass_task(self) -> None:
        trials = [_trial('Q001', i + 1, 'PASS', '{"a": 1}') for i in range(5)]
        task = analyze_task(trials, is_json_task=True, grading_status=READY_DETERMINISTIC)
        self.assertEqual(task.success_rate, 1.0)
        self.assertEqual(task.initial_attempt_success, 1.0)
        self.assertTrue(task.any_pass_5)
        self.assertTrue(task.all_pass_5)
        self.assertFalse(task.grader_flip)
        self.assertEqual(task.unique_raw_output_count, 1)
        self.assertEqual(task.unique_canonical_json_count, 1)
        self.assertEqual(task.failure_code_distribution, {})

    def test_q011_pattern_fixture(self) -> None:
        # PASS / FAIL / PASS / PASS / FAIL with distinct outputs.
        verdicts = ['PASS', 'FAIL', 'PASS', 'PASS', 'FAIL']
        trials = [
            _trial('Q011', i + 1, verdict, f'output variant {i}')
            for i, verdict in enumerate(verdicts)
        ]
        task = analyze_task(trials, grading_status=READY_DETERMINISTIC)
        self.assertAlmostEqual(task.success_rate or 0.0, 0.6)
        self.assertEqual(task.initial_attempt_success, 1.0)
        self.assertTrue(task.any_pass_5)
        self.assertFalse(task.all_pass_5)
        self.assertTrue(task.grader_flip)
        self.assertEqual(task.unique_raw_output_count, 5)
        self.assertEqual(len(task.failure_code_distribution), 1)

    def test_all_fail_task(self) -> None:
        task = analyze_task(
            _fails('Q071', [f'wrong {i}' for i in range(5)]),
            grading_status=READY_DETERMINISTIC,
        )
        self.assertEqual(task.success_rate, 0.0)
        self.assertFalse(task.any_pass_5)
        self.assertFalse(task.all_pass_5)
        self.assertFalse(task.grader_flip)

    def test_recovery_pattern(self) -> None:
        trials = [_trial('Q019', 1, 'FAIL', 'bad')] + [
            _trial('Q019', i, 'PASS', 'good') for i in (2, 3, 4, 5)
        ]
        task = analyze_task(trials, grading_status=READY_DETERMINISTIC)
        self.assertEqual(task.initial_attempt_success, 0.0)
        self.assertTrue(task.any_pass_5)
        self.assertFalse(task.all_pass_5)
        self.assertTrue(task.grader_flip)

    def test_judge_status_defers_despite_passing_subchecks(self) -> None:
        details = [
            {'grader_type': 'fact_check', 'passed': True, 'detail': 'ok',
             'violations': []},
            {'grader_type': 'rubric_judge', 'passed': True,
             'detail': 'deferrable', 'violations': []},
        ]
        trials = [
            _trial('Q051', i + 1, 'NEEDS_JUDGE', f'essay {i}', details=details)
            for i in range(5)
        ]
        task = analyze_task(trials, grading_status=READY_JUDGE)
        self.assertTrue(task.is_judge_task)
        self.assertIsNone(task.success_rate)
        self.assertIsNone(task.all_pass_5)
        self.assertIsNone(task.grader_flip)
        self.assertEqual(task.deterministic_sub_passes, 5)
        self.assertEqual(task.deterministic_sub_total, 5)
        self.assertEqual(task.deterministic_sub_pass_rate, 1.0)
        self.assertEqual(task.judge_deferrals, 5)

    def test_judge_status_fail_contributes_zero_to_denominators(self) -> None:
        # The locked regression case: deterministic subcheck failed once,
        # then deferred twice. Global contribution must be exactly zero.
        fail_details = [
            {'grader_type': 'fact_check', 'passed': False, 'detail': 'wrong',
             'violations': ['bad fact']},
            {'grader_type': 'rubric_judge', 'passed': True,
             'detail': 'deferrable', 'violations': []},
        ]
        defer_details = [
            {'grader_type': 'fact_check', 'passed': True, 'detail': 'ok',
             'violations': []},
            {'grader_type': 'rubric_judge', 'passed': True,
             'detail': 'deferrable', 'violations': []},
        ]
        trials = [
            _trial('Q051', 1, 'FAIL', 'bad', details=fail_details),
            _trial('Q051', 2, 'NEEDS_JUDGE', 'essay', details=defer_details),
            _trial('Q051', 3, 'NEEDS_JUDGE', 'essay2', details=defer_details),
        ]
        report = analyze_experiment(
            'r', {'Q051': analyze_task(trials, grading_status=READY_JUDGE)}
        )
        self.assertEqual(report.deterministic_tasks, {})
        self.assertEqual(list(report.judge_tasks), ['Q051'])
        self.assertIsNone(report.mean_success_rate)
        self.assertIsNone(report.grader_flip_rate)
        pre = report.judge_prechecks
        self.assertEqual(pre.judge_task_trials, 3)
        self.assertEqual(pre.deterministic_precheck_passes, 2)
        self.assertEqual(pre.deterministic_precheck_total, 3)
        self.assertAlmostEqual(pre.deterministic_precheck_pass_rate or 0.0, 2 / 3)
        self.assertEqual(pre.deterministic_precheck_fail_count, 1)
        self.assertEqual(pre.judge_deferrals, 2)
        self.assertAlmostEqual(pre.judge_deferral_rate or 0.0, 2 / 3)

    def test_unknown_status_raises(self) -> None:
        trials = [_trial('QX', 1, 'PASS', 'x')]
        with self.assertRaises(UnknownStatusError):
            analyze_task(trials, grading_status='READY_SOMEDAY')
        with self.assertRaises(UnknownStatusError):
            analyze_grouped('r', {'QX': trials}, {})

    def test_canonical_json_counts_semantics(self) -> None:
        trials = [
            _trial('Q001', 1, 'PASS', '{"a": 1, "b": 2}'),
            _trial('Q001', 2, 'PASS', '{"b": 2, "a": 1}'),
            _trial('Q001', 3, 'PASS', '{"a": 1, "b": 3}'),
            _trial('Q001', 4, 'PASS', 'not json'),
            _trial('Q001', 5, 'PASS', 'not json'),
        ]
        task = analyze_task(trials, is_json_task=True, grading_status=READY_DETERMINISTIC)
        self.assertEqual(task.unique_raw_output_count, 4)
        self.assertEqual(task.unique_canonical_json_count, 3)

    def test_prefill_stratified_insufficient_n(self) -> None:
        trials = [
            _trial('Q001', 1, 'PASS', 'a', prefill=190.0, cache='UNCACHED', uncached=19),
            _trial('Q001', 2, 'PASS', 'a', prefill=195.0, cache='UNCACHED', uncached=19),
            _trial('Q001', 3, 'PASS', 'a', prefill=None, cache='PARTIAL', uncached=None),
            _trial('Q001', 4, 'PASS', 'a', prefill=300.0, cache='FULLY_CACHED', uncached=0),
            _trial('Q001', 5, 'PASS', 'a', prefill=None, cache='FULLY_CACHED', uncached=0),
        ]
        task = analyze_task(trials, grading_status=READY_DETERMINISTIC)
        uncached = task.prefill_by_cache_state['UNCACHED']
        self.assertNotIsInstance(uncached, str)
        self.assertEqual(task.prefill_by_cache_state['PARTIAL'], 'insufficient_n')
        self.assertEqual(
            task.cache_state_distribution,
            {'UNCACHED': 2, 'PARTIAL': 1, 'FULLY_CACHED': 2},
        )
        self.assertEqual(task.uncached_counts_by_state['UNCACHED'], [19, 19])
        self.assertNotIn('prefill_stats', set(task.__dict__))


class ReliabilityExperimentTests(unittest.TestCase):
    def _analyze(self, tasks: dict[str, object]) -> object:
        analyses = {}
        for tid, trials in tasks.items():
            analyses[tid] = analyze_task(trials, grading_status=READY_DETERMINISTIC)  # type: ignore[arg-type]
        return analyze_experiment('reliability-v1', analyses)

    def test_aggregate_math(self) -> None:
        q011 = analyze_task(
            [
                _trial('Q011', 1, 'PASS', 'a'),
                _trial('Q011', 2, 'FAIL', 'b'),
                _trial('Q011', 3, 'PASS', 'c'),
                _trial('Q011', 4, 'PASS', 'd'),
                _trial('Q011', 5, 'FAIL', 'e'),
            ],
            grading_status=READY_DETERMINISTIC,
        )
        q019 = analyze_task(
            [_trial('Q019', 1, 'FAIL', 'x')]
            + [_trial('Q019', i, 'PASS', 'y') for i in (2, 3, 4, 5)],
            grading_status=READY_DETERMINISTIC,
        )
        q001 = analyze_task(
            [_trial('Q001', i + 1, 'PASS', 'z') for i in range(5)],
            grading_status=READY_DETERMINISTIC,
        )
        report = analyze_experiment(
            'reliability-v1', {'Q011': q011, 'Q019': q019, 'Q001': q001}
        )
        self.assertAlmostEqual(
            report.mean_success_rate or 0.0, (0.6 + 0.8 + 1.0) / 3
        )
        self.assertAlmostEqual(
            report.mean_initial_attempt_success or 0.0, (1.0 + 0.0 + 1.0) / 3
        )
        self.assertAlmostEqual(report.mean_any_pass_5 or 0.0, 1.0)
        self.assertAlmostEqual(report.mean_all_pass_5 or 0.0, 1.0 / 3)
        self.assertAlmostEqual(report.grader_flip_rate or 0.0, 2.0 / 3)
        self.assertEqual(report.flipped_tasks, ['Q011', 'Q019'])
        self.assertEqual(report.later_trial_recovery_fraction, '1/1')
        self.assertEqual(report.later_trial_recovery_rate, 1.0)
        self.assertEqual(report.judge_tasks, {})

    def test_judge_tasks_separated_by_status(self) -> None:
        det = analyze_task(
            [_trial('Q001', i + 1, 'PASS', 'z') for i in range(5)],
            grading_status=READY_DETERMINISTIC,
        )
        failing_judge = analyze_task(
            _fails('Q051', [f'bad {i}' for i in range(5)]),
            grading_status=READY_JUDGE,
        )
        report = analyze_experiment('r', {'Q001': det, 'Q051': failing_judge})
        self.assertEqual(list(report.deterministic_tasks), ['Q001'])
        self.assertEqual(list(report.judge_tasks), ['Q051'])
        self.assertEqual(report.mean_success_rate, 1.0)
        self.assertEqual(report.grader_flip_rate, 0.0)
        self.assertEqual(report.judge_prechecks.judge_task_trials, 5)
        self.assertEqual(report.judge_prechecks.deterministic_precheck_passes, 0)

    def test_status_driven_grouped_analysis(self) -> None:
        grouped = {
            'Q001': [_trial('Q001', i + 1, 'PASS', 'z') for i in range(5)],
            'Q051': _fails('Q051', [f'bad {i}' for i in range(5)]),
        }
        statuses = {'Q001': READY_DETERMINISTIC, 'Q051': READY_JUDGE}
        report = analyze_grouped('r', grouped, statuses)
        self.assertEqual(list(report.deterministic_tasks), ['Q001'])
        self.assertEqual(list(report.judge_tasks), ['Q051'])


class ProvenanceTests(unittest.TestCase):
    def _with_freeze(self, tmp: str) -> None:
        specs = Path(tmp, 'evals', 'specs')
        dataset_dir = Path(tmp, 'evals', 'datasets', 'eval-v1')
        specs.mkdir(parents=True)
        dataset_dir.mkdir(parents=True)
        spec_path = specs / 'eval-v1-grading.yaml'
        dataset_path = dataset_dir / 'executable-v1.jsonl'
        spec_path.write_text(
            'spec_version: eval-v1\nstatus: FROZEN\ntasks: {}\n',
            encoding='utf-8',
        )
        dataset_path.write_text('', encoding='utf-8')
        Path(specs / 'eval-v1-grading.freeze.json').write_text(
            json.dumps({'artifacts': {
                'eval-v1-grading.yaml': file_sha256(spec_path),
                'executable-v1.jsonl': file_sha256(dataset_path),
            }}),
            encoding='utf-8',
        )

    def test_non_git_dir_fails_closed(self) -> None:
        with TemporaryDirectory() as tmp:
            self._with_freeze(tmp)
            with self.assertRaises(DirtyWorktreeError):
                collect_provenance(tmp, {'experiment': 'x'})
            provenance = collect_provenance(tmp, {'experiment': 'x'}, allow_dirty=True)
            self.assertTrue(provenance.analysis_worktree_dirty)
            self.assertIsNone(provenance.analysis_code_git_commit)
            self.assertEqual(
                provenance.grading_spec_hash,
                file_sha256(Path(tmp) / 'evals/specs/eval-v1-grading.yaml'),
            )
            self.assertEqual(
                provenance.dataset_hash,
                file_sha256(Path(tmp) / 'evals/datasets/eval-v1/executable-v1.jsonl'),
            )
            self.assertEqual(len(provenance.experiment_config_hash), 64)

    def test_untracked_json_does_not_dirty(self) -> None:
        from analysis.reliability import git_worktree_status

        with TemporaryDirectory() as tmp:
            self._with_freeze(tmp)
            subprocess.run(['git', 'init', '-q'], cwd=tmp, check=True,
                           capture_output=True)
            subprocess.run(['git', 'config', 'user.email', 't@t'],
                           cwd=tmp, check=True, capture_output=True)
            subprocess.run(['git', 'config', 'user.name', 't'],
                           cwd=tmp, check=True, capture_output=True)
            Path(tmp, 'results', 'summaries').mkdir(parents=True)
            Path(tmp, 'results', 'summaries', 'out.json').write_text(
                '{}', encoding='utf-8'
            )
            subprocess.run(['git', 'add', '-A'], cwd=tmp, check=True,
                           capture_output=True)
            subprocess.run(['git', 'commit', '-qm', 'init'], cwd=tmp, check=True,
                           capture_output=True)
            # New untracked result JSON: clean.
            Path(tmp, 'results', 'summaries', 'new.json').write_text(
                '{}', encoding='utf-8'
            )
            commit, dirty = git_worktree_status(tmp)
            self.assertFalse(dirty)
            self.assertTrue(commit)
            # New untracked python file: dirty (could alter behavior).
            Path(tmp, 'sneaky.py').write_text('x = 1\n', encoding='utf-8')
            _, dirty = git_worktree_status(tmp)
            self.assertTrue(dirty)
            # Tracked modification: dirty.
            freeze_path = Path(tmp, 'evals', 'specs', 'eval-v1-grading.freeze.json')
            freeze_path.write_text(
                freeze_path.read_text(encoding='utf-8') + ' ', encoding='utf-8'
            )
            _, dirty = git_worktree_status(tmp)
            self.assertTrue(dirty)


if __name__ == '__main__':
    unittest.main()
