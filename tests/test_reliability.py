"""Offline reliability tests: synthetic trials, no live model."""

from __future__ import annotations

import unittest

from analysis.reliability import (
    TrialRecord,
    analyze_experiment,
    analyze_task,
)


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
        task = analyze_task(trials, is_json_task=True)
        self.assertEqual(task.success_rate, 1.0)
        self.assertEqual(task.initial_attempt_success, 1.0)
        self.assertTrue(task.final_success)
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
        task = analyze_task(trials)
        self.assertAlmostEqual(task.success_rate or 0.0, 0.6)
        self.assertEqual(task.initial_attempt_success, 1.0)
        self.assertTrue(task.final_success)  # pass@5
        self.assertFalse(task.all_pass_5)  # pass^5
        self.assertTrue(task.grader_flip)
        self.assertEqual(task.unique_raw_output_count, 5)
        self.assertEqual(len(task.failure_code_distribution), 1)

    def test_all_fail_task(self) -> None:
        task = analyze_task(_fails('Q071', [f'wrong {i}' for i in range(5)]))
        self.assertEqual(task.success_rate, 0.0)
        self.assertFalse(task.final_success)
        self.assertFalse(task.all_pass_5)
        self.assertFalse(task.grader_flip)

    def test_recovery_pattern(self) -> None:
        trials = [_trial('Q019', 1, 'FAIL', 'bad')] + [
            _trial('Q019', i, 'PASS', 'good') for i in (2, 3, 4, 5)
        ]
        task = analyze_task(trials)
        self.assertEqual(task.initial_attempt_success, 0.0)
        self.assertTrue(task.final_success)
        self.assertFalse(task.all_pass_5)
        self.assertTrue(task.grader_flip)

    def test_judge_task_excluded_with_sub_report(self) -> None:
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
        task = analyze_task(trials)
        self.assertTrue(task.is_judge_task)
        self.assertIsNone(task.success_rate)
        self.assertIsNone(task.all_pass_5)
        self.assertIsNone(task.grader_flip)
        self.assertEqual(task.deterministic_sub_pass_rate, 1.0)
        self.assertEqual(task.judge_deferrals, 5)
        self.assertEqual(task.unique_raw_output_count, 5)

    def test_canonical_json_counts_semantics(self) -> None:
        trials = [
            _trial('Q001', 1, 'PASS', '{"a": 1, "b": 2}'),
            _trial('Q001', 2, 'PASS', '{"b": 2, "a": 1}'),
            _trial('Q001', 3, 'PASS', '{"a": 1, "b": 3}'),
            _trial('Q001', 4, 'PASS', 'not json'),
            _trial('Q001', 5, 'PASS', 'not json'),
        ]
        task = analyze_task(trials, is_json_task=True)
        self.assertEqual(task.unique_raw_output_count, 4)
        # Key-order pair collapses; unparseable pair collapses into one class.
        self.assertEqual(task.unique_canonical_json_count, 3)

    def test_prefill_stratified_insufficient_n(self) -> None:
        trials = [
            _trial('Q001', 1, 'PASS', 'a', prefill=190.0, cache='UNCACHED'),
            _trial('Q001', 2, 'PASS', 'a', prefill=195.0, cache='UNCACHED'),
            _trial('Q001', 3, 'PASS', 'a', prefill=None, cache='PARTIAL'),
            _trial('Q001', 4, 'PASS', 'a', prefill=300.0, cache='FULLY_CACHED'),
            _trial('Q001', 5, 'PASS', 'a', prefill=None, cache='FULLY_CACHED'),
        ]
        task = analyze_task(trials)
        uncached = task.prefill_by_cache_state['UNCACHED']
        self.assertNotIsInstance(uncached, str)
        self.assertEqual(task.prefill_by_cache_state['PARTIAL'], 'insufficient_n')
        self.assertEqual(
            task.cache_state_distribution,
            {'UNCACHED': 2, 'PARTIAL': 1, 'FULLY_CACHED': 2},
        )
        # No pooled prefill column may exist anywhere in the task report.
        self.assertNotIn('prefill_stats', set(task.__dict__))


class ReliabilityExperimentTests(unittest.TestCase):
    def test_aggregate_math(self) -> None:
        q011 = analyze_task(
            [
                _trial('Q011', 1, 'PASS', 'a'),
                _trial('Q011', 2, 'FAIL', 'b'),
                _trial('Q011', 3, 'PASS', 'c'),
                _trial('Q011', 4, 'PASS', 'd'),
                _trial('Q011', 5, 'FAIL', 'e'),
            ]
        )
        q019 = analyze_task(
            [_trial('Q019', 1, 'FAIL', 'x')]
            + [_trial('Q019', i, 'PASS', 'y') for i in (2, 3, 4, 5)]
        )
        q001 = analyze_task([_trial('Q001', i + 1, 'PASS', 'z') for i in range(5)])
        report = analyze_experiment(
            'reliability-v1', {'Q011': q011, 'Q019': q019, 'Q001': q001}
        )
        self.assertAlmostEqual(
            report.mean_success_rate or 0.0, (0.6 + 0.8 + 1.0) / 3
        )
        self.assertAlmostEqual(
            report.mean_initial_attempt_success or 0.0, (1.0 + 0.0 + 1.0) / 3
        )
        self.assertAlmostEqual(report.mean_final_success or 0.0, 1.0)
        self.assertAlmostEqual(report.mean_all_pass_5 or 0.0, 1.0 / 3)
        self.assertAlmostEqual(report.grader_flip_rate or 0.0, 2.0 / 3)
        self.assertEqual(report.flipped_tasks, ['Q011', 'Q019'])
        # Only Q019 failed trial 1 and recovered in trials 2-5.
        self.assertEqual(report.retry_recovery_fraction, '1/1')
        self.assertEqual(report.retry_recovery_rate, 1.0)
        self.assertEqual(report.judge_tasks, {})

    def test_judge_tasks_separated(self) -> None:
        det = analyze_task([_trial('Q001', i + 1, 'PASS', 'z') for i in range(5)])
        judge = analyze_task(
            [_trial('Q059', i + 1, 'NEEDS_JUDGE', f'e{i}') for i in range(5)]
        )
        report = analyze_experiment('r', {'Q001': det, 'Q059': judge})
        self.assertEqual(list(report.deterministic_tasks), ['Q001'])
        self.assertEqual(list(report.judge_tasks), ['Q059'])
        self.assertEqual(report.mean_success_rate, 1.0)
        self.assertEqual(report.grader_flip_rate, 0.0)


if __name__ == '__main__':
    unittest.main()
