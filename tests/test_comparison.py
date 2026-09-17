"""Paired comparison tests: transitions, percentiles, buckets, McNemar."""

from __future__ import annotations

import unittest

from analysis.comparison import (
    ComparisonTrial,
    compare_executions,
    display_execution_id,
    mcnemar_exact_p,
    summarize_performance,
)


def _trial(
    task_id: str,
    trial: int,
    verdict: str,
    *,
    raw: str = 'out',
    decode: float | None = 37.5,
    ttft: float | None = 100.0,
    eval_count: int | None = 20,
    family: str = 'DETERMINISTIC_EXACT',
    suite: str = 'core-v1',
    difficulty: str = 'medium',
) -> ComparisonTrial:
    return ComparisonTrial(
        task_id=task_id, trial=trial, verdict=verdict, raw_output=raw,
        decode_tok_s=decode, ttft_ms=ttft, eval_count=eval_count,
        primary_class=family, suite=suite, difficulty=difficulty,
    )


def _grouped(
    outcomes: dict[str, list[str]], **kwargs: object
) -> dict[str, list[ComparisonTrial]]:
    return {
        task_id: [
            _trial(task_id, i + 1, verdict, **kwargs)  # type: ignore[arg-type]
            for i, verdict in enumerate(verdicts)
        ]
        for task_id, verdicts in outcomes.items()
    }


class McNemarTests(unittest.TestCase):
    def test_empty_discordance_is_one(self) -> None:
        self.assertEqual(mcnemar_exact_p(0, 0), 1.0)

    def test_hand_computed_table(self) -> None:
        # b=2, c=6, n=8: 2*(C(8,0)+C(8,1)+C(8,2))/256 = 74/256.
        self.assertAlmostEqual(mcnemar_exact_p(2, 6), 74 / 256)
        self.assertAlmostEqual(mcnemar_exact_p(6, 2), 74 / 256)

    def test_caps_at_one(self) -> None:
        self.assertEqual(mcnemar_exact_p(4, 4), 1.0)


class PercentileTests(unittest.TestCase):
    def test_summaries(self) -> None:
        summary = summarize_performance([10.0, 20.0, 30.0, 40.0])
        self.assertEqual(summary.n, 4)
        self.assertEqual(summary.median, 25.0)
        self.assertEqual(summary.p95, 40.0)
        self.assertAlmostEqual(summary.mean or 0.0, 25.0)

    def test_empty(self) -> None:
        summary = summarize_performance([None, None])
        self.assertEqual(summary.n, 0)
        self.assertIsNone(summary.median)


class TransitionTests(unittest.TestCase):
    def test_all_four_cells_and_net_gain(self) -> None:
        base = _grouped({
            'PP': ['PASS', 'PASS', 'PASS'],
            'FF': ['FAIL', 'FAIL', 'FAIL'],
            'FP': ['FAIL', 'FAIL', 'FAIL'],
            'PF': ['PASS', 'PASS', 'PASS'],
        })
        cand = _grouped({
            'PP': ['PASS', 'PASS', 'PASS'],
            'FF': ['FAIL', 'FAIL', 'FAIL'],
            'FP': ['PASS', 'PASS', 'PASS'],
            'PF': ['FAIL', 'FAIL', 'FAIL'],
        })
        report = compare_executions(
            experiment_spec_id='full-baseline-v1',
            base_execution_id='full-baseline-v1__qwen3-4b-q4',
            candidate_execution_id='full-baseline-v1__qwen3-4b-q5',
            base_grouped=base, cand_grouped=cand,
        )
        matrix = report.transitions_strict
        self.assertEqual(matrix.pass_to_pass, ['PP'])
        self.assertEqual(matrix.fail_to_fail, ['FF'])
        self.assertEqual(matrix.fail_to_pass, ['FP'])
        self.assertEqual(matrix.pass_to_fail, ['PF'])
        self.assertEqual(matrix.net_task_gain, 0)
        self.assertEqual((report.mcnemar_b, report.mcnemar_c, report.mcnemar_n), (1, 1, 2))
        self.assertEqual(report.all_pass_delta, 0)
        self.assertEqual(report.trial_accuracy_delta, 0.0)

    def test_strict_vs_any_pass_notion(self) -> None:
        base = _grouped({'FX': ['FAIL', 'FAIL', 'FAIL']})
        cand = _grouped({'FX': ['PASS', 'FAIL', 'FAIL']})
        report = compare_executions(
            experiment_spec_id='s', base_execution_id='b',
            candidate_execution_id='c', base_grouped=base, cand_grouped=cand,
        )
        # Strict: still FAIL (not all pass). Any-pass: improved.
        self.assertEqual(report.transitions_strict.fail_to_fail, ['FX'])
        self.assertEqual(report.transitions_any_pass.fail_to_pass, ['FX'])

    def test_judge_tasks_never_enter(self) -> None:
        # load_comparison_trials filters by status; compare_executions only
        # sees what it is given: assert common-task intersection behavior.
        base = _grouped({'A': ['PASS']})
        cand = _grouped({'A': ['PASS'], 'B': ['PASS']})
        report = compare_executions(
            experiment_spec_id='s', base_execution_id='b',
            candidate_execution_id='c', base_grouped=base, cand_grouped=cand,
        )
        self.assertEqual(report.base.deterministic_tasks, 1)
        self.assertEqual(report.candidate.deterministic_tasks, 1)

    def test_display_alias_is_presentation_only(self) -> None:
        self.assertEqual(
            display_execution_id('full-baseline-v1'),
            'full-baseline-v1__qwen3-4b-q4',
        )
        self.assertEqual(
            display_execution_id('full-baseline-v1__qwen3-4b-q5'),
            'full-baseline-v1__qwen3-4b-q5',
        )

    def test_token_buckets(self) -> None:
        base = {'A': [
            _trial('A', 1, 'PASS', eval_count=2, decode=70.0),
            _trial('A', 2, 'PASS', eval_count=20, decode=38.0),
            _trial('A', 3, 'PASS', eval_count=200, decode=37.0),
        ]}
        cand = {'A': [
            _trial('A', 1, 'PASS', eval_count=2, decode=70.0),
            _trial('A', 2, 'PASS', eval_count=20, decode=38.0),
            _trial('A', 3, 'PASS', eval_count=200, decode=37.0),
        ]}
        report = compare_executions(
            experiment_spec_id='s', base_execution_id='b',
            candidate_execution_id='c', base_grouped=base, cand_grouped=cand,
        )
        buckets = report.base.decode_by_token_bucket
        self.assertEqual(buckets['1-4'].median, 70.0)
        self.assertEqual(buckets['5-32'].median, 38.0)
        self.assertEqual(buckets['129+'].median, 37.0)
        self.assertEqual(buckets['33-128'].n, 0)


if __name__ == '__main__':
    unittest.main()
