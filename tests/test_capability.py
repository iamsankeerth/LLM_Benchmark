"""Capability summary tests: integer relations, status-driven populations."""

from __future__ import annotations

import unittest
from collections import Counter
from pathlib import Path

from analysis.capability import build_capability_summary
from analysis.reliability import (
    READY_DETERMINISTIC,
    READY_JUDGE,
    SummaryProvenance,
    UnknownStatusError,
    analyze_task,
)
from tests.test_reliability import _fails, _trial


def _provenance() -> SummaryProvenance:
    return SummaryProvenance(
        grading_spec_hash='g' * 64,
        dataset_hash='d' * 64,
        experiment_config_hash='c' * 64,
        analysis_code_git_commit='a' * 40,
        analysis_worktree_dirty=False,
    )


def _statuses(*det: str, judge: tuple[str, ...] = ()) -> dict[str, str]:
    return {t: READY_DETERMINISTIC for t in det} | {
        t: READY_JUDGE for t in judge
    }


class CapabilitySummaryTests(unittest.TestCase):
    def test_integer_relations_mirror_baseline_shape(self) -> None:
        # Mirrors the live baseline structure: deterministic tasks with
        # mixed trial outcomes plus READY_JUDGE tasks failing deterministically
        # (Q014/Q051 shape). Integers asserted exactly; rates derived.
        tasks = {
            'QA': analyze_task(
                [_trial('QA', 1, 'PASS', 'a'),
                 _trial('QA', 2, 'PASS', 'a'),
                 _trial('QA', 3, 'PASS', 'a')],
                grading_status=READY_DETERMINISTIC,
            ),
            'QB': analyze_task(
                [_trial('QB', 1, 'FAIL', 'a'),
                 _trial('QB', 2, 'PASS', 'b'),
                 _trial('QB', 3, 'PASS', 'b')],
                grading_status=READY_DETERMINISTIC,
            ),
            'QC': analyze_task(
                _fails('QC', ['x', 'y', 'z']),
                grading_status=READY_DETERMINISTIC,
            ),
            'QJ': analyze_task(
                _fails('QJ', ['j1', 'j2', 'j3']),
                grading_status=READY_JUDGE,
            ),
        }
        statuses = _statuses('QA', 'QB', 'QC', judge=('QJ',))
        summary = build_capability_summary('exp', tasks, statuses, _provenance())
        # Exact integer relations: 5 passes over 9 deterministic trials.
        self.assertEqual(summary.passes, 5)
        self.assertEqual(summary.deterministic_trials, 9)
        self.assertEqual(summary.deterministic_tasks, 3)
        self.assertEqual(summary.tasks_any_pass_3, 2)
        self.assertEqual(summary.tasks_all_pass_3, 1)
        self.assertAlmostEqual(summary.deterministic_trial_accuracy or 0.0, 5 / 9)
        self.assertEqual(summary.deterministic_task_ids, ['QA', 'QB', 'QC'])
        self.assertEqual(summary.judge_task_ids, ['QJ'])
        pre = summary.judge_prechecks
        self.assertEqual(pre.judge_task_trials, 3)
        self.assertEqual(pre.deterministic_precheck_passes, 0)
        self.assertEqual(pre.deterministic_precheck_fail_count, 3)

    def test_locked_frozen_population_integers(self) -> None:
        # Integer relations derived from frozen, committed files (not live
        # DBs): guards the denominator structure at the source. The live
        # figures 53/117, 1/12, 1/7 were recomputed from persisted rows and
        # must satisfy these same relations.
        import yaml

        root = Path(__file__).resolve().parents[1]
        spec = yaml.safe_load(
            (root / 'evals/specs/eval-v1-grading.yaml').read_text(encoding='utf-8')
        )
        statuses = [e['grading_status'] for e in spec['tasks'].values()]
        # 68 READY_DETERMINISTIC + 4 READY_JUDGE = 72 mandatory.
        self.assertEqual(statuses.count('READY_DETERMINISTIC'), 68)
        self.assertEqual(statuses.count('READY_JUDGE'), 4)
        baseline = yaml.safe_load(
            (root / 'configs/creator-baseline-v1.yaml').read_text(encoding='utf-8')
        )
        task_ids = [str(t) for t in baseline['task_ids']]
        by_status = Counter(
            spec['tasks'][tid]['grading_status'] for tid in task_ids
        )
        # 43 - 4 judge = 39 deterministic; 39 x 3 trials = 117 denominator.
        self.assertEqual(len(task_ids), 43)
        self.assertEqual(by_status['READY_JUDGE'], 4)
        self.assertEqual(by_status['READY_DETERMINISTIC'], 39)
        self.assertEqual(39 * int(baseline['trials']), 117)
        reliability = yaml.safe_load(
            (root / 'configs/reliability-v1.yaml').read_text(encoding='utf-8')
        )
        rel_ids = [str(t).split('#')[0].strip() for t in reliability['task_ids']]
        rel_status = Counter(spec['tasks'][tid]['grading_status'] for tid in rel_ids)
        # 14 - 2 judge = 12 deterministic; 12 x 5 trials = 60 denominator.
        self.assertEqual(len(rel_ids), 14)
        self.assertEqual(rel_status['READY_JUDGE'], 2)
        self.assertEqual(rel_status['READY_DETERMINISTIC'], 12)
        self.assertEqual(12 * int(reliability['trials']), 60)

    def test_unknown_task_raises(self) -> None:
        tasks = {
            'QA': analyze_task(
                [_trial('QA', 1, 'PASS', 'a')],
                grading_status=READY_DETERMINISTIC,
            ),
        }
        with self.assertRaises(UnknownStatusError):
            build_capability_summary('exp', tasks, {}, _provenance())

    def test_analysis_population_mismatch_raises(self) -> None:
        tasks = {
            'QA': analyze_task(
                [_trial('QA', 1, 'PASS', 'a')],
                grading_status=READY_JUDGE,
            ),
        }
        with self.assertRaises(UnknownStatusError):
            build_capability_summary(
                'exp', tasks, {'QA': READY_DETERMINISTIC}, _provenance()
            )


if __name__ == '__main__':
    unittest.main()
