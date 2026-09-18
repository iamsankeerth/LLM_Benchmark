"""Failure-report tests: row-based decompositions on synthetic reviews."""

from __future__ import annotations

import unittest

from analysis.failure_report import (
    ReviewNotApprovedError,
    require_approved,
    table_contract_modes,
    table_judge_prechecks,
    table_substance,
    task_table,
)


def _task(
    tid: str, status: str, modes: list[str], label: str, **kwargs: object
) -> dict[str, object]:
    return {
        'task_id': tid,
        'task_status': status,
        'failed_trials': list(range(1, len(modes) + 1)),
        'auto_analysis': {
            'failed_rows': len(modes),
            'row_modes': modes,
            'generation_flags': kwargs.get('flags', []),
            'excerpts': [{'trial': 1, 'excerpt': 'out'}],
        },
        'review': {
            'substantive_answer_assessment': label,
            'assessment_author': 'draft_review',
            'mode_override': None,
            'notes': 'n',
        },
    }


def _review() -> dict[str, object]:
    return {
        'taxonomy_version': 'failure-modes-v1',
        'review_status': 'APPROVED',
        'tasks': [
            _task('QA', 'READY_DETERMINISTIC',
                  ['OUTPUT_CONTRACT'] * 3, 'CORRECT'),
            _task('QB', 'READY_DETERMINISTIC',
                  ['OUTPUT_CONTRACT', 'CONTENT_ERROR', 'OUTPUT_CONTRACT'],
                  'INCORRECT'),
            _task('QJ', 'READY_JUDGE', ['CONTENT_ERROR'] * 2, 'CORRECT'),
        ],
    }


class DecompositionTests(unittest.TestCase):
    def test_table1_row_based(self) -> None:
        table = table_contract_modes(_review())
        # 6 det rows: 5 contract + 1 content (QB t2 varies, not MIXED).
        self.assertEqual(table['denominator_rows'], 6)
        self.assertEqual(
            table['counts'], {'CONTENT_ERROR': 1, 'OUTPUT_CONTRACT': 5}
        )
        self.assertAlmostEqual(table['percent']['OUTPUT_CONTRACT'], 5 / 6, places=4)

    def test_table2_inherits_rows(self) -> None:
        table = table_substance(_review())
        # QA 3 rows CORRECT + QB 3 rows INCORRECT = 6; judge excluded.
        self.assertEqual(table['denominator_rows'], 6)
        self.assertEqual(table['counts'], {'CORRECT': 3, 'INCORRECT': 3})

    def test_judge_separate(self) -> None:
        table = table_judge_prechecks(_review())
        self.assertEqual(table['total_rows'], 2)
        self.assertEqual(len(table['tasks']), 1)

    def test_task_table_flags_distribution(self) -> None:
        table = {t['task_id']: t for t in task_table(_review())}
        self.assertFalse(table['QA']['multi_mode_across_trials'])
        self.assertTrue(table['QB']['multi_mode_across_trials'])
        self.assertEqual(
            table['QB']['mode_distribution'],
            {'CONTENT_ERROR': 1, 'OUTPUT_CONTRACT': 2},
        )

    def test_approval_gate(self) -> None:
        require_approved(_review())
        draft = dict(_review())
        draft['review_status'] = 'DRAFT'
        with self.assertRaises(ReviewNotApprovedError):
            require_approved(draft)


if __name__ == '__main__':
    unittest.main()
