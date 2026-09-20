"""Offline guards for retry-rescue-v2 representability resolution."""

from __future__ import annotations

import copy
import unittest
from typing import Any

from analysis.retry_rescue_v2 import (
    NUMERIC_CONSTRAINT,
    PROBE_COHORTS,
    constraint_sha256,
    freeze_matrix,
    is_bare_json_number,
)


def _draft_matrix() -> dict[str, Any]:
    primary_rows = [
        {'task_id': task_id, 'trial': trial, 'eligible': 'REQUIRES_LIVE_VERIFICATION'}
        for task_id, cohort in PROBE_COHORTS.items()
        if cohort == 'primary'
        for trial in (1, 2, 3)
    ]
    primary_rows.extend(
        {'task_id': f'S{index:03}', 'trial': 1, 'eligible': False}
        for index in range(21)
    )
    return {
        'source_population_rows': 39,
        'matrix_status': 'DRAFT_STATIC',
        'row_eligibility': primary_rows,
        'control_population': {
            'source_rows': 12,
            'matrix_status': 'DRAFT_STATIC',
            'row_eligibility': [
                *({'task_id': 'Q016', 'trial': trial, 'eligible': True} for trial in (1, 2, 3)),
                *({'task_id': 'Q023', 'trial': trial, 'eligible': False} for trial in (1, 2, 3)),
                *({'task_id': 'Q025', 'trial': trial, 'eligible': False} for trial in (1, 2, 3)),
                *({'task_id': 'Q026', 'trial': trial, 'eligible': 'REQUIRES_LIVE_VERIFICATION'} for trial in (1, 2, 3)),
            ],
        },
    }


def _records(compatible: dict[str, bool]) -> list[dict[str, Any]]:
    digest = constraint_sha256(NUMERIC_CONSTRAINT)
    return [
        {
            'task_id': task_id,
            'cohort': cohort,
            'constraint': NUMERIC_CONSTRAINT,
            'constraint_sha256': digest,
            'transport_completed': True,
            'representation_compatible': compatible[task_id],
            'frozen_grader_passed': False,
        }
        for task_id, cohort in PROBE_COHORTS.items()
    ]


class BareJsonNumberTests(unittest.TestCase):
    def test_accepts_only_standard_top_level_json_numbers(self) -> None:
        for output in ('136', '218.4', '-7', '1e3', ' 136\n'):
            self.assertTrue(is_bare_json_number(output), output)
        for output in (
            'true', 'NaN', 'Infinity', '"136"', '{"value":136}',
            'Answer: 136', '136 kg', '```json\n136\n```',
        ):
            self.assertFalse(is_bare_json_number(output), output)


class MatrixResolutionTests(unittest.TestCase):
    def test_resolves_whole_task_triplets_and_counts(self) -> None:
        compatible = {task_id: True for task_id in PROBE_COHORTS}
        compatible['Q020'] = False
        compatible['Q026'] = False
        matrix = freeze_matrix(
            _draft_matrix(), _records(compatible),
            evidence_path='results/reports/evidence.json', evidence_sha256='a' * 64,
            probe_code_git_commit='b' * 40,
        )
        self.assertEqual(matrix['matrix_status'], 'FROZEN')
        self.assertEqual(matrix['eligible_rows'], 15)
        self.assertEqual(matrix['ineligible_rows'], 24)
        self.assertEqual(matrix['requires_live_verification_rows'], 0)
        control = matrix['control_population']
        self.assertEqual(control['matrix_status'], 'FROZEN')
        self.assertEqual(control['eligible_rows'], 3)
        self.assertEqual(control['ineligible_rows'], 9)
        self.assertEqual(control['requires_live_verification_rows'], 0)

    def test_rejects_missing_or_alternate_probe_records(self) -> None:
        records = _records({task_id: True for task_id in PROBE_COHORTS})
        with self.assertRaisesRegex(ValueError, 'task set'):
            freeze_matrix(
                _draft_matrix(), records[:-1], evidence_path='e', evidence_sha256='a' * 64,
                probe_code_git_commit='b' * 40,
            )
        alternate = copy.deepcopy(records)
        alternate[0]['constraint'] = {'type': 'string'}
        with self.assertRaisesRegex(ValueError, 'alternate constraint'):
            freeze_matrix(
                _draft_matrix(), alternate, evidence_path='e', evidence_sha256='a' * 64,
                probe_code_git_commit='b' * 40,
            )


if __name__ == '__main__':
    unittest.main()
