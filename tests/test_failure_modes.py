"""Failure-mode taxonomy tests: mapping, MIXED, authorship boundary."""

from __future__ import annotations

import unittest
from typing import Any, cast

from analysis.failure_modes import (
    CONTENT_ERROR,
    LEXICAL_CONSTRAINT,
    MIXED,
    OUTPUT_CONTRACT,
    STRUCTURE_ERROR,
    UNKNOWN,
    mode_for_signature,
    modes_for_failed_row,
    review_generation_allowed,
    row_failure_mode,
    signature_for_detail,
    taxonomy_hash,
    validate_review_entry,
    validate_review_file,
)


def _detail(
    grader_type: str, passed: bool, violations: list[str] | None = None,
    detail: str = 'fail',
) -> dict[str, object]:
    return {
        'grader_type': grader_type, 'passed': passed,
        'detail': detail, 'violations': violations or [],
    }


class ModeMappingTests(unittest.TestCase):
    def test_bare_number_is_contract(self) -> None:
        sig = signature_for_detail(
            _detail('numeric', False, [], 'output is not a bare number: We are given')
        )
        self.assertEqual(mode_for_signature('numeric', sig), OUTPUT_CONTRACT)

    def test_time_unparseable_is_contract(self) -> None:
        sig = signature_for_detail(
            _detail('time', False, [], 'could not parse time from prose')
        )
        self.assertEqual(mode_for_signature('time', sig), OUTPUT_CONTRACT)

    def test_word_count_is_lexical(self) -> None:
        sig = signature_for_detail(
            _detail('constraints', False, ['word_count: expected 70, got 65'])
        )
        self.assertEqual(mode_for_signature('constraints', sig), LEXICAL_CONSTRAINT)

    def test_term_occurrence_is_lexical(self) -> None:
        sig = signature_for_detail(
            _detail('constraints', False, ['term_occurrence(latency): expected 1, got 2'])
        )
        self.assertEqual(mode_for_signature('constraints', sig), LEXICAL_CONSTRAINT)

    def test_structured_value_is_content(self) -> None:
        sig = signature_for_detail(
            _detail('structured', False, ["$.project: expected 'Atlas', got 'Project Atlas'"])
        )
        self.assertEqual(mode_for_signature('structured', sig), CONTENT_ERROR)

    def test_structured_shape_is_structure(self) -> None:
        sig = signature_for_detail(
            _detail('structured', False, ['missing field: date'])
        )
        self.assertEqual(mode_for_signature('structured', sig), STRUCTURE_ERROR)

    def test_fact_check_is_content(self) -> None:
        sig = signature_for_detail(_detail('fact_check', False, ['wrong fact']))
        self.assertEqual(mode_for_signature('fact_check', sig), CONTENT_ERROR)

    def test_unknown_kind_is_unknown(self) -> None:
        sig = signature_for_detail(
            _detail('constraints', False, ['hyperdrive_motivator: snafu'])
        )
        self.assertEqual(mode_for_signature('constraints', sig), UNKNOWN)

    def test_bare_wrong_token_is_content(self) -> None:
        # Q036 shape: clean single-token wrong answer is CONTENT_ERROR,
        # not a contract failure.
        signatures, modes = modes_for_failed_row(
            [_detail('exact', False, [], "expected 'UNKNOWN', got 'NO'")],
            raw_output='NO',
        )
        self.assertEqual(row_failure_mode(modes), CONTENT_ERROR)

    def test_exact_with_appendix_is_contract(self) -> None:
        signatures, modes = modes_for_failed_row(
            [_detail('exact', False, [], "expected 'CACHE', got 'CACHE\\n\\nWhy'")],
            raw_output='CACHE\n\nWhy: because.',
        )
        self.assertEqual(row_failure_mode(modes), OUTPUT_CONTRACT)

    def test_mixed_within_one_generation(self) -> None:
        signatures, modes = modes_for_failed_row([
            _detail('constraints', False, ['word_count: expected 70, got 65']),
            _detail('structured', False, ["$.x: expected 1, got 2"]),
        ])
        self.assertEqual(len(signatures), 2)
        self.assertEqual(row_failure_mode(modes), MIXED)

    def test_single_mode_row(self) -> None:
        _, modes = modes_for_failed_row([
            _detail('numeric', False, [], 'output is not a bare number: blah'),
        ])
        self.assertEqual(row_failure_mode(modes), OUTPUT_CONTRACT)

    def test_empty_modes_is_unknown(self) -> None:
        self.assertEqual(row_failure_mode([]), UNKNOWN)
        self.assertEqual(row_failure_mode([UNKNOWN]), UNKNOWN)

    def test_contradiction_evidence_preserved(self) -> None:
        # "The result is 66,080 ... therefore final answer = 65,080":
        # both values must survive in signatures/excerpt for the reviewer;
        # the CORRECT-vs-LIKELY_CORRECT call itself stays human.
        sig = signature_for_detail(
            _detail('numeric', False, [], 'output is not a bare number: result is 66,080')
        )
        self.assertIn('66,080', sig)
        self.assertEqual(mode_for_signature('numeric', sig), OUTPUT_CONTRACT)

    def test_taxonomy_hash_stable(self) -> None:
        self.assertEqual(taxonomy_hash(), taxonomy_hash())
        self.assertEqual(len(taxonomy_hash()), 64)


class AuthorshipBoundaryTests(unittest.TestCase):
    def _entry(self, **overrides: object) -> dict[str, object]:
        entry: dict[str, object] = {
            'task_id': 'Q020',
            'auto_analysis': {'failure_mode': OUTPUT_CONTRACT, 'signatures': []},
            'review': {
                'substantive_answer_assessment': 'CORRECT',
                'assessment_author': 'draft_review',
                'mode_override': None,
                'notes': 'Arithmetic reaches the required result.',
            },
        }
        entry.update(overrides)
        return entry

    def test_valid_entry(self) -> None:
        self.assertEqual(validate_review_entry(self._entry()), [])

    def test_missing_assessment_rejected(self) -> None:
        review = cast(dict[str, Any], self._entry()['review'])
        del review['substantive_answer_assessment']
        errors = validate_review_entry(self._entry(review=review))
        self.assertTrue(any('vocabulary' in e for e in errors))

    def test_missing_author_rejected(self) -> None:
        review = cast(dict[str, Any], self._entry()['review'])
        del review['assessment_author']
        errors = validate_review_entry(self._entry(review=review))
        self.assertTrue(any('assessment_author' in e for e in errors))

    def test_override_without_reason_rejected(self) -> None:
        review = cast(dict[str, Any], self._entry()['review'])
        review['mode_override'] = CONTENT_ERROR
        errors = validate_review_entry(self._entry(review=review))
        self.assertTrue(any('override_reason' in e for e in errors))

    def test_mode_in_review_section_rejected(self) -> None:
        review = cast(dict[str, Any], self._entry()['review'])
        review['failure_mode'] = OUTPUT_CONTRACT
        errors = validate_review_entry(self._entry(review=review))
        self.assertTrue(any('auto_analysis' in e for e in errors))

    def test_generation_gated_on_approval(self) -> None:
        base = {
            'taxonomy_version': 'failure-modes-v1',
            'review_status': 'DRAFT',
            'tasks': [self._entry()],
        }
        self.assertFalse(review_generation_allowed(base))
        self.assertTrue(review_generation_allowed({**base, 'review_status': 'APPROVED'}))
        bad = dict(base)
        bad['taxonomy_version'] = 'failure-modes-v0'
        bad['review_status'] = 'APPROVED'
        self.assertFalse(review_generation_allowed(bad))

    def test_file_requires_tasks(self) -> None:
        self.assertTrue(validate_review_file({'taxonomy_version': 'failure-modes-v1', 'review_status': 'DRAFT', 'tasks': []}))


if __name__ == '__main__':
    unittest.main()
