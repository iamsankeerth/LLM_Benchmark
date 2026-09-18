"""Failure-mode taxonomy tests: mapping, MIXED, authorship boundary."""

from __future__ import annotations

import unittest
from pathlib import Path
from typing import Any, cast

from analysis.failure_modes import (
    ASSERTED_CONTRADICTION,
    ASSERTED_SINGLE,
    CONTENT_ERROR,
    LEXICAL_CONSTRAINT,
    MIXED,
    NO_ASSERTED_VALUE,
    OUTPUT_CONTRACT,
    STRUCTURE_ERROR,
    TRUNCATED_AT_NUM_PREDICT,
    UNKNOWN,
    asserted_value_check,
    extract_asserted_value,
    mode_for_signature,
    modes_for_failed_row,
    review_generation_allowed,
    row_failure_mode,
    signature_for_detail,
    taxonomy_hash,
    truncation_flag,
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


class AssertedValueTests(unittest.TestCase):
    def test_explicit_wrong_final_numeric(self) -> None:
        # Q026 shape: "Final answer:" marker line, standalone 86.40 final.
        text = 'Some derivation with 0.90 and 0.96 in it.\nFinal answer:  \n86.40'
        asserted = extract_asserted_value(text, kind='numeric')
        self.assertEqual(asserted.status, ASSERTED_SINGLE)
        self.assertEqual(asserted.values, ('86.40',))

    def test_explicit_correct_final_numeric(self) -> None:
        text = 'Long derivation.\n✅ Final answer: **136**'
        asserted = extract_asserted_value(text, kind='numeric')
        self.assertEqual(asserted.status, ASSERTED_SINGLE)
        self.assertEqual(asserted.values, ('136',))

    def test_explicit_final_time(self) -> None:
        text = '### Final Answer:\n**10:37**'
        asserted = extract_asserted_value(text, kind='time')
        self.assertEqual(asserted.status, ASSERTED_SINGLE)
        self.assertEqual(asserted.values, ('10:37',))

    def test_derivation_only_no_assertion(self) -> None:
        text = 'Step 1 uses 50.4 extra hours.\nStep 2 adds 168 more.\nBut'
        asserted = extract_asserted_value(text, kind='numeric')
        self.assertEqual(asserted.status, NO_ASSERTED_VALUE)

    def test_time_fragments_never_count(self) -> None:
        text = 'Ends at 36.92 minutes after 10:00, roughly speaking.'
        asserted = extract_asserted_value(text, kind='time')
        self.assertEqual(asserted.status, NO_ASSERTED_VALUE)

    def test_conflicting_finals(self) -> None:
        text = 'Final answer: 66,080, revised to 65,080'
        asserted = extract_asserted_value(text, kind='numeric')
        self.assertEqual(asserted.status, ASSERTED_CONTRADICTION)
        self.assertEqual(set(asserted.values), {'66080', '65080'})

    def test_mismatch_check_numeric(self) -> None:
        grader = {'type': 'numeric', 'expected': 85.83, 'absolute_tolerance': 0.01}
        status, sigs, modes = asserted_value_check(
            'Work.\nFinal answer:  \n86.40', grader
        )
        self.assertEqual(status, ASSERTED_SINGLE)
        self.assertEqual(sigs, ['numeric:asserted_value_mismatch'])
        self.assertEqual(modes, [CONTENT_ERROR])

    def test_match_check_numeric(self) -> None:
        grader = {'type': 'numeric', 'expected': 136}
        status, sigs, modes = asserted_value_check(
            'Work.\n✅ Final answer: **136**', grader
        )
        self.assertEqual(status, ASSERTED_SINGLE)
        self.assertEqual(sigs, [])
        self.assertEqual(modes, [])

    def test_mismatch_check_time(self) -> None:
        grader = {'type': 'time', 'expected': '11:37', 'precision': 'minute',
                  'parsing': ['HH:MM']}
        status, sigs, modes = asserted_value_check(
            'Work.\n### Final Answer:\n**10:37**', grader
        )
        self.assertEqual(status, ASSERTED_SINGLE)
        self.assertEqual(sigs, ['time:asserted_value_mismatch'])
        self.assertEqual(modes, [CONTENT_ERROR])

    def test_contradiction_check(self) -> None:
        grader = {'type': 'numeric', 'expected': 66080}
        status, sigs, modes = asserted_value_check(
            'Final answer: 66,080, revised to 65,080', grader
        )
        self.assertEqual(status, ASSERTED_CONTRADICTION)
        self.assertEqual(sigs, ['numeric:value_contradiction'])

    def test_non_numeric_time_graders_untouched(self) -> None:
        status, sigs, modes = asserted_value_check(
            'Anything.', {'type': 'exact', 'expected': 'X'}
        )
        self.assertEqual((status, sigs, modes), (NO_ASSERTED_VALUE, [], []))

    def test_locked_q023_q026_q025_shapes(self) -> None:
        # Frozen-spec entries with persisted-output shapes.
        import yaml

        root = Path(__file__).resolve().parents[1]
        spec = yaml.safe_load(
            (root / 'evals/specs/eval-v1-grading.yaml').read_text(encoding='utf-8')
        )
        q023 = spec['tasks']['Q023']['graders'][0]
        status, _, modes = asserted_value_check(
            'Steps.\n### Final Answer:\n**10:37**', q023
        )
        self.assertEqual(status, ASSERTED_SINGLE)
        self.assertEqual(modes, [CONTENT_ERROR])
        q026 = spec['tasks']['Q026']['graders'][0]
        status, _, modes = asserted_value_check(
            'Steps.\nFinal answer:  \n86.40', q026
        )
        self.assertEqual(status, ASSERTED_SINGLE)
        self.assertEqual(modes, [CONTENT_ERROR])
        q025 = spec['tasks']['Q025']['graders'][0]
        status, sigs, modes = asserted_value_check(
            'So the first time is **12:36**?\n\nWait \u2014 is there a time **after 10:3',
            q025,
        )
        self.assertEqual((status, sigs, modes), (NO_ASSERTED_VALUE, [], []))


class TruncationFlagTests(unittest.TestCase):
    def test_truncated_when_length_and_budget_hit(self) -> None:
        self.assertEqual(
            truncation_flag('length', 512, 512), [TRUNCATED_AT_NUM_PREDICT]
        )

    def test_not_truncated_on_stop(self) -> None:
        self.assertEqual(truncation_flag('stop', 491, 512), [])

    def test_not_truncated_when_unmeasurable(self) -> None:
        self.assertEqual(truncation_flag('length', None, 512), [])
        self.assertEqual(truncation_flag(None, 512, 512), [])
        self.assertEqual(truncation_flag('length', 100, 512), [])


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
