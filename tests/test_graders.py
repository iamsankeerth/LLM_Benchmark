"""Golden-fixture tests: the evaluator is tested before it grades any model.

Each grader family gets a passing output, a failing output, and an edge case
drawn from real eval-v1 task shapes.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evals.graders.engine import (  # noqa: E402
    grade_constraints,
    grade_exact,
    grade_fact_check,
    grade_numeric,
    grade_refusal,
    grade_rubric_judge,
    grade_structured,
    grade_time,
    grade_tool_call,
    normalize_text,
    words,
)


class ExactGraderTests(unittest.TestCase):
    def test_pass_and_alternates(self) -> None:
        self.assertTrue(grade_exact('CACHE', {'type': 'exact', 'expected': 'CACHE'}).passed)
        self.assertTrue(grade_exact('  150GB\n', {'type': 'exact', 'expected': '150 GB', 'accepted_alternates': ['150GB']}).passed)
        self.assertFalse(grade_exact('cache', {'type': 'exact', 'expected': 'CACHE'}).passed)

    def test_numeric_alias(self) -> None:
        grader = {'type': 'exact', 'expected': '1e-4', 'accepted_alternates': ['0.0001'], 'numeric_alias_match': True}
        self.assertTrue(grade_exact('0.0001', grader).passed)
        self.assertTrue(grade_exact('1e-4', grader).passed)
        self.assertFalse(grade_exact('2e-4', grader).passed)

    def test_strict_no_trim(self) -> None:
        grader = {'type': 'exact', 'expected': '[1,2,3]', 'trim_outer_whitespace': False}
        self.assertTrue(grade_exact('[1,2,3]', grader).passed)
        self.assertFalse(grade_exact(' [1,2,3] ', grader).passed)


class NumericGraderTests(unittest.TestCase):
    def test_exact_and_tolerance(self) -> None:
        self.assertTrue(grade_numeric('136', {'type': 'numeric', 'expected': 136}).passed)
        self.assertTrue(grade_numeric(' 136 ', {'type': 'numeric', 'expected': 136}).passed)
        self.assertFalse(grade_numeric('137', {'type': 'numeric', 'expected': 136}).passed)
        self.assertTrue(grade_numeric('85.835', {'type': 'numeric', 'expected': 85.83, 'absolute_tolerance': 0.01}).passed)
        self.assertFalse(grade_numeric('85.85', {'type': 'numeric', 'expected': 85.83, 'absolute_tolerance': 0.01}).passed)

    def test_prose_rejection(self) -> None:
        self.assertFalse(grade_numeric('The answer is roughly 136 or so', {'type': 'numeric', 'expected': 136}).passed)
        self.assertFalse(grade_numeric('The answer is 136.', {'type': 'numeric', 'expected': 136}).passed)
        self.assertTrue(grade_numeric('136', {'type': 'numeric', 'expected': 136}).passed)


class TimeGraderTests(unittest.TestCase):
    def test_minute_precision_forms(self) -> None:
        grader = {'type': 'time', 'expected': '00:55', 'precision': 'minute', 'parsing': ['HH:MM']}
        self.assertTrue(grade_time('00:55', grader).passed)
        self.assertFalse(grade_time('0:55', grader).passed)
        self.assertFalse(grade_time('12:55 AM', grader).passed)

    def test_ampm_with_explicit_parsing(self) -> None:
        grader = {'type': 'time', 'expected': '11:37', 'precision': 'minute', 'parsing': ['HH:MM', 'H:MM AM', 'H:MM a.m.']}
        self.assertTrue(grade_time('11:37 AM', grader).passed)
        self.assertTrue(grade_time('11:37 a.m.', grader).passed)
        grader_24h = {'type': 'time', 'expected': '00:55', 'precision': 'minute', 'parsing': ['HH:MM']}
        self.assertFalse(grade_time('12:55 AM', grader_24h).passed)

    def test_no_mathematical_rounding(self) -> None:
        grader = {'type': 'time', 'expected': '11:37', 'precision': 'minute'}
        self.assertFalse(grade_time('11:36', grader).passed)
        self.assertFalse(grade_time('11:38', grader).passed)


class StructuredGraderTests(unittest.TestCase):
    def test_valid_object(self) -> None:
        grader = {
            'type': 'structured',
            'expected_output': {'ticket_id': 5512, 'customer': 'Noor', 'issue': 'refund not received', 'current_priority': 'urgent'},
            'required_fields': ['ticket_id', 'customer', 'issue', 'current_priority'],
            'allow_extra_fields': False,
        }
        result = grade_structured('{"ticket_id": 5512, "customer": "Noor", "issue": "refund not received", "current_priority": "urgent"}', grader)
        self.assertTrue(result.passed, result.violations)

    def test_missing_and_extra_fields(self) -> None:
        grader = {
            'type': 'structured',
            'expected_output': {'customer': 'Noor', 'issue': 'refund not received'},
            'required_fields': ['customer', 'issue'],
            'allow_extra_fields': False,
        }
        result = grade_structured('{"customer": "Noor", "issue": "refund not received", "extra": 1}', grader)
        self.assertFalse(result.passed)
        self.assertTrue(any('extra' in v for v in result.violations))
        result = grade_structured('{"customer": "Noor"}', grader)
        self.assertFalse(result.passed)
        self.assertTrue(any('missing required field' in v for v in result.violations))

    def test_code_fence_tolerated(self) -> None:
        grader = {
            'type': 'structured',
            'expected_output': {'accuracy_rate': 0.8, 'all_runs_consistent': False},
            'required_fields': ['accuracy_rate', 'all_runs_consistent'],
            'allow_extra_fields': False,
            'numeric_fields': {'accuracy_rate': {'absolute_tolerance': 0.0001}},
        }
        result = grade_structured('```json\n{"accuracy_rate": 0.80, "all_runs_consistent": false}\n```', grader)
        self.assertTrue(result.passed, result.violations)

    def test_unicode_field(self) -> None:
        grader = {
            'type': 'structured',
            'expected_output': {'name': 'João da Silva', 'city': 'São Paulo', 'score': 8.5},
            'required_fields': ['name', 'city', 'score'],
            'allow_extra_fields': False,
            'unicode_normalization': 'NFC',
        }
        self.assertTrue(grade_structured('{"name": "João da Silva", "city": "São Paulo", "score": 8.5}', grader).passed)
        self.assertFalse(grade_structured('{"name": "Joao da Silva", "city": "Sao Paulo", "score": 8.5}', grader).passed)

    def test_invalid_json(self) -> None:
        grader = {'type': 'structured', 'expected_output': {'a': 1}, 'required_fields': ['a'], 'allow_extra_fields': False}
        result = grade_structured('{"a": 1,}', grader)
        self.assertFalse(result.passed)
        self.assertIn('invalid JSON', result.detail)


class ToolCallGraderTests(unittest.TestCase):
    def test_exact_call(self) -> None:
        grader = {
            'type': 'tool_call',
            'expected_output': {'function': 'book_table', 'arguments': {'restaurant': 'Olive Bistro', 'people': 4}},
            'required_arguments': {'book_table': ['restaurant', 'people']},
        }
        self.assertTrue(grade_tool_call('{"function": "book_table", "arguments": {"restaurant": "Olive Bistro", "people": 4}}', grader).passed)
        self.assertFalse(grade_tool_call('{"function": "book_table", "arguments": {"restaurant": "Olive Bistro", "people": "four"}}', grader).passed)
        self.assertFalse(grade_tool_call('{"function": "book_table", "arguments": {"restaurant": "Olive Bistro", "people": 4, "time": "7pm"}}', grader).passed)

    def test_forbidden_function(self) -> None:
        grader = {
            'type': 'tool_call',
            'expected_output': {'function': 'list_transactions', 'arguments': {'account': 'savings', 'days': 1}},
            'required_arguments': {'list_transactions': ['account', 'days']},
            'forbidden_functions': ['transfer_money'],
        }
        result = grade_tool_call('{"function": "transfer_money", "arguments": {"account": "savings", "amount": 100}}', grader)
        self.assertFalse(result.passed)
        self.assertTrue(any('forbidden' in v for v in result.violations))


class ConstraintGraderTests(unittest.TestCase):
    def test_word_count_and_terms(self) -> None:
        grader = {
            'type': 'constraints',
            'constraints': [
                {'kind': 'word_count', 'value': 4},
                {'kind': 'forbidden_terms', 'terms': ['protocol']},
                {'kind': 'term_occurrence', 'term': 'latency', 'occurrence': 'exactly', 'count': 1},
            ],
        }
        self.assertTrue(grade_constraints('Low latency here today', grader).passed)
        self.assertFalse(grade_constraints('Latency matters; the protocol stacks latency up.', grader).passed)

    def test_sentences_and_questions(self) -> None:
        grader = {
            'type': 'constraints',
            'constraints': [
                {'kind': 'sentence_count', 'value': 3},
                {'kind': 'sentence_contains', 'index': 1, 'terms': ['chlorophyll']},
                {'kind': 'sentence_numeric_tokens', 'index': 2, 'min': 1, 'max': 1},
                {'kind': 'sentence_is_question', 'index': 3},
            ],
        }
        good = 'Chlorophyll drives the process. About 6 molecules cooperate. Does that surprise you?'
        self.assertTrue(grade_constraints(good, grader).passed)
        bad = 'Chlorophyll drives the process. About 6 or 7 molecules cooperate. Does that surprise you?'
        self.assertFalse(grade_constraints(bad, grader).passed)
        missing_term = 'Plants grow tall. About 6 molecules cooperate. Does that surprise you?'
        result = grade_constraints(missing_term, grader)
        self.assertFalse(result.passed)
        self.assertTrue(any('chlorophyll' in v for v in result.violations))

    def test_bullets_and_prefix(self) -> None:
        grader = {
            'type': 'constraints',
            'constraints': [
                {'kind': 'bullet_count', 'value': 2},
                {'kind': 'bullet_prefix', 'index': -1, 'prefix': 'Therefore'},
                {'kind': 'forbidden_chars', 'chars': ['q']},
            ],
        }
        good = '- TCP skips handshakes\n- Therefore latency drops'
        self.assertTrue(grade_constraints(good, grader).passed)
        self.assertFalse(grade_constraints('- TCP quick handshake\n- Therefore latency drops', grader).passed)
        self.assertFalse(grade_constraints('- TCP handshakes\n- extra bullet\n- Therefore latency drops', grader).passed)

    def test_markdown_table_and_end(self) -> None:
        grader = {
            'type': 'constraints',
            'constraints': [
                {'kind': 'markdown_table_shape', 'columns': ['SSD', 'HDD'], 'data_rows': 2},
                {'kind': 'trailing_literal', 'literal': 'END'},
                {'kind': 'forbidden_terms', 'terms': ['price']},
            ],
        }
        good = '| SSD | HDD |\n|---|---|\n| faster reads | slower reads |\n| no platters | spinning platters |\nEND'
        self.assertTrue(grade_constraints(good, grader).passed)
        self.assertFalse(grade_constraints('| SSD | HDD |\n|---|---|\n| low price | cheap |\nEND', grader).passed)

    def test_unicode_words(self) -> None:
        self.assertEqual(words('João São  test'), ['João', 'São', 'test'])
        self.assertEqual(words("don't stop"), ["don't", 'stop'])
        self.assertEqual(normalize_text('  spaced  \n'), 'spaced')

    def test_allowed_punctuation_markers_only(self) -> None:
        # Q015 shape: "No punctuation except the initial letter's period."
        grader = {
            'type': 'constraints',
            'constraints': [
                {'kind': 'allowed_punctuation', 'allowed': ['A', '.', 'B', 'C', 'D', 'E', 'F']}
            ],
        }
        good = 'A. Foo bar baz qux quux corge grault\nB. Second line here now today'
        self.assertTrue(grade_constraints(good, grader).passed)
        self.assertFalse(grade_constraints('A. Wow, really?', grader).passed)
        self.assertFalse(grade_constraints('A. Trailing period here.', grader).passed)

    def test_term_occurrence_across_json_values(self) -> None:
        # Q016 shape: exactly one JSON value may contain the word.
        grader = {
            'type': 'constraints',
            'constraints': [
                {'kind': 'term_occurrence', 'term': 'latency', 'occurrence': 'across_values', 'count': 1}
            ],
        }
        good = '{"benefit_1": "caching cuts latency drops", "benefit_2": "fewer backend calls daily", "risk": "stale data served"}'
        self.assertTrue(grade_constraints(good, grader).passed)
        both = '{"benefit_1": "lower latency here", "benefit_2": "latency again", "risk": "stale data"}'
        self.assertFalse(grade_constraints(both, grader).passed)
        self.assertFalse(grade_constraints('not json at all', grader).passed)

    def test_item_sentence_count_all_items(self) -> None:
        # Q017 shape: no index means every numbered item holds one sentence.
        grader = {
            'type': 'constraints',
            'constraints': [{'kind': 'item_sentence_count', 'value': 1}],
        }
        good = '1. Measure latency first.\n2. Check logs now.\n3. Split; retry.'
        self.assertTrue(grade_constraints(good, grader).passed)
        bad = '1. Measure latency first. Then record it.\n2. Check logs now.'
        result = grade_constraints(bad, grader)
        self.assertFalse(result.passed)
        self.assertTrue(any('item 1' in v for v in result.violations))
        self.assertFalse(grade_constraints('No numbered items here.', grader).passed)

    def test_bullet_prefix_first_and_last(self) -> None:
        last = {
            'type': 'constraints',
            'constraints': [{'kind': 'bullet_prefix', 'index': -1, 'prefix': 'Therefore'}],
        }
        self.assertTrue(grade_constraints('- Fast.\n- Therefore slow.', last).passed)
        self.assertFalse(grade_constraints('- Therefore fast.\n- Slow end.', last).passed)
        first = {
            'type': 'constraints',
            'constraints': [{'kind': 'bullet_prefix', 'index': 1, 'prefix': 'First'}],
        }
        self.assertTrue(grade_constraints('- First point.\n- Second point.', first).passed)
        self.assertFalse(grade_constraints('- Zeroth.\n- First later.', first).passed)

    def test_engine_covers_every_frozen_constraint_shape(self) -> None:
        # Regression lock: every (kind, key-shape) in the frozen grading spec
        # must be handled by the engine — no KeyError, no "unknown kind".
        # A gap here once crashed a live 129-generation baseline run (Q012).
        import yaml

        spec_path = Path(__file__).resolve().parents[1] / 'evals/specs/eval-v1-grading.yaml'
        spec = yaml.safe_load(spec_path.read_text(encoding='utf-8'))
        sample = 'Alpha beta gamma. Second sentence here? Third one follows!'
        checked = 0
        for task_id, entry in spec['tasks'].items():
            for grader_entry in entry.get('graders') or []:
                if grader_entry.get('type') != 'constraints':
                    continue
                result = grade_constraints(sample, grader_entry)
                self.assertNotIn(
                    'unknown constraint kind',
                    '; '.join(result.violations),
                    f'{task_id} uses an unhandled constraint kind',
                )
                checked += 1
        self.assertGreater(checked, 0)


class FactCheckGraderTests(unittest.TestCase):
    def test_required_facts_and_limit(self) -> None:
        grader = {
            'type': 'fact_check',
            'required_facts': ['₹250', 'institute portal', '30 September 2026'],
            'word_limit': {'max': 25},
        }
        good = 'Applications close 30 September 2026 at 5:00 PM. The ₹250 fee applies via institute portal.'
        self.assertTrue(grade_fact_check(good, grader).passed)
        long_bad = good + ' ' + 'extra filler ' * 30
        result = grade_fact_check(long_bad, grader)
        self.assertFalse(result.passed)
        self.assertTrue(any('word_limit' in v for v in result.violations))
        missing = 'Applications close 30 September 2026 via the portal with the fee.'
        result = grade_fact_check(missing, grader)
        self.assertFalse(result.passed)
        self.assertTrue(any('missing fact' in v for v in result.violations))

    def test_forbidden_facts(self) -> None:
        grader = {'type': 'fact_check', 'required_facts': ['pool'], 'forbidden_facts': ['database']}
        self.assertTrue(grade_fact_check('The connection pool shrank; rollbacks followed.', grader).passed)
        self.assertFalse(grade_fact_check('The connection pool shrank; the database was blamed.', grader).passed)


class RefusalGraderTests(unittest.TestCase):
    def test_refusal_detection(self) -> None:
        grader = {'type': 'refusal', 'refusal_expected': True}
        self.assertTrue(grade_refusal('UNKNOWN', grader).passed)
        self.assertTrue(grade_refusal('INSUFFICIENT_INFORMATION', grader).passed)
        self.assertFalse(grade_refusal('42 requests', grader).passed)
        grader = {'type': 'refusal', 'refusal_expected': False}
        self.assertTrue(grade_refusal('1e-4', grader).passed)
        self.assertFalse(grade_refusal('UNKNOWN', grader).passed)


class RubricJudgeTests(unittest.TestCase):
    def test_requires_criteria(self) -> None:
        grader = {'type': 'rubric_judge', 'rubric_required': True, 'criteria': {'concise': {'type': 'boolean'}}}
        self.assertTrue(grade_rubric_judge('any text', grader).passed)
        self.assertFalse(grade_rubric_judge('any text', {'type': 'rubric_judge'}).passed)


if __name__ == '__main__':
    unittest.main()
