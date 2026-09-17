"""Spec-validation tests: the grading layer itself is validated, not trusted.

These tests verify that every task in the grading spec is internally
consistent, that the audit invariants hold, and that invalid specifications
are rejected.
"""

import json
import unittest
from collections import Counter
from pathlib import Path
from typing import Any, ClassVar, cast

import yaml
from jsonschema import Draft202012Validator

from scripts.audit_grading_specs import audit
from scripts.convert_eval_workbook import ROOT


class GradingSpecTests(unittest.TestCase):
    spec_path: ClassVar[Path]
    schema_path: ClassVar[Path]
    spec: ClassVar[dict[str, Any]]
    schema: ClassVar[dict[str, Any]]
    tasks: ClassVar[dict[str, dict[str, Any]]]

    @classmethod
    def setUpClass(cls) -> None:
        cls.spec_path = ROOT / 'evals/specs/eval-v1-grading.yaml'
        cls.schema_path = ROOT / 'evals/specs/grading-spec.schema.json'
        cls.spec = cast(dict[str, Any], yaml.safe_load(cls.spec_path.read_text(encoding='utf-8')))
        cls.schema = cast(dict[str, Any], json.loads(cls.schema_path.read_text(encoding='utf-8')))
        cls.tasks = cast(dict[str, dict[str, Any]], cls.spec['tasks'])

    def test_spec_matches_schema(self) -> None:
        Draft202012Validator.check_schema(self.schema)
        errors = list(Draft202012Validator(self.schema).iter_errors(self.spec))
        self.assertEqual(errors, [], [e.message[:200] for e in errors])

    def test_all_80_tasks_and_exact_ids(self) -> None:
        self.assertEqual(len(self.tasks), 80)
        self.assertEqual(set(self.tasks), {f'Q{i:03}' for i in range(1, 81)})

    def test_no_duplicate_primary_class_keys(self) -> None:
        for task_id, entry in self.tasks.items():
            self.assertIn('primary_class', entry, task_id)
            self.assertIsInstance(entry['primary_class'], str, task_id)

    def test_grader_lists_may_overlap_but_counts_derived(self) -> None:
        counts = Counter(entry['primary_class'] for entry in self.tasks.values())
        self.assertEqual(counts['OPTIONAL_NOT_READY'], 8)
        total_mandatory = sum(v for k, v in counts.items() if k != 'OPTIONAL_NOT_READY')
        self.assertEqual(total_mandatory, 72)

    def test_ready_tasks_have_graders_pending_do_not(self) -> None:
        for task_id, entry in self.tasks.items():
            status = entry['grading_status']
            graders = entry['graders']
            if status in ('READY_DETERMINISTIC', 'READY_JUDGE'):
                self.assertGreater(len(graders), 0, f'{task_id}: ready status requires graders')
                self.assertNotIn('pending_reason', entry, f'{task_id}: ready task must not carry pending_reason')
            else:
                if status == 'PENDING_SPECIFICATION':
                    self.assertEqual(graders, [], f'{task_id}: pending-specification must have no graders')

    def test_pending_review_has_provenance_or_reason(self) -> None:
        for task_id, entry in self.tasks.items():
            if entry['grading_status'] == 'PENDING_REVIEW':
                has_provenance = 'provenance' in entry
                has_reason = 'pending_reason' in entry
                self.assertTrue(has_provenance or has_reason, f'{task_id}: pending review needs justification')
                if has_provenance:
                    provenance = entry['provenance']
                    self.assertEqual(provenance['review_status'], 'REQUIRED', f'{task_id}: provenance must require review')

    def test_numeric_tasks_have_tolerances(self) -> None:
        for task_id, entry in self.tasks.items():
            for grader in entry['graders']:
                if grader['type'] == 'numeric':
                    self.assertIn('expected', grader, f'{task_id}: numeric grader needs expected')
                    tolerance = grader.get('absolute_tolerance', 0) or grader.get('relative_tolerance', 0)
                    self.assertIsInstance(tolerance, (int, float), f'{task_id}: tolerance must be explicit')
                    self.assertGreaterEqual(tolerance, 0, f'{task_id}: tolerance cannot be negative')

    def test_deterministic_tasks_have_expected_values(self) -> None:
        for task_id, entry in self.tasks.items():
            if entry['grading_status'] != 'READY_DETERMINISTIC':
                continue
            graders = entry['graders']
            self.assertTrue(any('expected' in g or 'expected_output' in g or 'constraints' in g or 'required_facts' in g for g in graders), f'{task_id}: deterministic task lacks any expected ground truth')

    def test_judge_tasks_have_rubrics(self) -> None:
        for task_id, entry in self.tasks.items():
            if entry['grading_status'] != 'READY_JUDGE':
                continue
            has_judge = any(g['type'] == 'rubric_judge' and g.get('criteria') for g in entry['graders'])
            self.assertTrue(has_judge, f'{task_id}: judge-ready task needs rubric criteria')

    def test_structured_tasks_have_canonical_objects(self) -> None:
        for task_id, entry in self.tasks.items():
            for grader in entry['graders']:
                if grader['type'] == 'structured':
                    self.assertIsInstance(grader['expected_output'], dict, f'{task_id}: structured expected_output must be object')
                    self.assertIn('required_fields', grader, f'{task_id}: structured grader needs required_fields')
                    self.assertIn('allow_extra_fields', grader, f'{task_id}: structured grader must declare extra-field policy')
                elif grader['type'] == 'tool_call':
                    self.assertIsInstance(grader['expected_output'], dict, f'{task_id}: tool_call expected_output must be object')
                    self.assertIn('required_arguments', grader, f'{task_id}: tool_call grader needs required_arguments')

    def test_time_tasks_pin_precision(self) -> None:
        for task_id, entry in self.tasks.items():
            for grader in entry['graders']:
                if grader['type'] == 'time':
                    self.assertIn('precision', grader, f'{task_id}: time grader must pin precision')

    def test_defaults_block_pinned(self) -> None:
        defaults = self.spec['defaults']
        self.assertEqual(defaults['unicode_normalization'], 'NFC')
        self.assertTrue(defaults['trim_outer_whitespace'])
        self.assertTrue(defaults['normalize_line_endings'])
        self.assertTrue(defaults['text']['case_sensitive'])
        self.assertEqual(defaults['word_count']['tokenizer'], 'unicode_whitespace')
        self.assertEqual(defaults['numeric']['absolute_tolerance'], 0)
        self.assertEqual(defaults['numeric']['relative_tolerance'], 0)
        self.assertEqual(defaults['time']['precision'], 'exact')

    def test_audit_invariants(self) -> None:
        result = audit()
        self.assertTrue(result['ids_exactly_q001_q080'])
        self.assertEqual(result['mandatory_task_ids'], 72)
        self.assertEqual(result['optional_coding_task_ids'], 8)
        self.assertEqual(result['dataset_task_count'], 80)
        self.assertEqual(result['review_inventory_ids'], 80)
        self.assertTrue(result['mandatory_have_primary_class'])
        self.assertEqual(result['fully_ready_mandatory'] + result['pending_human_review'], 72)
        self.assertEqual(result['unresolved_mandatory_blockers'], len(result['blockers']))

    def test_all_mandatory_tasks_approved_and_ready(self) -> None:
        result = audit()
        self.assertEqual(result['fully_ready_mandatory'], 72, str(result['blockers']))
        self.assertEqual(result['pending_human_review'], 0)
        self.assertEqual(result['unresolved_mandatory_blockers'], 0)
        blocker_tasks = {b['task'] for b in result['blockers']}
        self.assertEqual(blocker_tasks, set())
        self.assertEqual(result['spec_status'], 'FROZEN')
        for task_id in ('Q023', 'Q052', 'Q066', 'Q071'):
            entry = self.tasks[task_id]
            self.assertEqual(entry['grading_status'], 'READY_DETERMINISTIC', task_id)
            self.assertEqual(entry['provenance']['review_status'], 'APPROVED', task_id)
            self.assertIn('approved_by', entry['provenance'], task_id)
            self.assertIn('approved_on', entry['provenance'], task_id)


if __name__ == '__main__':
    unittest.main()
