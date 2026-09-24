"""Eval-v1.1 overlay, satisfiability, and corrected-grader tests."""

from __future__ import annotations

import unittest
from pathlib import Path
from typing import Any, ClassVar

from evals.contract import EvalContract, load_eval_contract
from evals.graders.engine import grade_output
from scripts.audit_grading_specs import audit
from evals.verdicts import NEEDS_JUDGE, PASS, reduce_verdict


ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = ROOT / 'evals/specs/eval-v1.1-grading.yaml'


class EvalV11ContractTests(unittest.TestCase):
    contract: ClassVar[EvalContract]
    tasks: ClassVar[dict[str, dict[str, Any]]]

    @classmethod
    def setUpClass(cls) -> None:
        cls.contract = load_eval_contract(ROOT, SPEC_PATH)
        cls.tasks = cls.contract.grader_entries()

    def test_overlay_resolves_complete_population(self) -> None:
        self.assertEqual(self.contract.spec_version, 'eval-v1.1')
        self.assertEqual(len(self.tasks), 80)
        statuses = [
            entry['grading_status'] for entry in self.tasks.values()
        ]
        self.assertEqual(statuses.count('READY_DETERMINISTIC'), 67)
        self.assertEqual(statuses.count('READY_JUDGE'), 5)
        self.assertEqual(statuses.count('PENDING_SPECIFICATION'), 8)

    def test_q016_is_satisfiable_and_word_bounded(self) -> None:
        grader = self.tasks['Q016']['graders'][0]
        constraint = self.tasks['Q016']['graders'][1]['constraints'][1]
        output = (
            '{"benefit_1":"Caching reduces latency for repeated requests",'
            '"benefit_2":"It lowers backend workload across users",'
            '"risk":"Stale cache data may confuse users"}'
        )
        results = grade_output(output, self.tasks['Q016']['graders'])
        self.assertTrue(all(result.passed for result in results), results)
        self.assertEqual(reduce_verdict('READY_DETERMINISTIC', results), PASS)
        self.assertEqual(grader['expected_output'], {})
        self.assertEqual(constraint['match'], 'word')
        wrong = output.replace('latency', 'latencies')
        self.assertFalse(all(result.passed for result in grade_output(
            wrong, self.tasks['Q016']['graders']
        )))

    def test_q047_is_judge_required_with_fact_groups(self) -> None:
        entry = self.tasks['Q047']
        self.assertEqual(entry['primary_class'], 'JUDGE_REQUIRED')
        self.assertEqual(entry['grading_status'], 'READY_JUDGE')
        fact_grader = entry['graders'][0]
        output = (
            'Currently enrolled undergraduates must submit the non-refundable '
            'Rs 250 fee through the institute portal by 5 PM on 30 Sept 2026; '
            'email submissions are not accepted.'
        )
        results = grade_output(output, entry['graders'])
        self.assertTrue(results[0].passed, results[0].violations)
        self.assertEqual(reduce_verdict('READY_JUDGE', results), NEEDS_JUDGE)
        missing = output.replace('non-refundable ', '')
        self.assertFalse(grade_output(missing, entry['graders'])[0].passed)
        self.assertTrue(fact_grader['required_fact_groups'])

    def test_audit_resolves_v11_counts(self) -> None:
        result = audit(
            spec_path=SPEC_PATH,
            schema_path=ROOT / 'evals/specs/eval-v1.1-grading.schema.json',
            freeze_path=ROOT / 'evals/specs/eval-v1.1-grading.freeze.json',
        )
        self.assertEqual(result['spec_version'], 'eval-v1.1')
        self.assertEqual(result['grading_status_counts']['READY_DETERMINISTIC'], 67)
        self.assertEqual(result['grading_status_counts']['READY_JUDGE'], 5)
        self.assertEqual(result['unresolved_mandatory_blockers'], 0)

    def test_overlay_resolves_without_freeze(self) -> None:
        contract = load_eval_contract(ROOT, SPEC_PATH)
        self.assertEqual(contract.spec_version, 'eval-v1.1')


if __name__ == '__main__':
    unittest.main()
