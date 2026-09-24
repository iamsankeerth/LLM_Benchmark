"""Offline Judge Suite V1 protocol and source-population tests."""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, ClassVar

from analysis.judge import (
    JudgeProtocol,
    JudgeProtocolError,
    JudgeSourceItem,
    build_rubric_prompt,
    build_controlled_pairs,
    load_judge_protocol,
    load_source_items,
    parse_rubric_response,
)
from scripts.run_judge_local_panel import JudgeRunError, run_local_panel


ROOT = Path(__file__).resolve().parents[1]


class JudgeProtocolTests(unittest.TestCase):
    protocol: ClassVar[JudgeProtocol]
    items: ClassVar[list[JudgeSourceItem]]

    @classmethod
    def setUpClass(cls) -> None:
        cls.protocol = load_judge_protocol(ROOT)
        cls.items = load_source_items(
            ROOT, cls.protocol, completed_ids=['qwen3-4b-q4']
        )

    def test_source_population_and_prechecks(self) -> None:
        self.assertEqual(len(self.items), 15)
        self.assertEqual(
            {item.task_id for item in self.items},
            {'Q014', 'Q047', 'Q051', 'Q059', 'Q060'},
        )
        self.assertTrue(all(item.raw_output_sha256 for item in self.items))
        self.assertTrue(all(
            item.precheck_status in {'PRECHECK_FAIL', 'JUDGE_PENDING'}
            for item in self.items
        ))

    def test_prompt_is_blinded_and_response_schema_is_strict(self) -> None:
        item = next(item for item in self.items if item.task_id == 'Q059')
        rubric = {'criteria': self.protocol.rubrics['Q059']}
        prompt = build_rubric_prompt(item, rubric)
        self.assertNotIn(item.model_config_id, prompt)
        self.assertIn(item.raw_output, prompt)
        response: dict[str, Any] = {
            'schema_version': 'judge-rubric-v1',
            'item_id': item.source_item_id,
            'criteria': {
                criterion_id: {
                    'observed': True,
                    'confidence': 'high',
                    'evidence': [],
                }
                for criterion_id in self.protocol.rubrics['Q059']
            },
            'abstain': False,
        }
        parsed = parse_rubric_response(
            json.dumps(response), item_id=item.source_item_id,
            rubric=rubric, candidate=item.raw_output,
        )
        self.assertFalse(parsed['abstain'])
        response['criteria']['technically_sound']['evidence'] = ['not in candidate']
        with self.assertRaises(JudgeProtocolError):
            parse_rubric_response(
                json.dumps(response), item_id=item.source_item_id,
                rubric=rubric, candidate=item.raw_output,
            )

    def test_missing_human_calibration_refuses_before_calls(self) -> None:
        with TemporaryDirectory() as tmp:
            with self.assertRaises(JudgeRunError):
                run_local_panel(
                    ROOT, calibration_path=Path(tmp) / 'missing-calibration.json'
                )

    def test_controlled_pairs_are_deterministic(self) -> None:
        first = build_controlled_pairs(self.items, per_task=8)
        second = build_controlled_pairs(self.items, per_task=8)
        self.assertEqual([pair.pair_id for pair in first], [pair.pair_id for pair in second])
        self.assertEqual(len(first), 8 * len({
            item.task_id for item in self.items
            if item.precheck_status == 'JUDGE_PENDING'
        }))

    def test_rubric_file_covers_all_source_tasks(self) -> None:
        self.assertEqual(
            set(self.protocol.rubrics),
            set(self.protocol.document['source_task_ids']),
        )


if __name__ == '__main__':
    unittest.main()
