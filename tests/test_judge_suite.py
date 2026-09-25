"""Offline Judge Suite V1 protocol and source-population tests."""

from __future__ import annotations

import json
import re
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from typing import Any, ClassVar
from unittest.mock import patch

from analysis.judge import (
    JudgeProtocol,
    JudgeProtocolError,
    JudgeSourceItem,
    build_rubric_prompt,
    build_controlled_pairs,
    load_judge_protocol,
    load_source_items,
    parse_pair_response,
    parse_rubric_response,
    position_flip_rate,
    select_real_pairs,
)
from scripts.run_judge_local_panel import JudgeRunError, run_local_panel
from scripts.run_external_judge import _run_pairs, run_external_judge
from inference.external_judge import ExternalJudgeCall
from storage.judge import (
    open_judge_db,
    record_protocol,
    record_source_item,
    source_population_hash,
)


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
        self.assertIn('never a list', prompt)
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
        response['item_id'] = 'caller-binds-this-id'
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

    def test_parser_normalizes_common_reasoning_shapes(self) -> None:
        item = next(item for item in self.items if item.task_id == 'Q059')
        rubric = {'criteria': self.protocol.rubrics['Q059']}
        quote = item.raw_output[: min(12, len(item.raw_output))]
        response: dict[str, Any] = {
            'schema_version': 'judge-rubric-v1',
            'criteria': [
                {
                    'criterion_id': criterion_id,
                    'observed': True,
                    'confidence': 'high',
                    'evidence': [{'quote': quote}],
                }
                for criterion_id in self.protocol.rubrics['Q059']
            ],
            'abstain': False,
        }
        parsed = parse_rubric_response(
            json.dumps(response), item_id=item.source_item_id,
            rubric=rubric, candidate=item.raw_output,
        )
        self.assertEqual(set(parsed['criteria']), set(rubric['criteria']))
        pair = parse_pair_response(
            json.dumps({
                'schema_version': 'judge-pair-v1',
                'decision': 'A',
                'confidence': 'high',
                'evidence': {'A': 'alpha'},
                'abstain': False,
            }),
            pair_id='pair',
            candidates={'A': 'alpha text', 'B': 'beta text'},
        )
        self.assertEqual(pair['evidence'], [{'candidate': 'A', 'quote': 'alpha'}])

    def test_external_protocol_is_active_and_human_free(self) -> None:
        document = self.protocol.document
        self.assertEqual(document['judge_mode'], 'external_openai_compatible')
        self.assertFalse(document['local_judges_enabled'])
        self.assertEqual(document['external_judge']['model_id'], 'stealth/space-bunny-alpha')
        self.assertTrue(document['external_judge']['reasoning_enabled'])
        self.assertFalse(document['human_adjudication']['required'])

    def test_external_canary_uses_no_local_judges(self) -> None:
        class FakeClient:
            def __init__(self) -> None:
                self.config = SimpleNamespace(
                    model_id='stealth/space-bunny-alpha',
                    base_url='https://api.example.test/v1',
                    max_attempts=1,
                    max_calls=10,
                )
                self.calls_made = 0
                self.total_cost_usd = 0.0
                self.known_cost_calls = 0
                self.unknown_cost_calls = 0

            def judge(self, prompt: str, _schema: Any) -> ExternalJudgeCall:
                self.calls_made += 1
                match = re.search(r'ITEM_ID: ([0-9a-f]+)', prompt)
                if match is None:
                    raise AssertionError('judge prompt omitted item id')
                item_id = match.group(1)
                criterion_ids = re.findall(r'^- ([A-Za-z0-9_]+):', prompt, re.MULTILINE)
                payload = {
                    'schema_version': 'judge-rubric-v1',
                    'item_id': item_id,
                    'criteria': {
                        criterion_id: {
                            'observed': True, 'confidence': 'high', 'evidence': [],
                        }
                        for criterion_id in criterion_ids
                    },
                    'abstain': False,
                }
                return ExternalJudgeCall(
                    text=json.dumps(payload), request_id='fake', usage={},
                    attempts=1, estimated_cost_usd=0.0,
                )

        with TemporaryDirectory() as tmp:
            db_path = Path(tmp) / 'judge.db'
            connection = open_judge_db(db_path)
            record_protocol(
                connection,
                protocol=self.protocol,
                population_hash=source_population_hash(self.items),
                rubric_hash='test-rubric-hash',
                created_at_utc='2026-01-01T00:00:00+00:00',
            )
            for item in self.items:
                record_source_item(
                    connection, protocol_id='judge-suite-v1', item=item,
                )
            client = FakeClient()
            with patch(
                'scripts.run_external_judge.load_source_items', return_value=self.items,
            ), patch('scripts.run_external_judge.open_judge_db', return_value=connection):
                result = run_external_judge(ROOT, canary=True, client=client)  # type: ignore[arg-type]
            self.assertEqual(result['status'], 'CANARY_COMPLETE')
            self.assertEqual(result['calls_made'], 3)
            self.assertFalse(result['local_judges_used'])
            self.assertFalse(result['human_involvement'])

    def test_external_pair_orientations_and_aggregate_are_persisted(self) -> None:
        pair = select_real_pairs(self.items, per_task=1)[0]

        class FakePairClient:
            def __init__(self) -> None:
                self.config = SimpleNamespace(
                    model_id='stealth/space-bunny-alpha',
                    base_url='https://api.example.test/v1',
                    max_attempts=1,
                    max_calls=10,
                )
                self.calls_made = 0
                self.total_cost_usd = 0.0
                self.known_cost_calls = 0
                self.unknown_cost_calls = 0

            def judge(self, prompt: str, _schema: Any) -> ExternalJudgeCall:
                self.calls_made += 1
                match = re.search(r'PAIR_ID: ([0-9a-f]+)', prompt)
                if match is None:
                    raise AssertionError('pair prompt omitted pair id')
                payload: dict[str, Any] = {
                    'schema_version': 'judge-pair-v1',
                    'pair_id': match.group(1),
                    'decision': 'A',
                    'confidence': 'high',
                    'evidence': [],
                    'abstain': False,
                }
                return ExternalJudgeCall(
                    text=json.dumps(payload), request_id='fake', usage={},
                    attempts=1, estimated_cost_usd=0.0,
                )

        with TemporaryDirectory() as tmp:
            connection = open_judge_db(Path(tmp) / 'judge.db')
            client = FakePairClient()
            counts = _run_pairs(
                connection, client, protocol_id='judge-suite-v1', pairs=[pair],  # type: ignore[arg-type]
            )
            rows = connection.execute(
                'SELECT orientation,decision,parse_status FROM judge_pair_results'
                ' WHERE pair_id=? ORDER BY orientation', (pair.pair_id,),
            ).fetchall()
            call_rows = connection.execute(
                'SELECT status,parse_status FROM external_judge_calls'
                ' WHERE pair_id=?', (pair.pair_id,),
            ).fetchall()
            connection.close()
        self.assertEqual(counts['pairs'], 1)
        self.assertEqual(counts['third_calls'], 1)
        self.assertEqual(
            [tuple(row) for row in rows],
            [
                ('aggregate', 'LEFT', 'VALID'),
                ('forward', 'LEFT', 'VALID'),
                ('inverse', 'RIGHT', 'VALID'),
                ('third_call', 'LEFT', 'VALID'),
            ],
        )
        self.assertEqual(
            sorted(tuple(row) for row in call_rows),
            [('CALL_COMPLETE', 'VALID')] * 3,
        )
        self.assertEqual(position_flip_rate([
            {'pair_id': pair.pair_id, 'orientation': row[0], 'decision': row[1]}
            for row in rows
        ]), 1.0)

    def test_external_pair_parse_failure_is_recorded_without_aggregate_label(self) -> None:
        pair = select_real_pairs(self.items, per_task=1)[0]

        class FakePairClient:
            def __init__(self) -> None:
                self.config = SimpleNamespace(
                    model_id='stealth/space-bunny-alpha',
                    base_url='https://api.example.test/v1',
                    max_attempts=1,
                    max_calls=10,
                )
                self.calls_made = 0
                self.total_cost_usd = 0.0
                self.known_cost_calls = 0
                self.unknown_cost_calls = 0

            def judge(self, prompt: str, _schema: Any) -> ExternalJudgeCall:
                self.calls_made += 1
                if self.calls_made == 1:
                    return ExternalJudgeCall(
                        text='{}', request_id='fake', usage={}, attempts=1,
                        estimated_cost_usd=0.0,
                    )
                match = re.search(r'PAIR_ID: ([0-9a-f]+)', prompt)
                if match is None:
                    raise AssertionError('pair prompt omitted pair id')
                payload: dict[str, Any] = {
                    'schema_version': 'judge-pair-v1',
                    'pair_id': match.group(1),
                    'decision': 'A',
                    'confidence': 'high',
                    'evidence': [],
                    'abstain': False,
                }
                return ExternalJudgeCall(
                    text=json.dumps(payload), request_id='fake', usage={},
                    attempts=1, estimated_cost_usd=0.0,
                )

        with TemporaryDirectory() as tmp:
            connection = open_judge_db(Path(tmp) / 'judge.db')
            counts = _run_pairs(
                connection, FakePairClient(), protocol_id='judge-suite-v1', pairs=[pair],  # type: ignore[arg-type]
            )
            rows = connection.execute(
                'SELECT orientation,decision,parse_status FROM judge_pair_results'
                ' WHERE pair_id=? ORDER BY orientation', (pair.pair_id,),
            ).fetchall()
            call_rows = connection.execute(
                'SELECT status,parse_status FROM external_judge_calls'
                ' WHERE pair_id=?', (pair.pair_id,),
            ).fetchall()
            connection.close()
        self.assertEqual(counts['errors'], 1)
        self.assertEqual(counts['pairs'], 0)
        self.assertEqual(
            [tuple(row) for row in rows],
            [('aggregate', 'INVALID', 'INCOMPLETE'), ('inverse', 'RIGHT', 'VALID')],
        )
        self.assertIn(('PARSE_ERROR', 'INVALID'), [tuple(row) for row in call_rows])

    def test_missing_human_calibration_refuses_before_calls(self) -> None:
        with TemporaryDirectory() as tmp:
            with self.assertRaises(JudgeRunError):
                run_local_panel(
                    ROOT, calibration_path=Path(tmp) / 'missing-calibration.json'
                )

    def test_real_pair_selector_handles_small_eligible_groups(self) -> None:
        all_items = load_source_items(ROOT, self.protocol)
        pairs = select_real_pairs(all_items, per_task=10)
        self.assertEqual(len(pairs), 30)
        self.assertEqual(
            {task_id: sum(pair.task_id == task_id for pair in pairs) for task_id in {'Q014', 'Q059', 'Q060'}},
            {'Q014': 10, 'Q059': 10, 'Q060': 10},
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
