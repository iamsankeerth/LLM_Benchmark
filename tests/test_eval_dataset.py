import copy
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from typing import ClassVar, cast

from jsonschema import Draft202012Validator

from scripts.convert_eval_workbook import ROOT, TaskRecord, build, convert, digest, validate


class DatasetTests(unittest.TestCase):
    path: ClassVar[Path]
    output: ClassVar[Path]
    source: ClassVar[list[dict[str, object]]]
    tasks: ClassVar[list[TaskRecord]]

    @classmethod
    def setUpClass(cls) -> None:
        cls.path = ROOT / 'LocalLLM_Eval_Questions.xlsx'
        cls.output = ROOT / 'evals/datasets/eval-v1'
        cls.source, cls.tasks, _ = convert(cls.path)

    def test_counts_and_ids(self) -> None:
        self.assertEqual(Counter(cast(str, t['suite']) for t in self.tasks), {'core-v1': 52, 'stress-v1': 20, 'coding-v1': 8})
        self.assertEqual({cast(str, t['id']) for t in self.tasks}, {f'Q{i:03}' for i in range(1, 81)})
        self.assertEqual([cast(str, t['id']) for t in self.tasks if t['suite'] == 'coding-v1'], [f'Q{i:03}' for i in range(27, 35)])

    def test_source_fidelity(self) -> None:
        validate(self.source, self.tasks)
        for source, task in zip(self.source, self.tasks):
            question = cast(dict[str, str], source['question'])
            answer = cast(dict[str, str], source['answer_key'])
            self.assertEqual(source['id'], answer['ID'])
            self.assertEqual(task['graders_source'], question['Primary Grader'])
            self.assertEqual(task['grading_notes_source'], answer['Grading Notes'])

    def test_schema(self) -> None:
        schema = json.loads((self.output / 'task.schema.json').read_text(encoding='utf-8'))
        Draft202012Validator.check_schema(schema)
        validator = Draft202012Validator(schema)
        for task in self.tasks:
            validator.validate(task)
        invalid = copy.deepcopy(self.tasks[0])
        invalid['grading_status'] = 'READY_DETERMINISTIC'
        self.assertFalse(validator.is_valid(invalid))

    def test_unicode_and_serialization(self) -> None:
        data = (self.output / 'executable-v1.jsonl').read_bytes()
        self.assertFalse(data.startswith(b'\xef\xbb\xbf'))
        tasks = [json.loads(line) for line in data.decode('utf-8').splitlines()]
        self.assertEqual(tasks, self.tasks)
        self.assertIn('João', str(cast(dict[str, object], tasks[64])['prompt']))
        self.assertIn('₹', str(cast(dict[str, object], tasks[0])['prompt']))

    def test_reproducibility_and_source_hash(self) -> None:
        before = digest(self.path.read_bytes())
        manifest = build(self.path, self.output, check=True)
        source = cast(dict[str, object], manifest['source'])
        self.assertEqual(before, source['sha256'])
        self.assertEqual(before, digest(self.path.read_bytes()))

    def test_duplicate_and_prompt_changes_rejected(self) -> None:
        tasks = copy.deepcopy(self.tasks)
        tasks[1]['id'] = tasks[0]['id']
        with self.assertRaises(ValueError):
            validate(self.source, tasks)
        tasks = copy.deepcopy(self.tasks)
        tasks[0]['prompt'] = cast(str, tasks[0]['prompt']) + 'modified'
        with self.assertRaises(ValueError):
            validate(self.source, tasks)

    def test_refuse_overwrite_edits_and_frozen_release(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / 'task.schema.json').write_bytes((self.output / 'task.schema.json').read_bytes())
            build(self.path, output)
            data_path = output / 'executable-v1.jsonl'
            data_path.write_bytes(data_path.read_bytes() + b'\n')
            with self.assertRaises(ValueError):
                build(self.path, output)
            manifest_path = output / 'manifest.json'
            manifest = cast(dict[str, object], json.loads(manifest_path.read_text()))
            manifest['release_status'] = 'FROZEN'
            manifest_path.write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):
                build(self.path, output)

    def test_no_invented_grading_or_execution_settings(self) -> None:
        for task in self.tasks:
            self.assertEqual(task['grading_status'], 'PENDING_SPECIFICATION')
            self.assertIsNone(task['expected_output'])
            self.assertIsNone(task['context_size'])
            self.assertIsNone(task['timeout_seconds'])
            self.assertNotIn('baseline_repeats', task)
            self.assertTrue(task['pending_reasons'])


if __name__ == '__main__':
    unittest.main()
