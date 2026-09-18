"""Preflight tests: gate math, guards, no-DB side effects (mocked client)."""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from inference.ollama_client import GenerationResult
from scripts.preflight_context import NUM_CTX, main, run_preflight


def _fake_result(prompt_eval_count: int | None) -> GenerationResult:
    return GenerationResult(
        text='x', thinking='', done_reason='stop',
        prompt_eval_count=prompt_eval_count,
        request_start_ns=0, first_token_ns=1, request_end_ns=2,
    )


class PreflightTests(unittest.TestCase):
    def _run(self, counts: dict[str, int], num_predict: int = 2048) -> dict[str, object]:
        ordered = sorted(counts)

        def fake_generate(base_url: str, request: object, **kwargs: object) -> GenerationResult:
            task = ordered.pop(0)
            return _fake_result(counts[task])

        with patch('scripts.preflight_context.generate', side_effect=fake_generate):
            return run_preflight(
                'http://127.0.0.1:11434', 'qwen3-4b-q4', num_predict
            )

    def test_pass_with_headroom(self) -> None:
        import yaml

        root = Path(__file__).resolve().parents[1]
        spec = yaml.safe_load(
            (root / 'evals/specs/eval-v1-grading.yaml').read_text(encoding='utf-8')
        )
        mandatory = sorted(
            tid for tid, e in spec['tasks'].items()
            if e['grading_status'] in ('READY_DETERMINISTIC', 'READY_JUDGE')
        )
        counts = {tid: 150 for tid in mandatory}
        counts['Q008'] = 159
        document = self._run(counts)
        self.assertEqual(document['max_prompt_eval_count'], 159)
        self.assertEqual(document['max_task_id'], 'Q008')
        self.assertEqual(document['headroom_tokens'], NUM_CTX - 159 - 2048)
        self.assertTrue(document['preflight_pass'])
        self.assertEqual(document['num_ctx'], 4096)

    def test_fail_without_headroom(self) -> None:
        import yaml

        root = Path(__file__).resolve().parents[1]
        spec = yaml.safe_load(
            (root / 'evals/specs/eval-v1-grading.yaml').read_text(encoding='utf-8')
        )
        mandatory = sorted(
            tid for tid, e in spec['tasks'].items()
            if e['grading_status'] in ('READY_DETERMINISTIC', 'READY_JUDGE')
        )
        counts = {tid: 100 for tid in mandatory}
        counts['Q008'] = 2049
        document = self._run(counts)
        self.assertFalse(document['preflight_pass'])
        self.assertLess(int(str(document['headroom_tokens'])), 0)

    def test_missing_counter_is_error(self) -> None:
        def fake_generate(base_url: str, request: object, **kwargs: object) -> GenerationResult:
            return _fake_result(None)

        with patch('scripts.preflight_context.generate', side_effect=fake_generate):
            with self.assertRaises(Exception):
                run_preflight('http://127.0.0.1:11434', 'qwen3-4b-q4', 2048)

    def test_main_writes_artifact_and_exit_codes(self) -> None:
        with TemporaryDirectory() as tmp:
            out = str(Path(tmp) / 'probe.json')
            with patch(
                'scripts.preflight_context.run_preflight',
                return_value={
                    'model_config_id': 'm', 'template_sha256': 't' * 64,
                    'num_ctx': 4096, 'num_predict': 2048,
                    'max_prompt_eval_count': 159, 'max_task_id': 'Q008',
                    'headroom_tokens': 1889, 'preflight_pass': True,
                },
            ):
                code = main(['--model', 'qwen3-4b-q4', '--out', out])
            self.assertEqual(code, 0)
            document = json.loads(Path(out).read_text(encoding='utf-8'))
            self.assertTrue(document['preflight_pass'])
            for key in ('model_config_id', 'template_sha256', 'num_ctx',
                        'num_predict', 'max_prompt_eval_count', 'max_task_id',
                        'headroom_tokens', 'preflight_pass'):
                self.assertIn(key, document)


if __name__ == '__main__':
    unittest.main()
