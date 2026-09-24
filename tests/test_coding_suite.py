"""Safe coding extraction, static-check, and sandbox-gate tests."""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from analysis.coding import (
    CodingProtocolError,
    extract_code,
    load_fixture_manifest,
    static_check,
)
from inference.sandbox import SandboxRequest, SandboxUnavailable, run_isolated_tests
from scripts.run_coding_suite import CodingRunError, run_coding_model


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / 'evals/fixtures/coding-v1'
POLICY = {
    'max_source_bytes': 65536,
    'python_fence_languages': ['python', 'py'],
    'javascript_fence_languages': ['javascript', 'js'],
}


class CodingSuiteTests(unittest.TestCase):
    def test_extraction_accepts_raw_and_one_fence(self) -> None:
        raw = extract_code('def f():\n    return 1\n', language='python', done_reason='stop', policy=POLICY)
        fenced = extract_code('```python\ndef f():\n    return 1\n```', language='python', done_reason='stop', policy=POLICY)
        self.assertEqual(raw.source, fenced.source)
        self.assertEqual(fenced.extraction_method, 'single_whole_document_fence')

    def test_extraction_rejects_unsafe_forms(self) -> None:
        cases = (
            ('```python\na=1\n```\ntext', 'prose'),
            ('```python\na=1\n```\n```python\nb=2\n```', 'multiple'),
            ('```python\na=1', 'malformed'),
        )
        for output, label in cases:
            with self.subTest(label=label):
                with self.assertRaises(CodingProtocolError):
                    extract_code(output, language='python', done_reason='stop', policy=POLICY)
        with self.assertRaises(CodingProtocolError):
            extract_code('x=1', language='python', done_reason='length', policy=POLICY)

    def test_static_checks_do_not_execute_code(self) -> None:
        good = static_check('def f(x):\n    return x\n', language='python', entrypoint='f', rules=[])
        bad = static_check('import re\ndef f(x):\n    return x\n', language='python', entrypoint='f', rules=['no_re_import'])
        self.assertTrue(good['passed'])
        self.assertFalse(bad['passed'])

    def test_fixture_manifest_has_all_tasks(self) -> None:
        manifest = load_fixture_manifest(FIXTURES / 'fixture-manifest.json')
        self.assertEqual(set(manifest['tasks']), {f'Q{i:03d}' for i in range(27, 35)})

    def test_live_runner_refuses_unpinned_runtime_before_generation(self) -> None:
        with TemporaryDirectory() as tmp:
            with self.assertRaises(CodingRunError):
                run_coding_model(
                    ROOT, model_id='qwen3-4b-q4', db_path=Path(tmp) / 'coding.db',
                    generate_fn=lambda *_args, **_kwargs: self.fail('must not generate'),
                )

    def test_sandbox_requires_pinned_digest(self) -> None:
        request = SandboxRequest(
            language='python', entrypoint='f', candidate_source='def f():\n    return 1\n',
            test_source='', cases_json='[]', image_ref='python:3.12-slim',
            image_digest=None, policy={'whole_candidate_wall_seconds': 5},
        )
        with self.assertRaises(SandboxUnavailable):
            run_isolated_tests(request)


if __name__ == '__main__':
    unittest.main()
