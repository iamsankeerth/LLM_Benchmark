"""Safe coding extraction, static-check, and sandbox-gate tests."""

from __future__ import annotations

import json
import subprocess
import unittest
from pathlib import Path

from analysis.coding import (
    CodingProtocolError,
    extract_code,
    load_fixture_manifest,
    static_check,
)
from inference.sandbox import SandboxRequest, SandboxUnavailable, run_isolated_tests


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

    def test_live_runner_requires_pinned_runtime(self) -> None:
        manifest = load_fixture_manifest(FIXTURES / 'fixture-manifest.json')
        self.assertTrue(manifest['runtimes']['python']['digest'])
        self.assertTrue(manifest['runtimes']['node']['digest'])

    def test_sandbox_requires_pinned_digest(self) -> None:
        request = SandboxRequest(
            language='python', entrypoint='f', candidate_source='def f():\n    return 1\n',
            test_source='', cases_json='[]', image_ref='coding-worker-python:1',
            image_digest=None, policy={'whole_candidate_wall_seconds': 5},
        )
        with self.assertRaises(SandboxUnavailable):
            run_isolated_tests(request)

    def test_sandbox_command_enforces_docker_boundary(self) -> None:
        calls: list[list[str]] = []

        def runner(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
            calls.append(command)
            if 'inspect' in command:
                return subprocess.CompletedProcess(command, 0, 'sha256:abc\n', '')
            return subprocess.CompletedProcess(
                command, 0,
                json.dumps({'passed': True, 'tests': {'static_passed': 1, 'total': 1}}),
                '',
            )

        request = SandboxRequest(
            language='python', entrypoint='f', candidate_source='def f():\n    return 1\n',
            test_source='', cases_json='[]', image_ref='coding-worker-python:1',
            image_digest='sha256:abc',
            policy={
                'processes': 64, 'memory_mib': 256, 'cpus': 1,
                'file_descriptors': 32, 'tmpfs_mib': 16,
                'whole_candidate_wall_seconds': 5, 'output_bytes': 1024,
            },
        )
        result = run_isolated_tests(request, runner=runner)
        self.assertTrue(result['passed'])
        self.assertEqual(len(calls), 2)
        command = calls[1]
        for flag in ('-i', '--network', '--read-only', '--cap-drop',
                     '--security-opt', '--pids-limit', '--memory', '--cpus',
                     '--tmpfs', '--user'):
            self.assertIn(flag, command)
        self.assertNotIn('--privileged', command)
        self.assertNotIn('docker.sock', ' '.join(command))
        self.assertNotIn('-v', command)

    def test_worker_test_failure_is_classified_as_task_failure(self) -> None:
        def runner(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
            if 'inspect' in command:
                return subprocess.CompletedProcess(command, 0, 'sha256:abc\n', '')
            return subprocess.CompletedProcess(
                command, 0,
                json.dumps({
                    'passed': False,
                    'failure_kind': 'WORKER_ERROR',
                    'tests': {'static_passed': 0, 'total': 1},
                }),
                '',
            )

        request = SandboxRequest(
            language='javascript', entrypoint='groupBy',
            candidate_source='export const other = 1;\n', test_source='',
            cases_json='[]', image_ref='coding-worker-node:1',
            image_digest='sha256:abc',
            policy={
                'processes': 64, 'memory_mib': 256, 'cpus': 1,
                'file_descriptors': 32, 'tmpfs_mib': 16,
                'whole_candidate_wall_seconds': 5, 'output_bytes': 1024,
            },
        )
        result = run_isolated_tests(request, runner=runner)
        self.assertEqual(result['status'], 'FAIL')
        self.assertEqual(result['failure_kind'], 'FUNCTIONAL_FAIL')


if __name__ == '__main__':
    unittest.main()
