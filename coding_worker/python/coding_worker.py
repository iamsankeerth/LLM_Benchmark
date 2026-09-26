from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


def _truncate(value: str, limit: int) -> str:
    return value if len(value.encode('utf-8')) <= limit else value.encode('utf-8')[:limit].decode('utf-8', errors='ignore')


def _result(
    *,
    passed: bool,
    failure_kind: str | None,
    stdout: str,
    stderr: str,
    total_tests: int,
    static_passed: int,
    policy: dict[str, Any],
) -> dict[str, Any]:
    output_limit = int(policy['output_bytes'])
    return {
        'passed': passed,
        'failure_kind': failure_kind,
        'stdout': _truncate(stdout, output_limit),
        'stderr': _truncate(stderr, output_limit),
        'tests': {
            'static_passed': static_passed,
            'total': total_tests,
        },
    }


def main() -> int:
    try:
        request = json.load(sys.stdin)
        if not isinstance(request, dict):
            raise ValueError('request is not an object')
        if request.get('language') != 'python':
            raise ValueError('python worker received another language')
        candidate_source = str(request['candidate_source'])
        test_source = str(request['test_source'])
        cases_json = str(request['cases_json'])
        policy = request['policy']
        if not isinstance(policy, dict):
            raise ValueError('policy is not an object')
        if len(candidate_source.encode('utf-8')) > int(policy['candidate_source_bytes']):
            raise ValueError('candidate source exceeds policy')
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(json.dumps({
            'passed': False,
            'failure_kind': 'WORKER_INPUT_ERROR',
            'error': str(exc),
            'tests': {'static_passed': 0, 'total': 0},
        }))
        return 0

    work = Path(tempfile.mkdtemp(prefix='coding-', dir='/tmp'))
    try:
        (work / 'candidate.py').write_text(candidate_source, encoding='utf-8')
        (work / 'test_candidate.py').write_text(test_source, encoding='utf-8')
        (work / 'cases.json').write_text(cases_json, encoding='utf-8')
        env = dict(os.environ)
        env['HOME'] = '/tmp'
        env['PYTHONDONTWRITEBYTECODE'] = '1'
        try:
            completed = subprocess.run(
                [sys.executable, '-m', 'unittest', '-v', 'test_candidate'],
                cwd=work,
                env=env,
                capture_output=True,
                text=True,
                timeout=float(policy['whole_candidate_wall_seconds']),
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            print(json.dumps(_result(
                passed=False,
                failure_kind='TIMEOUT',
                stdout=str(exc.stdout or ''),
                stderr=str(exc.stderr or ''),
                total_tests=0,
                static_passed=0,
                policy=policy,
            )))
            return 0
        combined = f'{completed.stdout}\n{completed.stderr}'
        match = re.search(r'^Ran (\d+) tests?', combined, re.MULTILINE)
        total_tests = int(match.group(1)) if match else 0
        passed = completed.returncode == 0 and total_tests > 0
        print(json.dumps(_result(
            passed=passed,
            failure_kind=None if passed else 'FUNCTIONAL_FAIL',
            stdout=completed.stdout,
            stderr=completed.stderr,
            total_tests=total_tests,
            static_passed=total_tests if passed else 0,
            policy=policy,
        )))
        return 0
    finally:
        for path in work.glob('*'):
            path.unlink(missing_ok=True)
        work.rmdir()


if __name__ == '__main__':
    raise SystemExit(main())
