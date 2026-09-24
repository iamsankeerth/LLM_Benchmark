"""Run the repository's platform-neutral offline quality gate."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    commands: list[list[str]] = [
        [sys.executable, '-m', 'ruff', 'check', '.'],
        [sys.executable, '-m', 'mypy', 'analysis', 'evals', 'inference', 'scripts', 'storage', 'tests'],
        [sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-p', 'test_*.py'],
        [sys.executable, 'scripts/convert_eval_workbook.py', '--check'],
        [sys.executable, 'scripts/audit_grading_specs.py', '--json'],
        [
            sys.executable,
            'scripts/audit_grading_specs.py',
            '--spec', 'evals/specs/eval-v1.1-grading.yaml',
            '--schema', 'evals/specs/eval-v1.1-grading.schema.json',
            '--freeze', 'evals/specs/eval-v1.1-grading.freeze.json',
            '--json',
        ],
        [sys.executable, 'scripts/audit_long_context.py', '--json'],
    ]
    for command in commands:
        print(f'QUALITY GATE: {" ".join(command)}', flush=True)
        result = subprocess.run(command, cwd=ROOT)
        if result.returncode != 0:
            print(f'QUALITY GATE FAILED: {command[0]} {command[1:]}', flush=True)
            return result.returncode
    print('QUALITY GATE PASS', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
