"""Run trusted golden coding fixtures outside the model sandbox boundary."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis.coding import load_fixture_manifest


def validate(root: Path) -> None:
    fixture_root = root / 'evals/fixtures/coding-v1'
    golden_root = root / 'tests/fixtures/coding-v1/golden'
    manifest = load_fixture_manifest(fixture_root / 'fixture-manifest.json')
    for task_id, task in sorted(manifest['tasks'].items()):
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            (work / 'cases.json').write_text(
                (fixture_root / task['fixture_path']).read_text(encoding='utf-8'),
                encoding='utf-8',
            )
            if task['language'] == 'python':
                shutil.copyfile(golden_root / f'{task_id}.py', work / 'candidate.py')
                shutil.copyfile(fixture_root / task['test_path'], work / 'test_candidate.py')
                command = [sys.executable, '-m', 'unittest', 'test_candidate.py']
            else:
                shutil.copyfile(golden_root / f'{task_id}.mjs', work / 'candidate.mjs')
                shutil.copyfile(fixture_root / task['test_path'], work / 'test_candidate.mjs')
                shutil.copyfile(fixture_root / task['fixture_path'], work / 'cases.json')
                command = ['node', '--test', 'test_candidate.mjs']
            result = subprocess.run(command, cwd=work, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(
                    f'{task_id} golden fixture failed:\n{result.stdout}\n{result.stderr}'
                )


def main(argv: list[str] | None = None) -> int:
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description='Validate coding golden fixtures')
    parser.parse_args(argv)
    try:
        validate(root)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f'CODING GOLDEN FIXTURES FAILED: {exc}', flush=True)
        return 2
    print('coding golden fixtures: PASS')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
