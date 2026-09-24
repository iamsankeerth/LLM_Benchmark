"""Audit Coding v1 fixture, policy, and runtime-pin readiness."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml

from analysis.coding import load_fixture_manifest


ROOT = Path(__file__).resolve().parents[1]


def _fixture_bundle_hash(root: Path) -> str:
    fixture_root = root / 'evals/fixtures/coding-v1'
    paths = sorted(
        path for path in fixture_root.rglob('*')
        if path.is_file() and path.name not in {'fixture-manifest.json', 'fixture-hashes.json'}
    )
    payload = '\n'.join(
        f'{path.relative_to(fixture_root).as_posix()}|{hashlib.sha256(path.read_bytes()).hexdigest()}'
        for path in paths
    )
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


def audit() -> dict[str, Any]:
    spec_path = ROOT / 'evals/specs/coding-v1-grading.yaml'
    schema_path = ROOT / 'evals/specs/coding-v1.schema.json'
    spec = yaml.safe_load(spec_path.read_text(encoding='utf-8'))
    schema = json.loads(schema_path.read_text(encoding='utf-8'))
    Draft202012Validator.check_schema(schema)
    errors = [f'{list(error.absolute_path)}: {error.message}' for error in Draft202012Validator(schema).iter_errors(spec)]
    if errors:
        raise ValueError(f'coding spec schema errors: {errors[:5]}')
    fixture_path = ROOT / 'evals/fixtures/coding-v1/fixture-manifest.json'
    manifest = load_fixture_manifest(fixture_path)
    blockers: list[str] = []
    if manifest['source_executable_sha256'] != hashlib.sha256(
        (ROOT / 'evals/datasets/eval-v1/executable-v1.jsonl').read_bytes()
    ).hexdigest():
        blockers.append('source executable hash drift')
    if manifest.get('bundle_sha256') != _fixture_bundle_hash(ROOT):
        blockers.append('fixture bundle hash drift')
    for task_id, task in manifest['tasks'].items():
        for field in ('fixture_path', 'test_path'):
            path = ROOT / 'evals/fixtures/coding-v1' / task[field]
            if not path.is_file():
                blockers.append(f'{task_id}: missing {field}')
    if spec['host_execution_allowed'] is not False or spec['oci_worker_required'] is not True:
        blockers.append('unsafe execution policy')
    runtime_status = {
        key: value.get('status') for key, value in manifest['runtimes'].items()
    }
    live_ready = all(value.get('digest') for value in manifest['runtimes'].values())
    return {
        'spec_version': spec['spec_version'],
        'spec_status': spec['status'],
        'task_count': len(manifest['tasks']),
        'fixture_files_verified': not any('missing' in blocker for blocker in blockers),
        'runtime_status': runtime_status,
        'live_ready': live_ready,
        'blockers': blockers,
        'audit_pass': not blockers,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Audit Coding v1')
    parser.add_argument('--json', action='store_true')
    args = parser.parse_args(argv)
    try:
        result = audit()
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f'CODING AUDIT FAILED: {exc}', flush=True)
        return 2
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(f"coding-v1: {result['task_count']} tasks, audit={'PASS' if result['audit_pass'] else 'FAIL'}, live_ready={result['live_ready']}")
    return 0 if result['audit_pass'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
