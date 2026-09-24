"""Audit the long-context-v1 synthetic dataset and grading contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml

from evals.graders.engine import words


ROOT = Path(__file__).resolve().parents[1]


def audit() -> dict[str, Any]:
    spec_path = ROOT / 'evals/specs/long-context-v1-grading.yaml'
    schema_path = ROOT / 'evals/specs/long-context-v1.schema.json'
    spec = yaml.safe_load(spec_path.read_text(encoding='utf-8'))
    schema = json.loads(schema_path.read_text(encoding='utf-8'))
    Draft202012Validator.check_schema(schema)
    errors = [f'{list(error.absolute_path)}: {error.message}' for error in Draft202012Validator(schema).iter_errors(spec)]
    if errors:
        raise ValueError(f'long-context spec schema errors: {errors[:5]}')
    dataset_dir = ROOT / 'evals/datasets/long-context-v1'
    manifest = json.loads((dataset_dir / 'dataset-manifest.json').read_text(encoding='utf-8'))
    tasks = [json.loads(line) for line in (dataset_dir / 'tasks.jsonl').read_text(encoding='utf-8').splitlines() if line]
    expected_ids = {f'LC{index:03d}' for index in range(1, 22)}
    blockers: list[str] = []
    if len(tasks) != 21 or {task['id'] for task in tasks} != expected_ids:
        blockers.append('task IDs/count are not exactly LC001..LC021')
    if manifest['tasks_sha256'] != hashlib.sha256((dataset_dir / 'tasks.jsonl').read_bytes()).hexdigest():
        blockers.append('tasks hash mismatch')
    lengths = Counter(task['source_length'] for task in tasks)
    archetypes = Counter(task['archetype'] for task in tasks)
    if lengths != Counter({'S': 7, 'M': 7, 'L': 7}):
        blockers.append(f'length distribution mismatch: {dict(lengths)}')
    if any(count != 3 for count in archetypes.values()) or len(archetypes) != 7:
        blockers.append(f'archetype distribution mismatch: {dict(archetypes)}')
    for task in tasks:
        document_path = dataset_dir / task['document_path']
        if not document_path.is_file():
            blockers.append(f'{task["id"]}: missing document')
            continue
        document_bytes = document_path.read_bytes()
        document = document_bytes.decode('utf-8').rstrip('\n')
        if hashlib.sha256(document_bytes).hexdigest() != task['document_sha256']:
            blockers.append(f'{task["id"]}: document hash mismatch')
        if len(words(document)) != task['context_size']:
            blockers.append(f'{task["id"]}: word count mismatch')
        if document not in task['prompt']:
            blockers.append(f'{task["id"]}: full document not embedded in prompt')
        if task['grading_status'] != 'READY_DETERMINISTIC' or not task['graders']:
            blockers.append(f'{task["id"]}: missing deterministic grader')
    return {
        'spec_version': spec['spec_version'],
        'spec_status': spec['status'],
        'task_count': len(tasks),
        'length_counts': dict(lengths),
        'archetype_counts': dict(archetypes),
        'documents_verified': len(tasks) - sum('document' in blocker for blocker in blockers),
        'blockers': blockers,
        'audit_pass': not blockers,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Audit long-context-v1')
    parser.add_argument('--json', action='store_true')
    args = parser.parse_args(argv)
    try:
        result = audit()
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f'LONG-CONTEXT AUDIT FAILED: {exc}', flush=True)
        return 2
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(f"long-context-v1: {result['task_count']} tasks, audit={'PASS' if result['audit_pass'] else 'FAIL'}")
    return 0 if result['audit_pass'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
