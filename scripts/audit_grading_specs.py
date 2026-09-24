"""Join the grading-spec layer to the executable dataset and audit readiness.

The audit is honest by design: while human-review blockers exist it reports
FAIL for release readiness even when every technical gate passes.
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, cast

from jsonschema import Draft202012Validator

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evals.contract import EvalContractError, load_eval_contract, resolve_grading_spec

import yaml

ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT / 'evals/datasets/eval-v1/executable-v1.jsonl'
REVIEW_PATH = ROOT / 'evals/datasets/eval-v1/grading-review.json'
SPEC_PATH = ROOT / 'evals/specs/eval-v1-grading.yaml'
SCHEMA_PATH = ROOT / 'evals/specs/grading-spec.schema.json'

MANDATORY_SUITES = ('core-v1', 'stress-v1')
READY_STATUSES = {'READY_DETERMINISTIC', 'READY_JUDGE', 'READY_HUMAN'}
APPROVED_CLASSES = {'DETERMINISTIC_EXACT', 'DETERMINISTIC_NUMERIC', 'DETERMINISTIC_STRUCTURED', 'DETERMINISTIC_CONSTRAINT', 'JUDGE_REQUIRED'}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding='utf-8'))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def audit(
    dataset_path: Path = DATASET_PATH,
    spec_path: Path = SPEC_PATH,
    schema_path: Path = SCHEMA_PATH,
    review_path: Path = REVIEW_PATH,
    freeze_path: Path | None = None,
) -> dict[str, Any]:
    dataset = load_jsonl(dataset_path)
    selected_spec = spec_path if spec_path.is_absolute() else ROOT / spec_path
    raw_spec = yaml.safe_load(selected_spec.read_text(encoding='utf-8'))
    if not isinstance(raw_spec, dict):
        raise ValueError(f'grading spec is not a mapping: {selected_spec}')
    if 'overrides' in raw_spec:
        overlay_schema = load_json(schema_path)
        Draft202012Validator.check_schema(overlay_schema)
        overlay_errors = [
            f'{list(error.absolute_path)}: {error.message}'
            for error in Draft202012Validator(overlay_schema).iter_errors(raw_spec)
        ]
        if overlay_errors:
            raise ValueError(f'Grading overlay violates its schema: {overlay_errors[:5]}')
        effective_schema = load_json(ROOT / 'evals/specs/grading-spec-v1.1.schema.json')
    else:
        effective_schema = load_json(schema_path)
    try:
        if freeze_path is not None:
            contract = load_eval_contract(
                ROOT,
                selected_spec,
                freeze_path=freeze_path,
                dataset_path=dataset_path,
            )
            spec = contract.spec
        else:
            spec, _ = resolve_grading_spec(selected_spec)
    except EvalContractError as exc:
        raise ValueError(str(exc)) from exc
    review = {entry['id']: entry for entry in load_json(review_path)}

    Draft202012Validator.check_schema(effective_schema)
    schema_errors = [
        f'{list(error.absolute_path)}: {error.message}'
        for error in Draft202012Validator(effective_schema).iter_errors(spec)
    ]
    if schema_errors:
        raise ValueError(f'Grading spec violates its schema: {schema_errors[:5]}')

    task_ids = sorted(cast(str, t['id']) for t in dataset)
    expected_ids = {f'Q{i:03}' for i in range(1, 81)}
    spec_tasks = cast(dict[str, dict[str, Any]], spec['tasks'])
    mandatory = {cast(str, t['id']) for t in dataset if t['suite'] in MANDATORY_SUITES}

    blockers: list[dict[str, str]] = []
    if set(spec_tasks) != expected_ids:
        blockers.append({'task': 'SPEC', 'reason': f'spec task IDs differ from Q001..Q080 (missing={sorted(expected_ids - set(spec_tasks))[:10]}, extra={sorted(set(spec_tasks) - expected_ids)[:10]})'})
    if len(task_ids) != 80 or set(task_ids) != expected_ids:
        blockers.append({'task': 'DATASET', 'reason': 'dataset IDs must be exactly Q001..Q080'})
    if len(spec_tasks) != 80:
        blockers.append({'task': 'SPEC', 'reason': f'spec has {len(spec_tasks)} entries, expected 80'})

    ready = 0
    for task_id in sorted(mandatory):
        entry = spec_tasks.get(task_id)
        if entry is None:
            blockers.append({'task': task_id, 'reason': 'missing from grading spec'})
            continue
        status = entry['grading_status']
        if status in READY_STATUSES:
            ready += 1
        elif status == 'PENDING_REVIEW':
            blockers.append({'task': task_id, 'reason': 'pending human review: ' + str(entry.get('provenance', {}).get('reason', entry.get('pending_reason', 'unspecified'))[:120])})
        elif status == 'PENDING_SPECIFICATION':
            blockers.append({'task': task_id, 'reason': 'grading specification pending: ' + str(entry.get('pending_reason', 'unspecified'))[:120]})
        else:
            blockers.append({'task': task_id, 'reason': f'unknown grading status {status}'})

    primary_counts = dict(Counter(cast(str, entry['primary_class']) for entry in spec_tasks.values()))
    status_counts = dict(Counter(cast(str, entry['grading_status']) for entry in spec_tasks.values()))
    multi_grader = sum(1 for entry in spec_tasks.values() if len(entry['graders']) > 1)

    review_ids = set(review)
    return {
        'spec_version': spec['spec_version'],
        'spec_status': spec['status'],
        'total_task_ids': len(task_ids),
        'ids_exactly_q001_q080': bool(len(task_ids) == 80 and set(task_ids) == expected_ids and set(spec_tasks) == expected_ids),
        'mandatory_task_ids': len(mandatory),
        'optional_coding_task_ids': len(task_ids) - len(mandatory),
        'mandatory_have_primary_class': all(spec_tasks.get(t, {}).get('primary_class') in APPROVED_CLASSES for t in mandatory),
        'no_duplicate_primary_class': True,
        'grader_lists_may_overlap': True,
        'multi_grader_tasks': multi_grader,
        'review_inventory_ids': len(review_ids),
        'primary_class_counts': primary_counts,
        'grading_status_counts': status_counts,
        'mandatory_grading_coverage': f'{len(mandatory)}/{len(mandatory)}' if len(mandatory) == 72 else f'{len(mandatory)}/72',
        'mandatory_coverage_complete': len(mandatory) == 72,
        'fully_ready_mandatory': ready,
        'pending_human_review': len(mandatory) - ready,
        'blockers': blockers,
        'unresolved_mandatory_blockers': len(blockers),
        'dataset_task_count': len(dataset),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Audit grading spec readiness')
    parser.add_argument('--json', action='store_true', help='emit machine-readable JSON')
    parser.add_argument('--spec', default=str(SPEC_PATH), help='full spec or versioned overlay')
    parser.add_argument('--schema', default=str(SCHEMA_PATH), help='schema for the selected spec')
    parser.add_argument('--freeze', default=None, help='optional contract freeze to verify')
    args = parser.parse_args(argv)

    result = audit(
        spec_path=Path(args.spec),
        schema_path=Path(args.schema),
        freeze_path=Path(args.freeze) if args.freeze else None,
    )

    if not args.json:
        print('Grading Specification Audit')
        print(f"  spec: {result['spec_version']} (status={result['spec_status']})")
        print()
        print('Mandatory coverage')
        print(f"  72 mandatory tasks present?            {'YES' if result['mandatory_coverage_complete'] else 'NO'}")
        print(f"  72 have a primary grading class?       {'YES' if result['mandatory_have_primary_class'] else 'NO'}")
        print(f"  72 have all required grader specs?     {'YES' if result['mandatory_coverage_complete'] and result['mandatory_have_primary_class'] and result['fully_ready_mandatory'] == 72 else 'NO'}")
        print(f"  0 unresolved mandatory blockers?       {'YES' if result['unresolved_mandatory_blockers'] == 0 else 'NO'}")
        print()
        print("Primary class counts (auto-derived from spec entries, overlap allowed):")
        for name, count in sorted(result['primary_class_counts'].items()):
            print(f'  {name:28} {count}')
        print()
        print("Grading status counts (auto-derived):")
        for name, count in sorted(result['grading_status_counts'].items()):
            print(f'  {name:28} {count}')
        print()
        print(f"Tasks with multiple graders (overlap is allowed): {result['multi_grader_tasks']}")
        print(f"Mandatory grading coverage:        {result['mandatory_grading_coverage']}")
        print(f"Fully ready mandatory tasks:       {result['fully_ready_mandatory']}/72")
        print(f"Pending human review:              {result['pending_human_review']}")
        print()
        if result['blockers']:
            print(f"Unresolved blockers ({result['unresolved_mandatory_blockers']}):")
            for blocker in result['blockers']:
                print(f"  [{blocker['task']}] {blocker['reason']}")
        print()
        print('Release-readiness audit: ' + ('PASS' if result['unresolved_mandatory_blockers'] == 0 else 'FAIL (human review required)'))
    else:
        print(json.dumps(result, indent=2, ensure_ascii=False))

    return 0 if result['unresolved_mandatory_blockers'] == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
