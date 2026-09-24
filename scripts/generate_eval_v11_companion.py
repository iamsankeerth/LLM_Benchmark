"""Generate corrected Eval-v1.1 companion summaries from the regrade overlay."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evals.contract import load_eval_contract


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _pareto_ids(rows: list[dict[str, Any]], *, speed: bool) -> list[str]:
    ids: list[str] = []
    for row in rows:
        accuracy = float(row['deterministic_trial_accuracy'])
        resource = float(row['decode_median_tok_s'] if speed else row['peak_vram_mib'])
        dominated = False
        for other in rows:
            if other is row:
                continue
            other_accuracy = float(other['deterministic_trial_accuracy'])
            other_resource = float(
                other['decode_median_tok_s'] if speed else other['peak_vram_mib']
            )
            if speed:
                better_or_equal = (
                    other_accuracy >= accuracy and other_resource >= resource
                )
                strictly_better = (
                    other_accuracy > accuracy or other_resource > resource
                )
            else:
                better_or_equal = (
                    other_accuracy >= accuracy and other_resource <= resource
                )
                strictly_better = (
                    other_accuracy > accuracy or other_resource < resource
                )
            if better_or_equal and strictly_better:
                dominated = True
                break
        if not dominated:
            ids.append(str(row['model_config_id']))
    return sorted(ids)


def build_companion_report(
    *,
    root: Path,
    overlay_path: Path,
    historical_report_path: Path,
) -> dict[str, Any]:
    contract = load_eval_contract(
        root,
        'evals/specs/eval-v1.1-grading.yaml',
        freeze_path='evals/specs/eval-v1.1-grading.freeze.json',
    )
    overlay = json.loads(overlay_path.read_text(encoding='utf-8'))
    historical = json.loads(historical_report_path.read_text(encoding='utf-8'))
    old_rows = {
        str(row['model_config_id']): row
        for row in historical['registry_table_14_rows']
        if row.get('eligibility_status') == 'ELIGIBLE_GPU'
    }
    statuses = contract.statuses
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in overlay['records']:
        grouped.setdefault(str(record['execution_id']), []).append(record)
    summaries: list[dict[str, Any]] = []
    for model_id, old_row in sorted(old_rows.items()):
        execution_id = f'full-baseline-v2__{model_id}'
        records = grouped.get(execution_id, [])
        by_task: dict[str, list[dict[str, Any]]] = {}
        for record in records:
            by_task.setdefault(str(record['task_id']), []).append(record)
        deterministic = [
            task_id for task_id, status in statuses.items()
            if status == 'READY_DETERMINISTIC' and task_id in by_task
        ]
        judge = [
            task_id for task_id, status in statuses.items()
            if status == 'READY_JUDGE' and task_id in by_task
        ]
        passes = sum(
            1 for task_id in deterministic for record in by_task[task_id]
            if record['new_verdict'] == 'PASS'
        )
        trials = sum(len(by_task[task_id]) for task_id in deterministic)
        any_pass = sum(
            1 for task_id in deterministic
            if any(record['new_verdict'] == 'PASS' for record in by_task[task_id])
        )
        all_pass = sum(
            1 for task_id in deterministic
            if all(record['new_verdict'] == 'PASS' for record in by_task[task_id])
        )
        judge_prechecks = sum(
            1 for task_id in judge for record in by_task[task_id]
            if record['new_verdict'] != 'NEEDS_JUDGE'
        )
        summaries.append({
            'model_config_id': model_id,
            'family': old_row['family'],
            'quantization': old_row['quantization'],
            'deterministic_tasks': len(deterministic),
            'deterministic_trials': trials,
            'passes': passes,
            'deterministic_trial_accuracy': passes / trials if trials else None,
            'tasks_any_pass_3': any_pass,
            'tasks_all_pass_3': all_pass,
            'judge_tasks': len(judge),
            'judge_precheck_non_deferrals': judge_prechecks,
            'decode_median_tok_s': old_row['decode_median_tok_s'],
            'decode_p95_tok_s': old_row['decode_p95_tok_s'],
            'ttft_median_ms': old_row['ttft_median_ms'],
            'ttft_p95_ms': old_row['ttft_p95_ms'],
            'peak_vram_mib': old_row['peak_vram_mib'],
            'model_size_bytes': old_row['model_size_bytes'],
        })
    return {
        'title': 'Eval-v1.1 corrected V2 companion report',
        'historical_source': str(historical_report_path.relative_to(root)).replace('\\', '/'),
        'provenance': {
            'grading_spec_version': contract.spec_version,
            'grading_spec_hash': contract.hashes['spec_sha256'],
            'dataset_hash': contract.hashes['dataset_sha256'],
            'grading_freeze_hash': contract.hashes['freeze_sha256'],
            'regrade_overlay_hash': _file_hash(overlay_path),
            'historical_report_hash': _file_hash(historical_report_path),
            'model_generation_calls': 0,
        },
        'population': {
            'mandatory_tasks': 72,
            'deterministic_tasks': 67,
            'judge_tasks': 5,
            'trials_per_task': 3,
            'deterministic_trials_per_model': 201,
            'judge_trials_per_model': 15,
        },
        'models': summaries,
        'pareto': {
            'speed_accuracy': _pareto_ids(summaries, speed=True),
            'vram_accuracy': _pareto_ids(summaries, speed=False),
        },
        'interpretation': 'Corrected grading changes denominators and Q016 outcomes; this is not a new model generation run.',
    }


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        '# Eval-v1.1 Corrected V2 Companion Report',
        '',
        'Corrected grading-only view of the sealed V2 outputs. No model generation calls were made.',
        '',
        '| Model | Deterministic passes | Accuracy | Any-pass | All-pass | Judge tasks |',
        '|---|---:|---:|---:|---:|---:|',
    ]
    for row in report['models']:
        accuracy = row['deterministic_trial_accuracy']
        lines.append(
            f"| {row['model_config_id']} | {row['passes']}/{row['deterministic_trials']} | "
            f"{accuracy:.3f} | {row['tasks_any_pass_3']}/{row['deterministic_tasks']} | "
            f"{row['tasks_all_pass_3']}/{row['deterministic_tasks']} | {row['judge_tasks']} |"
        )
    lines += [
        '',
        f"Speed/accuracy Pareto: {', '.join(report['pareto']['speed_accuracy'])}",
        f"VRAM/accuracy Pareto: {', '.join(report['pareto']['vram_accuracy'])}",
        '',
        'Historical Eval-v1 reports remain unchanged. This companion is a separate Eval-v1.1 artifact.',
    ]
    return '\n'.join(lines) + '\n'


def main(argv: list[str] | None = None) -> int:
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description='Generate Eval-v1.1 companion report')
    parser.add_argument('--overlay', default='results/reports/eval-v1.1-regrade-overlay.json')
    parser.add_argument('--historical', default='results/reports/sweep-v2-report.json')
    parser.add_argument('--out-json', default='results/reports/eval-v1.1-companion-report.json')
    parser.add_argument('--out-md', default='results/reports/eval-v1.1-companion-report.md')
    args = parser.parse_args(argv)
    report = build_companion_report(
        root=root,
        overlay_path=root / args.overlay,
        historical_report_path=root / args.historical,
    )
    for output, content in (
        (root / args.out_json, json.dumps(report, indent=2, sort_keys=True) + '\n'),
        (root / args.out_md, _markdown(report)),
    ):
        if output.exists():
            print(f'REPORT REFUSED: refusing to overwrite {output}', flush=True)
            return 2
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(content, encoding='utf-8')
    print(f"companion report: {len(report['models'])} eligible models")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
