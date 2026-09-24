"""Generate the long-context-v1 model report from persisted rows."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis.long_context import load_long_context_tasks, summarize_long_context


class LongContextReportError(ValueError):
    """Persisted long-context evidence is incomplete or contaminated."""


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_report(root: Path, db_path: Path, model_id: str) -> dict[str, Any]:
    tasks = load_long_context_tasks(root)
    execution_id = f'long-context-v1__{model_id}'
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        warmups = conn.execute(
            'SELECT COUNT(*) FROM runs WHERE experiment_id=? AND run_kind=\'WARMUP\'',
            (execution_id,),
        ).fetchone()[0]
        rows = conn.execute(
            'SELECT task_id, trial, status, grader_verdict, raw_output,'
            ' temperature, model_digest FROM runs WHERE experiment_id=?'
            ' AND is_warmup=0 ORDER BY task_id, trial',
            (execution_id,),
        ).fetchall()
    finally:
        conn.close()
    expected = {(task.task_id, trial) for task in tasks for trial in (1, 2, 3)}
    observed = {(str(row['task_id']), int(row['trial'])) for row in rows}
    if warmups != 2 or len(rows) != 63 or observed != expected:
        raise LongContextReportError(
            f'{execution_id}: expected 2 warmups and 63 measured rows, got '
            f'{warmups}/{len(rows)}'
        )
    if any(row['status'] != 'COMPLETE' or row['grader_verdict'] == 'ERROR' for row in rows):
        raise LongContextReportError(f'{execution_id}: error rows present')
    result = summarize_long_context([dict(row) for row in rows], tasks)
    result.update({
        'schema_version': 'long-context-report-v1',
        'study': 'long-context-v1',
        'execution_id': execution_id,
        'model_config_id': model_id,
        'dataset_sha256': _hash(root / 'evals/datasets/long-context-v1/tasks.jsonl'),
        'grading_spec_sha256': _hash(root / 'evals/specs/long-context-v1-grading.yaml'),
        'request_accounting': {
            'canonical_probe': 1,
            'task_preflight_probes': len(tasks),
            'warmups': 2,
            'measured': len(tasks) * 3,
            'total': 1 + len(tasks) + 2 + len(tasks) * 3,
        },
        'claim_scope': 'configured 4096-token window; no native 8K/32K claim',
    })
    return result


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        '# Long-Context V1 Report', '',
        report['claim_scope'], '',
        '| Length | Passes | Trials | Accuracy |', '|---|---:|---:|---:|',
    ]
    for length, row in report['by_length'].items():
        lines.append(f"| {length} | {row['passes']} | {row['trials']} | {row['accuracy']:.3f} |")
    lines += ['', '| Archetype | Passes | Trials | Accuracy |', '|---|---:|---:|---:|']
    for archetype, row in report['by_archetype'].items():
        lines.append(f"| {archetype} | {row['passes']} | {row['trials']} | {row['accuracy']:.3f} |")
    return '\n'.join(lines) + '\n'


def main(argv: list[str] | None = None) -> int:
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description='Generate long-context-v1 report')
    parser.add_argument('--db', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--out-json', required=True)
    parser.add_argument('--out-md', required=True)
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args(argv)
    try:
        report = build_report(root, Path(args.db), args.model)
    except (OSError, sqlite3.Error, LongContextReportError, ValueError) as exc:
        print(f'LONG-CONTEXT REPORT REFUSED: {exc}', flush=True)
        return 2
    for path, content in (
        (root / args.out_json, json.dumps(report, indent=2, sort_keys=True) + '\n'),
        (root / args.out_md, _markdown(report)),
    ):
        if path.exists() and not args.force:
            print(f'LONG-CONTEXT REPORT REFUSED: exists {path}', flush=True)
            return 2
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding='utf-8')
    print(f"long-context report: {report['trial_count']} trials")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
