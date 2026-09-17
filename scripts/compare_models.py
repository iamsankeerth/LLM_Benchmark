"""Paired comparison between two executions of one frozen contract."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml

from analysis.comparison import (
    compare_executions,
    comparison_to_console,
    comparison_to_json,
    load_comparison_trials,
    load_execution_evidence,
)
from storage.db import connect


def load_task_meta(root: Path) -> dict[str, dict[str, str]]:
    spec = yaml.safe_load(
        open(root / 'evals/specs/eval-v1-grading.yaml', encoding='utf-8')
    )
    meta: dict[str, dict[str, str]] = {
        str(tid): {
            'grading_status': str(entry['grading_status']),
            'primary_class': str(entry['primary_class']),
        }
        for tid, entry in spec['tasks'].items()
    }
    for line in open(
        root / 'evals/datasets/eval-v1/executable-v1.jsonl', encoding='utf-8'
    ):
        row = json.loads(line)
        task_id = str(row['id'])
        if task_id in meta:
            meta[task_id]['suite'] = str(row['suite'])
            meta[task_id]['difficulty'] = str(row['difficulty'])
    return meta


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Compare two model executions')
    parser.add_argument('--db-base', required=True)
    parser.add_argument('--db-candidate', required=True)
    parser.add_argument('--base', required=True, help='base execution id')
    parser.add_argument('--candidate', required=True, help='candidate execution id')
    parser.add_argument('--spec', default='full-baseline-v1')
    parser.add_argument('--out', default=None)
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parent.parent
    meta = load_task_meta(root)
    base_conn = connect(args.db_base)
    base_grouped = load_comparison_trials(base_conn, args.base, meta)
    base_size, base_digest, base_peaks = load_execution_evidence(base_conn, args.base)
    base_conn.close()
    cand_conn = connect(args.db_candidate)
    cand_grouped = load_comparison_trials(cand_conn, args.candidate, meta)
    cand_size, cand_digest, cand_peaks = load_execution_evidence(
        cand_conn, args.candidate
    )
    cand_conn.close()
    if not base_grouped or not cand_grouped:
        print('missing measured rows on one side; refusing to compare')
        return 2
    report = compare_executions(
        experiment_spec_id=args.spec,
        base_execution_id=args.base,
        candidate_execution_id=args.candidate,
        base_grouped=base_grouped,
        cand_grouped=cand_grouped,
        base_size_bytes=base_size,
        cand_size_bytes=cand_size,
        base_artifact_digest=base_digest,
        cand_artifact_digest=cand_digest,
        base_vram_peaks=base_peaks,
        cand_vram_peaks=cand_peaks,
    )
    out_path = (
        Path(args.out)
        if args.out
        else root / 'results/summaries' / f'{args.base}-vs-{args.candidate}.json'
    )
    out_path.write_text(comparison_to_json(report) + '\n', encoding='utf-8')
    print(comparison_to_console(report))
    print(f'report: {out_path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
