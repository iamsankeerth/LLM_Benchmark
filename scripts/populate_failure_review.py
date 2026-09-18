"""One-shot scaffold: populate the failure-review file's objective half.

Writes auto_analysis (DB verdicts, grader violations, signatures, auto
mode, trial counts, excerpts) for every persisted FAIL row. The review
section is scaffolded with nulls: substantive assessment is human-authored
and must never be auto-filled.
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml

from analysis.failure_modes import (
    TAXONOMY_VERSION,
    FailedRowFeatures,
    extract_failed_rows,
    validate_review_file,
)
from analysis.reliability import load_spec_statuses
from storage.db import connect


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description='Scaffold a failure review')
    parser.add_argument('--execution-id', default='full-baseline-v1')
    parser.add_argument('--db', default=None)
    parser.add_argument('--out', default=None)
    parser.add_argument('--label', default=None)
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parent.parent
    db_path = args.db or str(root / f'results/local/{args.execution_id}.db')
    conn = connect(db_path)
    statuses = load_spec_statuses(str(root / 'evals/specs/eval-v1-grading.yaml'))
    spec = yaml.safe_load(
        open(root / 'evals/specs/eval-v1-grading.yaml', encoding='utf-8')
    )
    graders_map = {
        str(tid): [dict(g) for g in (entry.get('graders') or [])]
        for tid, entry in spec['tasks'].items()
    }
    rows = extract_failed_rows(conn, args.execution_id, statuses, graders_map=graders_map)
    conn.close()
    grouped: dict[str, list[FailedRowFeatures]] = defaultdict(list)
    for row in rows:
        grouped[row.task_id].append(row)
    det = sum(1 for r in rows if r.task_status == 'READY_DETERMINISTIC')
    judge = sum(1 for r in rows if r.task_status == 'READY_JUDGE')
    print(f'failed rows: {len(rows)} (deterministic={det}, judge={judge})')
    print(f'tasks: {len(grouped)}')
    out = (
        Path(args.out)
        if args.out
        else root / 'analysis/reviews/qwen-q4-failure-review.yaml'
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    # Preserve human-authored review sections across re-scaffolds: only
    # auto_analysis regenerates. Authorship boundary holds by construction.
    prior_reviews: dict[str, object] = {}
    prior_status = 'DRAFT'
    if out.exists():
        prior_doc = yaml.safe_load(out.read_text(encoding='utf-8'))
        if isinstance(prior_doc, dict):
            prior_status = str(prior_doc.get('review_status', 'DRAFT'))
            for task in prior_doc.get('tasks') or []:
                if isinstance(task, dict) and 'task_id' in task:
                    prior_reviews[str(task['task_id'])] = task.get('review')
    tasks = []
    for task_id in sorted(grouped):
        task_rows = grouped[task_id]
        review = prior_reviews.get(task_id) or {
            'substantive_answer_assessment': None,
            'assessment_author': None,
            'mode_override': None,
            'override_reason': None,
            'notes': None,
        }
        tasks.append(
            {
                'task_id': task_id,
                'task_status': task_rows[0].task_status,
                'failed_trials': [r.trial for r in task_rows],
                'auto_analysis': {
                    'failed_rows': len(task_rows),
                    'benchmark_verdicts': [r.benchmark_verdict for r in task_rows],
                    'row_modes': [r.failure_mode for r in task_rows],
                    'signatures': sorted(
                        {s for r in task_rows for s in r.signatures}
                    ),
                    'excerpts': [
                        {
                            'trial': r.trial,
                            'excerpt': r.excerpt,
                            'row_mode': r.failure_mode,
                            'asserted_value_status': r.asserted_value_status,
                            'generation_flags': list(r.generation_flags),
                        }
                        for r in task_rows
                    ],
                    'generation_flags': sorted(
                        {f for r in task_rows for f in r.generation_flags}
                    ),
                },
                'review': review,
            }
        )
    review = {
        'artifact': (args.label or 'qwen3-4b-q4 failure review (post-hoc diagnostic)'),
        'execution_id': args.execution_id,
        'taxonomy_version': TAXONOMY_VERSION,
        'review_status': prior_status,
        'populations': {
            'ready_deterministic_fail_rows': det,
            'ready_judge_precheck_fail_rows': judge,
            'total_fail_rows': len(rows),
        },
        'tasks': tasks,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        yaml.safe_dump(review, allow_unicode=True, sort_keys=False), encoding='utf-8'
    )
    errors = validate_review_file(review)
    # DRAFT with null assessments is expected to fail assessment validation;
    # only structural errors matter here.
    structural = [e for e in errors if 'vocabulary' not in e and 'author' not in e]
    if structural:
        print('STRUCTURAL ERRORS:', structural)
        return 1
    print(f'wrote {out} (DRAFT, assessments pending)')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
