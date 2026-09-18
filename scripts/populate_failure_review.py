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


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    conn = connect(str(root / 'results/local/full-baseline-v1.db'))
    statuses = load_spec_statuses(str(root / 'evals/specs/eval-v1-grading.yaml'))
    rows = extract_failed_rows(conn, 'full-baseline-v1', statuses)
    conn.close()
    grouped: dict[str, list[FailedRowFeatures]] = defaultdict(list)
    for row in rows:
        grouped[row.task_id].append(row)
    det = sum(1 for r in rows if r.task_status == 'READY_DETERMINISTIC')
    judge = sum(1 for r in rows if r.task_status == 'READY_JUDGE')
    print(f'failed rows: {len(rows)} (deterministic={det}, judge={judge})')
    print(f'tasks: {len(grouped)}')
    tasks = []
    for task_id in sorted(grouped):
        task_rows = grouped[task_id]
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
                        {'trial': r.trial, 'excerpt': r.excerpt}
                        for r in task_rows
                    ],
                },
                'review': {
                    'substantive_answer_assessment': None,
                    'assessment_author': None,
                    'mode_override': None,
                    'override_reason': None,
                    'notes': None,
                },
            }
        )
    review = {
        'artifact': 'qwen3-4b-q4 failure review (post-hoc diagnostic)',
        'execution_id': 'full-baseline-v1',
        'taxonomy_version': TAXONOMY_VERSION,
        'review_status': 'DRAFT',
        'populations': {
            'ready_deterministic_fail_rows': det,
            'ready_judge_precheck_fail_rows': judge,
            'total_fail_rows': len(rows),
        },
        'tasks': tasks,
    }
    out = root / 'analysis/reviews/qwen-q4-failure-review.yaml'
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
