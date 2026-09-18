"""Generate the failure-analysis JSON + Markdown from an APPROVED review."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml

from analysis.failure_modes import TAXONOMY_VERSION, taxonomy_hash
from analysis.failure_report import (
    ReviewNotApprovedError,
    decomposition_shell,
    require_approved,
    table_contract_modes,
    table_judge_prechecks,
    table_substance,
    task_table,
)
from analysis.reliability import git_worktree_status

# Licensed lineage statements (verified 2026-09-18 from persisted DBs:
# 0 verdict diffs across 216 row identities V1<->V2; Q025 outputs differ).
LINEAGE_STATEMENTS = [
    'Across all 216 measured rows, the persisted benchmark verdict was '
    'identical between V1 and V2.',
    'Within the 204 READY_DETERMINISTIC rows, the PASS/FAIL matrix was '
    'identical row-for-row.',
]

THREATS_TO_VALIDITY = [
    'Substring semantics are frozen for eval-v1: required/forbidden term '
    'matching is substring-based (e.g. "plant" matches inside "plants", '
    '"carried forward" does not satisfy "carry", "over two hours" does not '
    'satisfy "longer than two hours"). Human interpretation is token-level; '
    'the grader is character-level. This is intentional and versioned, not '
    'a defect to hotfix — but cross-study comparisons must account for it.',
    'Generation-cap entanglement: Q025 trials hitting num_predict truncation '
    'cannot establish what final the model would have asserted with a larger '
    'budget. Truncation is recorded orthogonally (TRUNCATED_AT_NUM_PREDICT) '
    'and never merged into failure modes.',
    'Strict output contracts (bare numbers, exact tokens, ISO dates) punish '
    'verbosity and paraphrase even when substance is correct. The 52.5% '
    'contract accuracy therefore understates reasoning capability by design; '
    'Table 2 quantifies the gap.',
    'Single-model scope: every substantive label below describes Qwen3-4B '
    'Q4 outputs only. The deterministic mode taxonomy is frozen for reuse; '
    'human substantive labels must never be auto-generated for other models.',
]


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Generate failure report')
    parser.add_argument('--review', required=True)
    parser.add_argument('--out-json', default=None)
    parser.add_argument('--out-md', default=None)
    parser.add_argument('--allow-dirty', action='store_true')
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parent.parent
    review_path = Path(args.review)
    review = yaml.safe_load(review_path.read_text(encoding='utf-8'))
    try:
        require_approved(review)
    except ReviewNotApprovedError as exc:
        print(f'REPORT REFUSED: {exc}', flush=True)
        return 2
    table1 = table_contract_modes(review)
    table2 = table_substance(review)
    table3 = table_judge_prechecks(review)
    tasks = task_table(review)
    commit, dirty = git_worktree_status(str(root))
    if dirty and not args.allow_dirty:
        print('REPORT REFUSED: dirty worktree (use --allow-dirty for dev only)',
              flush=True)
        return 3
    freeze = json.loads(
        (root / 'evals/specs/eval-v1-grading.freeze.json').read_text(encoding='utf-8')
    )
    provenance = {
        'grading_spec_hash': freeze['artifacts']['eval-v1-grading.yaml'],
        'dataset_hash': freeze['artifacts']['executable-v1.jsonl'],
        'taxonomy_version': TAXONOMY_VERSION,
        'taxonomy_hash': taxonomy_hash(),
        'review_file': str(review_path),
        'review_file_hash': _file_hash(review_path),
        'review_status': review.get('review_status'),
        'reviewed_by': review.get('reviewed_by'),
        'reviewed_on': review.get('reviewed_on'),
        'analysis_code_git_commit': commit,
        'analysis_worktree_dirty': dirty,
    }
    document = {
        'title': 'Why Qwen3-4B Q4 Fails the Frozen Eval Contract',
        'diagnostic_only': True,
        'contract_accuracy_note': '52.5% benchmark result untouched by this artifact',
        'table_contract_modes_97_rows': table1,
        'table_substance_97_rows': table2,
        'table_judge_prechecks_6_rows': table3,
        'task_table': tasks,
        'lineage_v1_v2_statements': LINEAGE_STATEMENTS,
        'threats_to_validity': THREATS_TO_VALIDITY,
        'decomposition_shell': decomposition_shell(table1),
        'taxonomy_disposition': 'failure-modes-v1: CONFIRMED_UNCHANGED',
        'provenance': provenance,
    }
    out_json = Path(args.out_json) if args.out_json else (
        root / 'results/reports/qwen3-4b-q4-failure-analysis.json'
    )
    out_json.write_text(json.dumps(document, indent=2) + '\n', encoding='utf-8')

    lines = [
        '# Why Qwen3-4B Q4 Fails the Frozen Eval Contract',
        '',
        '> Post-hoc diagnostic artifact. The 52.5% benchmark result is '
        'untouched; categories below never mutate verdicts, graders, or specs.',
        '',
        'Central finding: Qwen Q4 frequently reaches the correct semantic '
        'answer but fails the required output contract.',
        '',
        '## Table 1 — contract decomposition (97 deterministic FAIL rows)',
        '',
    ]
    for mode in sorted(table1['counts']):
        count = table1['counts'][mode]
        pct = table1['percent'][mode]
        lines.append(f'- {mode}: {count} rows ({pct:.1%})')
    lines += [
        '',
        '## Table 2 — substantive assessment (97 rows inherit task labels)',
        '',
    ]
    for label in sorted(table2['counts']):
        count = table2['counts'][label]
        pct = table2['percent'][label]
        lines.append(f'- {label}: {count} rows ({pct:.1%})')
    lines += [
        '',
        '## Table 3 — judge precheck failures (6 rows, descriptive only)',
        '',
    ]
    for task in table3['tasks']:
        lines.append(
            f"- {task['task_id']}: substantive={task['substantive']}, "
            f"trials={task['failed_trials']}, "
            f"deferrals tracked in capability summary"
        )
    lines += [
        '',
        '## Per-task failure summaries',
        '',
    ]
    for task in tasks:
        lines.append(
            f"### {task['task_id']} [{task['task_status']}] — "
            f"{task['substantive']}"
        )
        lines.append(
            f'- trials: {task["failed_trials"]} '
            f'modes: {task["mode_distribution"]} '
            f'multi_mode: {task["multi_mode_across_trials"]} '
            f'flags: {task["generation_flags"]}'
        )
        if task['notes']:
            lines.append(f'- notes: {task["notes"]}')
        lines.append('')
    lines += [
        '## Lineage context (V1 512-token vs V2 2048-token)',
        '',
    ]
    lines += [f'- {statement}' for statement in LINEAGE_STATEMENTS]
    lines += [
        '- 34/35 failed tasks byte-identical across lineages; Q025 diverged '
        '(completed wrong final + mid-derivation assertions into truncation).',
        '',
        '## Threats to validity',
        '',
    ]
    lines += [f'- {threat}' for threat in THREATS_TO_VALIDITY]
    lines += [
        '',
        '## Cross-model decomposition shell',
        '',
        'Qwen Q4 column live under frozen failure-modes-v1; Q5+ reserved.',
        '',
        '## Provenance',
        '',
        '```json',
        json.dumps(provenance, indent=2),
        '```',
        '',
    ]
    out_md = Path(args.out_md) if args.out_md else (
        root / 'results/reports/qwen3-4b-q4-failure-analysis.md'
    )
    out_md.write_text('\n'.join(lines), encoding='utf-8')
    print(f'wrote {out_json}', flush=True)
    print(f'wrote {out_md}', flush=True)
    print(f"Table 1 denominator: {table1['denominator_rows']} rows", flush=True)
    print(f"Table 2 denominator: {table2['denominator_rows']} rows", flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
