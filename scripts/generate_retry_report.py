"""Generate retry-rescue-v1 report (canonical analysis artifact)."""

from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

REPORT_PATH = Path('results/reports/retry-rescue-v1.json')
MD_PATH = Path('results/reports/retry-rescue-v1.md')

V2_SEALED_COMMIT = '1d63305fadd416387f218ad8ba868e85a1345382'


def main() -> int:
    root = Path('.')
    spec = yaml.safe_load(open(root / 'evals/specs/eval-v1-grading.yaml', encoding='utf-8'))
    retry_db = str(root / 'results/local/retry-rescue-v1__qwen3-4b-q4.db')
    baseline_db = str(root / 'results/local/full-baseline-v2__qwen3-4b-q4.db')
    # Derive populations from DB row modes (with asserted-value)
    from analysis.failure_modes import extract_failed_rows
    from analysis.reliability import load_spec_statuses

    statuses = load_spec_statuses(str(root / 'evals/specs/eval-v1-grading.yaml'))
    graders_map = {tid: entry.get('graders', []) for tid, entry in spec['tasks'].items()}
    from storage.db import connect

    conn = connect(baseline_db)
    rows = extract_failed_rows(conn, 'full-baseline-v2__qwen3-4b-q4', statuses, graders_map=graders_map)
    conn.close()
    primary_ids = {(r.task_id, r.trial) for r in rows if r.failure_mode == 'OUTPUT_CONTRACT'}
    control_ids = {(r.task_id, r.trial) for r in rows if r.failure_mode == 'MIXED'}
    assert len(primary_ids) == 39, len(primary_ids)
    assert len(control_ids) == 12, len(control_ids)

    # Load retry DB
    conn = sqlite3.connect(retry_db)
    conn.row_factory = sqlite3.Row
    retry_rows = conn.execute('SELECT task_id, trial, grader_verdict, raw_output, prompt FROM runs WHERE experiment_id=? ORDER BY task_id, trial', ('retry-rescue-v1__qwen3-4b-q4',)).fetchall()
    conn.close()
    assert len(retry_rows) == 51, len(retry_rows)
    primary_retry = [r for r in retry_rows if (r['task_id'], r['trial']) in primary_ids]
    control_retry = [r for r in retry_rows if (r['task_id'], r['trial']) in control_ids]
    assert len(primary_retry) == 39
    assert len(control_retry) == 12
    primary_recovered = sum(1 for r in primary_retry if r['grader_verdict'] == 'PASS')
    control_recovered = sum(1 for r in control_retry if r['grader_verdict'] == 'PASS')
    # Content-change classification for primary recovered
    baseline_map = {}
    conn = sqlite3.connect(baseline_db)
    for tid, trial, raw in conn.execute('SELECT task_id, trial, raw_output FROM runs WHERE experiment_id=? AND is_warmup=0', ('full-baseline-v2__qwen3-4b-q4',)):
        baseline_map[(tid, trial)] = raw
    conn.close()
    from analysis.retry_report import classify_content_change

    contract_only = 0
    with_change = 0
    unknown = 0
    for r in primary_retry:
        if r['grader_verdict'] != 'PASS':
            continue
        base = baseline_map.get((r['task_id'], r['trial']), '')
        gtype = spec['tasks'][r['task_id']]['graders'][0].get('type', 'exact') if spec['tasks'][r['task_id']].get('graders') else 'exact'
        bucket = classify_content_change(base, r['raw_output'], gtype)
        if bucket == 'recovered_contract_only':
            contract_only += 1
        elif bucket == 'recovered_with_content_change':
            with_change += 1
        else:
            unknown += 1

    baseline_passes = 107
    baseline_trials = 204
    post_retry_passes = baseline_passes + primary_recovered
    post_rate = round(post_retry_passes / baseline_trials, 4)
    uplift = round(post_rate - 0.525, 4)

    # Hashes for provenance
    commit = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True, timeout=10).stdout.strip()
    doc = {
        'title': 'Retry-Rescue V1: Qwen Q4 contract-aware prompt retry',
        'experiment': 'retry-rescue-v1__qwen3-4b-q4',
        'base_execution': 'full-baseline-v2__qwen3-4b-q4',
        'baseline_deterministic_trial_accuracy': 0.525,
        'baseline_passes': baseline_passes,
        'baseline_trials': baseline_trials,
        'populations': {
            'primary': {'failure_mode': 'OUTPUT_CONTRACT', 'rows': 39, 'recovered': primary_recovered, 'recovery_rate': round(primary_recovered/39, 4)},
            'control': {'failure_mode': 'MIXED', 'rows': 12, 'recovered': control_recovered, 'recovery_rate': round(control_recovered/12, 4) if 12 else None},
        },
        'post_retry': {
            'post_retry_passes': post_retry_passes,
            'post_retry_pass_rate': post_rate,
            'absolute_uplift': uplift,
        },
        'content_change': {
            'recovered_contract_only': contract_only,
            'recovered_with_content_change': with_change,
            'unknown': unknown,
        },
        'provenance': {
            'analysis_code_git_commit': commit,
            'v2_sealed_commit': V2_SEALED_COMMIT,
            'retry_template_hashes': 'frozen in configs/retry-rescue-v1.yaml',
        },
        'note': 'Baseline rows immutable; control never pooled; single retry per identity',
    }
    REPORT_PATH.write_text(json.dumps(doc, indent=2) + '\n', encoding='utf-8')
    md_lines = [
        '# Retry-Rescue V1: Qwen Q4 contract-aware prompt retry',
        '',
        f'Primary recovery: {primary_recovered} / 39 = {round(primary_recovered/39,4):.1%}',
        f'Control recovery: {control_recovered} / 12 = {round(control_recovered/12,4):.1%}' if 12 else 'Control: n/a',
        f'Post-retry pass rate: {post_retry_passes} / {baseline_trials} = {post_rate:.1%} (baseline 52.5%, uplift {uplift:+.1%})',
        '',
        f'Content-change among recovered: contract_only={contract_only}, with_change={with_change}, unknown={unknown}',
        '',
        'Baseline immutable; single retry per identity; frozen grader exactly.',
    ]
    MD_PATH.write_text('\n'.join(md_lines) + '\n', encoding='utf-8')
    print(f'wrote {REPORT_PATH}: primary {primary_recovered}/39 control {control_recovered}/12 post {post_rate}')
    print(f'wrote {MD_PATH}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
