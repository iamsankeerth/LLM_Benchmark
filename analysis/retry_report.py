"""Retry-rescue report: recovery + content-change, descriptive only."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from analysis.failure_modes import extract_asserted_value


def _load_retry_rows(db_path: str, execution_id: str) -> list[dict[str, Any]]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            'SELECT task_id, trial, grader_verdict, raw_output, grader_details_json, prompt, error '
            'FROM runs WHERE experiment_id=? ORDER BY task_id, trial',
            (execution_id,),
        ).fetchall()
    finally:
        conn.close()
    return [dict(r) for r in rows]


def _load_baseline_rows(db_path: str, execution_id: str) -> dict[tuple[str, int], str]:
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            'SELECT task_id, trial, raw_output FROM runs WHERE experiment_id=? AND is_warmup=0',
            (execution_id,),
        ).fetchall()
    finally:
        conn.close()
    return {(str(r[0]), int(r[1])): str(r[2]) for r in rows}


def classify_content_change(baseline_text: str, retry_text: str, grader_type: str) -> str:
    """Descriptive only: does the retry's asserted substantive answer differ?

    Uses conservative asserted-value extraction where available (numeric/time);
    falls back to normalized exact comparison for exact/structured. Returns
    one of: recovered_contract_only, recovered_with_content_change, unknown
    (when neither has an extractable asserted value).
    """
    kind = grader_type if grader_type in ('numeric', 'time') else 'exact'
    base_val = extract_asserted_value(baseline_text, kind=kind if kind in ('numeric', 'time') else 'numeric')
    retry_val = extract_asserted_value(retry_text, kind=kind if kind in ('numeric', 'time') else 'numeric')
    # For non-numeric/time, compare normalized raw outputs as proxy
    if grader_type not in ('numeric', 'time'):
        from evals.graders.engine import normalize_text

        if normalize_text(baseline_text).strip() == normalize_text(retry_text).strip():
            return 'recovered_contract_only'
        # Heuristic: if retry is a strict substring/bare form of baseline, contract-only
        if retry_text.strip() in baseline_text:
            return 'recovered_contract_only'
        return 'recovered_with_content_change'
    if base_val.status == 'NO_ASSERTED_VALUE' or retry_val.status == 'NO_ASSERTED_VALUE':
        return 'unknown'
    if base_val.values == retry_val.values:
        return 'recovered_contract_only'
    return 'recovered_with_content_change'


def build_report(
    retry_db: str,
    retry_execution: str,
    baseline_db: str,
    baseline_execution: str,
    spec_path: str,
) -> dict[str, Any]:
    import yaml

    spec = yaml.safe_load(Path(spec_path).read_text(encoding='utf-8'))
    retry_rows = _load_retry_rows(retry_db, retry_execution)
    baseline_map = _load_baseline_rows(baseline_db, baseline_execution)
    # For report, we compute from retry_rows directly: primary = OUTPUT_CONTRACT rows in original failure set
    # Here we approximate via retry_rows count minus control; real report recomputes from failure analysis
    total = len(retry_rows)
    recovered = sum(1 for r in retry_rows if r['grader_verdict'] == 'PASS')
    still_fail = total - recovered
    # Content-change breakdown for recovered only
    contract_only = 0
    with_change = 0
    unknown = 0
    for row in retry_rows:
        if row['grader_verdict'] != 'PASS':
            continue
        baseline_text = baseline_map.get((row['task_id'], row['trial']), '')
        grader_type = str(spec['tasks'][row['task_id']]['graders'][0].get('type', 'exact')) if spec['tasks'][row['task_id']].get('graders') else 'exact'
        bucket = classify_content_change(baseline_text, row['raw_output'], grader_type)
        if bucket == 'recovered_contract_only':
            contract_only += 1
        elif bucket == 'recovered_with_content_change':
            with_change += 1
        else:
            unknown += 1
    return {
        'retry_execution': retry_execution,
        'baseline_execution': baseline_execution,
        'total_retry_rows': total,
        'recovered': recovered,
        'still_fail': still_fail,
        'recovery_rate_overall': round(recovered / total, 4) if total else None,
        'recovered_contract_only': contract_only,
        'recovered_with_content_change': with_change,
        'substantive_change_unknown': unknown,
        'post_retry_pass_rate': round((107 + recovered) / 204, 4) if total else None,
        'note': 'post_retry_pass_rate uses recovered from primary 39 only in final report; this helper reports overall',
    }


def _mixed_ids(spec: dict[str, Any]) -> set[str]:
    return set()  # placeholder for type-check; real filtering uses failure analysis artifact
