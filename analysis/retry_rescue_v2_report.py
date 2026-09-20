"""Read-only A/B/C reducer for retry-rescue-v2."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from analysis.comparison import mcnemar_exact_p, performance_snapshot
from analysis.retry_report import classify_content_change


def _readonly_rows(
    db_path: str, execution_id: str, identities: set[tuple[str, int]],
) -> dict[tuple[str, int], dict[str, Any]]:
    uri = f'file:{Path(db_path).resolve().as_posix()}?mode=ro'
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            'SELECT task_id, trial, grader_verdict, raw_output, is_warmup, '
            'decode_tok_s, ttft_ms, eval_count, vram_peak_mib '
            'FROM runs WHERE experiment_id=? AND is_warmup=0',
            (execution_id,),
        ).fetchall()
    finally:
        conn.close()
    selected = {
        (str(row['task_id']), int(row['trial'])): dict(row)
        for row in rows
        if (str(row['task_id']), int(row['trial'])) in identities
    }
    return selected


def _require_identities(
    label: str, rows: dict[tuple[str, int], dict[str, Any]], expected: set[tuple[str, int]],
) -> None:
    if set(rows) != expected:
        raise ValueError(f'{label} identity mismatch')


def _side(rows: dict[tuple[str, int], dict[str, Any]]) -> dict[str, Any]:
    recovered = sum(row['grader_verdict'] == 'PASS' for row in rows.values())
    return {
        'recovered': recovered,
        'total': len(rows),
        'performance': performance_snapshot(
            [row['decode_tok_s'] for row in rows.values()],
            [row['ttft_ms'] for row in rows.values()],
            [row['eval_count'] for row in rows.values()],
            [row['vram_peak_mib'] for row in rows.values()],
        ),
    }


def _pair_counts(
    b_rows: dict[tuple[str, int], dict[str, Any]],
    c_rows: dict[tuple[str, int], dict[str, Any]],
) -> dict[str, float | int]:
    both = b_only = c_only = neither = 0
    for identity in sorted(b_rows):
        b_pass = b_rows[identity]['grader_verdict'] == 'PASS'
        c_pass = c_rows[identity]['grader_verdict'] == 'PASS'
        if b_pass and c_pass:
            both += 1
        elif b_pass:
            b_only += 1
        elif c_pass:
            c_only += 1
        else:
            neither += 1
    return {
        'both_PASS': both,
        'B_only_PASS': b_only,
        'C_only_PASS': c_only,
        'neither_PASS': neither,
        'mcnemar_exact_p': mcnemar_exact_p(b_only, c_only),
    }


def _content_change(
    a_rows: dict[tuple[str, int], dict[str, Any]],
    c_rows: dict[tuple[str, int], dict[str, Any]],
    grader_types: dict[str, str],
) -> dict[str, int]:
    counts = {
        'contract_only': 0,
        'recovered_with_content_change': 0,
        'unknown': 0,
    }
    for identity, c_row in c_rows.items():
        if c_row['grader_verdict'] != 'PASS':
            continue
        bucket = classify_content_change(
            str(a_rows[identity]['raw_output']), str(c_row['raw_output']),
            grader_types[identity[0]],
        )
        if bucket == 'recovered_contract_only':
            counts['contract_only'] += 1
        else:
            counts[bucket] += 1
    return counts


def build_v2_report(
    *,
    baseline_db: str,
    baseline_execution: str,
    retry_db: str,
    retry_execution: str,
    constrained_db: str,
    constrained_execution: str,
    primary_identities: Iterable[tuple[str, int]],
    control_identities: Iterable[tuple[str, int]],
    grader_types: dict[str, str],
) -> dict[str, Any]:
    """Build a scoped report without initializing or migrating any input DB."""
    primary = set(primary_identities)
    control = set(control_identities)
    if primary & control:
        raise ValueError('primary and control identities overlap')
    a_primary = _readonly_rows(baseline_db, baseline_execution, primary)
    b_primary = _readonly_rows(retry_db, retry_execution, primary)
    c_primary = _readonly_rows(constrained_db, constrained_execution, primary)
    _require_identities('A primary', a_primary, primary)
    _require_identities('B primary', b_primary, primary)
    _require_identities('C primary', c_primary, primary)
    if any(row['grader_verdict'] == 'PASS' for row in a_primary.values()):
        raise ValueError('A primary must be 0 PASS by construction')
    b_control = _readonly_rows(retry_db, retry_execution, control)
    c_control = _readonly_rows(constrained_db, constrained_execution, control)
    _require_identities('B control', b_control, control)
    _require_identities('C control', c_control, control)
    return {
        'primary': {
            'A': _side(a_primary),
            'B': _side(b_primary),
            'C': _side(c_primary),
            'paired_B_C': _pair_counts(b_primary, c_primary),
            'content_change': _content_change(a_primary, c_primary, grader_types),
        },
        'controls': {'B': _side(b_control), 'C': _side(c_control)},
    }
