"""Failure-report computation: row-based decompositions from an approved review.

Table 1 (contract) and Table 2 (substance) are computed over ROWS, not
tasks: each failed row inherits its task's substantive label, and each row
contributes its own mode. Task counts (22/3/10) never enter percentages.
Judge rows form a separate descriptive section. Generation requires an
APPROVED, valid review file.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Mapping

from analysis.failure_modes import (
    TAXONOMY_VERSION,
    review_generation_allowed,
)


class ReviewNotApprovedError(Exception):
    pass


def require_approved(review: Mapping[str, Any]) -> None:
    if not review_generation_allowed(review):
        raise ReviewNotApprovedError(
            'report generation requires an APPROVED, valid review file; '
            f"got status={review.get('review_status')!r}"
        )


def _det_tasks(review: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return [
        t for t in review.get('tasks', [])
        if isinstance(t, dict) and t.get('task_status') == 'READY_DETERMINISTIC'
    ]


def _judge_tasks(review: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return [
        t for t in review.get('tasks', [])
        if isinstance(t, dict) and t.get('task_status') == 'READY_JUDGE'
    ]


def table_contract_modes(review: Mapping[str, Any]) -> dict[str, Any]:
    """Table 1: row-mode share over READY_DETERMINISTIC failed rows."""
    modes: Counter[str] = Counter()
    total = 0
    for task in _det_tasks(review):
        auto = task.get('auto_analysis', {})
        row_modes = auto.get('row_modes', [])
        if isinstance(row_modes, list):
            for mode in row_modes:
                modes[str(mode)] += 1
                total += 1
    return {
        'denominator_rows': total,
        'counts': dict(sorted(modes.items())),
        'percent': {
            mode: round(count / total, 4) if total else None
            for mode, count in sorted(modes.items())
        },
    }


def table_substance(review: Mapping[str, Any]) -> dict[str, Any]:
    """Table 2: substantive share over rows (tasks propagate to their rows)."""
    labels: Counter[str] = Counter()
    total = 0
    for task in _det_tasks(review):
        auto = task.get('auto_analysis', {})
        failed_rows = auto.get('failed_rows', 0)
        label = task.get('review', {}).get('substantive_answer_assessment')
        if isinstance(failed_rows, int) and isinstance(label, str):
            labels[label] += failed_rows
            total += failed_rows
    return {
        'denominator_rows': total,
        'counts': dict(sorted(labels.items())),
        'percent': {
            label: round(count / total, 4) if total else None
            for label, count in sorted(labels.items())
        },
    }


def table_judge_prechecks(review: Mapping[str, Any]) -> dict[str, Any]:
    """Table 3: descriptive section over READY_JUDGE failed rows."""
    rows: list[dict[str, Any]] = []
    for task in _judge_tasks(review):
        auto = task.get('auto_analysis', {})
        rows.append({
            'task_id': task.get('task_id'),
            'failed_trials': task.get('failed_trials', []),
            'row_modes': auto.get('row_modes', []),
            'generation_flags': auto.get('generation_flags', []),
            'substantive': task.get('review', {}).get(
                'substantive_answer_assessment'),
            'notes': task.get('review', {}).get('notes'),
        })
    return {'tasks': rows, 'total_rows': sum(len(task.get('failed_trials', [])) for task in _judge_tasks(review))}


def task_table(review: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Per-task derived summaries (distributions, never auto-MIXED)."""
    table: list[dict[str, Any]] = []
    for task in review.get('tasks', []):
        if not isinstance(task, dict):
            continue
        auto = task.get('auto_analysis', {})
        row_modes = [str(m) for m in auto.get('row_modes', [])]
        dist: Counter[str] = Counter(row_modes)
        excerpts = auto.get('excerpts', [])
        representative = ''
        if isinstance(excerpts, list) and excerpts:
            first = excerpts[0]
            if isinstance(first, dict):
                representative = str(first.get('excerpt', ''))[:150]
        table.append({
            'task_id': task.get('task_id'),
            'task_status': task.get('task_status'),
            'failed_trials': task.get('failed_trials', []),
            'mode_distribution': dict(sorted(dist.items())),
            'multi_mode_across_trials': len(dist) > 1,
            'generation_flags': auto.get('generation_flags', []),
            'substantive': task.get('review', {}).get('substantive_answer_assessment'),
            'assessment_author': task.get('review', {}).get('assessment_author'),
            'mode_override': task.get('review', {}).get('mode_override'),
            'notes': task.get('review', {}).get('notes'),
            'representative_excerpt': representative,
        })
    return table


def decomposition_shell(table1: Mapping[str, Any]) -> dict[str, Any]:
    """Cross-model shell: Q4 live; later models slot under frozen taxonomy."""
    return {
        'taxonomy_version': TAXONOMY_VERSION,
        'columns': {
            'qwen3-4b-q4': {
                'denominator_rows': table1.get('denominator_rows'),
                'mode_counts': table1.get('counts'),
            },
        },
        'reserved': ['qwen3-4b-q5'],
    }
