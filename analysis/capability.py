"""Capability summaries: status-driven deterministic aggregates.

Population contract (same as reliability):
- READY_DETERMINISTIC tasks form every denominator (trial accuracy,
  any_pass_3, all_pass_3).
- READY_JUDGE tasks appear only under judge_prechecks (descriptive).
- Unknown task IDs raise UnknownStatusError; verdicts never imply membership.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

from analysis.reliability import (
    PASS,
    READY_DETERMINISTIC,
    READY_JUDGE,
    SummaryProvenance,
    TaskReliability,
    UnknownStatusError,
    build_judge_prechecks,
)

TRIALS_PER_TASK = 3


@dataclass
class CapabilitySummary:
    experiment_id: str
    deterministic_task_ids: list[str]
    judge_task_ids: list[str]
    # Integer relations; rates derived in code, never hand-rounded.
    passes: int
    deterministic_trials: int
    tasks_any_pass_3: int
    tasks_all_pass_3: int
    deterministic_tasks: int
    judge_prechecks: Any
    provenance: SummaryProvenance

    @property
    def deterministic_trial_accuracy(self) -> float | None:
        if not self.deterministic_trials:
            return None
        return self.passes / self.deterministic_trials

    @property
    def any_pass_3_rate(self) -> float | None:
        if not self.deterministic_tasks:
            return None
        return self.tasks_any_pass_3 / self.deterministic_tasks

    @property
    def all_pass_3_rate(self) -> float | None:
        if not self.deterministic_tasks:
            return None
        return self.tasks_all_pass_3 / self.deterministic_tasks


def build_capability_summary(
    experiment_id: str,
    tasks: dict[str, TaskReliability],
    statuses: Mapping[str, str],
    provenance: SummaryProvenance,
) -> CapabilitySummary:
    """Build the capability summary; every task must resolve to a status."""
    for task_id in tasks:
        if task_id not in statuses:
            raise UnknownStatusError(
                f'task {task_id} has no frozen grading_status; refusing to infer'
            )
        expected_judge = statuses[task_id] == READY_JUDGE
        if tasks[task_id].is_judge_task != expected_judge:
            raise UnknownStatusError(
                f'task {task_id} analysis/population mismatch: '
                f'verdict-implied judge={tasks[task_id].is_judge_task} vs '
                f'status {statuses[task_id]}'
            )
    det = sorted(
        tid for tid, t in tasks.items()
        if statuses[tid] == READY_DETERMINISTIC
    )
    judge = sorted(
        tid for tid, t in tasks.items() if statuses[tid] == READY_JUDGE
    )
    passes = sum(
        1 for tid in det for v in tasks[tid].verdicts if v == PASS
    )
    trials = sum(tasks[tid].n for tid in det)
    any_pass = sum(
        1 for tid in det if any(v == PASS for v in tasks[tid].verdicts)
    )
    all_pass = sum(
        1 for tid in det
        if tasks[tid].n > 0 and all(v == PASS for v in tasks[tid].verdicts)
    )
    return CapabilitySummary(
        experiment_id=experiment_id,
        deterministic_task_ids=det,
        judge_task_ids=judge,
        passes=passes,
        deterministic_trials=trials,
        tasks_any_pass_3=any_pass,
        tasks_all_pass_3=all_pass,
        deterministic_tasks=len(det),
        judge_prechecks=build_judge_prechecks(
            {tid: tasks[tid] for tid in judge}
        ),
        provenance=provenance,
    )


def capability_to_json(summary: CapabilitySummary) -> str:
    import json

    def encode(value: Any) -> Any:
        if hasattr(value, '__dataclass_fields__'):
            return {k: encode(v) for k, v in asdict(value).items()}
        if isinstance(value, dict):
            return {str(k): encode(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [encode(v) for v in value]
        return value

    document = encode(summary)
    document['derived_rates'] = {
        'deterministic_trial_accuracy': summary.deterministic_trial_accuracy,
        'any_pass_3_rate': summary.any_pass_3_rate,
        'all_pass_3_rate': summary.all_pass_3_rate,
    }
    return json.dumps(
        {k: document[k] for k in sorted(document)}, indent=2, sort_keys=False
    )


def summarize_capability_console(summary: CapabilitySummary) -> str:
    def fmt(value: float | None) -> str:
        return f'{value:.3f}' if value is not None else 'n/a'

    pre = summary.judge_prechecks
    return '\n'.join(
        [
            f'Capability: {summary.experiment_id}',
            f'  deterministic: {summary.deterministic_tasks} tasks, '
            f'{summary.deterministic_trials} trials',
            f'  passes: {summary.passes} '
            f'(accuracy {fmt(summary.deterministic_trial_accuracy)})',
            f'  any_pass_3: {summary.tasks_any_pass_3} '
            f'({fmt(summary.any_pass_3_rate)})',
            f'  all_pass_3: {summary.tasks_all_pass_3} '
            f'({fmt(summary.all_pass_3_rate)})',
            f'  judge tasks: {len(summary.judge_task_ids)} '
            f'{summary.judge_task_ids} (EXCLUDED from denominators)',
            f'  judge prechecks: {pre.deterministic_precheck_passes}/'
            f'{pre.deterministic_precheck_total} sub-pass, '
            f'{pre.judge_deferrals} deferrals',
        ]
    )


# Re-export for the thin wrapper script.
__all__ = [
    'CapabilitySummary',
    'build_capability_summary',
    'capability_to_json',
    'summarize_capability_console',
]
