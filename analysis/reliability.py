"""Reliability analysis: behavioral vs performance stability.

Behavioral stability (verdict/output/failure-code variance) and performance
stability (decode/TTFT/E2E/prefill variance) are reported in separate
sections; "reproducibility" is never used unqualified.

Conventions:
- Judge tasks (any NEEDS_JUDGE verdict) are excluded from every PASS/FAIL
  denominator. Mixed tasks additionally report deterministic sub-results.
- Prefill is stratified by cache state; no pooled prefill statistic exists.
  States with n<2 report "insufficient_n" instead of stdev/CV.
- Trials are observational: trial 1 is the initial attempt, trials 2-5 the
  follow-ups. There is no harness retry loop; recovery rates describe
  observed patterns, not retry mechanics.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import statistics
from dataclasses import asdict, dataclass
from typing import Any

from evals.graders.engine import normalize_text

NEEDS_JUDGE = 'NEEDS_JUDGE'
PASS = 'PASS'


@dataclass
class TrialRecord:
    task_id: str
    trial: int
    verdict: str
    raw_output: str
    grader_details: list[dict[str, Any]]
    decode_tok_s: float | None
    ttft_ms: float | None
    client_e2e_ms: float | None
    prefill_compute_tok_s: float | None
    prefill_cache_state: str | None
    ram_peak_mb: float | None
    vram_peak_mib: float | None


@dataclass
class DistributionStats:
    n: int
    mean: float | None
    stdev: float | None
    cv: float | None


@dataclass
class TaskReliability:
    task_id: str
    n: int
    verdicts: list[str]
    is_judge_task: bool
    # Behavioral (None for judge tasks: excluded from denominators).
    success_rate: float | None
    initial_attempt_success: float | None
    final_success: bool | None
    all_pass_5: bool | None
    grader_flip: bool | None
    unique_raw_output_count: int
    unique_normalized_output_count: int
    unique_canonical_json_count: int | None
    failure_code_distribution: dict[str, int]
    deterministic_sub_pass_rate: float | None
    judge_deferrals: int
    # Performance.
    decode_stats: DistributionStats
    ttft_stats: DistributionStats
    e2e_stats: DistributionStats
    prefill_by_cache_state: dict[str, DistributionStats | str]
    cache_state_distribution: dict[str, int]


@dataclass
class ExperimentReliability:
    experiment_id: str
    deterministic_tasks: dict[str, TaskReliability]
    judge_tasks: dict[str, TaskReliability]
    mean_success_rate: float | None
    mean_initial_attempt_success: float | None
    mean_final_success: float | None
    mean_all_pass_5: float | None
    grader_flip_rate: float | None
    flipped_tasks: list[str]
    retry_recovery_rate: float | None
    retry_recovery_fraction: str


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def _dist(values: list[float | None]) -> DistributionStats:
    observed = [v for v in values if v is not None]
    n = len(observed)
    if n == 0:
        return DistributionStats(n=0, mean=None, stdev=None, cv=None)
    mean = statistics.fmean(observed)
    if n < 2:
        return DistributionStats(n=n, mean=mean, stdev=None, cv=None)
    stdev = statistics.pstdev(observed)
    return DistributionStats(
        n=n, mean=mean, stdev=stdev, cv=(stdev / mean if mean > 0 else None)
    )


def _canonical_json_class(raw: str) -> str:
    try:
        parsed = json.loads(raw)
    except (ValueError, TypeError):
        return 'UNPARSEABLE'
    try:
        return json.dumps(parsed, sort_keys=True, ensure_ascii=False)
    except (ValueError, TypeError):
        return 'UNPARSEABLE'


def _failure_signature(detail: dict[str, Any]) -> str:
    grader_type = str(detail.get('grader_type', 'unknown'))
    if detail.get('passed', True):
        return f'{grader_type}:pass'
    violations = detail.get('violations') or []
    if violations:
        return f'{grader_type}:{str(violations[0])[:120]}'
    return f"{grader_type}:{str(detail.get('detail', 'fail'))[:120]}"


def analyze_task(
    trials: list[TrialRecord], *, is_json_task: bool = False
) -> TaskReliability:
    """Analyze one task's trials (ordered by trial number)."""
    ordered = sorted(trials, key=lambda t: t.trial)
    verdicts = [t.verdict for t in ordered]
    n = len(ordered)
    is_judge = any(v == NEEDS_JUDGE for v in verdicts)

    yields: dict[str, Any] = {
        'success_rate': None,
        'initial_attempt_success': None,
        'final_success': None,
        'all_pass_5': None,
        'grader_flip': None,
    }
    if not is_judge and n > 0:
        passes = [v == PASS for v in verdicts]
        yields['success_rate'] = sum(passes) / n
        yields['initial_attempt_success'] = 1.0 if passes[0] else 0.0
        yields['final_success'] = any(passes)
        yields['all_pass_5'] = all(passes)
        yields['grader_flip'] = any(p != passes[0] for p in passes[1:])

    failure_codes: dict[str, int] = {}
    det_passes: list[bool] = []
    deferrals = 0
    for trial, verdict in zip(ordered, verdicts):
        if verdict == NEEDS_JUDGE:
            deferrals += 1
        sub = [
            d for d in trial.grader_details if d.get('grader_type') != 'rubric_judge'
        ]
        if sub:
            det_passes.append(all(bool(d.get('passed', False)) for d in sub))
        for detail in trial.grader_details:
            if not detail.get('passed', True):
                sig = _failure_signature(detail)
                failure_codes[sig] = failure_codes.get(sig, 0) + 1

    raw_hashes = {_sha(t.raw_output) for t in ordered}
    norm_hashes = {_sha(normalize_text(t.raw_output)) for t in ordered}
    canonical: int | None = None
    if is_json_task:
        canonical = len({_sha(_canonical_json_class(t.raw_output)) for t in ordered})

    prefill_groups: dict[str, list[float | None]] = {}
    cache_dist: dict[str, int] = {}
    for trial in ordered:
        state = trial.prefill_cache_state or 'UNKNOWN'
        cache_dist[state] = cache_dist.get(state, 0) + 1
        prefill_groups.setdefault(state, []).append(trial.prefill_compute_tok_s)
    prefill_by_state: dict[str, DistributionStats | str] = {}
    for state, values in prefill_groups.items():
        stats = _dist(values)
        prefill_by_state[state] = stats if stats.n >= 2 else 'insufficient_n'

    return TaskReliability(
        task_id=ordered[0].task_id if ordered else '',
        n=n,
        verdicts=verdicts,
        is_judge_task=is_judge,
        success_rate=yields['success_rate'],
        initial_attempt_success=yields['initial_attempt_success'],
        final_success=yields['final_success'],
        all_pass_5=yields['all_pass_5'],
        grader_flip=yields['grader_flip'],
        unique_raw_output_count=len(raw_hashes),
        unique_normalized_output_count=len(norm_hashes),
        unique_canonical_json_count=canonical,
        failure_code_distribution=failure_codes,
        deterministic_sub_pass_rate=(
            sum(det_passes) / len(det_passes) if det_passes else None
        ),
        judge_deferrals=deferrals,
        decode_stats=_dist([t.decode_tok_s for t in ordered]),
        ttft_stats=_dist([t.ttft_ms for t in ordered]),
        e2e_stats=_dist([t.client_e2e_ms for t in ordered]),
        prefill_by_cache_state=prefill_by_state,
        cache_state_distribution=cache_dist,
    )


def analyze_experiment(
    experiment_id: str, tasks: dict[str, TaskReliability]
) -> ExperimentReliability:
    """Aggregate task analyses; judge tasks never enter pass denominators."""
    det = {tid: t for tid, t in tasks.items() if not t.is_judge_task}
    judge = {tid: t for tid, t in tasks.items() if t.is_judge_task}

    def mean(values: list[float | None]) -> float | None:
        observed = [v for v in values if v is not None]
        return statistics.fmean(observed) if observed else None

    flips = sorted(
        tid for tid, t in det.items() if t.grader_flip is True
    )
    failed_first = [t for t in det.values() if t.initial_attempt_success == 0.0]
    recovered = [t for t in failed_first if t.final_success is True]

    return ExperimentReliability(
        experiment_id=experiment_id,
        deterministic_tasks=det,
        judge_tasks=judge,
        mean_success_rate=mean([t.success_rate for t in det.values()]),
        mean_initial_attempt_success=mean(
            [t.initial_attempt_success for t in det.values()]
        ),
        mean_final_success=mean(
            [1.0 if t.final_success else 0.0 for t in det.values()]
        ),
        mean_all_pass_5=mean([1.0 if t.all_pass_5 else 0.0 for t in det.values()]),
        grader_flip_rate=(len(flips) / len(det) if det else None),
        flipped_tasks=flips,
        retry_recovery_rate=(
            len(recovered) / len(failed_first) if failed_first else None
        ),
        retry_recovery_fraction=f'{len(recovered)}/{len(failed_first)}',
    )


def load_trials(conn: sqlite3.Connection, experiment_id: str) -> dict[str, list[TrialRecord]]:
    """Load measured (non-warmup) trials grouped by task."""
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            'SELECT task_id, trial, grader_verdict, raw_output,'
            ' grader_details_json, decode_tok_s, ttft_ms, client_e2e_ms,'
            ' prefill_compute_tok_s, prefill_cache_state, ram_peak_mb,'
            ' vram_peak_mib FROM runs WHERE experiment_id=? AND is_warmup=0'
            ' ORDER BY task_id, trial',
            (experiment_id,),
        ).fetchall()
    finally:
        conn.row_factory = None
    grouped: dict[str, list[TrialRecord]] = {}
    for row in rows:
        try:
            details = json.loads(row['grader_details_json'] or '[]')
        except ValueError:
            details = []
        record = TrialRecord(
            task_id=str(row['task_id']),
            trial=int(row['trial']),
            verdict=str(row['grader_verdict']),
            raw_output=str(row['raw_output'] or ''),
            grader_details=details if isinstance(details, list) else [],
            decode_tok_s=row['decode_tok_s'],
            ttft_ms=row['ttft_ms'],
            client_e2e_ms=row['client_e2e_ms'],
            prefill_compute_tok_s=row['prefill_compute_tok_s'],
            prefill_cache_state=row['prefill_cache_state'],
            ram_peak_mb=row['ram_peak_mb'],
            vram_peak_mib=row['vram_peak_mib'],
        )
        grouped.setdefault(record.task_id, []).append(record)
    return grouped


def report_to_json(report: ExperimentReliability) -> str:
    def encode(value: Any) -> Any:
        if isinstance(value, (DistributionStats, TaskReliability, ExperimentReliability)):
            return {k: encode(v) for k, v in asdict(value).items()}
        if isinstance(value, dict):
            return {str(k): encode(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [encode(v) for v in value]
        return value

    return json.dumps(encode(report), indent=2, sort_keys=True)


def summarize_console(report: ExperimentReliability) -> str:
    lines = [
        f"Reliability: {report.experiment_id}",
        f"  deterministic tasks: {len(report.deterministic_tasks)}"
        f'  judge tasks: {len(report.judge_tasks)}',
    ]

    def fmt(value: float | None) -> str:
        return f'{value:.3f}' if value is not None else 'n/a'

    lines += [
        f'  mean success_rate: {fmt(report.mean_success_rate)}',
        f'  mean initial_attempt: {fmt(report.mean_initial_attempt_success)}',
        f'  mean final (pass@5): {fmt(report.mean_final_success)}',
        f'  mean all_pass_5 (pass^5): {fmt(report.mean_all_pass_5)}',
        f'  grader_flip_rate: {fmt(report.grader_flip_rate)} {report.flipped_tasks}',
        f'  retry_recovery: {report.retry_recovery_fraction} '
        f"({fmt(report.retry_recovery_rate)})",
        '  per-task:',
    ]
    for task_id in sorted(report.deterministic_tasks):
        task = report.deterministic_tasks[task_id]
        lines.append(
            f"    {task_id}: sr={fmt(task.success_rate)} "
            f"pass@5={task.final_success} pass^5={task.all_pass_5} "
            f"flip={task.grader_flip} raw_n={task.unique_raw_output_count} "
            f"norm_n={task.unique_normalized_output_count}"
        )
    for task_id in sorted(report.judge_tasks):
        task = report.judge_tasks[task_id]
        lines.append(
            f'    {task_id}: JUDGE-ONLY-VARIATION deferrals={task.judge_deferrals} '
            f'det_sub={fmt(task.deterministic_sub_pass_rate)} '
            f'raw_n={task.unique_raw_output_count}'
        )
    return '\n'.join(lines)
