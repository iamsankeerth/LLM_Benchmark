"""Reliability analysis: behavioral vs performance stability.

Behavioral stability (verdict/output/failure-code variance) and performance
stability (decode/TTFT/E2E/prefill variance) are reported in separate
sections; "reproducibility" is never used unqualified.

Population contract (status-driven, never verdict-driven):
- READY_DETERMINISTIC tasks form every PASS/FAIL denominator.
- READY_JUDGE tasks are excluded from all global denominators and reported
  under judge_prechecks with deterministic sub-results and deferral counts.
- A task ID that does not resolve to exactly one frozen grading_status is a
  hard error. Verdicts never imply population membership.
- Prefill is stratified by cache state and reported alongside uncached token
  counts; no pooled prefill statistic exists. States with n<2 report
  "insufficient_n" instead of stdev/CV.
- Trials are observational: trial 1 is the initial attempt, later trials the
  follow-ups. There is no harness retry loop; later-trial recovery describes
  observed patterns, not retry mechanics.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import statistics
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

from evals.graders.engine import normalize_text
from storage.manifest import hash_experiment_config

NEEDS_JUDGE = 'NEEDS_JUDGE'
PASS = 'PASS'
READY_DETERMINISTIC = 'READY_DETERMINISTIC'
READY_JUDGE = 'READY_JUDGE'
KNOWN_STATUSES = frozenset({READY_DETERMINISTIC, READY_JUDGE})


class UnknownStatusError(ValueError):
    """A task ID did not resolve to exactly one frozen grading_status."""


class DirtyWorktreeError(RuntimeError):
    """Refusal to stamp a release summary from a dirty worktree."""


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
    prompt_eval_uncached_count: int | None
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
    grading_status: str
    n: int
    verdicts: list[str]
    is_judge_task: bool
    # Behavioral (None for judge tasks: excluded from denominators).
    success_rate: float | None
    initial_attempt_success: float | None
    any_pass_5: bool | None
    all_pass_5: bool | None
    grader_flip: bool | None
    unique_raw_output_count: int
    unique_normalized_output_count: int
    unique_canonical_json_count: int | None
    failure_code_distribution: dict[str, int]
    deterministic_sub_passes: int
    deterministic_sub_total: int
    deterministic_sub_pass_rate: float | None
    judge_deferrals: int
    # Performance.
    decode_stats: DistributionStats
    ttft_stats: DistributionStats
    e2e_stats: DistributionStats
    prefill_by_cache_state: dict[str, DistributionStats | str]
    uncached_counts_by_state: dict[str, list[int]]
    cache_state_distribution: dict[str, int]


@dataclass
class JudgePrechecks:
    judge_task_trials: int
    deterministic_precheck_passes: int
    deterministic_precheck_total: int
    deterministic_precheck_pass_rate: float | None
    deterministic_precheck_fail_count: int
    judge_deferrals: int
    judge_deferral_rate: float | None
    per_task: dict[str, dict[str, Any]]


@dataclass
class ExperimentReliability:
    experiment_id: str
    deterministic_tasks: dict[str, TaskReliability]
    judge_tasks: dict[str, TaskReliability]
    judge_prechecks: JudgePrechecks
    mean_success_rate: float | None
    mean_initial_attempt_success: float | None
    mean_any_pass_5: float | None
    mean_all_pass_5: float | None
    grader_flip_rate: float | None
    flipped_tasks: list[str]
    later_trial_recovery_rate: float | None
    later_trial_recovery_fraction: str


@dataclass
class SummaryProvenance:
    grading_spec_hash: str
    dataset_hash: str
    experiment_config_hash: str
    analysis_code_git_commit: str | None
    analysis_worktree_dirty: bool


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


def git_worktree_status(repo_dir: str) -> tuple[str | None, bool]:
    """Return (HEAD commit, is_dirty); (None, True) when git is unavailable."""
    try:
        head = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            cwd=repo_dir, capture_output=True, text=True, timeout=15,
        )
        porcelain = subprocess.run(
            ['git', 'status', '--porcelain'],
            cwd=repo_dir, capture_output=True, text=True, timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return None, True
    if head.returncode != 0:
        return None, True
    commit = head.stdout.strip() or None
    dirty = porcelain.returncode != 0 or bool(porcelain.stdout.strip())
    return commit, dirty


def collect_provenance(
    repo_dir: str,
    experiment_config: Mapping[str, Any],
    *,
    allow_dirty: bool = False,
) -> SummaryProvenance:
    """Build the summary provenance block; fail closed on a dirty worktree
    unless the development-only allow_dirty override is set (stamped true)."""
    freeze_path = Path(repo_dir) / 'evals/specs/eval-v1-grading.freeze.json'
    freeze = json.loads(freeze_path.read_text(encoding='utf-8'))
    commit, dirty = git_worktree_status(repo_dir)
    if dirty and not allow_dirty:
        raise DirtyWorktreeError(
            'refusing to stamp a summary from a dirty worktree '
            '(use --allow-dirty for development-only output)'
        )
    return SummaryProvenance(
        grading_spec_hash=str(freeze['artifacts']['eval-v1-grading.yaml']),
        dataset_hash=str(freeze['artifacts']['executable-v1.jsonl']),
        experiment_config_hash=hash_experiment_config(dict(experiment_config)),
        analysis_code_git_commit=commit,
        analysis_worktree_dirty=dirty,
    )


def analyze_task(
    trials: list[TrialRecord], *, is_json_task: bool = False, grading_status: str
) -> TaskReliability:
    """Analyze one task's trials (ordered by trial number).

    Population membership comes from the frozen grading_status only.
    Unknown statuses raise; verdicts never imply membership.
    """
    if grading_status not in KNOWN_STATUSES:
        task = trials[0].task_id if trials else '?'
        raise UnknownStatusError(
            f'task {task} resolved to {grading_status!r}, not a frozen '
            'READY_DETERMINISTIC/READY_JUDGE status'
        )
    ordered = sorted(trials, key=lambda t: t.trial)
    verdicts = [t.verdict for t in ordered]
    n = len(ordered)
    is_judge = grading_status == READY_JUDGE

    yields: dict[str, Any] = {
        'success_rate': None,
        'initial_attempt_success': None,
        'any_pass_5': None,
        'all_pass_5': None,
        'grader_flip': None,
    }
    if not is_judge and n > 0:
        passes = [v == PASS for v in verdicts]
        yields['success_rate'] = sum(passes) / n
        yields['initial_attempt_success'] = 1.0 if passes[0] else 0.0
        yields['any_pass_5'] = any(passes)
        yields['all_pass_5'] = all(passes)
        yields['grader_flip'] = any(p != passes[0] for p in passes[1:])

    failure_codes: dict[str, int] = {}
    det_passes = 0
    det_total = 0
    deferrals = 0
    for trial, verdict in zip(ordered, verdicts):
        if verdict == NEEDS_JUDGE:
            deferrals += 1
        sub = [
            d for d in trial.grader_details if d.get('grader_type') != 'rubric_judge'
        ]
        if sub:
            det_total += 1
            if all(bool(d.get('passed', False)) for d in sub):
                det_passes += 1
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
    uncached_groups: dict[str, list[int]] = {}
    cache_dist: dict[str, int] = {}
    for trial in ordered:
        state = trial.prefill_cache_state or 'UNKNOWN'
        cache_dist[state] = cache_dist.get(state, 0) + 1
        prefill_groups.setdefault(state, []).append(trial.prefill_compute_tok_s)
        if trial.prompt_eval_uncached_count is not None:
            uncached_groups.setdefault(state, []).append(
                trial.prompt_eval_uncached_count
            )
    prefill_by_state: dict[str, DistributionStats | str] = {}
    for state, values in prefill_groups.items():
        stats = _dist(values)
        prefill_by_state[state] = stats if stats.n >= 2 else 'insufficient_n'

    return TaskReliability(
        task_id=ordered[0].task_id if ordered else '',
        grading_status=grading_status,
        n=n,
        verdicts=verdicts,
        is_judge_task=is_judge,
        success_rate=yields['success_rate'],
        initial_attempt_success=yields['initial_attempt_success'],
        any_pass_5=yields['any_pass_5'],
        all_pass_5=yields['all_pass_5'],
        grader_flip=yields['grader_flip'],
        unique_raw_output_count=len(raw_hashes),
        unique_normalized_output_count=len(norm_hashes),
        unique_canonical_json_count=canonical,
        failure_code_distribution=failure_codes,
        deterministic_sub_passes=det_passes,
        deterministic_sub_total=det_total,
        deterministic_sub_pass_rate=(
            det_passes / det_total if det_total else None
        ),
        judge_deferrals=deferrals,
        decode_stats=_dist([t.decode_tok_s for t in ordered]),
        ttft_stats=_dist([t.ttft_ms for t in ordered]),
        e2e_stats=_dist([t.client_e2e_ms for t in ordered]),
        prefill_by_cache_state=prefill_by_state,
        uncached_counts_by_state=uncached_groups,
        cache_state_distribution=cache_dist,
    )


def build_judge_prechecks(judge: dict[str, TaskReliability]) -> JudgePrechecks:
    """Separately labeled aggregate over READY_JUDGE tasks only.

    Descriptive information without contaminating the deterministic
    capability denominator.
    """
    trials = sum(t.n for t in judge.values())
    passes = sum(t.deterministic_sub_passes for t in judge.values())
    total = sum(t.deterministic_sub_total for t in judge.values())
    deferrals = sum(t.judge_deferrals for t in judge.values())
    per_task = {
        task_id: {
            'n': task.n,
            'verdicts': list(task.verdicts),
            'deterministic_precheck_passes': task.deterministic_sub_passes,
            'deterministic_precheck_total': task.deterministic_sub_total,
            'deterministic_precheck_pass_rate': task.deterministic_sub_pass_rate,
            'judge_deferrals': task.judge_deferrals,
            'global_deterministic_denominator': 'EXCLUDED',
        }
        for task_id, task in sorted(judge.items())
    }
    return JudgePrechecks(
        judge_task_trials=trials,
        deterministic_precheck_passes=passes,
        deterministic_precheck_total=total,
        deterministic_precheck_pass_rate=(passes / total if total else None),
        deterministic_precheck_fail_count=total - passes,
        judge_deferrals=deferrals,
        judge_deferral_rate=(deferrals / trials if trials else None),
        per_task=per_task,
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
    recovered = [t for t in failed_first if t.any_pass_5 is True]

    return ExperimentReliability(
        experiment_id=experiment_id,
        deterministic_tasks=det,
        judge_tasks=judge,
        judge_prechecks=build_judge_prechecks(judge),
        mean_success_rate=mean([t.success_rate for t in det.values()]),
        mean_initial_attempt_success=mean(
            [t.initial_attempt_success for t in det.values()]
        ),
        mean_any_pass_5=mean(
            [1.0 if t.any_pass_5 else 0.0 for t in det.values()]
        ),
        mean_all_pass_5=mean([1.0 if t.all_pass_5 else 0.0 for t in det.values()]),
        grader_flip_rate=(len(flips) / len(det) if det else None),
        flipped_tasks=flips,
        later_trial_recovery_rate=(
            len(recovered) / len(failed_first) if failed_first else None
        ),
        later_trial_recovery_fraction=f'{len(recovered)}/{len(failed_first)}',
    )


def load_spec_statuses(spec_path: str) -> dict[str, str]:
    """Load frozen task_id -> grading_status (read-only join source)."""
    import yaml

    spec = yaml.safe_load(open(spec_path, encoding='utf-8'))
    return {str(tid): str(entry['grading_status']) for tid, entry in spec['tasks'].items()}


def load_trials(conn: sqlite3.Connection, experiment_id: str) -> dict[str, list[TrialRecord]]:
    """Load measured (non-warmup) trials grouped by task."""
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            'SELECT task_id, trial, grader_verdict, raw_output,'
            ' grader_details_json, decode_tok_s, ttft_ms, client_e2e_ms,'
            ' prefill_compute_tok_s, prefill_cache_state,'
            ' prompt_eval_uncached_count, ram_peak_mb,'
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
            prompt_eval_uncached_count=row['prompt_eval_uncached_count'],
            ram_peak_mb=row['ram_peak_mb'],
            vram_peak_mib=row['vram_peak_mib'],
        )
        grouped.setdefault(record.task_id, []).append(record)
    return grouped


def analyze_grouped(
    experiment_id: str,
    grouped: dict[str, list[TrialRecord]],
    statuses: Mapping[str, str],
    *,
    json_tasks: frozenset[str] = frozenset(),
) -> ExperimentReliability:
    """Status-driven analysis; unknown task IDs raise UnknownStatusError."""
    analyses: dict[str, TaskReliability] = {}
    for task_id, trials in grouped.items():
        if task_id not in statuses:
            raise UnknownStatusError(
                f'task {task_id} has no frozen grading_status; refusing to infer'
            )
        analyses[task_id] = analyze_task(
            trials, is_json_task=task_id in json_tasks,
            grading_status=statuses[task_id],
        )
    return analyze_experiment(experiment_id, analyses)


def report_to_json(
    report: ExperimentReliability, provenance: SummaryProvenance
) -> str:
    def encode(value: Any) -> Any:
        if isinstance(
            value,
            (
                DistributionStats, TaskReliability, ExperimentReliability,
                JudgePrechecks, SummaryProvenance,
            ),
        ):
            return {k: encode(v) for k, v in asdict(value).items()}
        if isinstance(value, dict):
            return {str(k): encode(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [encode(v) for v in value]
        return value

    return json.dumps(
        {'report': encode(report), 'provenance': encode(provenance)},
        indent=2, sort_keys=True,
    )


def summarize_console(report: ExperimentReliability) -> str:
    lines = [
        f"Reliability: {report.experiment_id}",
        f"  deterministic tasks: {len(report.deterministic_tasks)}"
        f'  judge tasks: {len(report.judge_tasks)}',
    ]

    def fmt(value: float | None) -> str:
        return f'{value:.3f}' if value is not None else 'n/a'

    pre = report.judge_prechecks
    lines += [
        f'  mean success_rate: {fmt(report.mean_success_rate)}',
        f'  mean initial_attempt: {fmt(report.mean_initial_attempt_success)}',
        f'  mean any_pass_5: {fmt(report.mean_any_pass_5)}',
        f'  mean all_pass_5: {fmt(report.mean_all_pass_5)}',
        f'  grader_flip_rate: {fmt(report.grader_flip_rate)} {report.flipped_tasks}',
        f'  later_trial_recovery: {report.later_trial_recovery_fraction} '
        f"({fmt(report.later_trial_recovery_rate)})",
        f'  judge prechecks: {pre.deterministic_precheck_passes}/'
        f'{pre.deterministic_precheck_total} sub-pass, '
        f'{pre.judge_deferrals} deferrals',
        '  per-task:',
    ]
    for task_id in sorted(report.deterministic_tasks):
        task = report.deterministic_tasks[task_id]
        lines.append(
            f"    {task_id}: sr={fmt(task.success_rate)} "
            f"any5={task.any_pass_5} all5={task.all_pass_5} "
            f"flip={task.grader_flip} raw_n={task.unique_raw_output_count} "
            f"norm_n={task.unique_normalized_output_count}"
        )
    for task_id in sorted(report.judge_tasks):
        task = report.judge_tasks[task_id]
        lines.append(
            f'    {task_id}: JUDGE deferrals={task.judge_deferrals} '
            f'det_sub={fmt(task.deterministic_sub_pass_rate)} '
            f'raw_n={task.unique_raw_output_count} (EXCLUDED from denominators)'
        )
    return '\n'.join(lines)
