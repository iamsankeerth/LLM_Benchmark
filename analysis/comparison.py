"""Paired model comparison across two executions of one frozen contract.

Primary task outcome (strict, locked): PASS iff all trials PASS.
Secondary: ANY_PASS iff >=1 trial PASS. The two notions are never mixed.

Transition cells use strict outcomes over READY_DETERMINISTIC tasks present
in both executions. net_task_gain = b - c. The exact McNemar p-value on
discordant pairs is supporting texture (n=68 caveat), never the headline:
transition IDs carry more weight than the p-value.

"Grader-family performance" (structured/exact/constraint/numeric) uses
overlapping denominators by design; footnoted, never called a class split.
"""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import statistics
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

from analysis.reliability import PASS
from evals.graders.engine import normalize_text

# Presentation-only aliases for legacy single-ID executions. Display layer
# only: stored IDs, manifests, DB rows and historical summaries never change.
LEGACY_DISPLAY_ALIASES = {
    'full-baseline-v1': 'full-baseline-v1__qwen3-4b-q4',
}


def display_execution_id(execution_id: str) -> str:
    return LEGACY_DISPLAY_ALIASES.get(execution_id, execution_id)


@dataclass
class ComparisonTrial:
    task_id: str
    trial: int
    verdict: str
    raw_output: str
    decode_tok_s: float | None
    ttft_ms: float | None
    eval_count: int | None
    primary_class: str
    suite: str
    difficulty: str


@dataclass
class TransitionMatrix:
    pass_to_pass: list[str] = field(default_factory=list)
    fail_to_fail: list[str] = field(default_factory=list)
    fail_to_pass: list[str] = field(default_factory=list)  # b
    pass_to_fail: list[str] = field(default_factory=list)  # c

    @property
    def b(self) -> int:
        return len(self.fail_to_pass)

    @property
    def c(self) -> int:
        return len(self.pass_to_fail)

    @property
    def net_task_gain(self) -> int:
        return self.b - self.c


@dataclass
class PerformanceSummary:
    n: int
    median: float | None
    p25: float | None
    p75: float | None
    p95: float | None
    mean: float | None  # diagnostic only, never the headline


@dataclass
class ModelSide:
    execution_id: str
    display_id: str
    deterministic_tasks: int
    deterministic_trials: int
    passes: int
    trial_accuracy: float | None
    tasks_all_pass: int
    tasks_any_pass: int
    flips: list[str]
    decode: PerformanceSummary
    ttft: PerformanceSummary
    decode_by_token_bucket: dict[str, PerformanceSummary]
    peak_vram_mib: float | None
    model_size_bytes: int | None
    model_artifact_digest: str | None


@dataclass
class ComparisonReport:
    experiment_spec_id: str
    base_execution_id: str
    candidate_execution_id: str
    base: ModelSide
    candidate: ModelSide
    trial_accuracy_delta: float | None
    all_pass_delta: int
    any_pass_delta: int
    transitions_strict: TransitionMatrix
    transitions_any_pass: TransitionMatrix
    mcnemar_b: int
    mcnemar_c: int
    mcnemar_n: int
    mcnemar_exact_p: float
    grader_family: dict[str, dict[str, Any]]
    suite_deltas: dict[str, dict[str, Any]]
    difficulty_deltas: dict[str, dict[str, Any]]
    systematic_fail_both: list[str]
    mean_unique_normalized_outputs_base: float | None
    mean_unique_normalized_outputs_candidate: float | None


def _nearest_rank(sorted_vals: list[float], probability: float) -> float | None:
    if not sorted_vals:
        return None
    index = math.ceil(probability * len(sorted_vals)) - 1
    return sorted_vals[max(0, min(index, len(sorted_vals) - 1))]


def summarize_performance(values: list[float | None]) -> PerformanceSummary:
    observed = sorted(v for v in values if v is not None)
    n = len(observed)
    if not observed:
        return PerformanceSummary(n=0, median=None, p25=None, p75=None, p95=None, mean=None)
    quartiles = statistics.quantiles(observed, n=4) if n >= 4 else [None, None, None]
    return PerformanceSummary(
        n=n,
        median=statistics.median(observed),
        p25=quartiles[0],
        p75=quartiles[2],
        p95=_nearest_rank(observed, 0.95),
        mean=statistics.fmean(observed),
    )


TOKEN_BUCKETS = ('1-4', '5-32', '33-128', '129+')


def _bucket(eval_count: int | None) -> str | None:
    if eval_count is None:
        return None
    if eval_count <= 4:
        return '1-4'
    if eval_count <= 32:
        return '5-32'
    if eval_count <= 128:
        return '33-128'
    return '129+'


def mcnemar_exact_p(b: int, c: int) -> float:
    """Exact two-sided McNemar on discordant pairs; n==0 -> 1.0, capped 1.0."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1))
    return min(1.0, float(2.0 * tail / 2**n))


def _strict_outcomes(
    grouped: dict[str, list[ComparisonTrial]],
) -> dict[str, bool]:
    return {
        task_id: all(t.verdict == PASS for t in trials)
        for task_id, trials in grouped.items()
    }


def _any_outcomes(
    grouped: dict[str, list[ComparisonTrial]],
) -> dict[str, bool]:
    return {
        task_id: any(t.verdict == PASS for t in trials)
        for task_id, trials in grouped.items()
    }


def _build_matrix(
    base_out: dict[str, bool], cand_out: dict[str, bool]
) -> TransitionMatrix:
    matrix = TransitionMatrix()
    for task_id in sorted(set(base_out) & set(cand_out)):
        cell = (base_out[task_id], cand_out[task_id])
        if cell == (True, True):
            matrix.pass_to_pass.append(task_id)
        elif cell == (False, False):
            matrix.fail_to_fail.append(task_id)
        elif cell == (False, True):
            matrix.fail_to_pass.append(task_id)
        else:
            matrix.pass_to_fail.append(task_id)
    return matrix


def _side(
    execution_id: str,
    grouped: dict[str, list[ComparisonTrial]],
    *,
    size_bytes: int | None,
    artifact_digest: str | None,
    vram_peaks: list[float | None],
) -> ModelSide:
    trials = [t for trials in grouped.values() for t in trials]
    passes = sum(1 for t in trials if t.verdict == PASS)
    strict = _strict_outcomes(grouped)
    flips = sorted(
        task_id
        for task_id, ts in grouped.items()
        if len({t.verdict for t in ts}) > 1
    )
    buckets: dict[str, list[float | None]] = {b: [] for b in TOKEN_BUCKETS}
    for t in trials:
        bucket = _bucket(t.eval_count)
        if bucket is not None:
            buckets[bucket].append(t.decode_tok_s)
    peaks = [v for v in vram_peaks if v is not None]
    return ModelSide(
        execution_id=execution_id,
        display_id=display_execution_id(execution_id),
        deterministic_tasks=len(grouped),
        deterministic_trials=len(trials),
        passes=passes,
        trial_accuracy=(passes / len(trials) if trials else None),
        tasks_all_pass=sum(1 for v in strict.values() if v),
        tasks_any_pass=sum(1 for task_id in grouped if any(
            t.verdict == PASS for t in grouped[task_id])),
        flips=flips,
        decode=summarize_performance([t.decode_tok_s for t in trials]),
        ttft=summarize_performance([t.ttft_ms for t in trials]),
        decode_by_token_bucket={
            bucket: summarize_performance(vals) for bucket, vals in buckets.items()
        },
        peak_vram_mib=(max(peaks) if peaks else None),
        model_size_bytes=size_bytes,
        model_artifact_digest=artifact_digest,
    )


def _family_table(
    base_grouped: dict[str, list[ComparisonTrial]],
    cand_grouped: dict[str, list[ComparisonTrial]],
) -> dict[str, dict[str, Any]]:
    table: dict[str, dict[str, Any]] = {}
    families = sorted(
        {t.primary_class for trials in base_grouped.values() for t in trials}
        | {t.primary_class for trials in cand_grouped.values() for t in trials}
    )
    for family in families:
        base_trials = [t for trials in base_grouped.values() for t in trials
                       if t.primary_class == family]
        cand_trials = [t for trials in cand_grouped.values() for t in trials
                       if t.primary_class == family]
        base_rate = (
            sum(1 for t in base_trials if t.verdict == PASS) / len(base_trials)
            if base_trials else None
        )
        cand_rate = (
            sum(1 for t in cand_trials if t.verdict == PASS) / len(cand_trials)
            if cand_trials else None
        )
        table[family] = {
            'note': 'overlapping denominators by design; not a class split',
            'base_trials': len(base_trials),
            'candidate_trials': len(cand_trials),
            'base_rate': base_rate,
            'candidate_rate': cand_rate,
            'delta': (cand_rate - base_rate
                      if base_rate is not None and cand_rate is not None else None),
        }
    return table


def _strata_table(
    base_grouped: dict[str, list[ComparisonTrial]],
    cand_grouped: dict[str, list[ComparisonTrial]],
    key: str,
) -> dict[str, dict[str, Any]]:
    table: dict[str, dict[str, Any]] = {}
    strata = sorted(
        {getattr(t, key) for trials in base_grouped.values() for t in trials}
        | {getattr(t, key) for trials in cand_grouped.values() for t in trials}
    )
    for stratum in strata:
        base_trials = [t for trials in base_grouped.values() for t in trials
                       if getattr(t, key) == stratum]
        cand_trials = [t for trials in cand_grouped.values() for t in trials
                       if getattr(t, key) == stratum]
        base_rate = (
            sum(1 for t in base_trials if t.verdict == PASS) / len(base_trials)
            if base_trials else None
        )
        cand_rate = (
            sum(1 for t in cand_trials if t.verdict == PASS) / len(cand_trials)
            if cand_trials else None
        )
        table[stratum] = {
            'base_rate': base_rate, 'candidate_rate': cand_rate,
            'delta': (cand_rate - base_rate
                      if base_rate is not None and cand_rate is not None else None),
        }
    return table


def _mean_unique_normalized(grouped: dict[str, list[ComparisonTrial]]) -> float | None:
    if not grouped:
        return None
    counts = [
        len({
            hashlib.sha256(normalize_text(t.raw_output).encode('utf-8')).hexdigest()
            for t in trials
        })
        for trials in grouped.values()
    ]
    return statistics.fmean(counts)


def compare_executions(
    *,
    experiment_spec_id: str,
    base_execution_id: str,
    candidate_execution_id: str,
    base_grouped: dict[str, list[ComparisonTrial]],
    cand_grouped: dict[str, list[ComparisonTrial]],
    base_size_bytes: int | None = None,
    cand_size_bytes: int | None = None,
    base_artifact_digest: str | None = None,
    cand_artifact_digest: str | None = None,
    base_vram_peaks: list[float | None] | None = None,
    cand_vram_peaks: list[float | None] | None = None,
) -> ComparisonReport:
    """Paired comparison over READY_DETERMINISTIC tasks present in both."""
    common = sorted(set(base_grouped) & set(cand_grouped))
    base_common = {t: base_grouped[t] for t in common}
    cand_common = {t: cand_grouped[t] for t in common}
    base_side = _side(
        base_execution_id, base_common, size_bytes=base_size_bytes,
        artifact_digest=base_artifact_digest,
        vram_peaks=base_vram_peaks or [],
    )
    cand_side = _side(
        candidate_execution_id, cand_common, size_bytes=cand_size_bytes,
        artifact_digest=cand_artifact_digest,
        vram_peaks=cand_vram_peaks or [],
    )
    strict = _build_matrix(_strict_outcomes(base_common), _strict_outcomes(cand_common))
    any_pass = _build_matrix(_any_outcomes(base_common), _any_outcomes(cand_common))
    systematic = sorted(
        t for t in common
        if all(x.verdict != PASS for x in base_common[t] + cand_common[t])
    )
    acc_delta = None
    if base_side.trial_accuracy is not None and cand_side.trial_accuracy is not None:
        acc_delta = cand_side.trial_accuracy - base_side.trial_accuracy
    return ComparisonReport(
        experiment_spec_id=experiment_spec_id,
        base_execution_id=base_execution_id,
        candidate_execution_id=candidate_execution_id,
        base=base_side,
        candidate=cand_side,
        trial_accuracy_delta=acc_delta,
        all_pass_delta=cand_side.tasks_all_pass - base_side.tasks_all_pass,
        any_pass_delta=cand_side.tasks_any_pass - base_side.tasks_any_pass,
        transitions_strict=strict,
        transitions_any_pass=any_pass,
        mcnemar_b=strict.b,
        mcnemar_c=strict.c,
        mcnemar_n=strict.b + strict.c,
        mcnemar_exact_p=mcnemar_exact_p(strict.b, strict.c),
        grader_family=_family_table(base_common, cand_common),
        suite_deltas=_strata_table(base_common, cand_common, 'suite'),
        difficulty_deltas=_strata_table(base_common, cand_common, 'difficulty'),
        systematic_fail_both=systematic,
        mean_unique_normalized_outputs_base=_mean_unique_normalized(base_common),
        mean_unique_normalized_outputs_candidate=_mean_unique_normalized(cand_common),
    )


def load_comparison_trials(
    conn: sqlite3.Connection,
    execution_id: str,
    task_meta: Mapping[str, Mapping[str, str]],
) -> dict[str, list[ComparisonTrial]]:
    """Load READY_DETERMINISTIC measured trials with task metadata join."""
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            'SELECT task_id, trial, grader_verdict, raw_output, decode_tok_s,'
            ' ttft_ms, eval_count FROM runs WHERE experiment_id=? AND is_warmup=0'
            ' ORDER BY task_id, trial',
            (execution_id,),
        ).fetchall()
    finally:
        conn.row_factory = None
    grouped: dict[str, list[ComparisonTrial]] = {}
    for row in rows:
        task_id = str(row['task_id'])
        meta = task_meta.get(task_id)
        if meta is None or meta.get('grading_status') != 'READY_DETERMINISTIC':
            continue
        grouped.setdefault(task_id, []).append(
            ComparisonTrial(
                task_id=task_id, trial=int(row['trial']),
                verdict=str(row['grader_verdict']),
                raw_output=str(row['raw_output'] or ''),
                decode_tok_s=row['decode_tok_s'], ttft_ms=row['ttft_ms'],
                eval_count=row['eval_count'],
                primary_class=str(meta.get('primary_class', '')),
                suite=str(meta.get('suite', '')),
                difficulty=str(meta.get('difficulty', '')),
            )
        )
    return grouped


def load_execution_evidence(
    conn: sqlite3.Connection, execution_id: str
) -> tuple[int | None, str | None, list[float | None]]:
    """Model size bytes, artifact digest, VRAM peaks from persisted rows."""
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            'SELECT eligibility_evidence_json, model_digest, vram_peak_mib'
            ' FROM runs WHERE experiment_id=? AND is_warmup=0',
            (execution_id,),
        ).fetchall()
    finally:
        conn.row_factory = None
    size: int | None = None
    digest: str | None = None
    peaks: list[float | None] = []
    for row in rows:
        peaks.append(row['vram_peak_mib'])
        if digest is None and row['model_digest']:
            digest = str(row['model_digest'])
        if size is None and row['eligibility_evidence_json']:
            try:
                evidence = json.loads(row['eligibility_evidence_json'])
            except ValueError:
                continue
            raw_size = evidence.get('size')
            if isinstance(raw_size, int) and raw_size > 0:
                size = raw_size
    return size, digest, peaks


def comparison_to_json(report: ComparisonReport) -> str:
    def encode(value: Any) -> Any:
        if hasattr(value, '__dataclass_fields__'):
            return {k: encode(v) for k, v in asdict(value).items()}
        if isinstance(value, dict):
            return {str(k): encode(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [encode(v) for v in value]
        return value

    return json.dumps(encode(report), indent=2, sort_keys=True)


def comparison_to_console(report: ComparisonReport) -> str:
    def fmt(value: float | None, digits: int = 3) -> str:
        return f'{value:.{digits}f}' if value is not None else 'n/a'

    m = report.transitions_strict
    lines = [
        f'Paired comparison: {report.base.display_id}  vs  {report.candidate.display_id}',
        f'  deterministic tasks in common: {report.base.deterministic_tasks}',
        f'  trial accuracy: {fmt(report.base.trial_accuracy)} -> '
        f'{fmt(report.candidate.trial_accuracy)} '
        f'(delta {fmt(report.trial_accuracy_delta, 4)})',
        f'  all_pass_3: {report.base.tasks_all_pass} -> {report.candidate.tasks_all_pass} '
        f'(delta {report.all_pass_delta:+d})',
        f'  any_pass_3: {report.base.tasks_any_pass} -> {report.candidate.tasks_any_pass} '
        f'(delta {report.any_pass_delta:+d})',
        '  transition matrix (strict all_pass_3):',
        f'    PASS->PASS {len(m.pass_to_pass)}   FAIL->FAIL {len(m.fail_to_fail)}',
        f'    FAIL->PASS {m.b} {m.fail_to_pass}   PASS->FAIL {m.c} {m.pass_to_fail}',
        f'    net_task_gain: {m.net_task_gain:+d}',
        f'    McNemar supporting only: b={report.mcnemar_b} c={report.mcnemar_c} '
        f'n={report.mcnemar_n} p={report.mcnemar_exact_p:.4f} (n=68 caveat)',
        '  grader-family performance (overlapping denominators):',
    ]
    for family in sorted(report.grader_family):
        row = report.grader_family[family]
        lines.append(
            f'    {family}: {fmt(row["base_rate"])} -> {fmt(row["candidate_rate"])} '
            f'(delta {fmt(row["delta"], 4)})'
        )
    for label, table in (('suite', report.suite_deltas), ('difficulty', report.difficulty_deltas)):
        lines.append(f'  {label} deltas:')
        for key in sorted(table):
            lines.append(
                f'    {key}: {fmt(table[key]["base_rate"])} -> '
                f'{fmt(table[key]["candidate_rate"])} '
                f'(delta {fmt(table[key]["delta"], 4)})'
            )
    for label, side in (('base', report.base), ('candidate', report.candidate)):
        lines.append(
            f'  {label} decode tok/s: median {fmt(side.decode.median, 1)} '
            f'P25-P75 {fmt(side.decode.p25, 1)}-{fmt(side.decode.p75, 1)} '
            f'P95 {fmt(side.decode.p95, 1)} (mean {fmt(side.decode.mean, 1)} diagnostic)'
        )
        lines.append(
            f'  {label} TTFT ms: median {fmt(side.ttft.median, 1)} '
            f'P95 {fmt(side.ttft.p95, 1)}'
        )
        buckets = ', '.join(
            f'{b}: {fmt(side.decode_by_token_bucket[b].median, 1)}'
            for b in ('1-4', '5-32', '33-128', '129+')
        )
        lines.append(f'  {label} decode by tokens [{buckets}]')
        lines.append(
            f'  {label} peak VRAM MiB: {fmt(side.peak_vram_mib, 0)} '
            f'model size bytes: {side.model_size_bytes}'
        )
    lines.append(
        f'  systematic fail both: {len(report.systematic_fail_both)} '
        f'{report.systematic_fail_both[:12]}'
    )
    lines.append(
        f'  flips: base {report.base.flips} candidate {report.candidate.flips}'
    )
    return '\n'.join(lines)
