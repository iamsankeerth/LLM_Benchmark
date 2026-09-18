"""Sweep lifecycle primitives: Ollama ops, disk, state, verify, delete gate.

All subprocess access flows through an injectable runner for offline tests.
Live Ollama is never touched by the unit suite.
"""

from __future__ import annotations

import json
import shutil
import sqlite3
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Mapping, Sequence

# Lifecycle states (per model). -ING = in progress (crash: restart stage);
# bare names = stage complete (resume continues from the next stage).
PENDING = 'PENDING'
DOWNLOADING = 'DOWNLOADING'
DOWNLOAD_FAILED = 'DOWNLOAD_FAILED'
PULLED = 'PULLED'
DERIVING = 'DERIVING'
MANUAL_PIN_REQUIRED = 'MANUAL_PIN_REQUIRED'
DERIVED = 'DERIVED'
ADAPTER_MISMATCH = 'ADAPTER_MISMATCH'
ELIGIBLE = 'ELIGIBLE'
PREFLIGHTING = 'PREFLIGHTING'
PREFLIGHT_FAILED = 'PREFLIGHT_FAILED'
PREFLIGHTED = 'PREFLIGHTED'
SMOKING = 'SMOKING'
SMOKED = 'SMOKED'
WARMING_UP = 'WARMING_UP'
BENCHMARKING = 'BENCHMARKING'
BENCHMARK_ERROR = 'BENCHMARK_ERROR'
BENCHMARKED = 'BENCHMARKED'
VALIDATED = 'VALIDATED'
SUMMARIZED = 'SUMMARIZED'
VERIFYING = 'VERIFYING'
VERIFY_FAILED = 'VERIFY_FAILED'
COMPLETE = 'COMPLETE'
COMPLETE_INELIGIBLE = 'COMPLETE_INELIGIBLE'

# Ordered stages for --stop-after and resume mapping.
STAGES = (
    'pulled', 'derived', 'eligible', 'preflighted', 'smoked',
    'benchmarked', 'validated', 'summarized', 'verified',
    'unloaded', 'deleted', 'complete',
)

# Lifecycle value recorded when --stop-after halts after a stage.
STOPPED_STATE = {
    'pulled': PULLED,
    'derived': DERIVED,
    'eligible': ELIGIBLE,
    'preflighted': PREFLIGHTED,
    'smoked': SMOKED,
    'benchmarked': BENCHMARKED,
    'validated': VALIDATED,
    'summarized': SUMMARIZED,
    'verified': 'VERIFIED',
    'unloaded': 'UNLOADED',
    'deleted': 'DELETED',
    'complete': COMPLETE,
}

# Resume: lifecycle -> first stage to (re)run. Pull/show are cheap and
# idempotent, so resume re-enters through them; warmups re-run every
# invocation by the runner's session rule (see run_benchmark).
RESTART_STAGE = {
    PENDING: 'pulled',
    DOWNLOADING: 'pulled',
    DOWNLOAD_FAILED: 'pulled',
    PULLED: 'derived',
    DERIVING: 'derived',
    MANUAL_PIN_REQUIRED: 'derived',
    DERIVED: 'eligible',
    ADAPTER_MISMATCH: 'derived',
    ELIGIBLE: 'preflighted',
    PREFLIGHTING: 'preflighted',
    PREFLIGHT_FAILED: 'preflighted',
    PREFLIGHTED: 'smoked',
    SMOKING: 'smoked',
    SMOKED: 'benchmarked',
    WARMING_UP: 'benchmarked',
    BENCHMARKING: 'benchmarked',
    BENCHMARK_ERROR: 'benchmarked',
    BENCHMARKED: 'validated',
    VALIDATED: 'summarized',
    SUMMARIZED: 'verified',
    VERIFYING: 'verified',
    VERIFY_FAILED: 'verified',
    'UNLOADED': 'deleted',
    'DELETED': 'complete',
}

TERMINAL_OK = frozenset({COMPLETE, COMPLETE_INELIGIBLE})


@dataclass
class ModelLifecycle:
    model_config_id: str
    lifecycle: str = PENDING
    execution_id: str | None = None
    detail: str = ''
    verified: bool = False


@dataclass
class SweepState:
    sweep_id: str
    order: list[str] = field(default_factory=list)
    models: dict[str, ModelLifecycle] = field(default_factory=dict)
    ineligible: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)

    @property
    def completed(self) -> list[str]:
        return [mid for mid in self.order if self.models[mid].lifecycle == COMPLETE]

    @property
    def next_model(self) -> str | None:
        for mid in self.order:
            if self.models[mid].lifecycle not in TERMINAL_OK:
                return mid
        return None


def new_sweep_state(sweep_id: str, order: Sequence[str]) -> SweepState:
    return SweepState(
        sweep_id=sweep_id,
        order=list(order),
        models={mid: ModelLifecycle(mid) for mid in order},
    )


def save_sweep_state(path: str | Path, state: SweepState) -> None:
    document = {
        'sweep_id': state.sweep_id,
        'order': state.order,
        'models': {
            mid: asdict(lc) for mid, lc in state.models.items()
        },
        'ineligible': state.ineligible,
        'failed': state.failed,
    }
    file = Path(path)
    file.parent.mkdir(parents=True, exist_ok=True)
    tmp = file.with_suffix('.tmp')
    tmp.write_text(json.dumps(document, indent=2), encoding='utf-8')
    tmp.replace(file)


def load_sweep_state(path: str | Path) -> SweepState | None:
    file = Path(path)
    if not file.exists():
        return None
    document = json.loads(file.read_text(encoding='utf-8'))
    state = SweepState(
        sweep_id=str(document['sweep_id']),
        order=[str(m) for m in document['order']],
        ineligible=[str(m) for m in document.get('ineligible', [])],
        failed=[str(m) for m in document.get('failed', [])],
    )
    for mid, raw in document.get('models', {}).items():
        if isinstance(raw, dict):
            state.models[str(mid)] = ModelLifecycle(
                model_config_id=str(raw.get('model_config_id', mid)),
                lifecycle=str(raw.get('lifecycle', PENDING)),
                execution_id=raw.get('execution_id'),
                detail=str(raw.get('detail', '')),
                verified=bool(raw.get('verified', False)),
            )
    return state


RunFn = Callable[..., subprocess.CompletedProcess[str]]


def default_run(
    argv: Sequence[str], *, timeout_s: float = 3600.0
) -> subprocess.CompletedProcess[str]:
    # Explicit UTF-8 with replacement: ollama progress output breaks the
    # Windows locale decoder (cp1252) inside subprocess reader threads.
    return subprocess.run(
        list(argv), capture_output=True, encoding='utf-8',
        errors='replace', timeout=timeout_s,
    )


def _output(proc: subprocess.CompletedProcess[str]) -> str:
    return ((proc.stdout or '') + '\n' + (proc.stderr or '')).strip()


def ollama_pull(
    ollama_bin: str, identifier: str, run: RunFn = default_run
) -> tuple[bool, str]:
    proc = run([ollama_bin, 'pull', identifier], timeout_s=7200.0)
    output = _output(proc)
    if proc.returncode != 0:
        return False, f'DOWNLOAD_FAILED: {output[-500:]}'
    return True, 'pulled'


def ollama_stop(
    ollama_bin: str, identifier: str, run: RunFn = default_run
) -> tuple[bool, str]:
    proc = run([ollama_bin, 'stop', identifier], timeout_s=300.0)
    output = _output(proc)
    if proc.returncode != 0:
        return False, f'stop failed: {output[-500:]}'
    return True, 'stopped'


def ollama_remove(
    ollama_bin: str, identifier: str, run: RunFn = default_run
) -> tuple[bool, str]:
    proc = run([ollama_bin, 'rm', identifier], timeout_s=600.0)
    output = _output(proc)
    if proc.returncode != 0:
        return False, f'remove failed: {output[-500:]}'
    return True, 'removed'


def ollama_model_present(
    ollama_bin: str, identifier: str, run: RunFn = default_run
) -> bool:
    proc = run([ollama_bin, 'list'], timeout_s=120.0)
    if proc.returncode != 0:
        return True  # fail closed: assume present, do not advance blindly
    return identifier in (proc.stdout or '')


def disk_free_bytes(path: str | Path) -> int:
    return shutil.disk_usage(str(path)).free


def disk_reclaimed_ok(free_before_download: int, free_after_delete: int) -> bool:
    """Deletion verified when post-delete free space recovers to within
    1 GiB of the pre-download level (log/overhead slack)."""
    return free_after_delete >= free_before_download - (1024**3)


def set_lifecycle(
    state: SweepState, model_config_id: str, lifecycle: str, detail: str = ''
) -> None:
    state.models[model_config_id].lifecycle = lifecycle
    if detail:
        state.models[model_config_id].detail = detail
    if lifecycle == COMPLETE_INELIGIBLE and model_config_id not in state.ineligible:
        state.ineligible.append(model_config_id)
    if lifecycle in (DOWNLOAD_FAILED, BENCHMARK_ERROR, VERIFY_FAILED,
                     MANUAL_PIN_REQUIRED, ADAPTER_MISMATCH, PREFLIGHT_FAILED):
        if model_config_id not in state.failed:
            state.failed.append(model_config_id)
    if lifecycle in TERMINAL_OK and model_config_id in state.failed:
        state.failed.remove(model_config_id)


@dataclass
class VerificationResult:
    ok: bool
    checks: dict[str, bool]
    detail: str = ''


def verify_execution_persisted(
    db_path: str | Path,
    execution_id: str,
    statuses: Mapping[str, str],
    *,
    expected_det_tasks: int,
    expected_judge_tasks: int,
    trials_per_task: int,
    required_files: Mapping[str, str | Path],
) -> VerificationResult:
    """Persistence gate: every artifact + row count proven before deletion.

    required_files maps check-name -> path that must exist and be non-empty.
    """
    checks: dict[str, bool] = {}
    try:
        conn = sqlite3.connect(str(db_path))
    except sqlite3.Error as exc:
        return VerificationResult(False, {'db_open': False}, str(exc))
    try:
        conn.row_factory = sqlite3.Row
        try:
            measured = conn.execute(
                'SELECT task_id, trial, grader_verdict, status FROM runs'
                ' WHERE experiment_id=? AND is_warmup=0',
                (execution_id,),
            ).fetchall()
        finally:
            conn.row_factory = None
    except sqlite3.Error as exc:
        conn.close()
        return VerificationResult(False, {'db_read': False}, str(exc))
    conn.close()
    det_ids = {t for t, s in statuses.items() if s == 'READY_DETERMINISTIC'}
    judge_ids = {t for t, s in statuses.items() if s == 'READY_JUDGE'}
    det_rows = [r for r in measured if str(r['task_id']) in det_ids]
    judge_rows = [r for r in measured if str(r['task_id']) in judge_ids]
    unknown_rows = [
        r for r in measured
        if str(r['task_id']) not in statuses
    ]
    checks['det_rows'] = len(det_rows) == expected_det_tasks * trials_per_task
    checks['judge_rows'] = len(judge_rows) == expected_judge_tasks * trials_per_task
    checks['unknown_rows_zero'] = not unknown_rows
    identities = [(str(r['task_id']), int(r['trial'])) for r in measured]
    checks['no_duplicate_identities'] = len(identities) == len(set(identities))
    checks['no_error_rows'] = all(
        str(r['status']) != 'ERROR' and str(r['grader_verdict']) != 'ERROR'
        for r in measured
    )
    for name, file_path in required_files.items():
        candidate = Path(str(file_path))
        checks[f'file:{name}'] = candidate.is_file() and candidate.stat().st_size > 0
    ok = all(checks.values())
    missing = sorted(k for k, v in checks.items() if not v)
    return VerificationResult(
        ok, checks,
        '' if ok else f'verification failed: {missing}',
    )


def guard_deletion(verified: bool) -> None:
    """Deletion is forbidden until persistence verification succeeds."""
    if not verified:
        raise RuntimeError(
            'Refusing model deletion: benchmark persistence is not verified.'
        )


def wait_until_unloaded(
    is_absent: Callable[[], bool], *, timeout_s: float = 180.0,
    poll_s: float = 5.0,
) -> bool:
    """Poll until the model leaves /api/ps (or equivalent absence check)."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if is_absent():
            return True
        time.sleep(poll_s)
    return is_absent()
