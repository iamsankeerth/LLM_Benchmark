"""Status-driven grading verdict reducer.

The runner never branches on task IDs. Given the frozen spec's
``grading_status`` and the deterministic ``grade_output`` results, the
verdict is exactly one of:

- PENDING_SPECIFICATION / PENDING_REVIEW -> REFUSED_NOT_EXECUTABLE
- any deterministic grader fails                -> FAIL
- all deterministic pass + any rubric_judge     -> NEEDS_JUDGE
- all deterministic pass + no judge required    -> PASS

Generation/runtime errors map to ERROR at the call site (no grader ran).
``rubric_judge`` results are deferral markers: their ``passed=True`` can
never produce PASS, including for mixed deterministic+judge tasks.
"""

from __future__ import annotations

from evals.graders.engine import GraderResult

REFUSED_NOT_EXECUTABLE = 'REFUSED_NOT_EXECUTABLE'
ERROR = 'ERROR'
FAIL = 'FAIL'
NEEDS_JUDGE = 'NEEDS_JUDGE'
PASS = 'PASS'

REFUSED_STATUSES = frozenset({'PENDING_SPECIFICATION', 'PENDING_REVIEW'})


def reduce_verdict(grading_status: str, results: list[GraderResult]) -> str:
    """Reduce spec status + grader results to a single verdict string."""
    if grading_status in REFUSED_STATUSES:
        return REFUSED_NOT_EXECUTABLE
    if not results:
        raise ValueError(f'executable task {grading_status=} produced no grader results')
    deterministic = [r for r in results if r.grader_type != 'rubric_judge']
    if any(not r.passed for r in deterministic):
        return FAIL
    if any(r.grader_type == 'rubric_judge' for r in results):
        return NEEDS_JUDGE
    return PASS
