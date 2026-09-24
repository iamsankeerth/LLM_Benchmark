"""Isolated OCI coding-worker contract; host execution is intentionally unavailable."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from typing import Any, Callable, Mapping


class SandboxUnavailable(RuntimeError):
    """The pinned isolated worker cannot be used safely."""


@dataclass(frozen=True)
class SandboxRequest:
    language: str
    entrypoint: str
    candidate_source: str
    test_source: str
    cases_json: str
    image_ref: str
    image_digest: str | None
    policy: Mapping[str, Any]


def run_isolated_tests(
    request: SandboxRequest,
    *,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> dict[str, Any]:
    """Run a candidate only through an explicit OCI worker command.

    This function never invokes Python, Node, a shell, or a subprocess on the
    host with candidate source. The worker orchestration is supplied by the
    isolated runner and must enforce the policy independently.
    """
    if not request.image_digest:
        raise SandboxUnavailable('OCI image digest is not pinned')
    if request.image_ref == '':
        raise SandboxUnavailable('OCI image reference is empty')
    command = [
        request.image_ref,
        'python', '-m', 'coding_worker',
        '--language', request.language,
        '--entrypoint', request.entrypoint,
    ]
    completed = runner(
        command,
        input=json.dumps({
            'candidate_source': request.candidate_source,
            'test_source': request.test_source,
            'cases_json': request.cases_json,
            'policy': dict(request.policy),
        }),
        capture_output=True,
        text=True,
        timeout=float(request.policy['whole_candidate_wall_seconds']) + 30.0,
        check=False,
    )
    if completed.returncode != 0:
        return {
            'status': 'SANDBOX_ERROR',
            'returncode': completed.returncode,
            'stdout': completed.stdout,
            'stderr': completed.stderr,
        }
    try:
        result = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise SandboxUnavailable('OCI worker returned invalid JSON') from exc
    if not isinstance(result, dict):
        raise SandboxUnavailable('OCI worker result is not an object')
    return result
