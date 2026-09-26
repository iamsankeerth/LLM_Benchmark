"""Isolated Docker coding-worker contract; host execution is unavailable."""

from __future__ import annotations

import json
import os
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


def _run_docker(
    command: list[str],
    *,
    runner: Callable[..., subprocess.CompletedProcess[str]],
    input_text: str | None = None,
    timeout_s: float = 60.0,
) -> subprocess.CompletedProcess[str]:
    try:
        return runner(
            command,
            input=input_text,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise SandboxUnavailable(f'Docker invocation failed: {type(exc).__name__}') from exc


def _verify_image(
    request: SandboxRequest,
    *,
    runner: Callable[..., subprocess.CompletedProcess[str]],
    docker_bin: str,
) -> None:
    if not request.image_digest:
        raise SandboxUnavailable('OCI image digest is not pinned')
    if not request.image_ref:
        raise SandboxUnavailable('OCI image reference is empty')
    inspected = _run_docker(
        [docker_bin, 'image', 'inspect', '--format', '{{.Id}}', request.image_ref],
        runner=runner,
        timeout_s=30.0,
    )
    if inspected.returncode != 0:
        raise SandboxUnavailable('OCI image cannot be inspected')
    observed = inspected.stdout.strip()
    expected = request.image_digest.removeprefix('sha256:')
    observed_id = observed.removeprefix('sha256:')
    if observed_id != expected:
        raise SandboxUnavailable('OCI image digest does not match the pinned digest')


def _docker_command(
    request: SandboxRequest,
    *,
    docker_bin: str,
) -> list[str]:
    policy = request.policy
    return [
        docker_bin, 'run', '--rm', '-i',
        '--network', 'none',
        '--read-only',
        '--cap-drop', 'ALL',
        '--security-opt', 'no-new-privileges',
        '--pids-limit', str(int(policy['processes'])),
        '--memory', f"{int(policy['memory_mib'])}m",
        '--cpus', str(policy.get('cpus', 1)),
        '--ulimit', f"nofile={int(policy['file_descriptors'])}:{int(policy['file_descriptors'])}",
        '--tmpfs', f"/tmp:rw,noexec,nosuid,size={int(policy['tmpfs_mib'])}m",
        '--user', '65532:65532',
        '--env', 'HOME=/tmp',
        '--env', 'PYTHONDONTWRITEBYTECODE=1',
        request.image_ref,
    ]


def run_isolated_tests(
    request: SandboxRequest,
    *,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    docker_bin: str | None = None,
) -> dict[str, Any]:
    """Run a candidate only inside a pinned Docker worker."""
    binary = docker_bin or os.environ.get('CODING_DOCKER_BIN', 'docker')
    _verify_image(request, runner=runner, docker_bin=binary)
    policy = request.policy
    payload = json.dumps({
        'candidate_source': request.candidate_source,
        'test_source': request.test_source,
        'cases_json': request.cases_json,
        'language': request.language,
        'entrypoint': request.entrypoint,
        'policy': dict(policy),
    })
    try:
        completed = _run_docker(
            _docker_command(request, docker_bin=binary),
            runner=runner,
            input_text=payload,
            timeout_s=float(policy['whole_candidate_wall_seconds']) + 30.0,
        )
    except SandboxUnavailable:
        raise
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
        raise SandboxUnavailable('OCI worker result was not an object')
    if not isinstance(result.get('status'), str):
        failure_kind = result.get('failure_kind')
        tests = result.get('tests')
        total_tests = (
            int(tests.get('total', 0))
            if isinstance(tests, dict) else 0
        )
        if failure_kind == 'WORKER_ERROR' and total_tests > 0:
            result['failure_kind'] = 'FUNCTIONAL_FAIL'
            result['status'] = 'FAIL'
        elif failure_kind in {'WORKER_ERROR', 'WORKER_INPUT_ERROR'}:
            result['status'] = 'SANDBOX_ERROR'
        else:
            result['status'] = 'PASS' if result.get('passed') else 'FAIL'
    return result
