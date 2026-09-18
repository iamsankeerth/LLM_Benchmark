"""Transient-fault classification and bounded retry policy.

Retryable faults are transport-level only (timeouts, resets, truncated
streams, transient 5xx, pull/rm interruptions). Everything semantic
(4xx, malformed data, config/hash/dataset mismatches, persistence
failures) fails closed immediately. Retries never change benchmark
configuration and never persist partial output.
"""

from __future__ import annotations

import time
from typing import Callable, TypeVar

import requests

from inference.ollama_client import (
    MalformedChunkError,
    OllamaHTTPError,
    OllamaTimeoutError,
    TruncatedStreamError,
)

# Pull backoff schedule: 30s, 2m, 5m (3 attempts after the initial try).
PULL_BACKOFFS = (30.0, 120.0, 300.0)
# Model removal retries on transient failure.
RM_ATTEMPTS = 3
# Same-identity generation retries before unload/reload + final attempt.
GENERATION_RETRIES = 2

T = TypeVar('T')


def is_retryable(exc: BaseException) -> bool:
    """True only for transport-level faults. Fail closed otherwise."""
    if isinstance(exc, OllamaTimeoutError):
        return True
    if isinstance(exc, TruncatedStreamError):
        return True
    if isinstance(exc, OllamaHTTPError):
        return exc.status_code >= 500
    if isinstance(exc, MalformedChunkError):
        return False
    if isinstance(exc, (requests.Timeout, requests.ConnectionError)):
        return True
    if isinstance(exc, (TimeoutError, ConnectionError)):
        # Raw transport errors that escaped wrapping. Callers invoke this
        # classifier only around transport operations (pull/generate/rm),
        # so these flavors are retryable here.
        return True
    return False


def error_kind(exc: BaseException) -> str:
    """Stable short label for retry telemetry."""
    if isinstance(exc, OllamaTimeoutError):
        return 'timeout'
    if isinstance(exc, TruncatedStreamError):
        return 'truncated_stream'
    if isinstance(exc, OllamaHTTPError):
        return f'http_{exc.status_code}'
    if isinstance(exc, MalformedChunkError):
        return 'malformed_chunk'
    return type(exc).__name__


def run_with_retries(
    action: Callable[[], T],
    *,
    attempts: int,
    backoffs: tuple[float, ...] = (),
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    """Run action, retrying retryable faults with backoff.

    attempts counts total tries (initial + retries). Non-retryable faults
    raise immediately. attempts exhausted raises the last error.
    """
    last_error: BaseException | None = None
    for attempt in range(max(1, attempts)):
        try:
            return action()
        except Exception as exc:  # noqa: BLE001 - classified below
            if not is_retryable(exc):
                raise
            last_error = exc
            if attempt < len(backoffs):
                sleep(backoffs[attempt])
    assert last_error is not None
    raise last_error
