"""Profiler: single home for every derived inference metric.

All formulas live in :func:`derive_metrics` and nowhere else:

- ``prefill_compute_tok_s = uncached_count / prompt_eval_seconds``
  (uncached tokens only; the naive count/duration ratio is deliberately
  NOT computed because ``prompt_eval_duration`` covers uncached tokens)
- ``decode_tok_s = eval_count / eval_seconds``
- ``decode_ms_per_token = eval_duration_ms / eval_count``
- ``ttft_ms = first_token_ns - request_start_ns`` (client, monotonic)
- ``client_e2e_ms = request_end_ns - request_start_ns`` (client, monotonic)

Unmeasurable quantities become None, never estimates. Division by zero
or missing counters yield None.
"""

from __future__ import annotations

from dataclasses import dataclass

from inference.ollama_client import GenerationResult

_NS_PER_S = 1_000_000_000
_NS_PER_MS = 1_000_000

# Reload evidence: a measured generation whose server load_duration exceeds
# this threshold is treated as proof the model was (re)loaded mid-run
# (eviction), triggering automatic re-warm. Named, frozen, and recorded in
# every manifest so re-warm events reproduce exactly. NOT part of the
# generation identity hash: it is runtime policy, not a trial condition.
RELOAD_EVIDENCE_LOAD_DURATION_MS = 1000.0


def needs_rewarm(
    server_load_duration_ms: float | None,
    threshold_ms: float = RELOAD_EVIDENCE_LOAD_DURATION_MS,
) -> bool:
    """True when a generation's load duration evidences a (re)load."""
    return (
        server_load_duration_ms is not None and server_load_duration_ms > threshold_ms
    )


@dataclass(frozen=True)
class ProfiledMetrics:
    ttft_ms: float | None
    client_e2e_ms: float
    server_total_duration_ms: float | None
    server_load_duration_ms: float | None
    prompt_eval_duration_ms: float | None
    eval_duration_ms: float | None
    prompt_eval_count: int | None
    prompt_eval_cached_count: int | None
    prompt_eval_uncached_count: int | None
    prompt_cache_ratio: float | None
    eval_count: int | None
    prefill_compute_tok_s: float | None
    prefill_cache_state: str  # UNCACHED | PARTIAL | FULLY_CACHED | UNKNOWN
    decode_tok_s: float | None
    decode_ms_per_token: float | None
    # Diagnostic only: application/streaming overhead, NOT network latency.
    client_overhead_ms: float | None


def _ms(nanoseconds: int | None) -> float | None:
    return nanoseconds / _NS_PER_MS if nanoseconds is not None else None


def _rate_per_second(count: int | None, duration_ns: int | None) -> float | None:
    if count is None or duration_ns is None:
        return None
    if count <= 0 or duration_ns <= 0:
        return None
    return count / (duration_ns / _NS_PER_S)


def derive_metrics(result: GenerationResult) -> ProfiledMetrics:
    count = result.prompt_eval_count
    cached = result.prompt_eval_cached_count
    uncached: int | None = None
    ratio: float | None = None
    if count is not None and cached is not None:
        if cached < 0 or cached > count:
            # Inconsistent server counters: refuse to derive cache figures.
            uncached = None
        else:
            uncached = count - cached
            ratio = (cached / count) if count > 0 else 0.0
    elif count is not None and cached is None:
        # Older servers omit the cached counter: every token was computed.
        uncached = count
        ratio = 0.0

    if uncached is None or count is None:
        cache_state = 'UNKNOWN'
    elif count == 0:
        cache_state = 'UNCACHED'
    elif cached is None:
        # Legacy server without the cached counter: all tokens computed.
        cache_state = 'UNCACHED'
    elif cached == 0:
        cache_state = 'UNCACHED'
    elif cached == count and count > 0:
        cache_state = 'FULLY_CACHED'
    else:
        cache_state = 'PARTIAL'

    prefill_tok_s = _rate_per_second(uncached, result.prompt_eval_duration_ns)

    decode_tok_s = _rate_per_second(result.eval_count, result.eval_duration_ns)
    ms_per_token: float | None = None
    if (
        result.eval_count is not None
        and result.eval_count > 0
        and result.eval_duration_ns is not None
        and result.eval_duration_ns > 0
    ):
        ms_per_token = (result.eval_duration_ns / _NS_PER_MS) / result.eval_count

    if result.first_token_ns is not None:
        ttft_ms: float | None = (result.first_token_ns - result.request_start_ns) / _NS_PER_MS
    else:
        ttft_ms = None
    e2e_ms = (result.request_end_ns - result.request_start_ns) / _NS_PER_MS

    server_total_ms = _ms(result.total_duration_ns)
    overhead: float | None = None
    if server_total_ms is not None:
        overhead = e2e_ms - server_total_ms

    return ProfiledMetrics(
        ttft_ms=ttft_ms,
        client_e2e_ms=e2e_ms,
        server_total_duration_ms=server_total_ms,
        server_load_duration_ms=_ms(result.load_duration_ns),
        prompt_eval_duration_ms=_ms(result.prompt_eval_duration_ns),
        eval_duration_ms=_ms(result.eval_duration_ns),
        prompt_eval_count=count,
        prompt_eval_cached_count=cached,
        prompt_eval_uncached_count=uncached,
        prompt_cache_ratio=ratio,
        eval_count=result.eval_count,
        prefill_compute_tok_s=prefill_tok_s,
        prefill_cache_state=cache_state,
        decode_tok_s=decode_tok_s,
        decode_ms_per_token=ms_per_token,
        client_overhead_ms=overhead,
    )
