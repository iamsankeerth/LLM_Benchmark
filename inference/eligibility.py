"""GPU-only eligibility: a config benchmarks only when fully VRAM-resident.

``num_gpu`` in the adapter is a request, not proof. Proof is /api/ps,
inspected ONLY after a canonical load probe using the exact pinned
effective options (adapter template/mode, num_ctx, num_predict=1,
num_gpu, temperature=0). A verdict measured on any other load is invalid.

Three-way taxonomy:
- model absent after canonical probe -> ELIGIBILITY_MEASUREMENT_ERROR
  (STOP, weights retained; never an ineligibility claim)
- present with residency == 1.0     -> ELIGIBLE_GPU_ONLY
- present with residency < 1.0      -> INELIGIBLE_GPU_ONLY (evidence kept)
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Callable, Mapping

from inference.ollama_client import GenerationRequest, GenerationResult, fetch_ps

ELIGIBLE_GPU = 'ELIGIBLE_GPU'
INELIGIBLE_GPU_ONLY = 'INELIGIBLE_GPU_ONLY'
UNMEASURABLE_FAIL_CLOSED = 'UNMEASURABLE_FAIL_CLOSED'
DIGEST_MISMATCH_FAIL_CLOSED = 'DIGEST_MISMATCH_FAIL_CLOSED'
ELIGIBILITY_MEASUREMENT_ERROR = 'ELIGIBILITY_MEASUREMENT_ERROR'
COMPLETE_INELIGIBLE_RUNTIME_HEADROOM = 'COMPLETE_INELIGIBLE_RUNTIME_HEADROOM'

# Operational canary: sustained generation under the exact canonical config.
# Not a benchmark task; transport/runtime proof only; no DB row is written.
# A short natural stop below the token floor is inconclusive, never a pass:
# the canary must observe sustained generation or fail closed.
CANARY_PROMPT = (
    'Explain in detail, in at least 300 words, how a relational database '
    'uses indexes to speed up queries. Cover B-tree structure, write '
    'amplification, and when an index hurts performance.'
)
CANARY_NUM_PREDICT = 512
CANARY_MIN_EVAL_TOKENS = 128

# Canonical eligibility probe: diagnostic single token. num_predict=1 keeps
# it cheap; residency is governed by num_ctx, which always equals the
# benchmark context so the KV allocation matches the measured runs.
CANONICAL_PROBE_NUM_PREDICT = 1
CANONICAL_PROBE_PROMPT = 'OK'
CANONICAL_PROBE_TEMPERATURE = 0.0


class ProbeConfigMismatchError(Exception):
    """A probe attempted with options differing from the pinned config."""


@dataclass(frozen=True)
class EffectiveEligibilityOptions:
    """Exact runtime options a canonical probe must use (from the adapter)."""

    mode: str
    num_ctx: int
    num_predict: int
    num_gpu: int | None
    temperature: float
    template_sha256: str
    stop_tokens: tuple[str, ...] = ()
    think: bool | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def effective_options_for(
    *,
    mode: str,
    num_ctx: int,
    num_gpu: int | None,
    temperature: float,
    template_sha256: str,
    stop_tokens: tuple[str, ...] = (),
    think: bool | None = None,
) -> EffectiveEligibilityOptions:
    return EffectiveEligibilityOptions(
        mode=mode,
        num_ctx=num_ctx,
        num_predict=CANONICAL_PROBE_NUM_PREDICT,
        num_gpu=num_gpu,
        temperature=temperature,
        template_sha256=template_sha256,
        stop_tokens=stop_tokens,
        think=think,
    )


@dataclass(frozen=True)
class EligibilityResult:
    eligible: bool
    status: str
    model_size_bytes: int | None
    size_vram_bytes: int | None
    gpu_residency_ratio: float | None
    observed_digest: str | None
    eligibility_evidence_json: str


def _positive_int(value: Any) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return value
    return None


def evaluate_residency(
    ps_doc: dict[str, Any], model_identifier: str, *, expected_digest: str | None = None
) -> EligibilityResult:
    """Pure residency verdict from a /api/ps document (unit-testable)."""
    entries = ps_doc.get('models')
    match: dict[str, Any] | None = None
    if isinstance(entries, list):
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            if entry.get('model') == model_identifier or entry.get('name') == model_identifier:
                match = entry
                break
    evidence = json.dumps(match, sort_keys=True) if match is not None else '{}'
    if match is None:
        return EligibilityResult(False, UNMEASURABLE_FAIL_CLOSED, None, None, None, None, evidence)

    observed_digest = match.get('digest')
    digest = observed_digest if isinstance(observed_digest, str) else None
    if expected_digest is not None and digest != expected_digest:
        return EligibilityResult(
            False, DIGEST_MISMATCH_FAIL_CLOSED, None, None, None, digest, evidence
        )

    size = _positive_int(match.get('size'))
    size_vram = _positive_int(match.get('size_vram'))
    if size is None or size_vram is None:
        return EligibilityResult(
            False, UNMEASURABLE_FAIL_CLOSED, size, size_vram, None, digest, evidence
        )
    ratio = size_vram / size
    if size_vram == size:
        return EligibilityResult(True, ELIGIBLE_GPU, size, size_vram, ratio, digest, evidence)
    return EligibilityResult(
        False, INELIGIBLE_GPU_ONLY, size, size_vram, ratio, digest, evidence
    )


def check_eligibility(
    base_url: str, model_identifier: str, *, expected_digest: str | None = None
) -> EligibilityResult:
    """Fetch live /api/ps and evaluate residency (fail closed on any fault).

    NOTE: only valid immediately after a canonical load probe. Prefer
    run_canonical_eligibility(), which enforces the probe/config identity.
    """
    try:
        ps_doc = fetch_ps(base_url)
    except Exception:
        return EligibilityResult(
            False, UNMEASURABLE_FAIL_CLOSED, None, None, None, None, '{}'
        )
    return evaluate_residency(ps_doc, model_identifier, expected_digest=expected_digest)


@dataclass
class CanonicalEligibility:
    result: EligibilityResult
    effective_options: EffectiveEligibilityOptions
    rendered_prompt_sha256: str


@dataclass(frozen=True)
class CanaryOutcome:
    passed: bool
    eval_count: int | None
    done_reason: str | None
    failure_kind: str
    detail: str


def run_operational_canary(
    *,
    base_url: str,
    model_identifier: str,
    effective_options: EffectiveEligibilityOptions,
    render_prompt: Callable[[str], str],
    generate_fn: Callable[[str, GenerationRequest], GenerationResult] | None = None,
    timeout_s: float = 600.0,
) -> CanaryOutcome:
    """Sustained-generation canary under the exact canonical config.

    Full residency is necessary but not sufficient: this proves the loaded
    model can actually sustain canonical inference (Phi Q4 was 100%
    resident yet aborted streams with 95 MiB free). No DB row is written.
    done may be stop OR length (surviving the budget is the point); fewer
    than CANARY_MIN_EVAL_TOKENS observed tokens is inconclusive -> fail.
    """
    from inference.ollama_client import OllamaClientError

    options = effective_options
    prompt_text = (
        render_prompt(CANARY_PROMPT)
        if options.mode == 'raw'
        else CANARY_PROMPT
    )
    request = GenerationRequest(
        model=model_identifier,
        prompt=prompt_text,
        temperature=options.temperature,
        num_ctx=options.num_ctx,
        num_predict=CANARY_NUM_PREDICT,
        num_gpu=options.num_gpu,
        raw=(options.mode == 'raw'),
    )
    generate_call = generate_fn or (lambda url, req: _live_generate(url, req, timeout_s))
    try:
        result = generate_call(base_url, request)
    except OllamaClientError as exc:
        return CanaryOutcome(False, None, None, type(exc).__name__, str(exc)[:300])
    except Exception as exc:
        return CanaryOutcome(False, None, None, type(exc).__name__, str(exc)[:300])
    eval_count = result.eval_count
    if eval_count is not None and eval_count >= CANARY_MIN_EVAL_TOKENS:
        return CanaryOutcome(True, eval_count, result.done_reason, '', '')
    return CanaryOutcome(
        False, eval_count, result.done_reason, 'insufficient_output',
        f'only {eval_count} tokens observed (floor {CANARY_MIN_EVAL_TOKENS})',
    )


def run_canonical_eligibility(
    *,
    base_url: str,
    model_identifier: str,
    expected_digest: str,
    effective_options: EffectiveEligibilityOptions,
    render_prompt: Callable[[str], str],
    generate_fn: Callable[[str, GenerationRequest], GenerationResult] | None = None,
    fetch_ps_fn: Callable[[str], dict[str, Any]] | None = None,
    probe_options_override: Mapping[str, Any] | None = None,
    timeout_s: float = 300.0,
) -> CanonicalEligibility:
    """Load with the exact pinned options, then verdict from /api/ps.

    probe_options_override exists SOLELY for the regression fixture: any
    caller-supplied options differing from the pinned config raise
    ProbeConfigMismatchError before any verdict is produced. Production
    callers never pass it (options always derive from the adapter).
    """
    if probe_options_override is not None:
        override = dict(probe_options_override)
        pinned = effective_options.as_dict()
        if any(override.get(k) != pinned.get(k) for k in override):
            raise ProbeConfigMismatchError(
                'probe options differ from pinned config: '
                f'{override} != {pinned}'
            )
    options = effective_options
    is_raw = options.mode == 'raw'
    prompt_text = (
        render_prompt(CANONICAL_PROBE_PROMPT)
        if is_raw
        else CANONICAL_PROBE_PROMPT
    )
    request = GenerationRequest(
        model=model_identifier,
        prompt=prompt_text,
        temperature=options.temperature,
        num_ctx=options.num_ctx,
        num_predict=options.num_predict,
        num_gpu=options.num_gpu,
        stop=options.stop_tokens,
        raw=is_raw,
        think=options.think,
    )
    generate_call = generate_fn or (lambda url, req: _live_generate(url, req, timeout_s))
    generate_call(base_url, request)
    ps_call = fetch_ps_fn or fetch_ps
    try:
        ps_doc = ps_call(base_url)
    except Exception:
        empty = EligibilityResult(
            False, ELIGIBILITY_MEASUREMENT_ERROR, None, None, None, None, '{}'
        )
        return CanonicalEligibility(empty, options, _sha(prompt_text))
    result = evaluate_residency(ps_doc, model_identifier, expected_digest=expected_digest)
    if result.status == UNMEASURABLE_FAIL_CLOSED and not _ps_has_model(
        ps_doc, model_identifier
    ):
        # Distinguish "instrument says nothing is loaded" from other faults:
        # absent-after-probe is a measurement error, never an ineligibility.
        result = EligibilityResult(
            False, ELIGIBILITY_MEASUREMENT_ERROR, None, None, None, None,
            result.eligibility_evidence_json,
        )
    return CanonicalEligibility(result, options, _sha(prompt_text))


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def _live_generate(
    base_url: str, request: GenerationRequest, timeout_s: float
) -> GenerationResult:
    from inference.ollama_client import generate

    return generate(base_url, request, timeout_s=timeout_s)


def _ps_has_model(ps_doc: dict[str, Any], model_identifier: str) -> bool:
    entries = ps_doc.get('models')
    if not isinstance(entries, list):
        return False
    return any(
        isinstance(entry, dict)
        and (entry.get('model') == model_identifier or entry.get('name') == model_identifier)
        for entry in entries
    )
