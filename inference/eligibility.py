"""GPU-only eligibility: a config benchmarks only when fully VRAM-resident.

``num_gpu`` in the adapter is a request, not proof. Proof is /api/ps:
eligible iff ``size_vram == size`` (both positive) for the loaded model.
Anything unmeasurable fails closed (eligible=False), never silently.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from inference.ollama_client import fetch_ps

ELIGIBLE_GPU = 'ELIGIBLE_GPU'
INELIGIBLE_GPU_ONLY = 'INELIGIBLE_GPU_ONLY'
UNMEASURABLE_FAIL_CLOSED = 'UNMEASURABLE_FAIL_CLOSED'
DIGEST_MISMATCH_FAIL_CLOSED = 'DIGEST_MISMATCH_FAIL_CLOSED'


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
    """Fetch live /api/ps and evaluate residency (fail closed on any fault)."""
    try:
        ps_doc = fetch_ps(base_url)
    except Exception:
        return EligibilityResult(
            False, UNMEASURABLE_FAIL_CLOSED, None, None, None, None, '{}'
        )
    return evaluate_residency(ps_doc, model_identifier, expected_digest=expected_digest)
