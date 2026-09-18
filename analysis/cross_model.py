"""Cross-model sweep aggregation (offline, from committed summaries).

Canonical tables carry all 14 registry identities: 11 eligible configs
with measured values, 3 Phi configs preserved as INELIGIBLE_RUNTIME_HEADROOM
with null metrics and evidence refs. Denominators, skylines and rates
filter eligibility_status == ELIGIBLE_GPU explicitly — Phi rows never
enter a mean, skyline, or rate.

Failure taxonomy is automatic and symmetric (failure-modes-v1) for all 11
eligible configs. The Qwen Q4 human substantive review lives in its own
section and is never blended into the automatic columns.
"""

from __future__ import annotations

from typing import Any, Mapping

ELIGIBLE = 'ELIGIBLE_GPU'
INELIGIBLE_RUNTIME_HEADROOM = 'COMPLETE_INELIGIBLE_RUNTIME_HEADROOM'

METHODOLOGY_INCIDENTS = [
    {
        'id': 'wrong-canonical-load',
        'cause': 'Eligibility measured a model loaded under default options '
                 '(num_ctx=512, no num_gpu) instead of the pinned config.',
        'fixture': '512-context probe/config mismatch hard error; '
                   'captured size/size_vram numbers as poisoned input.',
        'rule': 'Canonical load probe with exact pinned options; explicit '
                'unload + ps-empty before any measured load.',
    },
    {
        'id': 'stale-state-loop',
        'cause': 'Main loop held a launch-time state copy, re-selecting a '
                 'terminal model forever.',
        'fixture': 'File-authoritative reload per iteration; terminal-state '
                   'transition invariant test.',
        'rule': 'State file reloaded every iteration; no terminal state '
                'transitions back to work.',
    },
    {
        'id': 'silent-gemma-metadata',
        'cause': '/api/show returned empty metadata for Gemma-3n (both '
                 'sources) although the model downloads and generates.',
        'fixture': 'Overlay-precedence regression test (pinned adapter '
                   'resolves with show mocked to fail).',
        'rule': 'Committed overlay takes precedence; interrogation failure '
                'with no pin is MANUAL_PIN_REQUIRED; human handbook pin '
                'with raw-canonical route for Gemma.',
    },
    {
        'id': 'first-token-trial',
        'cause': '1-token route trial falsely failed SmolLM2 Q6, whose first '
                 'token can be non-text on a healthy route.',
        'fixture': 'Trial-v1.1: 16-token budget; mid-derivation assertion test.',
        'rule': 'Trials prove transport, not brevity.',
    },
    {
        'id': 'phi-runtime-headroom',
        'cause': 'Phi Q4 100% resident yet aborted sustained generation '
                 'with ~95 MiB free; residency necessary but not sufficient.',
        'fixture': 'Residency-plus-canary amendment; RUNTIME_HEADROOM terminal.',
        'rule': 'Eligibility = full residency AND operational canary.',
    },
    {
        'id': 'ghost-deletion',
        'cause': 'rm success does not imply absence; list/ps can disagree.',
        'fixture': 'Ghost-state unit test (rm-ok + still-present).',
        'rule': 'Verified deletion guard; DELETION_FAILED never advances.',
    },
]

LIMITATIONS = [
    'Observed quantization differences within the tested paired '
    'deterministic tasks were small and were not statistically '
    'distinguishable under the exact McNemar tests used here.',
    'Temperature-0 inference showed high repeatability under this '
    'benchmark, hardware, and runtime configuration; no universal '
    'determinism claim is made.',
    'Single machine (RTX 2050 4 GB VRAM); feasibility findings are '
    'hardware-specific by construction.',
    'READY_JUDGE tasks are excluded from all capability denominators; '
    'qualitative outcomes remain deferred.',
    'Temperature study (0 vs 0.7), long-context, and judge-bias suites '
    'are out of scope for this lineage.',
    'Strict substring matchers (terms, dates, units) punish paraphrase; '
    'contract accuracy understates reasoning capability by design.',
]

NEXT_EXPERIMENTS = [
    'retry/rescue: 39/97 deterministic Qwen Q4 failure rows classified as '
    'pure OUTPUT_CONTRACT form the pre-registered recovery population.',
    'temperature 0 vs 0.7 on the frozen 72-task contract.',
    'long-context and reliability suites.',
]


def _num(value: Any) -> float | None:
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def build_registry_row(
    *,
    model_config_id: str,
    family: str,
    quantization: str,
    capability: Mapping[str, Any] | None,
    performance: Mapping[str, Any] | None,
    failure_modes: Mapping[str, Any] | None,
    manifest: Mapping[str, Any] | None,
    ineligible_evidence: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """One canonical 14-registry row (measured values or null + evidence)."""
    if ineligible_evidence is not None:
        return {
            'model_config_id': model_config_id,
            'family': family,
            'quantization': quantization,
            'eligibility_status': INELIGIBLE_RUNTIME_HEADROOM,
            'deterministic_trial_accuracy': None,
            'any_pass_3_rate': None,
            'all_pass_3_rate': None,
            'decode_median_tok_s': None,
            'decode_p95_tok_s': None,
            'ttft_median_ms': None,
            'peak_vram_mib': None,
            'model_size_bytes': None,
            'failure_modes': None,
            'evidence_ref': ineligible_evidence.get('artifact_path'),
            'residency_ratio': ineligible_evidence.get('residency_ratio'),
        }
    assert capability is not None and performance is not None
    rates = capability.get('derived_rates', {})
    decode = performance.get('decode_tok_s', {})
    ttft = performance.get('ttft_ms', {})
    modes = (failure_modes or {}).get('mode_counts_deterministic')
    return {
        'model_config_id': model_config_id,
        'family': family,
        'quantization': quantization,
        'eligibility_status': ELIGIBLE,
        'deterministic_trial_accuracy': rates.get('deterministic_trial_accuracy'),
        'any_pass_3_rate': rates.get('any_pass_3_rate'),
        'all_pass_3_rate': rates.get('all_pass_3_rate'),
        'decode_median_tok_s': decode.get('median'),
        'decode_p25_tok_s': decode.get('p25'),
        'decode_p75_tok_s': decode.get('p75'),
        'decode_p95_tok_s': decode.get('p95'),
        'ttft_median_ms': ttft.get('median'),
        'ttft_p95_ms': ttft.get('p95'),
        'peak_vram_mib': performance.get('peak_vram_mib'),
        'model_size_bytes': (manifest or {}).get('model_artifact_size_bytes'),
        'model_artifact_digest': (manifest or {}).get('model_artifact_digest'),
        'failure_modes': modes,
    }


def eligible_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [r for r in rows if r.get('eligibility_status') == ELIGIBLE]


def pareto_skyline(
    points: list[dict[str, Any]],
    *,
    x_key: str,
    y_key: str,
    y_bigger_better: bool,
) -> list[str]:
    """Skyline membership over eligible points.

    A point is dominated only when another eligible point is at least as
    good on both axes and strictly better on at least one. Exact ties
    remain co-members. x is always maximize (accuracy).
    """
    members: list[str] = []
    for candidate in points:
        cx, cy = _num(candidate.get(x_key)), _num(candidate.get(y_key))
        if cx is None or cy is None:
            continue
        dominated = False
        for other in points:
            if other is candidate:
                continue
            ox, oy = _num(other.get(x_key)), _num(other.get(y_key))
            if ox is None or oy is None:
                continue
            x_ok = ox >= cx
            y_ok = (oy >= cy) if y_bigger_better else (oy <= cy)
            x_strict = ox > cx
            y_strict = (oy > cy) if y_bigger_better else (oy < cy)
            if x_ok and y_ok and (x_strict or y_strict):
                dominated = True
                break
        if not dominated:
            members.append(str(candidate.get('model_config_id')))
    return sorted(members)


def summarize_quant_pairs(
    comparisons: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Aggregate anchor-paired comparison files (deltas + McNemar passthrough)."""
    rows: list[dict[str, Any]] = []
    for name in sorted(comparisons):
        doc = comparisons[name]
        matrix = doc.get('transitions_strict', {})
        rows.append({
            'comparison': name,
            'trial_accuracy_delta': doc.get('trial_accuracy_delta'),
            'all_pass_delta': doc.get('all_pass_delta'),
            'net_task_gain': len(matrix.get('fail_to_pass', []))
            - len(matrix.get('pass_to_fail', [])),
            'fail_to_pass': matrix.get('fail_to_pass', []),
            'pass_to_fail': matrix.get('pass_to_fail', []),
            'mcnemar_b': doc.get('mcnemar_b'),
            'mcnemar_c': doc.get('mcnemar_c'),
            'mcnemar_exact_p': doc.get('mcnemar_exact_p'),
        })
    return rows
