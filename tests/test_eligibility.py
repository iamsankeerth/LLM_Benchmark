"""Eligibility tests: live-observed Qwen residency locked as fixtures."""

from __future__ import annotations

import unittest
from typing import Any

from inference.adapters import get_model_config
from inference.eligibility import (
    DIGEST_MISMATCH_FAIL_CLOSED,
    ELIGIBLE_GPU,
    ELIGIBILITY_MEASUREMENT_ERROR,
    INELIGIBLE_GPU_ONLY,
    UNMEASURABLE_FAIL_CLOSED,
    CanonicalEligibility,
    EffectiveEligibilityOptions,
    ProbeConfigMismatchError,
    effective_options_for,
    evaluate_residency,
    run_canonical_eligibility,
)
from inference.ollama_client import GenerationRequest, GenerationResult

QWEN_ID = 'hf.co/bartowski/Qwen_Qwen3-4B-Instruct-2507-GGUF:Q4_K_M'
QWEN_DIGEST = '5cfd6a526dc35c24a21a8a457db3e8810165e93bbb134dd6a7ac20b34e97b25a'


def _ps_entry(**overrides: object) -> dict[str, Any]:
    entry: dict[str, Any] = {
        'name': QWEN_ID,
        'model': QWEN_ID,
        'size': 3178149969,
        'digest': QWEN_DIGEST,
        'size_vram': 3178149969,
        'context_length': 4096,
    }
    entry.update(overrides)
    return entry


class EligibilityTests(unittest.TestCase):
    def test_live_qwen_residency_is_eligible(self) -> None:
        # Exact /api/ps evidence observed 2026-09-17 at 100% GPU.
        result = evaluate_residency(
            {'models': [_ps_entry()]}, QWEN_ID, expected_digest=QWEN_DIGEST
        )
        self.assertTrue(result.eligible)
        self.assertEqual(result.status, ELIGIBLE_GPU)
        self.assertEqual(result.gpu_residency_ratio, 1.0)
        self.assertEqual(result.observed_digest, QWEN_DIGEST)

    def test_adapter_digest_matches_live_evidence(self) -> None:
        config = get_model_config('qwen3-4b-q4')
        self.assertEqual(config.ollama_model_digest, QWEN_DIGEST)

    def test_partial_offload_ineligible(self) -> None:
        result = evaluate_residency(
            {'models': [_ps_entry(size_vram=2000000000)]}, QWEN_ID
        )
        self.assertFalse(result.eligible)
        self.assertEqual(result.status, INELIGIBLE_GPU_ONLY)
        self.assertAlmostEqual(result.gpu_residency_ratio or 0.0, 2000000000 / 3178149969)

    def test_model_not_loaded_fails_closed(self) -> None:
        result = evaluate_residency({'models': []}, QWEN_ID)
        self.assertFalse(result.eligible)
        self.assertEqual(result.status, UNMEASURABLE_FAIL_CLOSED)

    def test_missing_sizes_fail_closed(self) -> None:
        entry = _ps_entry()
        del entry['size_vram']
        result = evaluate_residency({'models': [entry]}, QWEN_ID)
        self.assertFalse(result.eligible)
        self.assertEqual(result.status, UNMEASURABLE_FAIL_CLOSED)

    def test_zero_sizes_fail_closed(self) -> None:
        result = evaluate_residency(
            {'models': [_ps_entry(size=0, size_vram=0)]}, QWEN_ID
        )
        self.assertFalse(result.eligible)
        self.assertEqual(result.status, UNMEASURABLE_FAIL_CLOSED)

    def test_digest_mismatch_fails_closed(self) -> None:
        result = evaluate_residency(
            {'models': [_ps_entry()]}, QWEN_ID, expected_digest='0' * 64
        )
        self.assertFalse(result.eligible)
        self.assertEqual(result.status, DIGEST_MISMATCH_FAIL_CLOSED)

    def test_malformed_ps_doc_fails_closed(self) -> None:
        for doc in ({}, {'models': None}, {'models': ['nope']}):
            result = evaluate_residency(doc, QWEN_ID)
            self.assertFalse(result.eligible)
            self.assertEqual(result.status, UNMEASURABLE_FAIL_CLOSED)


def _fake_result() -> GenerationResult:
    return GenerationResult(
        text='ok', thinking='', done_reason='stop',
        request_start_ns=0, first_token_ns=1, request_end_ns=2,
    )


def _canonical_options() -> EffectiveEligibilityOptions:
    return effective_options_for(
        mode='raw', num_ctx=4096, num_gpu=99, temperature=0.0,
        template_sha256='t' * 64,
    )


class CanonicalProbeTests(unittest.TestCase):
    def _run(
        self, ps_doc: dict[str, Any], options: EffectiveEligibilityOptions | None = None
    ) -> CanonicalEligibility:
        def fake_generate(url: str, request: GenerationRequest) -> GenerationResult:
            self.captured = request
            return _fake_result()

        def fake_ps(url: str) -> dict[str, Any]:
            return ps_doc

        return run_canonical_eligibility(
            base_url='http://x', model_identifier=QWEN_ID,
            expected_digest=QWEN_DIGEST,
            effective_options=options if options is not None else _canonical_options(),
            render_prompt=lambda prompt: f'<|im_start|>user\n{prompt}<|im_end|>\n',
            generate_fn=fake_generate, fetch_ps_fn=fake_ps,
        )

    def test_full_residency_eligible(self) -> None:
        outcome = self._run({'models': [_ps_entry()]})
        self.assertTrue(outcome.result.eligible)
        self.assertEqual(outcome.result.status, ELIGIBLE_GPU)
        # Probe used the exact pinned options.
        request = self.captured
        self.assertEqual(request.num_ctx, 4096)
        self.assertEqual(request.num_predict, 1)
        self.assertEqual(request.num_gpu, 99)
        self.assertTrue(request.raw)

    def test_partial_residency_ineligible(self) -> None:
        outcome = self._run({'models': [_ps_entry(size_vram=2000000000)]})
        self.assertFalse(outcome.result.eligible)
        self.assertEqual(outcome.result.status, INELIGIBLE_GPU_ONLY)

    def test_absent_after_probe_is_measurement_error(self) -> None:
        outcome = self._run({'models': []})
        self.assertFalse(outcome.result.eligible)
        self.assertEqual(outcome.result.status, ELIGIBILITY_MEASUREMENT_ERROR)

    def test_probe_config_mismatch_is_hard_error(self) -> None:
        # The 512-context incident fixture: probe options differing from the
        # pinned config must raise before any verdict is produced.
        with self.assertRaises(ProbeConfigMismatchError):
            run_canonical_eligibility(
                base_url='http://x', model_identifier=QWEN_ID,
                expected_digest=QWEN_DIGEST,
                effective_options=_canonical_options(),
                render_prompt=lambda prompt: prompt,
                generate_fn=lambda url, req: _fake_result(),
                fetch_ps_fn=lambda url: {'models': [_ps_entry()]},
                probe_options_override={
                    'mode': 'raw', 'num_ctx': 512, 'num_predict': 1,
                    'num_gpu': None, 'temperature': 0.0,
                    'template_sha256': 't' * 64,
                },
            )

    def test_effective_options_recorded(self) -> None:
        outcome = self._run({'models': [_ps_entry()]})
        options = outcome.effective_options
        self.assertEqual(options.num_ctx, 4096)
        self.assertEqual(options.num_predict, 1)
        self.assertEqual(options.num_gpu, 99)


if __name__ == '__main__':
    unittest.main()
