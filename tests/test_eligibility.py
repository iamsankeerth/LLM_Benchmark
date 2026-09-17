"""Eligibility tests: live-observed Qwen residency locked as fixtures."""

from __future__ import annotations

import unittest
from typing import Any

from inference.adapters import get_model_config
from inference.eligibility import (
    DIGEST_MISMATCH_FAIL_CLOSED,
    ELIGIBLE_GPU,
    INELIGIBLE_GPU_ONLY,
    UNMEASURABLE_FAIL_CLOSED,
    evaluate_residency,
)

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


if __name__ == '__main__':
    unittest.main()
