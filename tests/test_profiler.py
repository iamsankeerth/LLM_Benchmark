"""Profiler formula tests with hand-computed fixtures (no live model)."""

from __future__ import annotations

import unittest

from inference.ollama_client import GenerationResult
from inference.profiler import derive_metrics
from inference.sysmon import SystemSampler, sample_vram_once


def _opt_int(value: object) -> int | None:
    if value is None:
        return None
    assert isinstance(value, int) and not isinstance(value, bool)
    return value


def _req_int(value: object) -> int:
    assert isinstance(value, int) and not isinstance(value, bool)
    return value


def _result(**overrides: object) -> GenerationResult:
    base: dict[str, object] = {
        'text': '42',
        'thinking': '',
        'done_reason': 'stop',
        'total_duration_ns': 9_352_113_600,
        'load_duration_ns': 9_179_720_100,
        'prompt_eval_count': 19,
        'prompt_eval_cached_count': 0,
        'prompt_eval_duration_ns': 99_204_000,
        'eval_count': 3,
        'eval_duration_ns': 67_262_000,
        'request_start_ns': 0,
        'first_token_ns': 9_292_400_000,
        'request_end_ns': 9_500_000_000,
    }
    base.update(overrides)
    return GenerationResult(
        text=str(base['text']),
        thinking=str(base['thinking']),
        done_reason=None
        if base['done_reason'] is None
        else str(base['done_reason']),
        total_duration_ns=_opt_int(base['total_duration_ns']),
        load_duration_ns=_opt_int(base['load_duration_ns']),
        prompt_eval_count=_opt_int(base['prompt_eval_count']),
        prompt_eval_cached_count=_opt_int(base['prompt_eval_cached_count']),
        prompt_eval_duration_ns=_opt_int(base['prompt_eval_duration_ns']),
        eval_count=_opt_int(base['eval_count']),
        eval_duration_ns=_opt_int(base['eval_duration_ns']),
        request_start_ns=_req_int(base['request_start_ns']),
        first_token_ns=_opt_int(base['first_token_ns']),
        request_end_ns=_req_int(base['request_end_ns']),
    )


class ProfilerFormulaTests(unittest.TestCase):
    def test_full_fixture_hand_computed(self) -> None:
        m = derive_metrics(_result())
        # 19 / 0.099204 = 191.5246...
        self.assertAlmostEqual(m.prefill_compute_tok_s or 0.0, 191.52, places=2)
        # 3 / 0.067262 = 44.6017...
        self.assertAlmostEqual(m.decode_tok_s or 0.0, 44.60, places=2)
        # 67.262 / 3 = 22.4207...
        self.assertAlmostEqual(m.decode_ms_per_token or 0.0, 22.42, places=2)
        self.assertAlmostEqual(m.ttft_ms or 0.0, 9292.4, places=1)
        self.assertAlmostEqual(m.client_e2e_ms, 9500.0, places=1)
        self.assertAlmostEqual(m.server_total_duration_ms or 0.0, 9352.1136, places=3)
        self.assertAlmostEqual(m.server_load_duration_ms or 0.0, 9179.7201, places=3)
        # 9500.0 - 9352.1136 = 147.8864
        self.assertAlmostEqual(m.client_overhead_ms or 0.0, 147.89, places=2)
        self.assertEqual(m.prompt_eval_uncached_count, 19)
        self.assertEqual(m.prompt_cache_ratio, 0.0)
        self.assertEqual(m.prefill_cache_state, 'UNCACHED')

    def test_partial_cache(self) -> None:
        m = derive_metrics(
            _result(prompt_eval_count=19, prompt_eval_cached_count=7)
        )
        self.assertEqual(m.prompt_eval_uncached_count, 12)
        self.assertAlmostEqual(m.prompt_cache_ratio or 0.0, 7 / 19)
        self.assertEqual(m.prefill_cache_state, 'PARTIAL')
        # 12 / 0.099204 = 120.963...
        self.assertAlmostEqual(m.prefill_compute_tok_s or 0.0, 120.96, places=2)

    def test_fully_cached_yields_no_prefill_rate(self) -> None:
        m = derive_metrics(
            _result(prompt_eval_count=19, prompt_eval_cached_count=19)
        )
        self.assertEqual(m.prefill_cache_state, 'FULLY_CACHED')
        self.assertEqual(m.prompt_eval_uncached_count, 0)
        self.assertIsNone(m.prefill_compute_tok_s)

    def test_unknown_cache_when_counters_absent(self) -> None:
        m = derive_metrics(
            _result(prompt_eval_count=None, prompt_eval_cached_count=None)
        )
        self.assertEqual(m.prefill_cache_state, 'UNKNOWN')
        self.assertIsNone(m.prefill_compute_tok_s)
        self.assertIsNone(m.prompt_cache_ratio)

    def test_legacy_server_without_cached_counter(self) -> None:
        m = derive_metrics(
            _result(prompt_eval_count=19, prompt_eval_cached_count=None)
        )
        self.assertEqual(m.prompt_eval_uncached_count, 19)
        self.assertEqual(m.prefill_cache_state, 'UNCACHED')

    def test_inconsistent_counters_refuse_derivation(self) -> None:
        m = derive_metrics(
            _result(prompt_eval_count=5, prompt_eval_cached_count=9)
        )
        self.assertEqual(m.prefill_cache_state, 'UNKNOWN')
        self.assertIsNone(m.prompt_eval_uncached_count)
        self.assertIsNone(m.prefill_compute_tok_s)

    def test_zero_eval_count_yields_no_decode_rate(self) -> None:
        m = derive_metrics(_result(eval_count=0, eval_duration_ns=0))
        self.assertIsNone(m.decode_tok_s)
        self.assertIsNone(m.decode_ms_per_token)

    def test_missing_durations_yield_none_not_zero(self) -> None:
        m = derive_metrics(
            _result(prompt_eval_duration_ns=None, eval_duration_ns=None,
                    total_duration_ns=None)
        )
        self.assertIsNone(m.prefill_compute_tok_s)
        self.assertIsNone(m.decode_tok_s)
        self.assertIsNone(m.server_total_duration_ms)
        self.assertIsNone(m.client_overhead_ms)

    def test_missing_first_token_yields_none_ttft(self) -> None:
        m = derive_metrics(_result(first_token_ns=None))
        self.assertIsNone(m.ttft_ms)
        self.assertAlmostEqual(m.client_e2e_ms, 9500.0, places=1)


class SystemSamplerTests(unittest.TestCase):
    def test_sampler_peak_monotonic(self) -> None:
        with SystemSampler(interval_s=0.01) as sampler:
            blob = bytearray(5_000_000)
            for index in range(len(blob)):
                blob[index] = index % 251
            sample = sampler.sample()
        self.assertGreaterEqual(sample.ram_peak_mb, sample.ram_baseline_mb)
        self.assertGreater(sample.ram_baseline_mb, 0.0)

    def test_vram_fields_consistent(self) -> None:
        with SystemSampler(interval_s=0.01) as sampler:
            sample = sampler.sample()
        if sample.nvml_available:
            self.assertIsNotNone(sample.vram_total_mib)
            self.assertIsNotNone(sample.vram_baseline_mib)
            self.assertIsNotNone(sample.vram_peak_mib)
        else:
            self.assertIsNone(sample.vram_baseline_mib)
            self.assertIsNone(sample.vram_peak_mib)
            self.assertIsNone(sample.vram_total_mib)

    def test_one_shot_vram_returns_pair(self) -> None:
        used, total = sample_vram_once()
        self.assertEqual(used is None, total is None)


if __name__ == '__main__':
    unittest.main()
