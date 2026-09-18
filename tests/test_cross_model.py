"""Cross-model aggregation tests: 14-row tables, Pareto, pairs (offline)."""

from __future__ import annotations

import unittest
from pathlib import Path

from analysis.cross_model import (
    build_registry_row,
    eligible_rows,
    pareto_skyline,
    summarize_quant_pairs,
)


def _measured(mid: str, acc: float, dec: float, vram: float) -> dict[str, object]:
    return build_registry_row(
        model_config_id=mid, family='f', quantization='Q',
        capability={'derived_rates': {
            'deterministic_trial_accuracy': acc,
            'any_pass_3_rate': acc, 'all_pass_3_rate': acc,
        }},
        performance={'decode_tok_s': {'median': dec}, 'ttft_ms': {'median': 50.0},
                     'peak_vram_mib': vram},
        failure_modes={'mode_counts_deterministic': {'OUTPUT_CONTRACT': 1}},
        manifest={'model_artifact_size_bytes': 100, 'model_artifact_digest': 'd'},
        ineligible_evidence=None,
    )


def _ineligible(mid: str) -> dict[str, object]:
    return build_registry_row(
        model_config_id=mid, family='f', quantization='Q',
        capability=None, performance=None, failure_modes=None, manifest=None,
        ineligible_evidence={'artifact_path': 'p', 'residency_ratio': 1.0},
    )


class RegistryTableTests(unittest.TestCase):
    def test_fourteen_rows_preserve_ineligible(self) -> None:
        rows = [_measured(f'm{i:02d}', 0.4, 40.0, 2000.0) for i in range(11)]
        rows += [_ineligible(f'x{i:02d}') for i in range(3)]
        self.assertEqual(len(rows), 14)
        eligible = eligible_rows(rows)
        self.assertEqual(len(eligible), 11)
        # Phi-style rows carry nulls + evidence, never zeros.
        null_row = rows[-1]
        self.assertIsNone(null_row['deterministic_trial_accuracy'])
        self.assertIsNone(null_row['decode_median_tok_s'])
        self.assertEqual(null_row['eligibility_status'],
                         'COMPLETE_INELIGIBLE_RUNTIME_HEADROOM')


class ParetoTests(unittest.TestCase):
    def _points(self) -> list[dict[str, object]]:
        return [
            {'model_config_id': 'fast-weak', 'a': 0.3, 's': 80.0, 'v': 2000.0},
            {'model_config_id': 'slow-strong', 'a': 0.5, 's': 35.0, 'v': 3600.0},
            {'model_config_id': 'dominated', 'a': 0.3, 's': 40.0, 'v': 3000.0},
            {'model_config_id': 'tie-a', 'a': 0.4, 's': 60.0, 'v': 2500.0},
            {'model_config_id': 'tie-b', 'a': 0.4, 's': 60.0, 'v': 2500.0},
        ]

    def test_speed_skyline(self) -> None:
        members = pareto_skyline(
            self._points(), x_key='a', y_key='s', y_bigger_better=True
        )
        self.assertIn('fast-weak', members)
        self.assertIn('slow-strong', members)
        self.assertNotIn('dominated', members)
        # Exact ties remain co-members.
        self.assertIn('tie-a', members)
        self.assertIn('tie-b', members)

    def test_vram_skyline(self) -> None:
        members = pareto_skyline(
            self._points(), x_key='a', y_key='v', y_bigger_better=False
        )
        self.assertIn('fast-weak', members)
        self.assertIn('slow-strong', members)
        self.assertNotIn('dominated', members)

    def test_missing_values_never_dominate(self) -> None:
        points = self._points() + [
            {'model_config_id': 'ghost', 'a': None, 's': None, 'v': None}
        ]
        members = pareto_skyline(
            points, x_key='a', y_key='s', y_bigger_better=True
        )
        self.assertNotIn('ghost', members)


class QuantPairTests(unittest.TestCase):
    def test_passthrough(self) -> None:
        comparisons = {
            'q4-vs-q5': {
                'trial_accuracy_delta': 0.02, 'all_pass_delta': 1,
                'transitions_strict': {
                    'fail_to_pass': ['Q1'], 'pass_to_fail': [],
                },
                'mcnemar_b': 1, 'mcnemar_c': 0, 'mcnemar_exact_p': 0.5,
            }
        }
        rows = summarize_quant_pairs(comparisons)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['net_task_gain'], 1)
        self.assertEqual(rows[0]['mcnemar_exact_p'], 0.5)


class ProvenanceSplitTests(unittest.TestCase):
    def test_report_pinned_freeze_live(self) -> None:
        # Report provenance must remain the sealed commit even when HEAD
        # is an arbitrary maintenance commit; freeze tracks live HEAD.
        import json

        V2_SEALED = '1d63305fadd416387f218ad8ba868e85a1345382'
        root = Path(__file__).resolve().parents[1]
        report = json.loads(
            (root / 'results/reports/sweep-v2-report.json').read_text(encoding='utf-8')
        )
        freeze = json.loads(
            (root / 'results/reports/sweep-v2-freeze.json').read_text(encoding='utf-8')
        )
        self.assertEqual(
            report['provenance']['analysis_code_git_commit'], V2_SEALED
        )
        self.assertRegex(freeze['analysis_code_git_commit'], r'^[0-9a-f]{40}$')
        self.assertNotEqual(
            report['provenance']['analysis_code_git_commit'],
            freeze['analysis_code_git_commit'],
        )


if __name__ == '__main__':
    unittest.main()
