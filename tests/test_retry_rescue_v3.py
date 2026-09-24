"""Offline contract tests for retry-rescue-v3 deterministic rendering."""

from __future__ import annotations

import inspect
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from analysis.retry_rescue_v3 import (
    RENDERED,
    UNRENDERABLE,
    load_v3_contract,
    load_source_rows,
    open_readonly_source,
    render_numeric,
)
from scripts.run_retry_rescue_v3 import build_v3_document


class RendererTests(unittest.TestCase):
    def test_accepts_explicit_or_terminal_finite_numbers(self) -> None:
        self.assertEqual(render_numeric('Work\nFinal answer: 218.40').output, '218.4')
        self.assertEqual(render_numeric('Work\n1e3').output, '1000')

    def test_rejects_prose_units_ambiguity_and_non_finite_forms(self) -> None:
        for text in (
            'Final answer: 136 kg', 'Final answer: NaN', 'Final answer: Infinity',
            'Final answer: 136\nFinal answer: 137', 'Derivation only: 136',
        ):
            self.assertEqual(render_numeric(text).status, UNRENDERABLE, text)

    def test_same_repeated_final_is_not_ambiguous(self) -> None:
        result = render_numeric('Final answer: 136\n136')
        self.assertEqual(result.status, RENDERED)
        self.assertEqual(result.output, '136')

    def test_renderer_has_no_expected_answer_or_grader_inputs(self) -> None:
        self.assertEqual(list(inspect.signature(render_numeric).parameters), ['raw_output'])


class SourceAndContractTests(unittest.TestCase):
    def test_repo_contract_selects_exact_frozen_18(self) -> None:
        contract = load_v3_contract('configs/retry-rescue-v3.yaml')
        self.assertEqual(len(contract.identities), 18)
        self.assertEqual({task_id for task_id, _ in contract.identities},
                         {'Q019', 'Q020', 'Q021', 'Q022', 'Q024', 'Q070'})
        self.assertEqual(len(load_source_rows(contract)), 18)

    def test_contract_refuses_source_hash_drift(self) -> None:
        with patch('analysis.retry_rescue_v3.sha256_file', return_value='0' * 64):
            with self.assertRaisesRegex(ValueError, 'baseline source hash mismatch'):
                load_v3_contract('configs/retry-rescue-v3.yaml')

    def test_report_generation_is_deterministic(self) -> None:
        config = Path('configs/retry-rescue-v3.yaml')
        self.assertEqual(build_v3_document(config), build_v3_document(config))

    def test_source_is_opened_read_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'source.db'
            writable = sqlite3.connect(path)
            writable.execute('CREATE TABLE rows (value TEXT)')
            writable.commit()
            writable.close()
            readonly = open_readonly_source(path)
            try:
                with self.assertRaises(sqlite3.OperationalError):
                    readonly.execute("INSERT INTO rows VALUES ('mutate')")
            finally:
                readonly.close()


if __name__ == '__main__':
    unittest.main()
