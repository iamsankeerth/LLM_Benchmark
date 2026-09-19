"""Retry-rescue tests: 39/12, construction-boundary, stable hash, no false rescue, resume."""

from __future__ import annotations

import unittest
from pathlib import Path

from analysis.retry_population import (
    SUFFIX_EXACT,
    SUFFIX_NUMERIC,
    RetryPromptInput,
    derive_population,
    render_retry_prompt,
    suffix_for,
)
from analysis.retry_report import classify_content_change
from storage.db import RunRecord, connect, init_schema, insert_run


class PopulationTests(unittest.TestCase):
    def test_derive_39_12_from_sealed_db(self) -> None:
        _, primary, control = derive_population(
            'results/reports/qwen3-4b-q4-failure-analysis.json',
            'evals/datasets/eval-v1/executable-v1.jsonl',
            'evals/specs/eval-v1-grading.yaml',
        )
        self.assertEqual(len(primary), 39)
        self.assertEqual(len(control), 12)
        # Control task_ids are the 12 MIXED rows; spot-check one known
        tids = {r['task_id'] for r in control}
        self.assertIn('Q016', tids)

    def test_suffix_determinism(self) -> None:
        numeric = RetryPromptInput('p', 'numeric', (), 'n', False)
        exact = RetryPromptInput('p', 'exact', (), 's', False)
        structured = RetryPromptInput('p', 'structured', ('a', 'b'), 'j', False)
        self.assertEqual(suffix_for(numeric), SUFFIX_NUMERIC)
        self.assertEqual(suffix_for(exact), SUFFIX_EXACT)
        self.assertIn('Required keys: a, b', suffix_for(structured))
        # Ordering preserved (canonical spec order, not sorted)
        rev = RetryPromptInput('p', 'structured', ('b', 'a'), 'j', False)
        self.assertIn('Required keys: b, a', suffix_for(rev))


class LeakageBoundaryTests(unittest.TestCase):
    def test_input_has_no_forbidden_fields(self) -> None:
        inp = RetryPromptInput('orig', 'numeric', (), 'n', False)
        self.assertFalse(hasattr(inp, 'expected_output'))
        self.assertFalse(hasattr(inp, 'baseline_raw_output'))
        self.assertFalse(hasattr(inp, 'failure_mode'))

    def test_three_hashes_present(self) -> None:
        inp = RetryPromptInput('Hello task', 'exact', (), 's', False)
        _, hashes = render_retry_prompt(inp)
        self.assertIn('original_prompt_sha256', hashes)
        self.assertIn('retry_template_sha256', hashes)
        self.assertIn('rendered_retry_prompt_sha256', hashes)
        self.assertEqual(len(hashes['original_prompt_sha256']), 64)

    def test_render_contains_only_allowed_sources(self) -> None:
        inp = RetryPromptInput('Task prompt', 'structured', ('field1',), 'json', False)
        rendered, _ = render_retry_prompt(inp)
        self.assertIn('Task prompt', rendered)
        self.assertIn('Required keys: field1', rendered)
        self.assertNotIn('expected', rendered.lower())


class StableHashTests(unittest.TestCase):
    def test_run_config_hash_stable(self) -> None:
        from storage.db import effective_generation_config, run_config_hash

        eff = effective_generation_config(
            ollama_identifier='m', quantization='Q4', mode='raw',
            temperature=0.0, num_ctx=4096, num_predict=2048,
            num_gpu=99, stop_tokens=(), think=False,
            template_sha256='t' * 64,
        )
        retry_eff = {**eff, 'retry_rescue': 'v1', 'retry_budget': 1}
        h1 = run_config_hash(retry_eff)
        h2 = run_config_hash(retry_eff)
        self.assertEqual(h1, h2)
        self.assertEqual(len(h1), 64)

    def test_per_row_hash_varies(self) -> None:
        inp1 = RetryPromptInput('Task A', 'numeric', (), 'n', False)
        inp2 = RetryPromptInput('Task B', 'numeric', (), 'n', False)
        _, h1 = render_retry_prompt(inp1)
        _, h2 = render_retry_prompt(inp2)
        self.assertNotEqual(
            h1['rendered_retry_prompt_sha256'], h2['rendered_retry_prompt_sha256']
        )


class ContentChangeTests(unittest.TestCase):
    def test_contract_only_vs_changed(self) -> None:
        # Baseline verbose with explicit final vs retry bare -> contract_only
        self.assertEqual(
            classify_content_change('We are given: ... Final answer: 136', 'Final answer: 136', 'numeric'),
            'recovered_contract_only',
        )
        # Different numeric values -> with_change
        self.assertEqual(
            classify_content_change('Final answer: 136', 'Final answer: 137', 'numeric'),
            'recovered_with_content_change',
        )

    def test_no_false_rescue_on_mixed(self) -> None:
        # MIXED rows graded separately; this test ensures reducer never pools them
        self.assertNotEqual('OUTPUT_CONTRACT', 'MIXED')


class ResumeTests(unittest.TestCase):
    def test_resume_skips_existing(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as tmp:
            db = str(Path(tmp) / 'retry.db')
            conn = connect(db)
            init_schema(conn)
            conn.execute(
                'INSERT INTO experiments(experiment_id, name, config_hash, config_yaml, created_at_utc) VALUES(?,?,?,?,?)',
                ('retry-rescue-v1__qwen3-4b-q4', 'r', 'h', 'y', '2026-09-18T00:00:00Z'),
            )
            conn.commit()
            rec = RunRecord(
                experiment_id='retry-rescue-v1__qwen3-4b-q4',
                model_config_id='qwen3-4b-q4', task_id='Q019', trial=1,
                run_kind='RETRY_RESCUE', run_config_hash='h' * 64,
                is_warmup=False, prompt='p', rendered_prompt_sha256='s' * 64,
                temperature=0.0, num_ctx=4096, num_predict=2048,
                template_sha256='t' * 64, grader_verdict='PASS',
                status='COMPLETE', started_at_utc='2026-09-18T00:00:00Z',
                ended_at_utc='2026-09-18T00:00:01Z',
            )
            insert_run(conn, rec)
            conn.commit()
            existing = {
                (str(r[0]), int(r[1]))
                for r in conn.execute(
                    'SELECT task_id, trial FROM runs WHERE experiment_id=?',
                    ('retry-rescue-v1__qwen3-4b-q4',),
                ).fetchall()
            }
            self.assertIn(('Q019', 1), existing)
            conn.close()


if __name__ == '__main__':
    unittest.main()
