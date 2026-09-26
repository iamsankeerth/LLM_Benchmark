"""Offline guards for retry-rescue-v2 representability resolution."""

from __future__ import annotations

import copy
import sqlite3
import tempfile
import unittest
from pathlib import Path
from typing import Any

from analysis.retry_rescue_v2 import (
    NUMERIC_CONSTRAINT,
    PROBE_COHORTS,
    canonical_format_bytes,
    canonical_format_mapping,
    format_mapping_sha256,
    load_v2_contract,
    reject_retry_suffix,
    validate_c_resume,
    v2_run_config_hash,
    constraint_sha256,
    freeze_matrix,
    is_bare_json_number,
)
from analysis.retry_rescue_v2_report import build_v2_report
from analysis.retry_population import load_task_meta
from inference.adapters import get_model_config
from scripts.probe_retry_rescue_v2 import build_cleanup_record
from scripts.run_retry_rescue_v2 import verify_prompt_invariance
from storage.db import RunRecord, connect, create_experiment, insert_run, migrate_v2_run_columns
from tests.platform_scope import requires_windows_digests


def _draft_matrix() -> dict[str, Any]:
    primary_rows = [
        {'task_id': task_id, 'trial': trial, 'eligible': 'REQUIRES_LIVE_VERIFICATION'}
        for task_id, cohort in PROBE_COHORTS.items()
        if cohort == 'primary'
        for trial in (1, 2, 3)
    ]
    primary_rows.extend(
        {'task_id': f'S{index:03}', 'trial': 1, 'eligible': False}
        for index in range(21)
    )
    return {
        'source_population_rows': 39,
        'matrix_status': 'DRAFT_STATIC',
        'row_eligibility': primary_rows,
        'control_population': {
            'source_rows': 12,
            'matrix_status': 'DRAFT_STATIC',
            'row_eligibility': [
                *({'task_id': 'Q016', 'trial': trial, 'eligible': True} for trial in (1, 2, 3)),
                *({'task_id': 'Q023', 'trial': trial, 'eligible': False} for trial in (1, 2, 3)),
                *({'task_id': 'Q025', 'trial': trial, 'eligible': False} for trial in (1, 2, 3)),
                *({'task_id': 'Q026', 'trial': trial, 'eligible': 'REQUIRES_LIVE_VERIFICATION'} for trial in (1, 2, 3)),
            ],
        },
    }


def _records(compatible: dict[str, bool]) -> list[dict[str, Any]]:
    digest = constraint_sha256(NUMERIC_CONSTRAINT)
    return [
        {
            'task_id': task_id,
            'cohort': cohort,
            'constraint': NUMERIC_CONSTRAINT,
            'constraint_sha256': digest,
            'transport_completed': True,
            'representation_compatible': compatible[task_id],
            'frozen_grader_passed': False,
        }
        for task_id, cohort in PROBE_COHORTS.items()
    ]


class BareJsonNumberTests(unittest.TestCase):
    def test_accepts_only_standard_top_level_json_numbers(self) -> None:
        for output in ('136', '218.4', '-7', '1e3', ' 136\n'):
            self.assertTrue(is_bare_json_number(output), output)
        for output in (
            'true', 'NaN', 'Infinity', '"136"', '{"value":136}',
            'Answer: 136', '136 kg', '```json\n136\n```',
        ):
            self.assertFalse(is_bare_json_number(output), output)


class MatrixResolutionTests(unittest.TestCase):
    def test_resolves_whole_task_triplets_and_counts(self) -> None:
        compatible = {task_id: True for task_id in PROBE_COHORTS}
        compatible['Q020'] = False
        compatible['Q026'] = False
        matrix = freeze_matrix(
            _draft_matrix(), _records(compatible),
            evidence_path='results/reports/evidence.json', evidence_sha256='a' * 64,
            probe_code_git_commit='b' * 40,
        )
        self.assertEqual(matrix['matrix_status'], 'FROZEN')
        self.assertEqual(matrix['eligible_rows'], 15)
        self.assertEqual(matrix['ineligible_rows'], 24)
        self.assertEqual(matrix['requires_live_verification_rows'], 0)
        control = matrix['control_population']
        self.assertEqual(control['matrix_status'], 'FROZEN')
        self.assertEqual(control['eligible_rows'], 3)
        self.assertEqual(control['ineligible_rows'], 9)
        self.assertEqual(control['requires_live_verification_rows'], 0)

    def test_rejects_missing_or_alternate_probe_records(self) -> None:
        records = _records({task_id: True for task_id in PROBE_COHORTS})
        with self.assertRaisesRegex(ValueError, 'task set'):
            freeze_matrix(
                _draft_matrix(), records[:-1], evidence_path='e', evidence_sha256='a' * 64,
                probe_code_git_commit='b' * 40,
            )
        alternate = copy.deepcopy(records)
        alternate[0]['constraint'] = {'type': 'string'}
        with self.assertRaisesRegex(ValueError, 'alternate constraint'):
            freeze_matrix(
                _draft_matrix(), alternate, evidence_path='e', evidence_sha256='a' * 64,
                probe_code_git_commit='b' * 40,
            )


class CleanupRecordTests(unittest.TestCase):
    def test_missing_baseline_never_claims_disk_reclaim(self) -> None:
        cleanup = build_cleanup_record(
            model_loaded_after_run=False,
            model_installed_after_run=False,
            free_before_bytes=None,
            free_after_bytes=123,
        )
        self.assertEqual(cleanup['cleanup_status'], 'MODEL_STATE_VERIFIED')
        self.assertFalse(cleanup['disk_reclaim_verified'])
        self.assertEqual(
            cleanup['disk_reclaim_status'],
            'NOT_VERIFIABLE_PRE_RUN_BASELINE_MISSING',
        )

    def test_snapshots_measure_reclaim_separately_from_model_state(self) -> None:
        cleanup = build_cleanup_record(
            model_loaded_after_run=False,
            model_installed_after_run=False,
            free_before_bytes=1000,
            free_after_bytes=1000,
        )
        self.assertEqual(cleanup['disk_reclaim_delta_bytes'], 0)
        self.assertTrue(cleanup['disk_reclaim_verified'])


class ContractTests(unittest.TestCase):
    def test_canonical_formats_and_mapping_are_order_independent(self) -> None:
        structured = {
            'additionalProperties': False,
            'properties': {'risk': {'type': 'string'}},
            'required': ['risk'],
            'type': 'object',
        }
        reordered = {
            'type': 'object', 'required': ['risk'],
            'properties': {'risk': {'type': 'string'}},
            'additionalProperties': False,
        }
        self.assertEqual(canonical_format_bytes(structured), canonical_format_bytes(reordered))
        first = canonical_format_mapping({'Q016': ('STRUCTURED_JSON', structured)})
        second = canonical_format_mapping({'Q016': ('STRUCTURED_JSON', reordered)})
        self.assertEqual(first, second)
        self.assertEqual(format_mapping_sha256(first), format_mapping_sha256(second))

    @requires_windows_digests
    def test_repo_contract_resolves_frozen_18_and_6_population(self) -> None:
        contract = load_v2_contract(
            'configs/retry-rescue-v2.yaml',
            'results/reports/retry-rescue-v2-capability-matrix.json',
        )
        self.assertEqual(len(contract.primary_identities), 18)
        self.assertEqual(len(contract.control_identities), 6)
        self.assertEqual(contract.format_for('Q019'), NUMERIC_CONSTRAINT)
        self.assertEqual(contract.format_for('Q016')['additionalProperties'], False)
        self.assertEqual(contract.format_kind_for('Q016'), 'STRUCTURED_JSON')

    def test_suffix_refusal(self) -> None:
        reject_retry_suffix('Original task prompt')
        with self.assertRaisesRegex(ValueError, 'retry suffix'):
            reject_retry_suffix('Original task prompt\n\nReturn only the final numeric value required by the task.')

    @requires_windows_digests
    def test_sealed_baseline_prompt_invariance(self) -> None:
        contract = load_v2_contract(
            'configs/retry-rescue-v2.yaml',
            'results/reports/retry-rescue-v2-capability-matrix.json',
        )
        prompts = verify_prompt_invariance(
            contract=contract,
            task_meta=load_task_meta('evals/datasets/eval-v1/executable-v1.jsonl'),
            config=get_model_config('qwen3-4b-q4'),
            baseline_db=Path('results/local/full-baseline-v2__qwen3-4b-q4.db'),
            baseline_execution='full-baseline-v2__qwen3-4b-q4',
        )
        self.assertEqual(len(prompts), 24)


class StorageMigrationTests(unittest.TestCase):
    def test_v2_columns_are_added_only_when_requested(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / 'v2.db')
            conn = connect(db_path)
            try:
                conn.execute('CREATE TABLE runs (run_id INTEGER PRIMARY KEY)')
                conn.commit()
                migrate_v2_run_columns(conn)
                columns = {row[1] for row in conn.execute('PRAGMA table_info(runs)')}
                self.assertTrue({
                    'original_prompt_sha256', 'response_format_kind',
                    'response_format_sha256',
                }.issubset(columns))
            finally:
                conn.close()

    def test_resume_refuses_format_or_identity_drift(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / 'v2.db')
            conn = connect(db_path)
            try:
                from storage.db import init_schema

                init_schema(conn)
                migrate_v2_run_columns(conn)
                create_experiment(conn, experiment_id='c', name='c', config_hash='c' * 64,
                                  config_yaml='{}', created_at_utc='2026-09-20T00:00:00Z')
                record = _record('c', 'Q019', 1, 'FAIL', '0')
                insert_run(conn, record)
                conn.commit()
                with self.assertRaisesRegex(ValueError, 'format hash mismatch'):
                    validate_c_resume(
                        conn, 'c', {('Q019', 1)}, 'r' * 64,
                        {'Q019': ('NUMBER', {'type': 'number'})},
                    )
            finally:
                conn.close()


def _record(
    experiment_id: str, task_id: str, trial: int, verdict: str, raw_output: str,
    *, warmup: bool = False,
) -> RunRecord:
    return RunRecord(
        experiment_id=experiment_id, model_config_id='qwen3-4b-q4', task_id=task_id,
        trial=trial, run_kind='WARMUP' if warmup else 'RETRY_RESCUE',
        run_config_hash=('w' if warmup else 'r') * 64, is_warmup=warmup,
        prompt=f'prompt {task_id}', rendered_prompt_sha256='p' * 64,
        original_prompt_sha256='o' * 64, temperature=0.0, num_ctx=4096,
        num_predict=2048, template_sha256='t' * 64, grader_verdict=verdict,
        status='COMPLETE', started_at_utc='2026-09-20T00:00:00Z',
        ended_at_utc='2026-09-20T00:00:01Z', raw_output=raw_output,
        response_format_kind='NUMBER', response_format_sha256='f' * 64,
    )


class ReducerTests(unittest.TestCase):
    def _db(self, path: Path, execution: str, rows: list[tuple[str, int, str, str]]) -> None:
        conn = connect(str(path))
        try:
            from storage.db import init_schema

            init_schema(conn)
            migrate_v2_run_columns(conn)
            create_experiment(conn, experiment_id=execution, name=execution,
                              config_hash='c' * 64, config_yaml='{}',
                              created_at_utc='2026-09-20T00:00:00Z')
            for task_id, trial, verdict, raw in rows:
                insert_run(conn, _record(execution, task_id, trial, verdict, raw))
            conn.commit()
        finally:
            conn.close()

    def test_reducer_requires_exact_scoped_identities_and_excludes_controls(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ids = [('Q019', 1), ('Q019', 2), ('Q019', 3)]
            controls = [('Q016', 1), ('Q016', 2), ('Q016', 3)]
            a_rows = [(task, trial, 'FAIL', 'Answer: 136') for task, trial in ids + controls]
            b_rows = [(task, trial, 'PASS' if trial == 1 else 'FAIL', '136') for task, trial in ids]
            b_rows += [(task, trial, 'FAIL', '{}') for task, trial in controls]
            c_rows = [(task, trial, 'PASS' if trial == 2 else 'FAIL', '136') for task, trial in ids]
            c_rows += [(task, trial, 'PASS', '{}') for task, trial in controls]
            self._db(root / 'a.db', 'a', a_rows)
            self._db(root / 'b.db', 'b', b_rows)
            self._db(root / 'c.db', 'c', c_rows)
            report = build_v2_report(
                baseline_db=str(root / 'a.db'), baseline_execution='a',
                retry_db=str(root / 'b.db'), retry_execution='b',
                constrained_db=str(root / 'c.db'), constrained_execution='c',
                primary_identities=ids, control_identities=controls,
                grader_types={'Q019': 'numeric'},
            )
            self.assertEqual(report['primary']['A']['recovered'], 0)
            self.assertEqual(report['primary']['B']['recovered'], 1)
            self.assertEqual(report['primary']['C']['recovered'], 1)
            self.assertEqual(report['primary']['paired_B_C']['B_only_PASS'], 1)
            self.assertEqual(report['primary']['paired_B_C']['C_only_PASS'], 1)
            self.assertEqual(report['controls']['C']['recovered'], 3)
            self.assertEqual(report['primary']['content_change']['contract_only'], 1)

    def test_reducer_rejects_missing_selected_identity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = [('Q019', 1, 'FAIL', '136')]
            self._db(root / 'a.db', 'a', rows)
            self._db(root / 'b.db', 'b', rows)
            self._db(root / 'c.db', 'c', rows)
            with self.assertRaisesRegex(ValueError, 'identity mismatch'):
                build_v2_report(
                    baseline_db=str(root / 'a.db'), baseline_execution='a',
                    retry_db=str(root / 'b.db'), retry_execution='b',
                    constrained_db=str(root / 'c.db'), constrained_execution='c',
                    primary_identities=[('Q019', 1), ('Q019', 2)],
                    control_identities=[], grader_types={'Q019': 'numeric'},
                )

    def test_read_only_inputs_cannot_be_migrated(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'sealed.db'
            connection = sqlite3.connect(path)
            try:
                connection.execute('CREATE TABLE runs (run_id INTEGER PRIMARY KEY)')
                connection.commit()
            finally:
                connection.close()
            uri = f'file:{path.as_posix()}?mode=ro'
            readonly = sqlite3.connect(uri, uri=True)
            try:
                with self.assertRaises(sqlite3.OperationalError):
                    migrate_v2_run_columns(readonly)
            finally:
                readonly.close()


class ExecutionIdentityTests(unittest.TestCase):
    def test_run_hash_binds_population_and_format_mapping(self) -> None:
        first = v2_run_config_hash(
            model_config_id='qwen3-4b-q4', temperature=0.0, num_ctx=4096,
            model_config_sha256='m' * 64, num_predict=2048, template_sha256='t' * 64, grader_spec_sha256='g' * 64,
            population_sha256='p' * 64, format_mapping_sha256='f' * 64,
        )
        changed = v2_run_config_hash(
            model_config_id='qwen3-4b-q4', temperature=0.0, num_ctx=4096,
            model_config_sha256='m' * 64, num_predict=2048, template_sha256='t' * 64, grader_spec_sha256='g' * 64,
            population_sha256='x' * 64, format_mapping_sha256='f' * 64,
        )
        self.assertEqual(len(first), 64)
        self.assertNotEqual(first, changed)


if __name__ == '__main__':
    unittest.main()
