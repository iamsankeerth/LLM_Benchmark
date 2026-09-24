"""Frozen temperature-study contract and fail-closed report tests."""

from __future__ import annotations

import json
import sqlite3
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import yaml

from analysis.temperature_study import TemperatureStudyError, validate_temperature_study_run
from evals.contract import load_eval_contract
from inference.adapters import get_model_config
from scripts.generate_temperature_study_report import generate_temperature_study_report
from storage.db import RunRecord, create_experiment, init_schema, insert_run
from storage.execution import ensure_provenance_table, record_execution_provenance


ROOT = Path(__file__).resolve().parents[1]
TASK_IDS = ['Q001', 'Q003', 'Q062', 'Q066', 'Q011', 'Q013', 'Q015', 'Q017',
            'Q019', 'Q023', 'Q026', 'Q035', 'Q039', 'Q042', 'Q071']


def _statuses() -> dict[str, str]:
    spec = yaml.safe_load(
        (ROOT / 'evals/specs/eval-v1-grading.yaml').read_text(encoding='utf-8')
    )
    return {str(task_id): str(entry['grading_status'])
            for task_id, entry in spec['tasks'].items()}


def _populate_arm(path: Path, *, arm: str, temperature: float, omit_last: bool = False) -> None:
    config = get_model_config('qwen3-4b-q4')
    execution_id = f'temperature-study-v1-{arm}__qwen3-4b-q4'
    conn = sqlite3.connect(path)
    init_schema(conn)
    ensure_provenance_table(conn)
    create_experiment(
        conn, experiment_id=execution_id, name=execution_id, config_hash='c' * 64,
        config_yaml='frozen', created_at_utc='2026-09-24T00:00:00Z',
    )
    record_execution_provenance(
        conn, execution_id=execution_id,
        experiment_spec_id=f'temperature-study-v1-{arm}',
        model_config_id='qwen3-4b-q4', experiment_config_hash='e' * 64,
        model_config_hash='m' * 64, model_artifact_digest=config.ollama_model_digest,
        created_at_utc='2026-09-24T00:00:00Z',
    )
    for trial in (1, 2):
        insert_run(conn, RunRecord(
            experiment_id=execution_id, model_config_id='qwen3-4b-q4',
            task_id=f'WARMUP-{trial}', trial=trial, run_kind='WARMUP',
            run_config_hash=f'{arm}-config', is_warmup=True, prompt='warmup',
            rendered_prompt_sha256=f'warmup-{trial}', temperature=temperature,
            num_ctx=config.num_ctx, num_predict=2048,
            template_sha256=config.template_sha256, grader_verdict='WARMUP',
            status='COMPLETE', started_at_utc='2026-09-24T00:00:00Z',
            ended_at_utc='2026-09-24T00:00:01Z', num_gpu=config.num_gpu,
            stop_tokens=list(config.stop_tokens), think=repr(config.think),
            model_digest=config.ollama_model_digest,
        ))
    pairs = [(task_id, trial) for task_id in TASK_IDS for trial in range(1, 6)]
    if omit_last:
        pairs.pop()
    for task_id, trial in pairs:
        insert_run(conn, RunRecord(
            experiment_id=execution_id, model_config_id='qwen3-4b-q4',
            task_id=task_id, trial=trial, run_kind='TEMPERATURE',
            run_config_hash=f'{arm}-config', is_warmup=False, prompt='prompt',
            rendered_prompt_sha256=f'prompt-{task_id}', temperature=temperature,
            num_ctx=config.num_ctx, num_predict=2048,
            template_sha256=config.template_sha256,
            grader_verdict='PASS' if trial % 2 else 'FAIL', status='COMPLETE',
            started_at_utc='2026-09-24T00:00:00Z', ended_at_utc='2026-09-24T00:00:01Z',
            num_gpu=config.num_gpu, stop_tokens=list(config.stop_tokens),
            think=repr(config.think), decode_tok_s=20.0, ttft_ms=50.0,
            eval_count=3, model_digest=config.ollama_model_digest,
        ))
    conn.commit()
    conn.close()
    contract = load_eval_contract(
        ROOT,
        'evals/specs/eval-v1.1-grading.yaml',
        freeze_path='evals/specs/eval-v1.1-grading.freeze.json',
    )
    manifest_dir = path.parent / 'experiment-manifests'
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        'app_git_commit': 'test-commit',
        'execution_id': execution_id,
        'experiment_spec_id': f'temperature-study-v1-{arm}',
        'model_config_id': 'qwen3-4b-q4',
        'model_config_hash': 'model-config-hash',
        'model_artifact_digest': config.ollama_model_digest,
        'model_identifier': config.ollama_identifier,
        'quantization': config.quantization,
        'template_sha256': config.template_sha256,
        'temperature': temperature,
        'num_ctx': config.num_ctx,
        'num_predict': 2048,
        'num_gpu': config.num_gpu,
        'stop_tokens': list(config.stop_tokens),
        'think': config.think,
        'contract_spec_version': contract.spec_version,
        'grading_spec_hash': contract.hashes['spec_sha256'],
        'dataset_hash': contract.hashes['dataset_sha256'],
        'eval_freeze_hash': contract.hashes['freeze_sha256'],
        'eval_freeze_record_sha256': contract.hashes['freeze_sha256'],
        'ollama_version': 'test-ollama',
        'python_version': 'test-python',
        'hardware_id': 'test-hardware',
        'power_mode': 'test-power',
    }
    (manifest_dir / f'{execution_id}.json').write_text(
        json.dumps(manifest), encoding='utf-8'
    )


class TemperatureStudyContractTests(unittest.TestCase):
    def test_scalar_arms_match_the_frozen_contract(self) -> None:
        statuses = _statuses()
        for arm in ('t0', 't07'):
            config = yaml.safe_load(
                (ROOT / f'configs/temperature-study-v1-{arm}.yaml').read_text(encoding='utf-8')
            )
            contract = validate_temperature_study_run(
                config, root=ROOT, model_config_id='qwen3-4b-q4',
                task_ids=[str(task_id) for task_id in config['task_ids']],
                temperature=float(config['temperature']), trials=int(config['trials']),
                num_predict=int(config['num_predict']), grading_statuses=statuses,
            )
            self.assertIsNotNone(contract)
            assert contract is not None
            self.assertEqual(contract['study'], 'temperature-study-v1')

    def test_partial_population_is_refused_before_execution(self) -> None:
        config = yaml.safe_load(
            (ROOT / 'configs/temperature-study-v1-t0.yaml').read_text(encoding='utf-8')
        )
        with self.assertRaisesRegex(TemperatureStudyError, 'task_ids'):
            validate_temperature_study_run(
                config, root=ROOT, model_config_id='qwen3-4b-q4',
                task_ids=[str(task_id) for task_id in config['task_ids'][:-1]],
                temperature=0.0, trials=5, num_predict=2048, grading_statuses=_statuses(),
            )


class TemperatureStudyReportTests(unittest.TestCase):
    def test_report_requires_exact_paired_population(self) -> None:
        with TemporaryDirectory() as tmp:
            base = Path(tmp) / 't0.db'
            candidate = Path(tmp) / 't07.db'
            _populate_arm(base, arm='t0', temperature=0.0)
            _populate_arm(candidate, arm='t07', temperature=0.7)
            report = generate_temperature_study_report(
                root=ROOT, db_t0=str(base), db_t07=str(candidate),
                config_t0=ROOT / 'configs/temperature-study-v1-t0.yaml',
                config_t07=ROOT / 'configs/temperature-study-v1-t07.yaml',
                manifest_root=Path(tmp),
            )
            self.assertEqual(report['population']['measured_rows_per_arm'], 75)
            self.assertEqual(sum(len(rows) for rows in report['trial_transitions'].values()), 75)

    def test_report_refuses_an_incomplete_arm(self) -> None:
        with TemporaryDirectory() as tmp:
            base = Path(tmp) / 't0.db'
            candidate = Path(tmp) / 't07.db'
            _populate_arm(base, arm='t0', temperature=0.0)
            _populate_arm(candidate, arm='t07', temperature=0.7, omit_last=True)
            with self.assertRaisesRegex(TemperatureStudyError, 'frozen 75 rows'):
                generate_temperature_study_report(
                    root=ROOT, db_t0=str(base), db_t07=str(candidate),
                    config_t0=ROOT / 'configs/temperature-study-v1-t0.yaml',
                    config_t07=ROOT / 'configs/temperature-study-v1-t07.yaml',
                    manifest_root=Path(tmp),
                )

    def test_report_refuses_non_temperature_row_drift(self) -> None:
        with TemporaryDirectory() as tmp:
            base = Path(tmp) / 't0.db'
            candidate = Path(tmp) / 't07.db'
            _populate_arm(base, arm='t0', temperature=0.0)
            _populate_arm(candidate, arm='t07', temperature=0.7)
            conn = sqlite3.connect(candidate)
            conn.execute("UPDATE runs SET template_sha256='wrong' WHERE is_warmup=0")
            conn.commit()
            conn.close()
            with self.assertRaisesRegex(TemperatureStudyError, 'non-temperature configuration drift'):
                generate_temperature_study_report(
                    root=ROOT, db_t0=str(base), db_t07=str(candidate),
                    config_t0=ROOT / 'configs/temperature-study-v1-t0.yaml',
                    config_t07=ROOT / 'configs/temperature-study-v1-t07.yaml',
                    manifest_root=Path(tmp),
                )
