"""Generate the scoped, read-only A/B/C retry-rescue-v2 report."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis.retry_rescue_v2 import load_v2_contract
from analysis.retry_rescue_v2_report import build_v2_report


ROOT = Path(__file__).resolve().parent.parent


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_head() -> str:
    return subprocess.run(
        ['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True,
        check=True, timeout=15,
    ).stdout.strip()


def generate_report(config_path: Path) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding='utf-8'))
    if not isinstance(config, dict):
        raise ValueError('V2 config must be a mapping')
    contract = load_v2_contract(str(config_path), str(ROOT / config['capability_matrix']))
    spec_path = ROOT / 'evals/specs/eval-v1-grading.yaml'
    spec = yaml.safe_load(spec_path.read_text(encoding='utf-8'))
    if not isinstance(spec, dict):
        raise ValueError('grading spec must be a mapping')
    grader_types = {
        task_id: str(spec['tasks'][task_id]['graders'][0]['type'])
        for task_id, _ in contract.primary_identities
    }
    report = build_v2_report(
        baseline_db=str(ROOT / 'results/local/full-baseline-v2__qwen3-4b-q4.db'),
        baseline_execution=str(config['base_execution']),
        retry_db=str(ROOT / 'results/local/retry-rescue-v1__qwen3-4b-q4.db'),
        retry_execution=str(config['retry_execution']),
        constrained_db=str(ROOT / 'results/local/retry-rescue-v2__qwen3-4b-q4.db'),
        constrained_execution=str(config['execution_id']),
        primary_identities=contract.primary_identities,
        control_identities=contract.control_identities,
        grader_types=grader_types,
    )
    return {
        'title': 'Retry-Rescue V2: constrained decoding',
        'experiment': config['execution_id'],
        'primary_scope_note': (
            'Results apply to the 18 of 39 OUTPUT_CONTRACT rows whose frozen '
            'output contracts are representable by the tested constrained-decoding mechanism.'
        ),
        'result': report,
        'provenance': {
            'v2_code_git_commit': _git_head(),
            'config_sha256': _sha256(config_path),
            'population_sha256': contract.population_sha256,
            'format_mapping_sha256': contract.format_mapping_sha256,
            'grading_spec_sha256': _sha256(spec_path),
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/retry-rescue-v2.yaml')
    parser.add_argument('--output', default='results/reports/retry-rescue-v2.json')
    args = parser.parse_args(argv)
    document = generate_report(ROOT / args.config)
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(document, indent=2) + '\n', encoding='utf-8')
    print(f'wrote {output}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
