"""Run retry-rescue-v3 without model, network, or database writes."""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis.retry_rescue_v3 import (
    RENDERED,
    canonical_json,
    load_source_rows,
    load_v3_contract,
    render_numeric,
)
from evals.graders.engine import grade_output


ROOT = Path(__file__).resolve().parent.parent


def _git_head() -> str:
    return subprocess.run(
        ['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True,
        check=True, timeout=15,
    ).stdout.strip()


def build_v3_document(config_path: Path) -> dict[str, Any]:
    contract = load_v3_contract(str(config_path))
    spec = yaml.safe_load(contract.grading_spec.read_text(encoding='utf-8'))
    if not isinstance(spec, dict):
        raise ValueError('V3 grading spec must be a mapping')
    source_rows = load_source_rows(contract)
    records: list[dict[str, Any]] = []
    recovered = unrenderable = rendered_failed = 0
    for task_id, trial in contract.identities:
        raw_output = source_rows[(task_id, trial)]
        rendered = render_numeric(raw_output)
        record: dict[str, Any] = {
            'task_id': task_id,
            'trial': trial,
            'source_raw_output_sha256': hashlib.sha256(raw_output.encode('utf-8')).hexdigest(),
            'renderer_status': rendered.status,
            'rendered_output': rendered.output,
        }
        if rendered.status != RENDERED:
            unrenderable += 1
            record['grader_verdict'] = 'UNRENDERABLE'
        else:
            details = grade_output(rendered.output or '', spec['tasks'][task_id]['graders'])
            passed = all(detail.passed for detail in details)
            record['grader_verdict'] = 'PASS' if passed else 'FAIL'
            record['grader_details'] = [
                {'grader_type': detail.grader_type, 'passed': detail.passed,
                 'detail': detail.detail, 'violations': detail.violations}
                for detail in details
            ]
            if passed:
                recovered += 1
            else:
                rendered_failed += 1
        records.append(record)
    return {
        'title': 'Retry-Rescue V3: deterministic numeric renderer',
        'experiment': 'retry-rescue-v3',
        'scope_note': (
            'Transforms only the sealed 18-row numeric primary intersection; '
            'this is not a fresh-generation retry estimate.'
        ),
        'summary': {
            'total': len(records),
            'recovered': recovered,
            'unrenderable': unrenderable,
            'rendered_but_grader_failed': rendered_failed,
        },
        'records': records,
        'provenance': {
            'v3_code_git_commit': _git_head(),
            'config_sha256': contract.config_sha256,
            'baseline_execution': contract.baseline_execution,
            'baseline_db_sha256': contract.baseline_db_sha256,
            'grading_spec_sha256': contract.grading_spec_sha256,
            'population_sha256': contract.population_sha256,
            'renderer_code_sha256': contract.renderer_code_sha256,
            'ollama_calls': 0,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/retry-rescue-v3.yaml')
    parser.add_argument('--output', default='results/reports/retry-rescue-v3.json')
    args = parser.parse_args(argv)
    output_path = ROOT / args.output
    if output_path.exists():
        raise FileExistsError(f'refusing to overwrite V3 artifact: {output_path}')
    document = build_v3_document(ROOT / args.config)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(canonical_json(document), encoding='utf-8')
    print(f'wrote {output_path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
