"""Aggregate the four-model Long Context v1 pilot reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def build(root: Path, report_paths: list[Path]) -> dict[str, Any]:
    reports = [json.loads(path.read_text(encoding='utf-8')) for path in report_paths]
    if len(reports) != 4 or any(report.get('trial_count') != 63 for report in reports):
        raise ValueError('pilot requires four complete 63-trial reports')
    return {
        'schema_version': 'long-context-pilot-v1',
        'suite': 'long-context-v1',
        'status': 'PILOT_COMPLETE',
        'model_count': 4,
        'task_count': 21,
        'trials_per_model': 63,
        'requests_per_model': 87,
        'models': [
            {
                'model_config_id': report['model_config_id'],
                'passes': report['passes'],
                'trial_accuracy': report['trial_accuracy'],
                'by_length': report['by_length'],
            }
            for report in sorted(reports, key=lambda value: value['model_config_id'])
        ],
        'claim_scope': 'Four-model pilot of the configured 4096-token full-context window; no native 8K/32K claim.',
    }


def main(argv: list[str] | None = None) -> int:
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description='Aggregate Long Context v1 pilot')
    parser.add_argument('--out', default='results/reports/long-context-v1-pilot.json')
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args(argv)
    paths = sorted((root / 'results/reports').glob('long-context-v1__*.json'))
    paths = [path for path in paths if 'pilot' not in path.name]
    if len(paths) != 4:
        print(f'LONG-CONTEXT PILOT REFUSED: expected 4 reports, found {len(paths)}', flush=True)
        return 2
    try:
        report = build(root, paths)
    except (OSError, ValueError, KeyError) as exc:
        print(f'LONG-CONTEXT PILOT REFUSED: {exc}', flush=True)
        return 2
    output = root / args.out
    if output.exists() and not args.force:
        print(f'LONG-CONTEXT PILOT REFUSED: exists {output}', flush=True)
        return 2
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    print('long-context pilot report: 4 models, 252 trials')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
