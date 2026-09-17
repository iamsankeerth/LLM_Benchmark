"""Run reliability analysis for one experiment and write the summary."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml

from analysis.reliability import (
    analyze_experiment,
    analyze_task,
    load_trials,
    report_to_json,
    summarize_console,
)
from storage.db import connect

# Tasks whose outputs are JSON: canonical-JSON variation counting applies.
JSON_TASKS = frozenset(
    {
        'Q001', 'Q002', 'Q003', 'Q004', 'Q005', 'Q006', 'Q007', 'Q008',
        'Q009', 'Q010', 'Q016', 'Q052', 'Q056', 'Q057', 'Q058', 'Q061',
        'Q062', 'Q065', 'Q066', 'Q069', 'Q080',
    }
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Analyze reliability runs')
    parser.add_argument('--db', required=True)
    parser.add_argument('--experiment', required=True)
    parser.add_argument('--out', default=None)
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parent.parent
    conn = connect(args.db)
    grouped = load_trials(conn, args.experiment)
    conn.close()
    if not grouped:
        print(f'no measured rows for experiment {args.experiment}')
        return 2
    analyses = {
        task_id: analyze_task(trials, is_json_task=task_id in JSON_TASKS)
        for task_id, trials in grouped.items()
    }
    report = analyze_experiment(args.experiment, analyses)
    document = report_to_json(report)
    out_path = (
        Path(args.out)
        if args.out
        else root / 'results/summaries' / f'{args.experiment}-reliability.json'
    )
    out_path.write_text(document + '\n', encoding='utf-8')
    print(summarize_console(report))
    print(f'report: {out_path}')
    # Cross-check the frozen reliability config when analyzing it.
    config_path = root / 'configs/reliability-v1.yaml'
    if args.experiment == 'reliability-v1' and config_path.exists():
        config = yaml.safe_load(open(config_path, encoding='utf-8'))
        frozen = set(str(t).split('#')[0].strip() for t in config['task_ids'])
        missing = frozen - set(grouped)
        extra = set(grouped) - frozen
        if missing or extra:
            print(f'CONFIG DRIFT: missing={sorted(missing)} extra={sorted(extra)}')
            return 3
        print('config check: analyzed tasks match frozen reliability-v1.yaml')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
