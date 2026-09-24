"""Run reliability analysis for one experiment and write the summary.

Population membership is status-driven (frozen grading spec, read-only);
unknown task IDs are a hard error. Summaries are stamped with a provenance
block; a dirty worktree refuses release output unless --allow-dirty is set
(development-only, stamped true).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml

from analysis.reliability import (
    DirtyWorktreeError,
    UnknownStatusError,
    analyze_grouped,
    collect_provenance,
    load_trials,
    report_to_json,
    summarize_console,
)
from evals.contract import EvalContractError, load_eval_contract
from storage.db import connect

# Tasks whose outputs are JSON: canonical-JSON variation counting applies.
JSON_TASKS = frozenset(
    {
        'Q001', 'Q002', 'Q003', 'Q004', 'Q005', 'Q006', 'Q007', 'Q008',
        'Q009', 'Q010', 'Q016', 'Q052', 'Q056', 'Q057', 'Q058', 'Q061',
        'Q062', 'Q065', 'Q066', 'Q069', 'Q080',
    }
)


def load_experiment_config(
    root: Path, experiment: str, *, config_override: str | None = None
) -> dict[str, Any]:
    candidates = []
    if config_override:
        candidates.append(Path(config_override))
    candidates.append(root / 'configs' / f'{experiment}.yaml')
    # Execution ids take the form <spec>__<model>: fall back to the spec file.
    if '__' in experiment:
        candidates.append(root / 'configs' / f'{experiment.split("__")[0]}.yaml')
    for path in candidates:
        if path.exists():
            config = yaml.safe_load(open(path, encoding='utf-8'))
            assert isinstance(config, dict)
            return {str(k): v for k, v in config.items()}
    return {'experiment': experiment}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Analyze reliability runs')
    parser.add_argument('--db', required=True)
    parser.add_argument('--experiment', required=True)
    parser.add_argument('--out', default=None)
    parser.add_argument('--config', default=None,
                        help='experiment spec YAML (default: configs/<spec>.yaml)')
    parser.add_argument('--spec', default='evals/specs/eval-v1-grading.yaml')
    parser.add_argument('--freeze', default='evals/specs/eval-v1-grading.freeze.json')
    parser.add_argument('--allow-dirty', action='store_true',
                        help='development-only: stamp dirty worktree instead of refusing')
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parent.parent
    conn = connect(args.db)
    grouped = load_trials(conn, args.experiment)
    conn.close()
    if not grouped:
        print(f'no measured rows for experiment {args.experiment}')
        return 2
    try:
        contract = load_eval_contract(
            root, args.spec, freeze_path=args.freeze
        )
    except EvalContractError as exc:
        print(f'CONTRACT REFUSED: {exc}')
        return 2
    statuses = dict(contract.statuses)
    try:
        report = analyze_grouped(
            args.experiment, grouped, statuses, json_tasks=JSON_TASKS
        )
    except UnknownStatusError as exc:
        print(f'STATUS ERROR: {exc}')
        return 4
    try:
        provenance = collect_provenance(
            str(root), load_experiment_config(root, args.experiment, config_override=args.config),
            allow_dirty=args.allow_dirty, spec_path=args.spec, freeze_path=args.freeze,
        )
    except DirtyWorktreeError as exc:
        print(f'PROVENANCE REFUSED: {exc}')
        return 5
    document = report_to_json(report, provenance)
    out_path = (
        Path(args.out)
        if args.out
        else root / 'results/summaries' / f'{args.experiment}-reliability.json'
    )
    out_path.write_text(document + '\n', encoding='utf-8')
    print(summarize_console(report))
    print(f'report: {out_path}')
    print(f'provenance: commit={provenance.analysis_code_git_commit} '
          f'dirty={provenance.analysis_worktree_dirty}')
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
