"""Status-driven capability summary for one experiment.

Reports deterministic_trial_accuracy / any_pass_3 / all_pass_3 over
READY_DETERMINISTIC tasks only, with READY_JUDGE rows under judge_prechecks.
Same provenance fail-closed behavior as the reliability analyzer.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis.capability import (
    build_capability_summary,
    capability_to_json,
    summarize_capability_console,
)
from analysis.reliability import (
    DirtyWorktreeError,
    UnknownStatusError,
    analyze_grouped,
    collect_provenance,
    load_spec_statuses,
    load_trials,
)
from scripts.analyze_reliability import JSON_TASKS, load_experiment_config
from storage.db import connect


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Summarize baseline capability')
    parser.add_argument('--db', required=True)
    parser.add_argument('--experiment', required=True)
    parser.add_argument('--out', default=None)
    parser.add_argument('--config', default=None,
                        help='experiment spec YAML (default: configs/<spec>.yaml)')
    parser.add_argument('--allow-dirty', action='store_true')
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parent.parent
    conn = connect(args.db)
    grouped = load_trials(conn, args.experiment)
    conn.close()
    if not grouped:
        print(f'no measured rows for experiment {args.experiment}')
        return 2
    statuses = load_spec_statuses(str(root / 'evals/specs/eval-v1-grading.yaml'))
    try:
        reliability = analyze_grouped(
            args.experiment, grouped, statuses, json_tasks=JSON_TASKS
        )
    except UnknownStatusError as exc:
        print(f'STATUS ERROR: {exc}')
        return 4
    tasks = {**reliability.deterministic_tasks, **reliability.judge_tasks}
    try:
        summary = build_capability_summary(
            args.experiment, tasks, statuses,
            collect_provenance(
                str(root), load_experiment_config(root, args.experiment, config_override=args.config),
                allow_dirty=args.allow_dirty,
            ),
        )
    except DirtyWorktreeError as exc:
        print(f'PROVENANCE REFUSED: {exc}')
        return 5
    out_path = (
        Path(args.out)
        if args.out
        else root / 'results/summaries' / f'{args.experiment}-capability.json'
    )
    out_path.write_text(capability_to_json(summary) + '\n', encoding='utf-8')
    print(summarize_capability_console(summary))
    print(f'report: {out_path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
