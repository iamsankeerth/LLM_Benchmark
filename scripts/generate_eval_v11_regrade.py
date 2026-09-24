"""Generate the append-only Eval-v1.1 regrade overlay."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis.regrade import RegradeError, build_overlay
from evals.contract import EvalContractError, load_eval_contract


def main(argv: list[str] | None = None) -> int:
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description='Generate Eval-v1.1 regrade overlay')
    parser.add_argument('--out', default='results/reports/eval-v1.1-regrade-overlay.json')
    parser.add_argument(
        '--spec', default='evals/specs/eval-v1.1-grading.yaml'
    )
    parser.add_argument(
        '--freeze', default='evals/specs/eval-v1.1-grading.freeze.json'
    )
    parser.add_argument(
        '--completed-ids',
        default='results/summaries/sweep-full-baseline-v2-complete.json',
    )
    args = parser.parse_args(argv)
    try:
        contract = load_eval_contract(
            root, args.spec, freeze_path=args.freeze
        )
        completed = json.loads(
            (root / args.completed_ids).read_text(encoding='utf-8')
        )['completed_ids']
        overlay = build_overlay(
            root=root, contract=contract, completed_ids=list(completed)
        )
    except (OSError, json.JSONDecodeError, EvalContractError, RegradeError) as exc:
        print(f'REGRADE REFUSED: {exc}', flush=True)
        return 2
    output = root / args.out
    if output.exists():
        print(f'REGRADE REFUSED: refusing to overwrite {output}', flush=True)
        return 2
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(overlay, indent=2, sort_keys=True) + '\n', encoding='utf-8'
    )
    print(
        f'regrade overlay: {overlay["record_count"]} records across '
        f'{overlay["source_db_count"]} source databases'
    )
    print(f'report: {output}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
