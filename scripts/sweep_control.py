"""Sweep control: pause / cancel-pause / status from a second terminal.

Pause is a request, not an interruption: the running sweep finishes its
current atomic unit, checkpoints, unloads (keeping weights), and reports
SAFE TO SHUT DOWN. Never power off before that banner.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from storage.pause import (
    cancel_pause,
    pause_requested,
    request_pause,
)
from storage.sweep import load_sweep_state


def _checkpoints_dir(root: Path) -> Path:
    return root / 'results/checkpoints'


def cmd_pause(args: argparse.Namespace, root: Path) -> int:
    request_pause(_checkpoints_dir(root), args.experiment)
    print('Pause requested.', flush=True)
    print('The sweep will stop at the next safe boundary.', flush=True)
    print('DO NOT shut down until the runner reports SAFE TO SHUT DOWN.',
          flush=True)
    return 0


def cmd_cancel_pause(args: argparse.Namespace, root: Path) -> int:
    if cancel_pause(_checkpoints_dir(root), args.experiment):
        print('Pending pause request cancelled.', flush=True)
    else:
        print('No pending pause request.', flush=True)
    return 0


def cmd_status(args: argparse.Namespace, root: Path) -> int:
    checkpoints = _checkpoints_dir(root)
    state = load_sweep_state(checkpoints / f'sweep-{args.experiment}.json')
    if state is None:
        print(f'no sweep state for experiment {args.experiment}')
        return 2
    current = state.next_model
    lines = [
        f'Sweep: {args.experiment}',
        f'Status: {state.status}',
        f'Model: {current}',
    ]
    if current is not None:
        db_path = root / 'results/local' / f'{args.experiment}__{current}.db'
        if db_path.exists():
            try:
                conn = sqlite3.connect(str(db_path))
                try:
                    count = conn.execute(
                        'SELECT COUNT(*) FROM runs WHERE experiment_id=?'
                        ' AND is_warmup=0',
                        (f'{args.experiment}__{current}',),
                    ).fetchone()[0]
                finally:
                    conn.close()
                lines.append(f'Progress: {count} measured rows')
            except sqlite3.Error:
                lines.append('Progress: db unreadable')
        lifecycle = state.models.get(current)
        if lifecycle is not None:
            lines.append(f'Lifecycle: {lifecycle.lifecycle}')
    lines.append(
        f"Pause requested: {'yes' if pause_requested(checkpoints, args.experiment) else 'no'}"
    )
    if state.status == 'PAUSED' and state.pause:
        lines.append(f"Paused from: {state.pause.get('paused_from')}")
        lines.append(f"Resume stage: {state.pause.get('resume_stage')}")
    print('\n'.join(lines), flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Sweep pause control')
    parser.add_argument('--experiment', default='full-baseline-v2')
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('pause', help='request a graceful pause')
    sub.add_parser('cancel-pause', help='withdraw a pending pause request')
    sub.add_parser('status', help='show sweep status and progress')
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parent.parent
    if args.command == 'pause':
        return cmd_pause(args, root)
    if args.command == 'cancel-pause':
        return cmd_cancel_pause(args, root)
    return cmd_status(args, root)


if __name__ == '__main__':
    raise SystemExit(main())
