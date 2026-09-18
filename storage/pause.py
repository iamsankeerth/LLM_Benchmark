"""Pause coordination: sentinel file, PAUSED state, signal discipline.

- Sentinel: results/checkpoints/pause-<spec>.request (atomic create).
- PAUSED is a parking bay: not COMPLETE/FAILED, never advances the registry.
- SIGINT discipline: the handler is FLAG-ONLY. It sets pause_requested and
  returns immediately — never touches SQLite, JSON, Ollama streams, or the
  DB. Normal control flow performs graceful_pause() at the next boundary.
  A second SIGINT restores hard-abort behavior.
"""

from __future__ import annotations

import json
import os
import signal
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from types import FrameType
from typing import Any


@dataclass
class PauseRecord:
    paused_from: str
    resume_stage: str
    model_config_id: str
    pause_reason: str = 'USER_REQUESTED'
    weights_retained: bool = True


def sentinel_path(checkpoints_dir: str | Path, experiment_spec: str) -> Path:
    return Path(checkpoints_dir) / f'pause-{experiment_spec}.request'


def request_pause(checkpoints_dir: str | Path, experiment_spec: str) -> Path:
    """Atomically create the pause request (tmp + rename)."""
    target = sentinel_path(checkpoints_dir, experiment_spec)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=str(target.parent), prefix=target.name + '.', suffix='.tmp'
    )
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as handle:
            handle.write('{"pause_requested": true}\n')
        Path(tmp_name).replace(target)
    except BaseException:
        try:
            Path(tmp_name).unlink(missing_ok=True)
        except OSError:
            pass
        raise
    return target


def cancel_pause(checkpoints_dir: str | Path, experiment_spec: str) -> bool:
    """Remove a pending request; True when one existed."""
    target = sentinel_path(checkpoints_dir, experiment_spec)
    if not target.exists():
        return False
    target.unlink()
    return True


def pause_requested(checkpoints_dir: str | Path, experiment_spec: str) -> bool:
    return sentinel_path(checkpoints_dir, experiment_spec).exists()


def consume_pause_request(checkpoints_dir: str | Path, experiment_spec: str) -> None:
    sentinel_path(checkpoints_dir, experiment_spec).unlink(missing_ok=True)


class PauseFlag:
    """Process-wide pause request. Set by CLI sentinel checks or by the
    flag-only SIGINT handler; read at safe boundaries."""

    def __init__(self) -> None:
        self.requested = False
        self._sigint_count = 0

    def request(self) -> None:
        self.requested = True

    def _sigint_handler(
        self, signum: int, frame: FrameType | None
    ) -> None:
        # FLAG-ONLY: never touch IO, SQLite, or subprocesses here. The
        # running unit continues; the next boundary performs the pause.
        self._sigint_count += 1
        if self._sigint_count == 1:
            self.requested = True
            print('\ngraceful pause requested (Ctrl+C); '
                  'finishing the current unit...', flush=True)
            return
        print('\nsecond interrupt: hard abort', flush=True)
        signal.signal(signal.SIGINT, signal.default_int_handler)
        raise KeyboardInterrupt

    def install_sigint_handler(self) -> None:
        self._sigint_count = 0
        signal.signal(signal.SIGINT, self._sigint_handler)

    def uninstall_sigint_handler(self) -> None:
        signal.signal(signal.SIGINT, signal.default_int_handler)


def pause_record_to_json(record: PauseRecord) -> str:
    return json.dumps(asdict(record), indent=2, sort_keys=True)


def pause_record_from_json(document: str) -> PauseRecord:
    raw: Any = json.loads(document)
    if not isinstance(raw, dict):
        raise ValueError('pause record must be an object')
    return PauseRecord(
        paused_from=str(raw.get('paused_from', '')),
        resume_stage=str(raw.get('resume_stage', '')),
        model_config_id=str(raw.get('model_config_id', '')),
        pause_reason=str(raw.get('pause_reason', 'USER_REQUESTED')),
        weights_retained=bool(raw.get('weights_retained', True)),
    )
