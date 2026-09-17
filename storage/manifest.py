"""Immutable per-experiment manifest: everything needed to reproduce a row.

Snapshots code version, frozen-eval hashes, model identity, server and
hardware fingerprints, and the full effective generation config. Live
environment facts are collected separately so tests stay offline.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class LiveEnvironment:
    ollama_version: str | None
    python_version: str
    hardware_id: str
    power_mode: str | None
    started_at_utc: str


@dataclass(frozen=True)
class ExperimentManifest:
    app_git_commit: str | None
    eval_freeze_hash: str
    grading_spec_hash: str
    dataset_hash: str
    # Identity triple: frozen contract x declared config x observed artifact.
    experiment_spec_id: str
    execution_id: str
    model_config_id: str
    experiment_config_hash: str
    model_config_hash: str
    model_artifact_digest: str | None
    model_artifact_size_bytes: int | None
    model_identifier: str
    quantization: str
    template_sha256: str
    ollama_version: str | None
    python_version: str
    hardware_id: str
    power_mode: str | None
    temperature: float
    num_ctx: int
    num_predict: int
    num_gpu: int | None
    stop_tokens: tuple[str, ...]
    think: bool | None
    thinking_source: str
    started_at_utc: str

    def to_json(self) -> str:
        document = asdict(self)
        document['stop_tokens'] = list(self.stop_tokens)
        return json.dumps(document, indent=2, sort_keys=True)


def hash_experiment_config(config: dict[str, Any]) -> str:
    canonical = json.dumps(config, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(canonical.encode('utf-8')).hexdigest()


def git_commit(repo_dir: str) -> str | None:
    try:
        out = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            cwd=repo_dir,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None if out.returncode == 0 else None


def collect_live_environment(*, ollama_version: str | None, started_at_utc: str) -> LiveEnvironment:
    """Best-effort fingerprints; unknown values are None, never guessed."""
    uname = platform.uname()
    hardware_id = f'{uname.system}-{uname.machine}-{uname.processor}'.strip('-')
    power_mode: str | None = None
    if uname.system == 'Windows':
        try:
            out = subprocess.run(
                ['powercfg', '/getactivescheme'],
                capture_output=True,
                text=True,
                timeout=15,
            )
            if out.returncode == 0 and out.stdout.strip():
                power_mode = out.stdout.strip().splitlines()[0].strip()
        except (OSError, subprocess.SubprocessError):
            power_mode = None
    return LiveEnvironment(
        ollama_version=ollama_version,
        python_version=sys.version.split()[0],
        hardware_id=hardware_id or 'unknown',
        power_mode=power_mode,
        started_at_utc=started_at_utc,
    )


def build_manifest(
    *,
    app_git_commit: str | None,
    eval_freeze_hash: str,
    grading_spec_hash: str,
    dataset_hash: str,
    experiment_spec_id: str,
    execution_id: str,
    model_config_id: str,
    model_config_hash: str,
    model_artifact_digest: str | None,
    model_artifact_size_bytes: int | None,
    model_identifier: str,
    quantization: str,
    template_sha256: str,
    temperature: float,
    num_ctx: int,
    num_predict: int,
    num_gpu: int | None,
    stop_tokens: tuple[str, ...],
    think: bool | None,
    thinking_source: str,
    experiment_config_hash: str,
    live: LiveEnvironment,
) -> ExperimentManifest:
    return ExperimentManifest(
        app_git_commit=app_git_commit,
        eval_freeze_hash=eval_freeze_hash,
        grading_spec_hash=grading_spec_hash,
        dataset_hash=dataset_hash,
        experiment_spec_id=experiment_spec_id,
        execution_id=execution_id,
        model_config_id=model_config_id,
        model_config_hash=model_config_hash,
        model_artifact_digest=model_artifact_digest,
        model_artifact_size_bytes=model_artifact_size_bytes,
        model_identifier=model_identifier,
        quantization=quantization,
        template_sha256=template_sha256,
        ollama_version=live.ollama_version,
        python_version=live.python_version,
        hardware_id=live.hardware_id,
        power_mode=live.power_mode,
        temperature=temperature,
        num_ctx=num_ctx,
        num_predict=num_predict,
        num_gpu=num_gpu,
        stop_tokens=stop_tokens,
        think=think,
        thinking_source=thinking_source,
        experiment_config_hash=experiment_config_hash,
        started_at_utc=live.started_at_utc,
    )
