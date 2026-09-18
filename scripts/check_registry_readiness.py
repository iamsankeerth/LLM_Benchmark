"""Registry readiness gate: prove all 14 identifiers live before the sweep.

Offline structural checks run in the unit suite. This script performs the
live half: for each registry entry, resolve the identifier, confirm the
exact artifact and requested quant file exist via the Hugging Face API
(metadata only, no weight download), and confirm instruct intent.
Any failure stops the sweep before Qwen Q4 begins.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import requests
import yaml

HF_API = 'https://huggingface.co/api/models'


def parse_ollama_identifier(identifier: str) -> tuple[str, str, str]:
    """Split hf.co/<org>/<repo>:<quant>; ValueError on placeholders/shapes."""
    if not identifier.startswith('hf.co/'):
        raise ValueError(f'not an hf.co identifier: {identifier!r}')
    if any(token in identifier for token in ('TODO', 'XXX', '...', '<', ' ')):
        raise ValueError(f'placeholder identifier: {identifier!r}')
    remainder = identifier[len('hf.co/'):]
    if ':' not in remainder or '/' not in remainder:
        raise ValueError(f'malformed identifier: {identifier!r}')
    repo_path, quant = remainder.rsplit(':', 1)
    if '/' not in repo_path or not quant:
        raise ValueError(f'malformed identifier: {identifier!r}')
    return repo_path, quant, identifier


def check_artifact_files(
    repo_path: str, quant: str, *, timeout_s: float = 60.0
) -> tuple[bool, str]:
    """Confirm a .gguf file for the quant exists in the repo (API metadata)."""
    try:
        response = requests.get(f'{HF_API}/{repo_path}', timeout=timeout_s)
    except requests.RequestException as exc:
        return False, f'api unreachable: {exc}'
    if response.status_code != 200:
        return False, f'api HTTP {response.status_code} for {repo_path}'
    try:
        siblings = response.json().get('siblings', [])
    except ValueError:
        return False, 'invalid api json'
    names = [
        str(s.get('rfilename', ''))
        for s in siblings
        if isinstance(s, dict)
    ]
    want = quant.upper() + '.GGUF'
    matches = [n for n in names if n.upper().endswith(want)]
    if not matches:
        return False, f'no *{quant}.gguf in {repo_path} ({len(names)} files listed)'
    return True, matches[0]


def check_entry(entry: dict[str, Any], *, timeout_s: float = 60.0) -> tuple[bool, str]:
    for key in ('model_config_id', 'family', 'quantization', 'ollama_identifier'):
        if not entry.get(key):
            return False, f'missing {key}'
    try:
        repo_path, quant, _ = parse_ollama_identifier(str(entry['ollama_identifier']))
    except ValueError as exc:
        return False, str(exc)
    if str(entry['quantization']) != quant:
        return False, (
            f"quantization {entry['quantization']!r} != identifier quant {quant!r}"
        )
    lowered = repo_path.lower()
    instruct_markers = (
        'instruct' in lowered
        or '-it-' in lowered
        or '_it_' in lowered
        or lowered.endswith(('-it', '_it'))
    )
    if not instruct_markers:
        return False, f'cannot confirm instruct intent in {repo_path!r}'
    ok, detail = check_artifact_files(repo_path, quant, timeout_s=timeout_s)
    if not ok:
        return False, detail
    return True, f'{repo_path}:{quant} -> {detail}'


def load_registry(path: Path) -> dict[str, Any]:
    registry = yaml.safe_load(open(path, encoding='utf-8'))
    assert isinstance(registry, dict)
    return registry


def check_registry(
    registry: dict[str, Any], *, timeout_s: float = 60.0
) -> tuple[int, list[str]]:
    """Structural gate (offline-capable part); returns (failures, messages)."""
    failures = 0
    messages: list[str] = []
    entries = registry.get('models', [])
    if len(entries) != 14:
        failures += 1
        messages.append(f'expected 14 registry entries, found {len(entries)}')
    ids = [str(e.get('model_config_id', '')) for e in entries]
    if len(set(ids)) != len(ids):
        failures += 1
        messages.append('duplicate model_config_ids')
    executions = [f"{registry.get('experiment_spec', 'full-baseline-v2')}__{i}" for i in ids]
    if len(set(executions)) != len(executions):
        failures += 1
        messages.append('duplicate execution ids')
    for entry in entries:
        if not isinstance(entry, dict):
            failures += 1
            messages.append('non-mapping registry entry')
            continue
        for key in ('model_config_id', 'family', 'quantization', 'ollama_identifier'):
            if not entry.get(key):
                failures += 1
                messages.append(f"{entry.get('model_config_id', '?')}: missing {key}")
    return failures, messages


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Registry readiness gate')
    parser.add_argument('--registry', default='configs/models-v2.yaml')
    parser.add_argument('--timeout-s', type=float, default=60.0)
    parser.add_argument('--structural-only', action='store_true')
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parent.parent
    registry = load_registry(root / args.registry)
    failures, messages = check_registry(registry)
    for message in messages:
        print(f'STRUCTURAL: {message}', flush=True)
    if args.structural_only:
        print(f'structural: {len(registry.get("models", []))} entries, '
              f'{failures} failures')
        return 1 if failures else 0
    for entry in registry.get('models', []):
        ok, detail = check_entry(entry, timeout_s=args.timeout_s)
        mark = 'READY' if ok else 'NOT READY'
        print(f'{mark}: {entry.get("model_config_id")} {detail}', flush=True)
        if not ok:
            failures += 1
    print(f'readiness: {len(registry.get("models", []))} entries, {failures} failures')
    return 1 if failures else 0


if __name__ == '__main__':
    raise SystemExit(main())
