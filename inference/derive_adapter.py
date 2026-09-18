"""Adapter auto-derivation for the unattended sweep (verify-then-trust).

Derivation interrogates the live model (/api/show), trials the default
chat route first, and falls back to raw only via the frozen ported-template
library keyed by family marker. Unknown family + chat failure ->
ManualPinRequired (loud stop, never a guessed template). Route selection
uses transport correctness only, never benchmark accuracy.

Overlays are create-once: derivation refuses when the overlay file exists;
resume re-interrogates live metadata and ADAPTER_MISMATCH fails closed on
any field drift. All Ollama access is injectable for offline tests.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from inference.adapters import (
    ModelConfig,
    _sha256_hex,
    config_from_overlay_json,
    config_to_overlay_json,
    validate_overlay_config,
)
from inference.ollama_client import GenerationRequest, GenerationResult


class ManualPinRequired(Exception):
    """Derivation cannot proceed without human template work."""


class OverlayExistsError(Exception):
    """Overlay already pinned: create-once forbids regeneration."""


@dataclass(frozen=True)
class AdapterMismatch:
    field: str
    pinned: str
    observed: str


# Frozen ported raw templates, keyed by family marker from show details.
# Only families a human has manually verified may appear here.
PORTED_RAW_TEMPLATES: dict[str, dict[str, str]] = {
    'qwen3': {
        'template_id': 'qwen3-chatml-no-think-v1',
        'template_text': '<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n',
    },
}

TRIAL_PROMPT = 'Return only the word OK.'


def parse_modelfile_stops(parameters: str) -> list[str]:
    """Extract stop "..." lines from the /api/show parameters blob."""
    stops: list[str] = []
    for line in parameters.split('\n'):
        match = re.match(r'^\s*stop\s+"(.*)"\s*$', line)
        if match and match.group(1) not in stops:
            stops.append(match.group(1))
    return stops


def family_markers(show_doc: Mapping[str, Any]) -> list[str]:
    details = show_doc.get('details', {})
    markers: list[str] = []
    if isinstance(details, dict):
        family = details.get('family')
        if isinstance(family, str):
            markers.append(family.lower())
        families = details.get('families', [])
        if isinstance(families, list):
            markers.extend(str(f).lower() for f in families if isinstance(f, str))
    return markers


def trial_default_route(
    trial_generate: Callable[[GenerationRequest], GenerationResult],
    model_identifier: str,
) -> bool:
    """Probe the Ollama-managed (non-raw) route: transport checks only."""
    try:
        result = trial_generate(
            GenerationRequest(
                model=model_identifier, prompt=TRIAL_PROMPT, temperature=0.0,
                num_ctx=512, num_predict=16, raw=False,
            )
        )
    except Exception:
        return False
    return bool(result.text.strip()) and result.done_reason in ('stop', 'length')


def derive_adapter(
    *,
    registry_entry: Mapping[str, Any],
    show_doc: Mapping[str, Any],
    observed_digest: str,
    trial_generate: Callable[[GenerationRequest], GenerationResult],
    overlay_path: str | Path,
) -> ModelConfig:
    """Derive + atomically write the overlay (create-once enforced).

    Raises OverlayExistsError (already pinned), ManualPinRequired (unknown
    raw fallback or failed verification), ValueError (malformed show doc).
    """
    overlay_file = Path(overlay_path)
    if overlay_file.exists():
        raise OverlayExistsError(f'overlay exists, refusing regeneration: {overlay_file}')
    config_id = str(registry_entry['model_config_id'])
    identifier = str(registry_entry['ollama_identifier'])
    quantization = str(registry_entry['quantization'])
    template = show_doc.get('template')
    if not isinstance(template, str) or not template:
        raise ValueError('show doc has no template')
    parameters = show_doc.get('parameters', '')
    stops = parse_modelfile_stops(parameters if isinstance(parameters, str) else '')
    if not stops:
        raise ValueError('show doc yielded no stop tokens')
    if trial_default_route(trial_generate, identifier):
        mode = 'chat'
        application = 'server_managed'
        template_id = f'auto-{config_id}-chat-v1'
        template_text = template
        template_source = 'ollama-show auto-derive (default route verified)'
        think: bool | None = None
    else:
        marker = next(
            (m for m in family_markers(show_doc) if m in PORTED_RAW_TEMPLATES), None
        )
        if marker is None:
            raise ManualPinRequired(
                f'{config_id}: default route failed and no ported raw template '
                f'for family markers {family_markers(show_doc)}'
            )
        ported = PORTED_RAW_TEMPLATES[marker]
        # Verify the ported template on the raw path before pinning.
        trial_request = GenerationRequest(
            model=identifier,
            prompt=ported['template_text'].replace('{prompt}', TRIAL_PROMPT),
            temperature=0.0, num_ctx=512, num_predict=16, raw=True, think=False,
        )
        try:
            trial_result = trial_generate(trial_request)
        except Exception as exc:
            raise ManualPinRequired(
                f'{config_id}: raw fallback trial failed: {exc}'
            ) from exc
        if not trial_result.text.strip():
            raise ManualPinRequired(f'{config_id}: raw fallback trial empty')
        mode = 'raw'
        application = 'client_rendered'
        template_id = ported['template_id']
        template_text = ported['template_text']
        template_source = (
            f'ported raw template for family {marker} (raw fallback verified)'
        )
        think = False
    config = ModelConfig(
        config_id=config_id,
        ollama_identifier=identifier,
        quantization=quantization,
        mode=mode,
        template_id=template_id,
        template_source=template_source,
        template_text=template_text,
        template_sha256=_sha256_hex(template_text),
        ollama_model_digest=observed_digest,
        num_ctx=4096,
        num_predict_default=512,
        num_gpu=99,
        temperature_default=0.0,
        stop_tokens=tuple(stops),
        think=think,
        thinking_support='supported',
        template_application=application,
        gpu_only=True,
    )
    validate_overlay_config(config)
    overlay_file.parent.mkdir(parents=True, exist_ok=True)
    tmp_file = overlay_file.with_suffix('.tmp')
    tmp_file.write_text(config_to_overlay_json(config) + '\n', encoding='utf-8')
    os.replace(tmp_file, overlay_file)
    # Round-trip the written file: what is pinned is what loads.
    reloaded = config_from_overlay_json(
        json.loads(overlay_file.read_text(encoding='utf-8'))
    )
    validate_overlay_config(reloaded)
    return reloaded


def verify_overlay_matches_live(
    config: ModelConfig,
    show_doc: Mapping[str, Any],
    observed_digest: str,
) -> list[AdapterMismatch]:
    """Compare pinned overlay against live interrogation (resume gate)."""
    mismatches: list[AdapterMismatch] = []
    template = show_doc.get('template')
    parameters = show_doc.get('parameters', '')

    def check(field: str, pinned: str, observed: str) -> None:
        if pinned != observed:
            mismatches.append(AdapterMismatch(field, pinned, observed))

    check('ollama_model_digest', config.ollama_model_digest, observed_digest)
    if isinstance(template, str) and config.template_application == 'server_managed':
        check(
            'template_sha256', config.template_sha256,
            hashlib.sha256(template.encode('utf-8')).hexdigest(),
        )
    if isinstance(parameters, str):
        check(
            'stop_tokens', ','.join(config.stop_tokens),
            ','.join(parse_modelfile_stops(parameters)),
        )
    return mismatches
