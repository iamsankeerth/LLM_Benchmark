"""Model adapter/config layer for LocalLLM Lab.

Each model gets an explicit ModelConfig: Ollama identifier, quantization,
raw/chat mode, byte-pinned prompt template (+ provenance), context size,
GPU-only requirement, stop tokens and thinking behavior. Qwen-specific
prompt handling lives here and never leaks into the generic client.

Template provenance for qwen3-4b-q4 (established 2026-09-17, live probes):
- /api/show template renders a lone user turn as
  ``<|im_start|>user\\n{content}<|im_end|>\\n<|im_start|>assistant\\n<think>\\n``
  i.e. generation starts inside an OPEN <think> block. With the default
  chat route the thinking parser swallowed the output (live: eval_count=1,
  response='' and thinking=''). The benchmark route is therefore raw with
  thinking disabled, using template A below (live: output exactly '42',
  thinking empty).
- stop tokens <|im_start|> / <|im_end|> come from the Modelfile PARAMETERS.
/api/ps evidence at 100% GPU: size == size_vram == 3178149969, digest
5cfd6a526dc3..., context_length 4096 (see eligibility).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ModelConfig:
    config_id: str
    ollama_identifier: str
    quantization: str
    mode: str  # 'raw' or 'chat'; the client never falls back silently
    template_id: str
    template_source: str
    template_text: str  # contains exactly one '{prompt}' placeholder
    template_sha256: str
    ollama_model_digest: str
    num_ctx: int
    num_predict_default: int
    num_gpu: int  # requested offload; residency verified via /api/ps
    temperature_default: float
    stop_tokens: tuple[str, ...]
    think: bool | None  # None -> model default (recorded as such)
    thinking_support: str  # 'supported' | 'unsupported'
    template_application: str  # 'server_managed' (chat) | 'client_rendered' (raw)
    gpu_only: bool = True


_QWEN3_TEMPLATE_A = '<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n'


def _sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


_QWEN3_4B_Q4 = ModelConfig(
    config_id='qwen3-4b-q4',
    ollama_identifier='hf.co/bartowski/Qwen_Qwen3-4B-Instruct-2507-GGUF:Q4_K_M',
    quantization='Q4_K_M',
    mode='raw',
    template_id='qwen3-chatml-no-think-v1',
    template_source=(
        'ollama-show template interrogated 2026-09-17; manual raw port with '
        'thinking disabled (template A); default chat route verified broken '
        '(empty response AND thinking at eval_count=1)'
    ),
    template_text=_QWEN3_TEMPLATE_A,
    template_sha256=_sha256_hex(_QWEN3_TEMPLATE_A),
    ollama_model_digest=(
        '5cfd6a526dc35c24a21a8a457db3e8810165e93bbb134dd6a7ac20b34e97b25a'
    ),
    num_ctx=4096,
    num_predict_default=512,
    num_gpu=99,
    temperature_default=0.0,
    stop_tokens=('<|im_start|>', '<|im_end|>'),
    think=False,
    thinking_support='supported',
    template_application='client_rendered',
    gpu_only=True,
)


_QWEN3_4B_Q5 = ModelConfig(
    config_id='qwen3-4b-q5',
    ollama_identifier='hf.co/bartowski/Qwen_Qwen3-4B-Instruct-2507-GGUF:Q5_K_M',
    quantization='Q5_K_M',
    mode='raw',
    template_id='qwen3-chatml-no-think-v1',
    template_source=(
        'ollama-show interrogated 2026-09-18: same qwen3 template mechanics '
        'and stops as qwen3-4b-q4 (trailer opens <think>); raw template A '
        'with thinking disabled verified live (output exactly 42)'
    ),
    template_text=_QWEN3_TEMPLATE_A,
    template_sha256=_sha256_hex(_QWEN3_TEMPLATE_A),
    ollama_model_digest=(
        '7b56805a15439abd489ff24764b46b349c2f5fa0d52cf41c66c6ba2492093138'
    ),
    num_ctx=4096,
    num_predict_default=512,
    num_gpu=99,
    temperature_default=0.0,
    stop_tokens=('<|im_start|>', '<|im_end|>'),
    think=False,
    thinking_support='supported',
    template_application='client_rendered',
    gpu_only=True,
)


MODEL_CONFIGS: dict[str, ModelConfig] = {
    _QWEN3_4B_Q4.config_id: _QWEN3_4B_Q4,
    _QWEN3_4B_Q5.config_id: _QWEN3_4B_Q5,
}

_OVERLAYS: dict[str, ModelConfig] = {}


def validate_overlay_config(config: ModelConfig) -> None:
    """Enforce the same invariants as coded entries, plus overlay rules."""
    if not config.config_id or not config.ollama_identifier:
        raise ValueError('overlay config requires config_id and ollama_identifier')
    if config.mode not in ('raw', 'chat'):
        raise ValueError(f"overlay mode must be raw|chat, got {config.mode!r}")
    if config.template_application not in ('server_managed', 'client_rendered'):
        raise ValueError(
            f'unknown template_application {config.template_application!r}'
        )
    if config.mode == 'chat' and config.template_application != 'server_managed':
        raise ValueError('chat mode requires server_managed template_application')
    if config.mode == 'raw' and config.template_application != 'client_rendered':
        raise ValueError('raw mode requires client_rendered template_application')
    if config.template_text.count('{prompt}') != 1 and config.mode == 'raw':
        raise ValueError('raw overlay template must contain one {prompt} slot')
    verify_template_integrity(config)
    if len(config.ollama_model_digest) != 64:
        raise ValueError('overlay digest must be 64 hex chars')
    if not config.stop_tokens:
        raise ValueError('overlay requires stop tokens')
    if config.config_id in MODEL_CONFIGS:
        raise ValueError(
            f'overlay {config.config_id!r} collides with a coded config'
        )


def config_to_overlay_json(config: ModelConfig) -> str:
    document: dict[str, Any] = {
        'config_id': config.config_id,
        'ollama_identifier': config.ollama_identifier,
        'quantization': config.quantization,
        'mode': config.mode,
        'template_id': config.template_id,
        'template_source': config.template_source,
        'template_text': config.template_text,
        'template_sha256': config.template_sha256,
        'ollama_model_digest': config.ollama_model_digest,
        'num_ctx': config.num_ctx,
        'num_predict_default': config.num_predict_default,
        'num_gpu': config.num_gpu,
        'temperature_default': config.temperature_default,
        'stop_tokens': list(config.stop_tokens),
        'think': config.think,
        'thinking_support': config.thinking_support,
        'template_application': config.template_application,
        'gpu_only': config.gpu_only,
    }
    return json.dumps(document, indent=2, sort_keys=True)


def config_from_overlay_json(document: dict[str, Any]) -> ModelConfig:
    return ModelConfig(
        config_id=str(document['config_id']),
        ollama_identifier=str(document['ollama_identifier']),
        quantization=str(document['quantization']),
        mode=str(document['mode']),
        template_id=str(document['template_id']),
        template_source=str(document['template_source']),
        template_text=str(document['template_text']),
        template_sha256=str(document['template_sha256']),
        ollama_model_digest=str(document['ollama_model_digest']),
        num_ctx=int(document['num_ctx']),
        num_predict_default=int(document['num_predict_default']),
        num_gpu=int(document['num_gpu']),
        temperature_default=float(document['temperature_default']),
        stop_tokens=tuple(str(s) for s in document['stop_tokens']),
        think=document['think'],
        thinking_support=str(document['thinking_support']),
        template_application=str(document['template_application']),
        gpu_only=bool(document.get('gpu_only', True)),
    )


def register_overlay(config: ModelConfig) -> None:
    validate_overlay_config(config)
    _OVERLAYS[config.config_id] = config


def load_overlays_from_dir(directory: str | Path) -> int:
    """Load and validate every overlay JSON; returns count loaded."""
    path = Path(directory)
    if not path.is_dir():
        return 0
    count = 0
    for file in sorted(path.glob('*.json')):
        document = json.loads(file.read_text(encoding='utf-8'))
        if not isinstance(document, dict):
            raise ValueError(f'overlay {file.name} is not an object')
        register_overlay(config_from_overlay_json(document))
        count += 1
    return count


def clear_overlays() -> None:
    """Test hook: drop all registered overlays."""
    _OVERLAYS.clear()


def get_model_config(config_id: str) -> ModelConfig:
    """Fetch a registered model config; KeyError lists what is available."""
    if config_id in MODEL_CONFIGS:
        return MODEL_CONFIGS[config_id]
    if config_id in _OVERLAYS:
        return _OVERLAYS[config_id]
    raise KeyError(
        f'unknown model config {config_id!r}; registered: '
        f'{sorted([*MODEL_CONFIGS, *_OVERLAYS])}'
    )


def render_prompt(config: ModelConfig, task_prompt: str) -> str:
    """Render a task prompt for sending.

    Raw mode substitutes the pinned template's single {prompt} slot
    (byte-exact). Chat mode returns the bare task prompt: the server owns
    templating (template_application == 'server_managed'), so there is
    nothing client-side to render. Either way the return value is exactly
    what the client transmits.
    """
    if config.mode == 'chat':
        return task_prompt
    if config.template_text.count('{prompt}') != 1:
        raise ValueError(f'template {config.template_id!r} must contain one {{prompt}} slot')
    return config.template_text.replace('{prompt}', task_prompt)


def rendered_prompt_sha256(rendered: str) -> str:
    return _sha256_hex(rendered)


def verify_template_integrity(config: ModelConfig) -> None:
    """Fail if the pinned template text no longer matches its recorded hash."""
    actual = _sha256_hex(config.template_text)
    if actual != config.template_sha256:
        raise ValueError(
            f'template {config.template_id!r} integrity failure: '
            f'recorded {config.template_sha256} != actual {actual}'
        )


def model_config_hash(config: ModelConfig) -> str:
    """Canonical hash of the DECLARED adapter configuration.

    Covers intent only: identifier, quantization, mode, template identity,
    context/output budgets, offload request, stops, think setting. The live
    model artifact digest is deliberately excluded (recorded separately as
    observed evidence: a tag could re-resolve to different bytes later).
    """
    import json

    canonical = json.dumps(
        {
            'ollama_identifier': config.ollama_identifier,
            'quantization': config.quantization,
            'mode': config.mode,
            'template_id': config.template_id,
            'template_sha256': config.template_sha256,
            'num_ctx': config.num_ctx,
            'num_predict_default': config.num_predict_default,
            'num_gpu': config.num_gpu,
            'temperature_default': config.temperature_default,
            'stop_tokens': list(config.stop_tokens),
            'think': config.think,
            'thinking_support': config.thinking_support,
            'gpu_only': config.gpu_only,
        },
        sort_keys=True,
        separators=(',', ':'),
    )
    return hashlib.sha256(canonical.encode('utf-8')).hexdigest()
