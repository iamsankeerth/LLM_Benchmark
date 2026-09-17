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
from dataclasses import dataclass


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
    gpu_only=True,
)


MODEL_CONFIGS: dict[str, ModelConfig] = {
    _QWEN3_4B_Q4.config_id: _QWEN3_4B_Q4,
}


def get_model_config(config_id: str) -> ModelConfig:
    """Fetch a registered model config; KeyError lists what is available."""
    try:
        return MODEL_CONFIGS[config_id]
    except KeyError as exc:
        raise KeyError(
            f'unknown model config {config_id!r}; registered: {sorted(MODEL_CONFIGS)}'
        ) from exc


def render_prompt(config: ModelConfig, task_prompt: str) -> str:
    """Render a task prompt through the model's pinned template (byte-exact)."""
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
