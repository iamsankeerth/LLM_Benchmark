"""Runtime-configured OpenAI-compatible external judge client."""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

import requests


class ExternalJudgeError(RuntimeError):
    """The external judge is unconfigured, unavailable, or over budget."""

    def __init__(
        self,
        message: str,
        *,
        request_id: str | None = None,
        usage: dict[str, Any] | None = None,
        estimated_cost_usd: float | None = None,
        response_sha256: str | None = None,
        attempts: int = 0,
    ) -> None:
        super().__init__(message)
        self.request_id = request_id
        self.usage = usage
        self.estimated_cost_usd = estimated_cost_usd
        self.response_sha256 = response_sha256
        self.attempts = attempts


@dataclass(frozen=True)
class ExternalJudgeConfig:
    base_url: str
    api_key: str
    model_id: str
    temperature: float = 0.0
    max_output_tokens: int = 768
    timeout_seconds: float = 120.0
    max_attempts: int = 3
    retry_statuses: frozenset[int] = frozenset({429, 500, 502, 503, 504})
    max_calls: int = 250
    max_cost_usd: float = 5.0
    cost_per_1k_tokens: float | None = None

    def __post_init__(self) -> None:
        if not self.base_url or not self.api_key or not self.model_id:
            raise ExternalJudgeError('external judge configuration is incomplete')
        if self.timeout_seconds <= 0:
            raise ExternalJudgeError('external judge timeout must be positive')
        if self.max_attempts < 1:
            raise ExternalJudgeError('external judge max_attempts must be positive')
        if self.max_calls < 0:
            raise ExternalJudgeError('external judge max_calls cannot be negative')
        if self.max_output_tokens < 1:
            raise ExternalJudgeError('external judge max_output_tokens must be positive')
        if self.max_cost_usd < 0:
            raise ExternalJudgeError('external judge max_cost_usd cannot be negative')
        if self.cost_per_1k_tokens is not None and self.cost_per_1k_tokens < 0:
            raise ExternalJudgeError('external judge token price cannot be negative')

    @classmethod
    def from_environment(cls, document: Mapping[str, Any]) -> ExternalJudgeConfig:
        external = document.get('external_judge')
        if not isinstance(external, Mapping):
            raise ExternalJudgeError('external_judge configuration is missing')
        base_env = str(external.get('base_url_env', 'JUDGE_API_BASE_URL'))
        key_env = str(external.get('api_key_env', 'JUDGE_API_KEY'))
        base_url = os.environ.get(base_env, '').strip()
        api_key = os.environ.get(key_env, '').strip()
        if not base_url:
            raise ExternalJudgeError(f'{base_env} is not set')
        if not api_key:
            raise ExternalJudgeError(f'{key_env} is not set')
        model_id = str(external.get('model_id', '')).strip()
        if not model_id:
            raise ExternalJudgeError('external judge model_id is not configured')
        runtime_model = os.environ.get('SPACE_BUNNY_MODEL_ID', model_id).strip()
        if runtime_model != model_id:
            raise ExternalJudgeError('SPACE_BUNNY_MODEL_ID does not match the frozen model_id')
        price_env = str(external.get('cost_per_1k_tokens_env', 'JUDGE_COST_PER_1K_TOKENS'))
        price_value = os.environ.get(price_env, '').strip()
        if not price_value:
            raise ExternalJudgeError(f'{price_env} is not set')
        return cls(
            base_url=base_url.rstrip('/'),
            api_key=api_key,
            model_id=model_id,
            temperature=float(external.get('temperature', 0.0)),
            max_output_tokens=int(external.get('max_output_tokens', 768)),
            timeout_seconds=float(external.get('timeout_seconds', 120.0)),
            max_attempts=int(external.get('max_attempts_per_request', 3)),
            retry_statuses=frozenset(
                int(value) for value in external.get('retry_statuses', [])
            ),
            max_calls=int(external.get('max_calls', 250)),
            max_cost_usd=float(external.get('max_cost_usd', 5.0)),
            cost_per_1k_tokens=float(price_value),
        )


@dataclass
class ExternalJudgeCall:
    text: str
    request_id: str | None
    usage: dict[str, Any]
    attempts: int
    estimated_cost_usd: float


@dataclass
class ExternalJudgeClient:
    config: ExternalJudgeConfig
    post_fn: Callable[..., Any] = requests.post
    sleep_fn: Callable[[float], None] = time.sleep
    calls_made: int = field(default=0, init=False)
    total_cost_usd: float = field(default=0.0, init=False)

    def _spend_call(self) -> None:
        if self.calls_made >= self.config.max_calls:
            raise ExternalJudgeError('external judge call budget exhausted')
        self.calls_made += 1

    def judge(self, prompt: str, schema: Mapping[str, Any]) -> ExternalJudgeCall:
        body = {
            'model': self.config.model_id,
            'messages': [
                {
                    'role': 'system',
                    'content': 'Return only JSON matching the supplied schema.',
                },
                {'role': 'user', 'content': prompt},
            ],
            'temperature': self.config.temperature,
            'max_tokens': self.config.max_output_tokens,
            'response_format': {
                'type': 'json_schema',
                'json_schema': {
                    'name': 'judge_response',
                    'strict': True,
                    'schema': dict(schema),
                },
            },
        }
        headers = {
            'Authorization': f'Bearer {self.config.api_key}',
            'Content-Type': 'application/json',
        }
        last_error = 'external judge request failed'
        for attempt in range(1, self.config.max_attempts + 1):
            self._spend_call()
            try:
                response = self.post_fn(
                    f'{self.config.base_url}/chat/completions',
                    json=body,
                    headers=headers,
                    timeout=self.config.timeout_seconds,
                )
            except requests.RequestException as exc:
                last_error = f'{type(exc).__name__}: {exc}'
                if attempt >= self.config.max_attempts:
                    raise ExternalJudgeError(last_error, attempts=attempt) from exc
                self.sleep_fn(float(2 ** (attempt - 1)))
                continue
            status = int(getattr(response, 'status_code', 0))
            if status in self.config.retry_statuses and attempt < self.config.max_attempts:
                last_error = f'external judge HTTP status {status}'
                self.sleep_fn(float(2 ** (attempt - 1)))
                continue
            if status < 200 or status >= 300:
                raise ExternalJudgeError(
                    f'external judge HTTP status {status}', attempts=attempt
                )
            try:
                document = response.json()
            except (ValueError, json.JSONDecodeError) as exc:
                raise ExternalJudgeError(
                    'external judge returned invalid JSON', attempts=attempt
                ) from exc
            try:
                text = str(document['choices'][0]['message']['content'])
            except (KeyError, IndexError, TypeError) as exc:
                raise ExternalJudgeError(
                    'external judge response has no message content', attempts=attempt
                ) from exc
            usage = document.get('usage', {}) if isinstance(document, dict) else {}
            if not isinstance(usage, dict):
                usage = {}
            try:
                total_tokens = int(usage.get('total_tokens', 0))
            except (TypeError, ValueError) as exc:
                raise ExternalJudgeError(
                    'external judge returned invalid token usage', attempts=attempt
                ) from exc
            if total_tokens < 0:
                raise ExternalJudgeError(
                    'external judge returned negative token usage', attempts=attempt
                )
            estimated_cost = (
                total_tokens / 1000.0 * self.config.cost_per_1k_tokens
                if self.config.cost_per_1k_tokens is not None else 0.0
            )
            self.total_cost_usd += estimated_cost
            headers_response = getattr(response, 'headers', {}) or {}
            request_id = headers_response.get('x-request-id')
            response_sha256 = hashlib.sha256(text.encode('utf-8')).hexdigest()
            if self.total_cost_usd > self.config.max_cost_usd:
                raise ExternalJudgeError(
                    'external judge cost budget exhausted',
                    request_id=request_id,
                    usage=usage,
                    estimated_cost_usd=estimated_cost,
                    response_sha256=response_sha256,
                    attempts=attempt,
                )
            return ExternalJudgeCall(
                text=text,
                request_id=request_id,
                usage=usage,
                attempts=attempt,
                estimated_cost_usd=estimated_cost,
            )
        raise ExternalJudgeError(last_error, attempts=self.config.max_attempts)
