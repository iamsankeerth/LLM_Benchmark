"""Streaming Ollama client for LocalLLM Lab.

The client always uses ``stream: true`` because time-to-first-token is
unobservable on the non-streaming route. It sends caller-rendered prompt
text only and never applies prompt templating itself (see adapters).

Timing uses ``time.perf_counter_ns`` (monotonic). Wall-clock UTC is left
to the persistence layer for audit columns.
"""

from __future__ import annotations

import codecs
import http.client
import json
import socket
import time
from dataclasses import dataclass, field
from typing import Any

import requests
import urllib3.exceptions as urllib3_exceptions

# Small read size so the first NDJSON chunk is observed immediately instead
# of waiting for a large buffer to fill (TTFT fidelity).
_READ_CHUNK_SIZE = 1024


class OllamaClientError(Exception):
    """Base class for all client-side Ollama failures."""


class OllamaHTTPError(OllamaClientError):
    def __init__(self, status_code: int, body: str) -> None:
        super().__init__(f'Ollama returned HTTP {status_code}: {body[:200]}')
        self.status_code = status_code
        self.body = body


class OllamaTimeoutError(OllamaClientError):
    pass


class MalformedChunkError(OllamaClientError):
    """A streamed NDJSON line was not valid JSON (data corruption)."""


class TruncatedStreamError(OllamaClientError):
    """Stream ended without a done:true chunk (disconnect or server bug)."""


@dataclass(frozen=True)
class GenerationRequest:
    model: str
    prompt: str  # fully rendered prompt text; client never templates
    temperature: float = 0.0
    num_ctx: int = 4096
    num_predict: int = 512
    stop: tuple[str, ...] = ()
    raw: bool = False
    # Requested GPU layer offload (Ollama num_gpu). This is a request only:
    # GPU residency must be verified via /api/ps (see eligibility).
    # None omits the option (server default).
    num_gpu: int | None = None
    # Passed through as options["think"] when not None. None means the
    # caller did not configure thinking: model default applies and the
    # run records thinking_source="model_default".
    think: bool | None = None


@dataclass
class GenerationResult:
    text: str
    thinking: str
    done_reason: str | None
    # Raw server-side counters from the done:true chunk (None if absent).
    total_duration_ns: int | None = None
    load_duration_ns: int | None = None
    prompt_eval_count: int | None = None
    prompt_eval_cached_count: int | None = None
    prompt_eval_duration_ns: int | None = None
    eval_count: int | None = None
    eval_duration_ns: int | None = None
    # Monotonic client-side timestamps (perf_counter_ns).
    request_start_ns: int = 0
    first_token_ns: int | None = None
    request_end_ns: int = 0
    raw_final: dict[str, Any] = field(default_factory=dict)


def _as_optional_int(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _caused_by_timeout(exc: BaseException) -> bool:
    """Walk the exception chain: streaming read timeouts often surface wrapped
    (requests ConnectionError / urllib3 ProtocolError around a socket timeout)."""
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(
            current,
            (
                requests.Timeout,
                urllib3_exceptions.TimeoutError,
                socket.timeout,
                TimeoutError,
            ),
        ):
            return True
        current = current.__cause__ or current.__context__
    return False


def _build_payload(request: GenerationRequest) -> dict[str, Any]:
    options: dict[str, Any] = {
        'temperature': request.temperature,
        'num_ctx': request.num_ctx,
        'num_predict': request.num_predict,
    }
    if request.stop:
        options['stop'] = list(request.stop)
    if request.think is not None:
        options['think'] = request.think
    if request.num_gpu is not None:
        options['num_gpu'] = request.num_gpu
    return {
        'model': request.model,
        'prompt': request.prompt,
        'stream': True,
        'raw': request.raw,
        'options': options,
    }


def generate(
    base_url: str, request: GenerationRequest, *, timeout_s: float = 300.0
) -> GenerationResult:
    """Run one streamed generation; raise on any transport/stream fault."""
    result = GenerationResult(text='', thinking='', done_reason=None)
    result.request_start_ns = time.perf_counter_ns()
    buffer = ''
    decoder = codecs.getincrementaldecoder('utf-8')()
    done_seen = False
    try:
        with requests.post(
            base_url.rstrip('/') + '/api/generate',
            json=_build_payload(request),
            stream=True,
            timeout=timeout_s,
        ) as response:
            if response.status_code != 200:
                raise OllamaHTTPError(response.status_code, response.text)
            try:
                for byte_chunk in response.iter_content(chunk_size=_READ_CHUNK_SIZE):
                    if not byte_chunk:
                        continue
                    buffer += decoder.decode(byte_chunk)
                    while '\n' in buffer:
                        line, buffer = buffer.split('\n', 1)
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            chunk = json.loads(line)
                        except json.JSONDecodeError as exc:
                            raise MalformedChunkError(
                                f'invalid NDJSON line: {line[:120]}'
                            ) from exc
                        if not isinstance(chunk, dict):
                            raise MalformedChunkError(
                                f'NDJSON line is not an object: {line[:120]}'
                            )
                        piece = chunk.get('response')
                        if isinstance(piece, str) and piece:
                            if result.first_token_ns is None:
                                result.first_token_ns = time.perf_counter_ns()
                            result.text += piece
                        think_piece = chunk.get('thinking')
                        if isinstance(think_piece, str) and think_piece:
                            result.thinking += think_piece
                        if chunk.get('done') is True:
                            done_seen = True
                            reason = chunk.get('done_reason')
                            result.done_reason = reason if isinstance(reason, str) else None
                            result.total_duration_ns = _as_optional_int(
                                chunk.get('total_duration')
                            )
                            result.load_duration_ns = _as_optional_int(chunk.get('load_duration'))
                            result.prompt_eval_count = _as_optional_int(
                                chunk.get('prompt_eval_count')
                            )
                            result.prompt_eval_cached_count = _as_optional_int(
                                chunk.get('prompt_eval_cached_count')
                            )
                            result.prompt_eval_duration_ns = _as_optional_int(
                                chunk.get('prompt_eval_duration')
                            )
                            result.eval_count = _as_optional_int(chunk.get('eval_count'))
                            result.eval_duration_ns = _as_optional_int(
                                chunk.get('eval_duration')
                            )
                            result.raw_final = chunk
                            break
                    if done_seen:
                        # Payload complete; nothing follows the final chunk.
                        break
            except OllamaClientError:
                raise
            except (
                requests.RequestException,
                urllib3_exceptions.HTTPError,
                http.client.HTTPException,
            ) as exc:
                if _caused_by_timeout(exc):
                    raise OllamaTimeoutError(
                        f'generation timed out after {timeout_s}s'
                    ) from exc
                raise TruncatedStreamError(
                    f'stream broke before done:true: {exc}'
                ) from exc
    except OllamaClientError:
        raise
    except requests.Timeout as exc:
        raise OllamaTimeoutError(f'generation timed out after {timeout_s}s') from exc
    except requests.RequestException as exc:
        if _caused_by_timeout(exc):
            raise OllamaTimeoutError(
                f'generation timed out after {timeout_s}s'
            ) from exc
        raise OllamaClientError(f'transport failure: {exc}') from exc
    finally:
        result.request_end_ns = time.perf_counter_ns()
    if not done_seen:
        raise TruncatedStreamError('stream ended without a done:true chunk')
    return result


def fetch_show(base_url: str, model: str, *, timeout_s: float = 30.0) -> dict[str, Any]:
    """Return the raw /api/show document (template discovery, digests)."""
    try:
        response = requests.post(
            base_url.rstrip('/') + '/api/show', json={'model': model}, timeout=timeout_s
        )
    except requests.Timeout as exc:
        raise OllamaTimeoutError(f'/api/show timed out after {timeout_s}s') from exc
    except requests.RequestException as exc:
        raise OllamaClientError(f'/api/show transport failure: {exc}') from exc
    if response.status_code != 200:
        raise OllamaHTTPError(response.status_code, response.text)
    try:
        document: Any = response.json()
    except ValueError as exc:
        raise MalformedChunkError('/api/show returned invalid JSON') from exc
    if not isinstance(document, dict):
        raise MalformedChunkError('/api/show returned a non-object document')
    return document


def fetch_ps(base_url: str, *, timeout_s: float = 30.0) -> dict[str, Any]:
    """Return the raw /api/ps document (residency evidence)."""
    try:
        response = requests.get(base_url.rstrip('/') + '/api/ps', timeout=timeout_s)
    except requests.Timeout as exc:
        raise OllamaTimeoutError(f'/api/ps timed out after {timeout_s}s') from exc
    except requests.RequestException as exc:
        raise OllamaClientError(f'/api/ps transport failure: {exc}') from exc
    if response.status_code != 200:
        raise OllamaHTTPError(response.status_code, response.text)
    try:
        document: Any = response.json()
    except ValueError as exc:
        raise MalformedChunkError('/api/ps returned invalid JSON') from exc
    if not isinstance(document, dict):
        raise MalformedChunkError('/api/ps returned a non-object document')
    return document
