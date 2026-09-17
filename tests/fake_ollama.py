"""Fake Ollama /api/generate streaming server for harness tests.

Stdlib only. Serves one scripted Scenario per server instance: an HTTP
status plus an ordered list of NDJSON chunks, each emitted after a
controlled delay. Fault modes (HTTP 500, malformed line, mid-stream
disconnect, missing done:true, timeouts) let the client tests separate
harness bugs from model weirdness without touching a live model.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import TracebackType
from typing import Any


@dataclass(frozen=True)
class ScriptedChunk:
    """One NDJSON line emitted after ``delay_ms`` milliseconds."""

    delay_ms: float
    payload: dict[str, Any] | str  # dict -> JSON line; str -> raw line verbatim


@dataclass
class Scenario:
    status_code: int = 200
    chunks: list[ScriptedChunk] = field(default_factory=list)
    # Simulate a mid-stream disconnect: close the connection right after
    # this many chunks instead of serving the rest of the script.
    close_after_chunks: int | None = None


def final_chunk(
    *,
    response: str = '',
    thinking: str = '',
    done_reason: str = 'stop',
    total_duration: int = 5_000_000_000,
    load_duration: int = 100_000_000,
    prompt_eval_count: int = 10,
    prompt_eval_duration: int = 200_000_000,
    eval_count: int = 20,
    eval_duration: int = 500_000_000,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a realistic Ollama done:true payload with known counters."""
    payload: dict[str, Any] = {
        'model': 'fake-model',
        'created_at': '2026-09-17T00:00:00Z',
        'response': response,
        'thinking': thinking,
        'done': True,
        'done_reason': done_reason,
        'total_duration': total_duration,
        'load_duration': load_duration,
        'prompt_eval_count': prompt_eval_count,
        'prompt_eval_duration': prompt_eval_duration,
        'eval_count': eval_count,
        'eval_duration': eval_duration,
    }
    if extra:
        payload.update(extra)
    return payload


def _make_handler(
    scenario: Scenario, record: dict[str, Any]
) -> type[BaseHTTPRequestHandler]:
    class FakeOllamaHandler(BaseHTTPRequestHandler):
        # HTTP/1.1 + chunked framing mirrors real Ollama streaming and, unlike
        # close-delimited HTTP/1.0, lets the client observe each chunk promptly
        # (fixed-size socket reads block until full on close-delimited bodies).
        protocol_version = 'HTTP/1.1'
        server_version = 'FakeOllama/1.0'

        def log_message(self, *args: Any) -> None:
            return None

        def _read_body(self) -> bytes:
            length = int(self.headers.get('Content-Length', '0'))
            return self.rfile.read(length) if length > 0 else b''

        def do_POST(self) -> None:
            record['request_count'] = int(record.get('request_count', 0)) + 1
            record['last_path'] = self.path
            record['last_body'] = self._read_body()
            if self.path != '/api/generate':
                self.send_response(404)
                self.send_header('Content-Length', '0')
                self.end_headers()
                return
            if scenario.status_code != 200:
                body = b'{"error":"scripted failure"}'
                self.send_response(scenario.status_code)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            self.send_response(200)
            self.send_header('Content-Type', 'application/x-ndjson')
            self.send_header('Transfer-Encoding', 'chunked')
            self.send_header('Connection', 'close')
            self.end_headers()
            chunks = scenario.chunks
            truncated = scenario.close_after_chunks is not None
            if truncated:
                chunks = chunks[: scenario.close_after_chunks]
            for chunk in chunks:
                time.sleep(chunk.delay_ms / 1000.0)
                if isinstance(chunk.payload, str):
                    line = chunk.payload
                else:
                    line = json.dumps(chunk.payload)
                data = line.encode('utf-8') + b'\n'
                self.wfile.write(f'{len(data):X}\r\n'.encode('ascii') + data + b'\r\n')
                self.wfile.flush()
            if not truncated:
                # Proper chunked terminator: clean EOF.
                self.wfile.write(b'0\r\n\r\n')
                self.wfile.flush()
            # Truncated scripts return without a terminator: the client
            # observes a mid-stream disconnect (no done:true chunk).
            self.close_connection = True

    return FakeOllamaHandler


class FakeOllamaServer:
    """Context-managed fake Ollama server bound to an ephemeral port."""

    def __init__(self, scenario: Scenario) -> None:
        self._scenario = scenario
        self._record: dict[str, Any] = {}
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def base_url(self) -> str:
        assert self._server is not None, 'server not started'
        address = self._server.server_address
        host = str(address[0])
        port = int(address[1])
        return f'http://{host}:{port}'

    @property
    def request_count(self) -> int:
        return int(self._record.get('request_count', 0))

    @property
    def last_request_json(self) -> dict[str, Any]:
        raw = self._record.get('last_body', b'')
        assert isinstance(raw, bytes)
        decoded = json.loads(raw.decode('utf-8'))
        assert isinstance(decoded, dict)
        return decoded

    def __enter__(self) -> FakeOllamaServer:
        handler = _make_handler(self._scenario, self._record)
        self._server = ThreadingHTTPServer(('127.0.0.1', 0), handler)
        self._server.daemon_threads = True
        self._thread = threading.Thread(
            target=self._server.serve_forever, kwargs={'poll_interval': 0.01}
        )
        self._thread.daemon = True
        self._thread.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
