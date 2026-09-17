"""Client tests against the fake Ollama streaming fixture.

Covers TTFT math, stream assembly, metadata extraction and every fault
mode: malformed line, HTTP 500, timeout, mid-stream disconnect, empty
response, missing done:true, zero eval_count, thinking field.
"""

from __future__ import annotations

import unittest

from inference.ollama_client import (
    GenerationRequest,
    MalformedChunkError,
    OllamaHTTPError,
    OllamaTimeoutError,
    TruncatedStreamError,
    generate,
)
from tests.fake_ollama import FakeOllamaServer, Scenario, ScriptedChunk, final_chunk

MODEL = 'fake-qwen-q4'


def _request(**overrides: object) -> GenerationRequest:
    base: dict[str, object] = {'model': MODEL, 'prompt': 'Say hi.'}
    base.update(overrides)
    return GenerationRequest(
        model=str(base['model']),
        prompt=str(base['prompt']),
    )


class ClientStreamTests(unittest.TestCase):
    def test_ok_stream_assembly_and_metadata(self) -> None:
        scenario = Scenario(
            chunks=[
                ScriptedChunk(100, {'response': 'Hello ', 'done': False}),
                ScriptedChunk(50, {'response': 'world', 'done': False}),
                ScriptedChunk(0, final_chunk(eval_count=4, eval_duration=100_000_000)),
            ]
        )
        with FakeOllamaServer(scenario) as server:
            result = generate(server.base_url, _request())
        self.assertEqual(result.text, 'Hello world')
        self.assertEqual(result.thinking, '')
        self.assertEqual(result.done_reason, 'stop')
        self.assertEqual(result.eval_count, 4)
        self.assertEqual(result.eval_duration_ns, 100_000_000)
        self.assertEqual(result.prompt_eval_count, 10)
        self.assertIsNotNone(result.first_token_ns)
        assert result.first_token_ns is not None
        # First token arrived ~100ms after request start (generous bounds).
        ttft_ms = (result.first_token_ns - result.request_start_ns) / 1e6
        self.assertGreaterEqual(ttft_ms, 50.0)
        self.assertLess(ttft_ms, 2000.0)
        self.assertLess(result.first_token_ns, result.request_end_ns)

    def test_first_token_observed_before_stream_end(self) -> None:
        # Chunk 1 arrives fast, then a long stall: TTFT must reflect chunk 1,
        # proving small-chunk reads instead of wait-for-buffer-fill.
        scenario = Scenario(
            chunks=[
                ScriptedChunk(50, {'response': 'early', 'done': False}),
                ScriptedChunk(2500, {'response': 'late', 'done': False}),
                ScriptedChunk(0, final_chunk()),
            ]
        )
        with FakeOllamaServer(scenario) as server:
            result = generate(server.base_url, _request(), timeout_s=30.0)
        assert result.first_token_ns is not None
        ttft_ms = (result.first_token_ns - result.request_start_ns) / 1e6
        e2e_ms = (result.request_end_ns - result.request_start_ns) / 1e6
        self.assertLess(ttft_ms, 2000.0)
        self.assertGreater(e2e_ms, 2000.0)
        self.assertEqual(result.text, 'earlylate')

    def test_request_payload_shape(self) -> None:
        scenario = Scenario(chunks=[ScriptedChunk(0, final_chunk())])
        with FakeOllamaServer(scenario) as server:
            generate(
                server.base_url,
                GenerationRequest(
                    model=MODEL,
                    prompt='Hi.',
                    temperature=0.0,
                    num_ctx=4096,
                    num_predict=64,
                    stop=('</s>',),
                    raw=True,
                ),
            )
            body = server.last_request_json
        self.assertEqual(body['model'], MODEL)
        self.assertEqual(body['prompt'], 'Hi.')
        self.assertTrue(body['stream'])
        self.assertTrue(body['raw'])
        self.assertEqual(body['options']['temperature'], 0.0)
        self.assertEqual(body['options']['num_ctx'], 4096)
        self.assertEqual(body['options']['num_predict'], 64)
        self.assertEqual(body['options']['stop'], ['</s>'])

    def test_malformed_line_raises(self) -> None:
        scenario = Scenario(
            chunks=[
                ScriptedChunk(0, {'response': 'ok', 'done': False}),
                ScriptedChunk(0, 'this is not json{{{'),
                ScriptedChunk(0, final_chunk()),
            ]
        )
        with FakeOllamaServer(scenario) as server:
            with self.assertRaises(MalformedChunkError):
                generate(server.base_url, _request())

    def test_http_500_raises(self) -> None:
        with FakeOllamaServer(Scenario(status_code=500)) as server:
            with self.assertRaises(OllamaHTTPError) as ctx:
                generate(server.base_url, _request())
        self.assertEqual(ctx.exception.status_code, 500)

    def test_timeout_raises(self) -> None:
        scenario = Scenario(chunks=[ScriptedChunk(3000, {'response': 'slow', 'done': False})])
        with FakeOllamaServer(scenario) as server:
            with self.assertRaises(OllamaTimeoutError):
                generate(server.base_url, _request(), timeout_s=0.3)

    def test_midstream_disconnect_raises(self) -> None:
        scenario = Scenario(
            chunks=[
                ScriptedChunk(0, {'response': 'partial', 'done': False}),
                ScriptedChunk(0, final_chunk()),
            ],
            close_after_chunks=1,
        )
        with FakeOllamaServer(scenario) as server:
            with self.assertRaises(TruncatedStreamError):
                generate(server.base_url, _request())

    def test_missing_done_raises(self) -> None:
        scenario = Scenario(
            chunks=[ScriptedChunk(0, {'response': 'no finale', 'done': False})]
        )
        with FakeOllamaServer(scenario) as server:
            with self.assertRaises(TruncatedStreamError):
                generate(server.base_url, _request())

    def test_empty_response_zero_eval_ok(self) -> None:
        scenario = Scenario(
            chunks=[
                ScriptedChunk(
                    0, final_chunk(response='', eval_count=0, eval_duration=0)
                )
            ]
        )
        with FakeOllamaServer(scenario) as server:
            result = generate(server.base_url, _request())
        self.assertEqual(result.text, '')
        self.assertEqual(result.eval_count, 0)
        self.assertIsNone(result.first_token_ns)

    def test_thinking_captured_separately(self) -> None:
        scenario = Scenario(
            chunks=[
                ScriptedChunk(0, {'thinking': 'hmm', 'done': False}),
                ScriptedChunk(0, {'response': 'answer', 'done': False}),
                ScriptedChunk(0, final_chunk(thinking='hmm', response='answer')),
            ]
        )
        with FakeOllamaServer(scenario) as server:
            result = generate(server.base_url, _request())
        self.assertEqual(result.thinking, 'hmmhmm')
        self.assertEqual(result.text, 'answeranswer')
        self.assertIsNotNone(result.first_token_ns)

    def test_missing_counters_become_none(self) -> None:
        scenario = Scenario(chunks=[ScriptedChunk(0, {'response': 'x', 'done': True})])
        with FakeOllamaServer(scenario) as server:
            result = generate(server.base_url, _request())
        self.assertEqual(result.text, 'x')
        self.assertIsNone(result.eval_count)
        self.assertIsNone(result.eval_duration_ns)
        self.assertIsNone(result.prompt_eval_count)
        self.assertIsNone(result.total_duration_ns)


if __name__ == '__main__':
    unittest.main()
