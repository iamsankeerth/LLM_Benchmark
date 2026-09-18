"""Retry classifier tests: per-type decisions, backoff behavior."""

from __future__ import annotations

import unittest

import requests

from inference.ollama_client import (
    MalformedChunkError,
    OllamaClientError,
    OllamaHTTPError,
    OllamaTimeoutError,
    TruncatedStreamError,
)
from inference.retry import (
    GENERATION_RETRIES,
    PULL_BACKOFFS,
    RM_ATTEMPTS,
    error_kind,
    is_retryable,
    run_with_retries,
)


class ClassifierTests(unittest.TestCase):
    def test_retryable_transports(self) -> None:
        self.assertTrue(is_retryable(OllamaTimeoutError('t')))
        self.assertTrue(is_retryable(TruncatedStreamError('t')))
        self.assertTrue(is_retryable(OllamaHTTPError(500, 'x')))
        self.assertTrue(is_retryable(OllamaHTTPError(503, 'x')))
        self.assertTrue(is_retryable(requests.Timeout()))
        self.assertTrue(is_retryable(requests.ConnectionError()))
        self.assertTrue(is_retryable(TimeoutError()))
        self.assertTrue(is_retryable(ConnectionResetError()))

    def test_non_retryable_semantics(self) -> None:
        self.assertFalse(is_retryable(OllamaHTTPError(400, 'x')))
        self.assertFalse(is_retryable(OllamaHTTPError(404, 'x')))
        self.assertFalse(is_retryable(MalformedChunkError('x')))
        self.assertFalse(is_retryable(OllamaClientError('x')))
        self.assertFalse(is_retryable(ValueError('x')))

    def test_error_kinds(self) -> None:
        self.assertEqual(error_kind(OllamaTimeoutError('t')), 'timeout')
        self.assertEqual(error_kind(OllamaHTTPError(503, 'x')), 'http_503')
        self.assertEqual(error_kind(MalformedChunkError('x')), 'malformed_chunk')

    def test_schedule_constants(self) -> None:
        self.assertEqual(PULL_BACKOFFS, (30.0, 120.0, 300.0))
        self.assertEqual(RM_ATTEMPTS, 3)
        self.assertEqual(GENERATION_RETRIES, 2)


class BackoffRunnerTests(unittest.TestCase):
    def test_retry_then_success(self) -> None:
        calls = {'n': 0}
        sleeps: list[float] = []

        def flaky() -> str:
            calls['n'] += 1
            if calls['n'] < 3:
                raise OllamaTimeoutError('blip')
            return 'ok'

        result = run_with_retries(
            flaky, attempts=3, backoffs=(5.0, 15.0),
            sleep=sleeps.append,
        )
        self.assertEqual(result, 'ok')
        self.assertEqual(calls['n'], 3)
        self.assertEqual(sleeps, [5.0, 15.0])

    def test_non_retryable_raises_immediately(self) -> None:
        calls = {'n': 0}

        def bad() -> str:
            calls['n'] += 1
            raise MalformedChunkError('corrupt')

        with self.assertRaises(MalformedChunkError):
            run_with_retries(bad, attempts=3, backoffs=(1.0,))
        self.assertEqual(calls['n'], 1)

    def test_exhaustion_raises_last(self) -> None:
        with self.assertRaises(OllamaTimeoutError):
            run_with_retries(
                lambda: (_ for _ in ()).throw(OllamaTimeoutError('down')),
                attempts=2, backoffs=(),
                sleep=lambda s: None,
            )


if __name__ == '__main__':
    unittest.main()
