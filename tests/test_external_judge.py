"""Offline tests for the external OpenAI-compatible judge client."""

from __future__ import annotations

import os
import unittest
from typing import Any
from unittest.mock import patch

from inference.external_judge import ExternalJudgeClient, ExternalJudgeConfig, ExternalJudgeError


class _Response:
    def __init__(
        self,
        status_code: int,
        content: str,
        usage: dict[str, Any] | None = None,
    ) -> None:
        self.status_code = status_code
        self.headers = {'x-request-id': 'req-1'}
        self._content = content
        self._usage = usage or {'total_tokens': 100}

    def json(self) -> dict[str, Any]:
        return {
            'choices': [{'message': {'content': self._content}}],
            'usage': self._usage,
        }


class ExternalJudgeClientTests(unittest.TestCase):
    def _config(self, **overrides: Any) -> ExternalJudgeConfig:
        values: dict[str, Any] = {
            'base_url': 'https://api.example.test/v1',
            'api_key': 'secret-key',
            'model_id': 'stealth/space-bunny-alpha',
            'max_calls': 3,
            'max_cost_usd': 5.0,
            'cost_per_1k_tokens': 0.01,
        }
        values.update(overrides)
        return ExternalJudgeConfig(**values)

    def test_request_is_pinned_and_does_not_log_key(self) -> None:
        seen: list[dict[str, Any]] = []

        def post(url: str, **kwargs: Any) -> _Response:
            seen.append({'url': url, **kwargs})
            return _Response(200, '{"ok":true}')

        client = ExternalJudgeClient(self._config(), post_fn=post, sleep_fn=lambda _: None)
        result = client.judge('exact candidate output', {'type': 'object'})
        self.assertEqual(result.text, '{"ok":true}')
        self.assertEqual(result.request_id, 'req-1')
        self.assertEqual(seen[0]['url'], 'https://api.example.test/v1/chat/completions')
        self.assertEqual(seen[0]['json']['model'], 'stealth/space-bunny-alpha')
        self.assertEqual(seen[0]['json']['temperature'], 0.0)
        self.assertIn('exact candidate output', seen[0]['json']['messages'][1]['content'])
        self.assertNotIn('secret-key', str(seen[0]['json']))

    def test_environment_does_not_require_a_token_price(self) -> None:
        document = {
            'external_judge': {
                'base_url_env': 'TEST_JUDGE_BASE_URL',
                'api_key_env': 'TEST_JUDGE_API_KEY',
                'model_id': 'stealth/space-bunny-alpha',
                'max_cost_usd': None,
                'cost_per_1k_tokens_env': None,
            },
        }
        with patch.dict(
            os.environ,
            {
                'TEST_JUDGE_BASE_URL': 'https://openrouter.ai/api/v1',
                'TEST_JUDGE_API_KEY': 'runtime-key',
                'SPACE_BUNNY_MODEL_ID': 'stealth/space-bunny-alpha',
            },
            clear=False,
        ):
            config = ExternalJudgeConfig.from_environment(document)
        self.assertIsNone(config.cost_per_1k_tokens)
        self.assertIsNone(config.max_cost_usd)

    def test_provider_reported_cost_is_recorded_without_token_price(self) -> None:
        client = ExternalJudgeClient(
            self._config(max_cost_usd=None, cost_per_1k_tokens=None),
            post_fn=lambda *_args, **_kwargs: _Response(
                200, '{"ok":true}', {'total_tokens': 100, 'cost': 0.25},
            ),
        )
        result = client.judge('candidate', {'type': 'object'})
        self.assertEqual(result.estimated_cost_usd, 0.25)
        self.assertEqual(result.cost_source, 'provider_usage')
        self.assertEqual(client.total_cost_usd, 0.25)
        self.assertEqual(client.known_cost_calls, 1)
        self.assertEqual(client.unknown_cost_calls, 0)

    def test_unknown_cost_stays_unknown_and_call_cap_still_applies(self) -> None:
        client = ExternalJudgeClient(
            self._config(max_cost_usd=None, cost_per_1k_tokens=None, max_calls=1),
            post_fn=lambda *_args, **_kwargs: _Response(200, '{"ok":true}'),
        )
        result = client.judge('candidate', {'type': 'object'})
        self.assertIsNone(result.estimated_cost_usd)
        self.assertEqual(result.cost_source, 'unavailable')
        self.assertIsNone(client.total_cost_usd)
        self.assertEqual(client.known_cost_calls, 0)
        self.assertEqual(client.unknown_cost_calls, 1)
        with self.assertRaises(ExternalJudgeError):
            client.judge('candidate', {'type': 'object'})

    def test_transient_retry_is_bounded(self) -> None:
        attempts: list[int] = []
        sleeps: list[float] = []

        def post(*_args: Any, **_kwargs: Any) -> _Response:
            attempts.append(1)
            return _Response(500 if len(attempts) < 2 else 200, '{}')

        client = ExternalJudgeClient(
            self._config(max_attempts=2), post_fn=post,
            sleep_fn=sleeps.append,
        )
        client.judge('candidate', {'type': 'object'})
        self.assertEqual(len(attempts), 2)
        self.assertEqual(sleeps, [1.0])

    def test_call_budget_refuses_before_request(self) -> None:
        client = ExternalJudgeClient(
            self._config(max_calls=0),
            post_fn=lambda *_args, **_kwargs: self.fail('must not call API'),
        )
        with self.assertRaises(ExternalJudgeError):
            client.judge('candidate', {'type': 'object'})

    def test_cost_budget_exposes_usage_without_storing_raw_response(self) -> None:
        client = ExternalJudgeClient(
            self._config(max_cost_usd=0.0005),
            post_fn=lambda *_args, **_kwargs: _Response(200, '{"secret":"response"}'),
        )
        with self.assertRaises(ExternalJudgeError) as context:
            client.judge('candidate', {'type': 'object'})
        self.assertEqual(client.calls_made, 1)
        self.assertEqual(client.total_cost_usd, 0.001)
        self.assertEqual(context.exception.usage, {'total_tokens': 100})
        self.assertIsNotNone(context.exception.response_sha256)
        self.assertNotIn('response', str(context.exception))

    def test_non_retryable_status_stops_immediately(self) -> None:
        attempts: list[int] = []

        def post(*_args: Any, **_kwargs: Any) -> _Response:
            attempts.append(1)
            return _Response(401, '{}')

        client = ExternalJudgeClient(
            self._config(max_attempts=3), post_fn=post, sleep_fn=lambda _: None,
        )
        with self.assertRaises(ExternalJudgeError) as context:
            client.judge('candidate', {'type': 'object'})
        self.assertEqual(len(attempts), 1)
        self.assertIn('401', str(context.exception))


if __name__ == '__main__':
    unittest.main()
