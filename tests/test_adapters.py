"""Adapter tests: registry completeness, byte-exact template rendering."""

from __future__ import annotations

import hashlib
import unittest

from inference.adapters import (
    MODEL_CONFIGS,
    ModelConfig,
    get_model_config,
    render_prompt,
    rendered_prompt_sha256,
    verify_template_integrity,
)

# Byte-exact expectation: the rendered prompt that produced output '42'
# in the 2026-09-17 live probe. Any template edit breaks this test.
PROBE_TASK_PROMPT = 'Return only the number 42 and nothing else.'
PROBE_RENDERED_BYTES = (
    '<|im_start|>user\n'
    'Return only the number 42 and nothing else.'
    '<|im_end|>\n'
    '<|im_start|>assistant\n'
)


class AdapterRegistryTests(unittest.TestCase):
    def test_registry_complete_and_valid(self) -> None:
        self.assertIn('qwen3-4b-q4', MODEL_CONFIGS)
        for config_id, config in MODEL_CONFIGS.items():
            self.assertEqual(config.config_id, config_id)
            self.assertTrue(config.ollama_identifier)
            self.assertTrue(config.quantization)
            self.assertIn(config.mode, ('raw', 'chat'))
            self.assertTrue(config.template_id)
            self.assertTrue(config.template_source)
            if config.mode == 'raw':
                self.assertEqual(config.template_text.count('{prompt}'), 1)
            else:
                self.assertEqual(config.template_application, 'server_managed')
            self.assertEqual(len(config.template_sha256), 64)
            self.assertEqual(len(config.ollama_model_digest), 64)
            self.assertGreater(config.num_ctx, 0)
            self.assertGreater(config.num_gpu, 0)
            self.assertTrue(config.stop_tokens)
            self.assertIn(config.thinking_support, ('supported', 'unsupported'))
            self.assertTrue(config.gpu_only)

    def test_qwen_config_matches_live_evidence(self) -> None:
        config = get_model_config('qwen3-4b-q4')
        self.assertEqual(
            config.ollama_identifier,
            'hf.co/bartowski/Qwen_Qwen3-4B-Instruct-2507-GGUF:Q4_K_M',
        )
        self.assertEqual(config.mode, 'raw')
        self.assertEqual(config.think, False)
        self.assertIn('<|im_end|>', config.stop_tokens)

    def test_qwen5_config_matches_live_evidence(self) -> None:
        config = get_model_config('qwen3-4b-q5')
        self.assertEqual(
            config.ollama_identifier,
            'hf.co/bartowski/Qwen_Qwen3-4B-Instruct-2507-GGUF:Q5_K_M',
        )
        self.assertEqual(config.mode, 'raw')
        self.assertEqual(config.think, False)
        # Same template mechanics as Q4: shared template id and bytes.
        q4 = get_model_config('qwen3-4b-q4')
        self.assertEqual(config.template_id, q4.template_id)
        self.assertEqual(config.template_sha256, q4.template_sha256)
        self.assertEqual(
            config.ollama_model_digest,
            '7b56805a15439abd489ff24764b46b349c2f5fa0d52cf41c66c6ba2492093138',
        )

    def test_unknown_config_raises_helpful_error(self) -> None:
        with self.assertRaises(KeyError) as ctx:
            get_model_config('nope')
        self.assertIn('qwen3-4b-q4', str(ctx.exception))


class TemplateRenderingTests(unittest.TestCase):
    def test_chat_mode_sends_bare_prompt(self) -> None:
        chat_like = ModelConfig(
            config_id='chat-probe', ollama_identifier='m', quantization='Q',
            mode='chat', template_id='t', template_source='s',
            template_text='GO TEMPLATE {{.X}} (no prompt slot)',
            template_sha256='0' * 64, ollama_model_digest='d' * 64,
            num_ctx=4096, num_predict_default=512, num_gpu=99,
            temperature_default=0.0, stop_tokens=('</s>',), think=None,
            thinking_support='supported', template_application='server_managed',
            gpu_only=True,
        )
        self.assertEqual(render_prompt(chat_like, 'Hello.'), 'Hello.')

    def test_render_is_byte_exact_probe_reproduction(self) -> None:
        config = get_model_config('qwen3-4b-q4')
        self.assertEqual(render_prompt(config, PROBE_TASK_PROMPT), PROBE_RENDERED_BYTES)

    def test_rendered_sha_matches_recorded_probe_digest(self) -> None:
        config = get_model_config('qwen3-4b-q4')
        rendered = render_prompt(config, PROBE_TASK_PROMPT)
        expected = hashlib.sha256(PROBE_RENDERED_BYTES.encode('utf-8')).hexdigest()
        self.assertEqual(rendered_prompt_sha256(rendered), expected)

    def test_template_integrity_self_check(self) -> None:
        for config in MODEL_CONFIGS.values():
            verify_template_integrity(config)

    def test_template_tamper_detected(self) -> None:
        config = get_model_config('qwen3-4b-q4')
        tampered = ModelConfig(
            config_id=config.config_id,
            ollama_identifier=config.ollama_identifier,
            quantization=config.quantization,
            mode=config.mode,
            template_id=config.template_id,
            template_source=config.template_source,
            template_text=config.template_text + ' ',
            template_sha256=config.template_sha256,
            ollama_model_digest=config.ollama_model_digest,
            num_ctx=config.num_ctx,
            num_predict_default=config.num_predict_default,
            num_gpu=config.num_gpu,
            temperature_default=config.temperature_default,
            stop_tokens=config.stop_tokens,
            think=config.think,
            thinking_support=config.thinking_support,
            template_application=config.template_application,
            gpu_only=config.gpu_only,
        )
        with self.assertRaises(ValueError):
            verify_template_integrity(tampered)


if __name__ == '__main__':
    unittest.main()
