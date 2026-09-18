"""Adapter overlay + derivation tests (offline, injected seams)."""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from inference.adapters import (
    clear_overlays,
    config_from_overlay_json,
    config_to_overlay_json,
    get_model_config,
    load_overlays_from_dir,
    register_overlay,
)
from inference.derive_adapter import (
    ManualPinRequired,
    OverlayExistsError,
    derive_adapter,
    family_markers,
    parse_modelfile_stops,
    trial_default_route,
    verify_overlay_matches_live,
)
from inference.ollama_client import GenerationRequest, GenerationResult


def _result(text: str, done_reason: str | None = 'stop') -> GenerationResult:
    return GenerationResult(
        text=text, thinking='', done_reason=done_reason,
        request_start_ns=0, first_token_ns=1, request_end_ns=2,
    )


def _show(family: str = 'qwen3', template: str = 'TEMPLATE {{.X}}') -> dict[str, object]:
    return {
        'template': template,
        'parameters': 'stop                           "<|im_start|>"\nstop                           "<|im_end|>"\ntemperature                    0.6\n',
        'details': {'family': family, 'families': [family]},
    }


ENTRY = {
    'model_config_id': 'llama3.2-3b-q4',
    'family': 'llama3.2-3b',
    'quantization': 'Q4_K_M',
    'ollama_identifier': 'hf.co/bartowski/Llama-3.2-3B-Instruct-GGUF:Q4_K_M',
}


class DeriveTests(unittest.TestCase):
    def setUp(self) -> None:
        clear_overlays()
        self.addCleanup(clear_overlays)

    def test_chat_route_derived_and_pinned(self) -> None:
        with TemporaryDirectory() as tmp:
            overlay = str(Path(tmp) / 'llama3.2-3b-q4.json')

            def trial(request: GenerationRequest) -> GenerationResult:
                self.assertFalse(request.raw)
                return _result('OK')

            config = derive_adapter(
                registry_entry=dict(ENTRY), show_doc=dict(_show('llama')),
                observed_digest='a' * 64, trial_generate=trial,
                overlay_path=overlay,
            )
            self.assertEqual(config.mode, 'chat')
            self.assertEqual(config.template_application, 'server_managed')
            self.assertEqual(config.template_text, 'TEMPLATE {{.X}}')
            self.assertEqual(config.stop_tokens, ('<|im_start|>', '<|im_end|>'))
            self.assertTrue(Path(overlay).exists())
            # Round-trip: pinned file loads identically.
            reloaded = config_from_overlay_json(
                json.loads(Path(overlay).read_text(encoding='utf-8'))
            )
            self.assertEqual(reloaded.template_sha256, config.template_sha256)

    def test_raw_fallback_for_known_family(self) -> None:
        with TemporaryDirectory() as tmp:
            overlay = str(Path(tmp) / 'q.json')

            def trial(request: GenerationRequest) -> GenerationResult:
                if not request.raw:
                    return _result('')
                return _result('OK')

            config = derive_adapter(
                registry_entry={**ENTRY, 'model_config_id': 'qwen3-4b-q9'},
                show_doc=dict(_show('qwen3')),
                observed_digest='b' * 64, trial_generate=trial,
                overlay_path=overlay,
            )
            self.assertEqual(config.mode, 'raw')
            self.assertEqual(config.template_application, 'client_rendered')
            self.assertIn('{prompt}', config.template_text)

    def test_unknown_family_fails_closed(self) -> None:
        with TemporaryDirectory() as tmp:
            def trial(request: GenerationRequest) -> GenerationResult:
                return _result('')

            with self.assertRaises(ManualPinRequired):
                derive_adapter(
                    registry_entry=dict(ENTRY), show_doc=dict(_show('mystery')),
                    observed_digest='c' * 64, trial_generate=trial,
                    overlay_path=str(Path(tmp) / 'x.json'),
                )
            self.assertFalse(Path(tmp, 'x.json').exists())

    def test_create_once_enforced(self) -> None:
        with TemporaryDirectory() as tmp:
            overlay = str(Path(tmp) / 'x.json')
            Path(overlay).write_text('{}', encoding='utf-8')
            with self.assertRaises(OverlayExistsError):
                derive_adapter(
                    registry_entry=dict(ENTRY), show_doc=dict(_show()),
                    observed_digest='d' * 64,
                    trial_generate=lambda req: _result('OK'),
                    overlay_path=overlay,
                )

    def test_overlay_mismatch_detected(self) -> None:
        with TemporaryDirectory() as tmp:
            overlay = str(Path(tmp) / 'llama3.2-3b-q4.json')

            def trial(request: GenerationRequest) -> GenerationResult:
                return _result('OK')

            config = derive_adapter(
                registry_entry=dict(ENTRY), show_doc=dict(_show('llama')),
                observed_digest='a' * 64, trial_generate=trial,
                overlay_path=overlay,
            )
            self.assertEqual(
                verify_overlay_matches_live(config, _show('llama'), 'a' * 64), []
            )
            mismatches = verify_overlay_matches_live(config, _show('llama'), 'f' * 64)
            self.assertEqual(len(mismatches), 1)
            self.assertEqual(mismatches[0].field, 'ollama_model_digest')
            changed = dict(_show('llama'))
            changed['parameters'] = 'stop                           "<|end|>"\n'
            mismatches = verify_overlay_matches_live(config, changed, 'a' * 64)
            self.assertTrue(any(m.field == 'stop_tokens' for m in mismatches))

    def test_overlay_collision_with_coded_rejected(self) -> None:
        qwen = get_model_config('qwen3-4b-q4')
        with self.assertRaises(ValueError):
            register_overlay(qwen)

    def test_overlay_round_trip_dir(self) -> None:
        with TemporaryDirectory() as tmp:
            overlay = str(Path(tmp) / 'llama3.2-3b-q4.json')

            def trial(request: GenerationRequest) -> GenerationResult:
                return _result('OK')

            config = derive_adapter(
                registry_entry=dict(ENTRY), show_doc=dict(_show('llama')),
                observed_digest='a' * 64, trial_generate=trial,
                overlay_path=overlay,
            )
            clear_overlays()
            self.assertEqual(load_overlays_from_dir(tmp), 1)
            fetched = get_model_config('llama3.2-3b-q4')
            self.assertEqual(fetched.template_sha256, config.template_sha256)
            self.assertEqual(
                json.loads(config_to_overlay_json(fetched))['mode'], 'chat'
            )


class ParseTests(unittest.TestCase):
    def test_stops_parsed(self) -> None:
        self.assertEqual(
            parse_modelfile_stops('stop  "<a>"\ntemperature  0.6\nstop "<b>"\n'),
            ['<a>', '<b>'],
        )
        self.assertEqual(parse_modelfile_stops('temperature 0.6\n'), [])

    def test_family_markers(self) -> None:
        self.assertIn('qwen3', family_markers(_show('Qwen3')))
        self.assertEqual(family_markers({}), [])

    def test_trial_transport_checks(self) -> None:
        good = lambda req: _result('OK')  # noqa: E731
        self.assertTrue(
            trial_default_route(good, 'm')
        )

        def boom(req: GenerationRequest) -> GenerationResult:
            raise RuntimeError('down')

        self.assertFalse(trial_default_route(boom, 'm'))
        self.assertFalse(
            trial_default_route(lambda req: _result(''), 'm')
        )
        self.assertFalse(
            trial_default_route(lambda req: _result('x', None), 'm')
        )


if __name__ == '__main__':
    unittest.main()
