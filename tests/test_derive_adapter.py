"""Adapter overlay + derivation tests (offline, injected seams)."""

from __future__ import annotations

import hashlib
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
            seen: list[GenerationRequest] = []
            unloads = {'n': 0}

            def trial(request: GenerationRequest) -> GenerationResult:
                seen.append(request)
                if not request.raw:
                    return _result('')
                return _result('OK')

            def ensure_unloaded() -> bool:
                unloads['n'] += 1
                return True

            config = derive_adapter(
                registry_entry={**ENTRY, 'model_config_id': 'qwen3-4b-q9'},
                show_doc=dict(_show('qwen3')),
                observed_digest='b' * 64, trial_generate=trial,
                overlay_path=overlay, ensure_unloaded=ensure_unloaded,
            )
            self.assertEqual(config.mode, 'raw')
            self.assertEqual(config.template_application, 'client_rendered')
            self.assertIn('{prompt}', config.template_text)
            # Unload rule: old chat load evicted before the raw trial.
            self.assertEqual(unloads['n'], 1)
            raw_trials = [r for r in seen if r.raw]
            self.assertEqual(len(raw_trials), 1)
            self.assertEqual(raw_trials[0].num_ctx, 4096)
            self.assertEqual(raw_trials[0].num_predict, 1)
            self.assertEqual(raw_trials[0].num_gpu, 99)

    def test_raw_fallback_aborts_when_unload_fails(self) -> None:
        with TemporaryDirectory() as tmp:
            def trial(request: GenerationRequest) -> GenerationResult:
                return _result('' if not request.raw else 'OK')

            with self.assertRaises(ManualPinRequired):
                derive_adapter(
                    registry_entry={**ENTRY, 'model_config_id': 'qwen3-4b-q9'},
                    show_doc=dict(_show('qwen3')),
                    observed_digest='b' * 64, trial_generate=trial,
                    overlay_path=str(Path(tmp) / 'q.json'),
                    ensure_unloaded=lambda: False,
                )

    def test_chat_trial_uses_canonical_options(self) -> None:
        with TemporaryDirectory() as tmp:
            seen: list[GenerationRequest] = []

            def trial(request: GenerationRequest) -> GenerationResult:
                seen.append(request)
                return _result('OK')

            derive_adapter(
                registry_entry=dict(ENTRY), show_doc=dict(_show('llama')),
                observed_digest='a' * 64, trial_generate=trial,
                overlay_path=str(Path(tmp) / 'x.json'),
            )
            self.assertEqual(len(seen), 1)
            self.assertEqual(seen[0].num_ctx, 4096)
            self.assertEqual(seen[0].num_predict, 1)
            self.assertEqual(seen[0].num_gpu, 99)
            self.assertFalse(seen[0].raw)

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


class GemmaPinTests(unittest.TestCase):
    # Byte-level fixture: the Gemma template is LF-only UTF-8, hashed from
    # parsed (not file) bytes so Windows CRLF can never slip in silently.
    EXPECTED_HEX = (
        '3c626f733e3c73746172745f6f665f7475726e3e757365720a'
        '7b70726f6d70747d3c656e645f6f665f7475726e3e0a3c737461'
        '72745f6f665f7475726e3e6d6f64656c0a'
    )
    EXPECTED_SHA = 'ca2868c165f954b50535658de436e3365800df16b36710454a1f651926b98774'

    def _overlay_doc(self) -> dict[str, object]:
        root = Path(__file__).resolve().parents[1]
        document = json.loads(
            (root / 'configs/adapters/gemma-3n-e2b-q4.json').read_text(encoding='utf-8')
        )
        assert isinstance(document, dict)
        return document

    def test_exact_template_bytes(self) -> None:
        document = self._overlay_doc()
        raw = str(document['template_text']).encode('utf-8')
        self.assertEqual(raw.hex(), self.EXPECTED_HEX)
        self.assertNotIn(b'\r', raw)
        self.assertEqual(hashlib.sha256(raw).hexdigest(), self.EXPECTED_SHA)
        self.assertEqual(document['template_sha256'], self.EXPECTED_SHA)

    def test_no_qwen_markers(self) -> None:
        document = self._overlay_doc()
        text = str(document['template_text'])
        self.assertNotIn('<|im_start|>', text)
        self.assertNotIn('<|im_end|>', text)
        self.assertEqual(text.count('{prompt}'), 1)

    def test_raw_canonical_provenance(self) -> None:
        document = self._overlay_doc()
        self.assertEqual(document['mode'], 'raw')
        self.assertEqual(document['template_application'], 'client_rendered')
        self.assertEqual(document['derivation_source'], 'human_pinned')
        self.assertEqual(document['review_status'], 'APPROVED')
        stops = document.get('stops_provenance', {})
        assert isinstance(stops, dict)
        self.assertEqual(stops.get('kind'), 'derived_from_pinned_template')

    def test_draft_human_pin_refused(self) -> None:
        from inference.derive_adapter import UnapprovedPinError, assert_overlay_approved

        with self.assertRaises(UnapprovedPinError):
            assert_overlay_approved({
                'config_id': 'x', 'derivation_source': 'human_pinned',
                'review_status': 'DRAFT',
            })
        # Machine-derived overlays are exempt (verified at derive time).
        assert_overlay_approved({'config_id': 'y'})


if __name__ == '__main__':
    unittest.main()
