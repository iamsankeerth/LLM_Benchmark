"""Registry readiness tests: structural gate offline, API parsing mocked."""

from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.check_registry_readiness import (
    check_artifact_files,
    check_entry,
    check_registry,
    load_registry,
    parse_ollama_identifier,
)


def _response(status: int, siblings: list[str]) -> object:
    class FakeResponse:
        status_code = status

        def json(self) -> object:
            return {'siblings': [{'rfilename': name} for name in siblings]}

    return FakeResponse()


class IdentifierParsingTests(unittest.TestCase):
    def test_valid_shapes(self) -> None:
        repo, quant, _ = parse_ollama_identifier(
            'hf.co/bartowski/Llama-3.2-3B-Instruct-GGUF:Q4_K_M'
        )
        self.assertEqual(repo, 'bartowski/Llama-3.2-3B-Instruct-GGUF')
        self.assertEqual(quant, 'Q4_K_M')
        repo, quant, _ = parse_ollama_identifier(
            'hf.co/bartowski/google_gemma-3n-E2B-it-GGUF:Q5_K_M'
        )
        self.assertEqual(repo, 'bartowski/google_gemma-3n-E2B-it-GGUF')
        self.assertEqual(quant, 'Q5_K_M')

    def test_placeholders_rejected(self) -> None:
        for bad in (
            'hf.co/TODO/repo:Q4_K_M',
            'hf.co/bartowski/repo:...',
            'ollama run llama3.2',
            'hf.co/bartowski/nocolon',
        ):
            with self.assertRaises(ValueError, msg=bad):
                parse_ollama_identifier(bad)


class ArtifactCheckTests(unittest.TestCase):
    def test_quant_file_found(self) -> None:
        with patch(
            'scripts.check_registry_readiness.requests.get',
            return_value=_response(200, [
                'Llama-3.2-3B-Instruct-Q4_K_M.gguf',
                'Llama-3.2-3B-Instruct-Q5_K_M.gguf',
                'README.md',
            ]),
        ):
            ok, detail = check_artifact_files(
                'bartowski/Llama-3.2-3B-Instruct-GGUF', 'Q5_K_M'
            )
        self.assertTrue(ok)
        self.assertIn('Q5_K_M', detail)

    def test_quant_file_missing(self) -> None:
        with patch(
            'scripts.check_registry_readiness.requests.get',
            return_value=_response(200, ['model-Q4_K_M.gguf']),
        ):
            ok, detail = check_artifact_files('org/repo', 'Q8_0')
        self.assertFalse(ok)
        self.assertIn('Q8_0', detail)

    def test_api_error(self) -> None:
        with patch(
            'scripts.check_registry_readiness.requests.get',
            return_value=_response(404, []),
        ):
            ok, _ = check_artifact_files('org/missing', 'Q4_K_M')
        self.assertFalse(ok)

    def test_instruct_intent_required(self) -> None:
        entry = {
            'model_config_id': 'x', 'family': 'y', 'quantization': 'Q4_K_M',
            'ollama_identifier': 'hf.co/bartowski/Some-Base-Model-GGUF:Q4_K_M',
        }
        with patch(
            'scripts.check_registry_readiness.requests.get',
            return_value=_response(200, ['x-Q4_K_M.gguf']),
        ):
            ok, detail = check_entry(entry)
        self.assertFalse(ok)
        self.assertIn('instruct', detail)

    def test_it_suffix_counts_as_instruct(self) -> None:
        entry = {
            'model_config_id': 'gemma-3n-e2b-q4', 'family': 'gemma-3n-e2b',
            'quantization': 'Q4_K_M',
            'ollama_identifier': 'hf.co/bartowski/google_gemma-3n-E2B-it-GGUF:Q4_K_M',
        }
        with patch(
            'scripts.check_registry_readiness.requests.get',
            return_value=_response(200, ['gemma-3n-e2b-it-q4_k_m.gguf']),
        ):
            ok, detail = check_entry(entry)
        self.assertTrue(ok, detail)

    def test_quant_mismatch_rejected(self) -> None:
        entry = {
            'model_config_id': 'x', 'family': 'y', 'quantization': 'Q5_K_M',
            'ollama_identifier': 'hf.co/bartowski/Some-Instruct-GGUF:Q4_K_M',
        }
        ok, _ = check_entry(entry)
        self.assertFalse(ok)


class StructuralGateTests(unittest.TestCase):
    def test_live_registry_structure(self) -> None:
        root = Path(__file__).resolve().parents[1]
        registry = load_registry(root / 'configs/models-v2.yaml')
        failures, messages = check_registry(registry)
        self.assertEqual((failures, messages), (0, []))
        entries = registry['models']
        self.assertEqual(len(entries), 14)
        quants = [e['quantization'] for e in entries]
        self.assertIn('Q8_0', quants)
        smol = [e for e in entries if e['family'] == 'smolm2-1.7b']
        self.assertEqual(len(smol), 4)
        self.assertTrue(all('nstruct' in e['ollama_identifier'] for e in smol))

    def test_duplicate_detection(self) -> None:
        failures, messages = check_registry({
            'experiment_spec': 's',
            'models': [
                {'model_config_id': 'a', 'family': 'f', 'quantization': 'Q',
                 'ollama_identifier': 'hf.co/o/r:Q'},
                {'model_config_id': 'a', 'family': 'f', 'quantization': 'Q',
                 'ollama_identifier': 'hf.co/o/r:Q'},
            ],
        })
        self.assertGreater(failures, 0)
        self.assertTrue(any('duplicate' in m for m in messages))


if __name__ == '__main__':
    unittest.main()
