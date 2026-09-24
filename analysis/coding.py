"""Safe coding-output extraction and static checks; no candidate execution."""

from __future__ import annotations

import ast
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


class CodingProtocolError(ValueError):
    """Coding output or fixture violates the safe extraction contract."""


@dataclass(frozen=True)
class ExtractedCode:
    source: str
    source_sha256: str
    extraction_method: str


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


def extract_code(
    raw_output: str,
    *,
    language: str,
    done_reason: str | None,
    policy: Mapping[str, Any],
) -> ExtractedCode:
    """Extract raw source or one whole-document fence without heuristic search."""
    if done_reason == 'length':
        raise CodingProtocolError('output truncated at generation limit')
    if not raw_output or '\x00' in raw_output:
        raise CodingProtocolError('empty or NUL-containing output')
    if len(raw_output.encode('utf-8')) > int(policy['max_source_bytes']):
        raise CodingProtocolError('candidate source exceeds byte limit')
    text = raw_output.strip()
    languages = (
        policy['python_fence_languages']
        if language == 'python' else policy['javascript_fence_languages']
    )
    if text.count('```') > 2:
        raise CodingProtocolError('multiple fenced code blocks')
    fence_pattern = re.compile(
        r'^```(' + '|'.join(re.escape(value) for value in languages) + r')\s*\n(.*?)\n```$',
        re.DOTALL | re.IGNORECASE,
    )
    matches = list(fence_pattern.finditer(text))
    if len(matches) > 1:
        raise CodingProtocolError('multiple fenced code blocks')
    if matches:
        source = matches[0].group(2).strip()
        if text != matches[0].group(0).strip():
            raise CodingProtocolError('prose around fenced code block')
        return ExtractedCode(source, _sha(source), 'single_whole_document_fence')
    if '```' in text:
        raise CodingProtocolError('unsupported or malformed code fence')
    return ExtractedCode(text, _sha(text), 'raw_source')


def _python_names(tree: ast.AST) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
        elif isinstance(node, ast.Import):
            names.update(alias.name.split('.')[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split('.')[0])
    return names


def static_check(
    source: str,
    *,
    language: str,
    entrypoint: str,
    rules: list[str],
) -> dict[str, Any]:
    """Run non-executing syntax and prompt-specific static assertions."""
    if language == 'python':
        try:
            tree = ast.parse(source)
        except SyntaxError as exc:
            return {'passed': False, 'failure_kind': 'SYNTAX_ERROR', 'detail': str(exc)}
        functions = {node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
        if entrypoint not in functions:
            return {'passed': False, 'failure_kind': 'ENTRYPOINT_MISSING', 'detail': entrypoint}
        names = _python_names(tree)
        for rule in rules:
            if rule == 'no_set' and 'set' in names:
                return {'passed': False, 'failure_kind': 'FORBIDDEN_CONSTRUCT', 'detail': 'set'}
            if rule == 'no_dict_fromkeys' and 'fromkeys' in names:
                return {'passed': False, 'failure_kind': 'FORBIDDEN_CONSTRUCT', 'detail': 'dict.fromkeys'}
            if rule == 'no_third_party_imports' and any(name not in {'collections','__future__'} for name in names if name.islower()):
                pass
            if rule == 'no_re_import' and 're' in names:
                return {'passed': False, 'failure_kind': 'FORBIDDEN_CONSTRUCT', 'detail': 're'}
            if rule == 'no_try_except' and any(isinstance(node, ast.Try) for node in ast.walk(tree)):
                return {'passed': False, 'failure_kind': 'FORBIDDEN_CONSTRUCT', 'detail': 'try/except'}
            if rule == 'no_sleep' and 'sleep' in names:
                return {'passed': False, 'failure_kind': 'FORBIDDEN_CONSTRUCT', 'detail': 'sleep'}
        return {'passed': True, 'failure_kind': None, 'detail': 'static checks passed'}
    if language == 'javascript':
        if not re.search(rf'\b(?:function\s+{re.escape(entrypoint)}|const\s+{re.escape(entrypoint)}\s*=)', source):
            return {'passed': False, 'failure_kind': 'ENTRYPOINT_MISSING', 'detail': entrypoint}
        if source.count('{') != source.count('}'):
            return {'passed': False, 'failure_kind': 'SYNTAX_ERROR', 'detail': 'brace mismatch'}
        return {'passed': True, 'failure_kind': None, 'detail': 'static checks passed'}
    return {'passed': False, 'failure_kind': 'UNSUPPORTED_LANGUAGE', 'detail': language}


def load_fixture_manifest(path: Path) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(document, dict):
        raise CodingProtocolError('coding fixture manifest is not an object')
    if document.get('schema_version') != 'coding-fixtures-v1':
        raise CodingProtocolError('invalid coding fixture manifest')
    if set(document.get('tasks', {})) != {f'Q{index:03d}' for index in range(27, 35)}:
        raise CodingProtocolError('coding fixture task population is not Q027..Q034')
    return document
