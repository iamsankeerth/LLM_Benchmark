"""Deterministic grader implementations for eval-v1.

Each grader evaluates raw model output against a spec-layer grader entry and
returns a GraderResult. Graders never mutate the spec; they only read it.
"""

import json
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, cast


@dataclass
class GraderResult:
    grader_type: str
    passed: bool
    detail: str
    violations: list[str] = field(default_factory=list)


def normalize_text(text: str, *, trim: bool = True) -> str:
    normalized = unicodedata.normalize('NFC', text)
    normalized = normalized.replace('\r\n', '\n').replace('\r', '\n')
    return normalized.strip() if trim else normalized


def words(text: str) -> list[str]:
    """Pinned tokenizer: NFC, trim, then contiguous non-whitespace sequences."""
    return re.findall(r'\S+', normalize_text(text))


def _coerce_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        match = re.search(r'[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?', value)
        if match:
            try:
                return float(match.group())
            except ValueError:
                return None
    return None


def _extract_json(text: str) -> Any:
    cleaned = normalize_text(text)
    fence = re.match(r'^```(?:json)?\s*\n?(.*?)\n?```$', cleaned, re.DOTALL)
    if fence:
        cleaned = fence.group(1).strip()
    return json.loads(cleaned)


def grade_exact(output: str, grader: dict[str, Any]) -> GraderResult:
    expected = grader['expected']
    case_sensitive = grader.get('case_sensitive', True)
    trim = grader.get('trim_outer_whitespace', True)
    candidate = output if not trim else normalize_text(output, trim=True)
    expected_text = str(expected)
    alternates = [str(a) for a in grader.get('accepted_alternates', [])]

    def compare(candidate_value: str, expected_value: str) -> bool:
        if not case_sensitive:
            return candidate_value.lower() == expected_value.lower()
        return candidate_value == expected_value

    if grader.get('numeric_alias_match'):
        produced_number = _coerce_number(output)
        if produced_number is not None:
            if _coerce_number(expected_text) == produced_number:
                return GraderResult('exact', True, f'numeric alias match to {expected_text}')
            for alternate in alternates:
                if _coerce_number(alternate) == produced_number:
                    return GraderResult('exact', True, f'numeric alias match: {alternate}')
        if compare(candidate, normalize_text(expected_text, trim=trim)):
            return GraderResult('exact', True, 'exact match')
        for alternate in alternates:
            if compare(candidate, normalize_text(alternate, trim=trim)):
                return GraderResult('exact', True, f'alternate form match: {alternate}')
        return GraderResult('exact', False, f'expected {expected_text!r}, got {candidate!r}')

    if compare(candidate, normalize_text(expected_text, trim=trim)):
        return GraderResult('exact', True, 'exact match')
    for alternate in alternates:
        if compare(candidate, normalize_text(alternate, trim=trim)):
            return GraderResult('exact', True, f'alternate form match: {alternate}')
    return GraderResult('exact', False, f'expected {expected_text!r}, got {candidate!r}')


_BARE_NUMBER_RE = re.compile(r'[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?')


def grade_numeric(output: str, grader: dict[str, Any]) -> GraderResult:
    expected = float(grader['expected'])
    abs_tol = float(grader.get('absolute_tolerance', 0))
    rel_tol = float(grader.get('relative_tolerance', 0))
    cleaned = normalize_text(output)
    cleaned = re.sub(r'[.!?]+$', '', cleaned).strip()
    if not _BARE_NUMBER_RE.fullmatch(cleaned):
        return GraderResult('numeric', False, f'output is not a bare number (prompts require only the number): {output!r}')
    produced = float(cleaned)
    allowed = abs_tol or (rel_tol * abs(expected))
    delta = abs(produced - expected)
    if delta <= max(abs_tol, rel_tol * abs(expected)):
        return GraderResult('numeric', True, f'{produced} within tolerance {allowed}')
    return GraderResult('numeric', False, f'expected {expected} (tol {allowed}), got {produced}')


_TIME_CORE_RE = re.compile(r'^\s*(\d{1,2}):(\d{2})(?::(\d{2}))?\s*(am|pm|a\.m\.|p\.m\.)?\s*$', re.IGNORECASE)


def _parse_time(text: str, parsing: list[str] | None = None) -> tuple[int, int, int] | None:
    """Parse time honoring the spec's declared accepted forms.

    'HH:MM'       -> exactly two hour digits, no meridiem
    'H:MM AM'     -> one-or-two hour digits, upper/lowercase am/pm allowed
    'H:MM a.m.'   -> dotted meridiem allowed
    Unlisted forms are rejected outright; no fallback guessing.
    """
    forms = parsing or ['HH:MM']
    match = _TIME_CORE_RE.match(normalize_text(text))
    if not match:
        return None
    hour_text = match.group(1)
    meridiem = match.group(4)
    allow_single_digit = False
    allow_meridiem = False
    for form in forms:
        compact = form.upper().replace(' ', '')
        if compact == 'HH:MM':
            continue
        if compact in ('H:MMAM', 'H:MMPM'):
            allow_single_digit = True
            allow_meridiem = True
        elif compact in ('H:MMA.M.', 'H:MMP.M.'):
            allow_single_digit = True
            allow_meridiem = True
    if not allow_single_digit and len(hour_text) != 2:
        return None
    if meridiem and not allow_meridiem:
        return None
    hour, minute, second = int(hour_text), int(match.group(2)), int(match.group(3) or 0)
    if meridiem:
        normalized = meridiem.replace('.', '').lower()
        if normalized == 'pm' and hour != 12:
            hour += 12
        elif normalized == 'am' and hour == 12:
            hour = 0
    return hour, minute, second


def grade_time(output: str, grader: dict[str, Any]) -> GraderResult:
    parsing = grader.get('parsing') or ['HH:MM']
    expected_parts = _parse_time(str(grader['expected']), parsing)
    if expected_parts is None:
        return GraderResult('time', False, f'bad expected time in spec: {grader["expected"]!r}')
    produced = _parse_time(output, parsing)
    if produced is None:
        return GraderResult('time', False, f'could not parse time from {output!r} using accepted forms {parsing}')
    precision = grader.get('precision', 'exact')
    if precision == 'minute':
        produced = (produced[0], produced[1], 0)
        expected_cmp = expected_parts[:2] + (0,)
    if produced == expected_cmp:
        return GraderResult('time', True, f'{produced[0]:02d}:{produced[1]:02d} matches at {precision} precision')
    return GraderResult('time', False, f'expected {expected_parts}, got {produced} (precision={precision}, no mathematical rounding)')


def _check_structured_value(expected: Any, actual: Any, path: str, violations: list[str]) -> None:
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            violations.append(f'{path}: expected object, got {type(actual).__name__}')
            return
        for key, sub_expected in expected.items():
            if key not in actual:
                violations.append(f'{path}.{key}: missing required field')
            else:
                _check_structured_value(sub_expected, actual[key], f'{path}.{key}', violations)
        return
    if isinstance(expected, list):
        if not isinstance(actual, list):
            violations.append(f'{path}: expected array, got {type(actual).__name__}')
            return
        if len(expected) != len(actual):
            violations.append(f'{path}: expected {len(expected)} items, got {len(actual)}')
            return
        remaining = list(actual)
        for index, sub_expected in enumerate(expected):
            matched = False
            for position, candidate in enumerate(remaining):
                trial: list[str] = []
                if _value_matches(sub_expected, candidate, trial):
                    remaining.pop(position)
                    matched = True
                    break
            if not matched:
                violations.append(f'{path}[{index}]: no unmatched element matches {sub_expected!r}')
        return
    final_trial: list[str] = []
    if not _value_matches(expected, actual, final_trial):
        violations.append(f'{path}: ' + '; '.join(final_trial))


def _value_matches(expected: Any, actual: Any, violations: list[str]) -> bool:
    if hasattr(expected, 'isoformat'):
        expected = expected.isoformat()
    if isinstance(expected, bool) or isinstance(actual, bool):
        if isinstance(expected, bool) and isinstance(actual, bool):
            return expected == actual
        violations.append(f'expected {expected!r} (type {type(expected).__name__}), got {actual!r}')
        return False
    if isinstance(expected, (int, float)):
        produced = _coerce_number(actual)
        if produced is None or abs(produced - float(expected)) > 1e-9:
            violations.append(f'expected {expected!r}, got {actual!r}')
            return False
        return True
    if isinstance(expected, str):
        if not isinstance(actual, str):
            violations.append(f'expected string, got {type(actual).__name__}')
            return False
        if normalize_text(expected) != normalize_text(actual):
            violations.append(f'expected {expected!r}, got {actual!r}')
            return False
        return True
    if expected == actual:
        return True
    violations.append(f'expected {expected!r}, got {actual!r}')
    return False


def grade_structured(output: str, grader: dict[str, Any]) -> GraderResult:
    try:
        parsed = _extract_json(output)
    except json.JSONDecodeError as error:
        return GraderResult('structured', False, f'invalid JSON: {error.msg} at line {error.lineno} column {error.colno}')
    if not isinstance(parsed, dict):
        return GraderResult('structured', False, f'expected JSON object, got {type(parsed).__name__}')
    violations: list[str] = []
    required_fields = cast(list[str], grader['required_fields'])
    for field_name in required_fields:
        if field_name not in parsed:
            violations.append(f'{field_name}: missing required field')
    if not grader.get('allow_extra_fields', True):
        expected_fields = set(required_fields)
        for open_value in grader.get('open_values', []):
            expected_fields.add(cast(str, open_value['field']))
        extras = set(parsed) - expected_fields
        if extras:
            violations.append(f'extra fields not allowed: {sorted(extras)}')
    expected_output = grader.get('expected_output') or {}
    _check_structured_value(expected_output, parsed, '$', violations)
    numeric_fields = grader.get('numeric_fields', {})
    for field_name, config in numeric_fields.items():
        if field_name in parsed:
            produced = _coerce_number(parsed[field_name])
            expected_number = _coerce_number(str(config.get('expected', parsed[field_name])))
            abs_tol = float(config.get('absolute_tolerance', 0))
            if produced is None or expected_number is None or abs(produced - expected_number) > abs_tol:
                violations.append(f'{field_name}: numeric mismatch or out of tolerance')
    open_values = grader.get('open_values', [])
    for open_value in open_values:
        field_name = cast(str, open_value['field'])
        value = parsed.get(field_name)
        if value is None:
            continue
        if not isinstance(value, str):
            violations.append(f'{field_name}: expected string value')
            continue
        count = len(words(value))
        word_range = open_value.get('word_range')
        if word_range and not (word_range['min'] <= count <= word_range['max']):
            violations.append(f'{field_name}: {count} words outside {word_range["min"]}-{word_range["max"]}')
        for term in open_value.get('required_terms', []):
            if term.lower() not in normalize_text(value).lower():
                violations.append(f'{field_name}: missing required term {term!r}')
    return GraderResult('structured', not violations, 'all field checks passed' if not violations else 'field violations', violations)


def grade_tool_call(output: str, grader: dict[str, Any]) -> GraderResult:
    try:
        parsed = _extract_json(output)
    except json.JSONDecodeError as error:
        return GraderResult('tool_call', False, f'invalid JSON: {error.msg}')
    if not isinstance(parsed, dict):
        return GraderResult('tool_call', False, f'expected JSON object, got {type(parsed).__name__}')
    violations: list[str] = []
    expected = grader['expected_output']
    if parsed.get('function') != expected['function']:
        violations.append(f'function: expected {expected["function"]!r}, got {parsed.get("function")!r}')
    for forbidden in grader.get('forbidden_functions', []):
        if parsed.get('function') == forbidden:
            violations.append(f'forbidden function selected: {forbidden}')
    arguments = parsed.get('arguments')
    if not isinstance(arguments, dict):
        violations.append('arguments: expected object')
    else:
        expected_args = expected['arguments']
        for key, value in expected_args.items():
            if key not in arguments:
                violations.append(f'arguments.{key}: missing')
            else:
                trial: list[str] = []
                if not _value_matches(value, arguments[key], trial):
                    violations.append(f'arguments.{key}: {"; ".join(trial)}')
        allowed_args = set(grader.get('required_arguments', {}).get(expected['function'], expected_args.keys()))
        extras = set(arguments) - set(allowed_args)
        if extras:
            violations.append(f'arguments: unexpected keys {sorted(extras)}')
    return GraderResult('tool_call', not violations, 'tool call matches' if not violations else 'violations', violations)


def grade_constraints(output: str, grader: dict[str, Any]) -> GraderResult:
    text = normalize_text(output)
    violations: list[str] = []
    lower = text.lower()
    for constraint in grader['constraints']:
        kind = constraint['kind']
        value = constraint.get('value')
        if kind == 'word_count':
            count = len(words(text))
            if count != value:
                violations.append(f'word_count: expected {value}, got {count}')
        elif kind == 'word_range':
            count = len(words(text))
            if not (constraint['min'] <= count <= constraint['max']):
                violations.append(f'word_range: {count} outside {constraint["min"]}-{constraint["max"]}')
        elif kind == 'word_limit':
            count = len(words(text))
            if count > constraint['max']:
                violations.append(f'word_limit: {count} exceeds {constraint["max"]}')
        elif kind == 'first_word':
            actual_words = words(text)
            if not actual_words or actual_words[0].rstrip('.,;:') != constraint['word']:
                violations.append(f'first_word: expected {constraint["word"]!r}, got {actual_words[0] if actual_words else None!r}')
        elif kind == 'last_word':
            actual_words = words(text)
            if not actual_words or actual_words[-1].rstrip('.,;:') != constraint['word']:
                violations.append(f'last_word: expected {constraint["word"]!r}, got {actual_words[-1] if actual_words else None!r}')
        elif kind == 'term_occurrence':
            term = constraint['term']
            occurrence = constraint.get('occurrence', 'exactly')
            expected_count = constraint['count']
            if occurrence == 'across_values':
                # Occurrence counted across JSON string values (e.g. Q016:
                # exactly one value may contain the word). Unparseable
                # output cannot satisfy a values-scoped assertion.
                try:
                    parsed = _extract_json(text)
                except ValueError:
                    parsed = None
                if not isinstance(parsed, dict):
                    violations.append(
                        f'term_occurrence({term}): across_values requires a JSON object'
                    )
                else:
                    hits = sum(
                        1
                        for value in parsed.values()
                        if isinstance(value, str)
                        and re.search(re.escape(term), value, re.IGNORECASE)
                    )
                    if hits != expected_count:
                        violations.append(
                            f'term_occurrence({term}): expected {expected_count} values, got {hits}'
                        )
            else:
                occurrences = len(re.findall(re.escape(term), text, re.IGNORECASE))
                if occurrence == 'exactly' and occurrences != expected_count:
                    violations.append(f'term_occurrence({term}): expected {expected_count}, got {occurrences}')
        elif kind == 'forbidden_terms':
            for term in constraint['terms']:
                if re.search(re.escape(term), text, re.IGNORECASE):
                    violations.append(f'forbidden_terms: found {term!r}')
        elif kind == 'allowed_punctuation':
            # Only line-initial markers (e.g. Q015 "A.".."F.") may carry
            # punctuation: strip ^[A-F]. markers, then any surviving
            # punctuation character is a violation.
            stripped = re.sub(r'(?m)^[A-F]\.', '', text)
            bad = sorted({c for c in stripped if not c.isalnum() and not c.isspace()})
            if bad:
                violations.append(f'allowed_punctuation: forbidden {bad}')
        elif kind == 'forbidden_chars':
            for char in constraint['chars']:
                if char in text:
                    violations.append(f'forbidden_chars: found {char!r}')
        elif kind == 'required_terms':
            for term in constraint['terms']:
                if term.lower() not in lower:
                    violations.append(f'required_terms: missing {term!r}')
        elif kind == 'first_prefix':
            first_line = text.split('\n', 1)[0].strip()
            if not first_line.startswith(constraint['prefix']):
                violations.append(f'first_prefix: expected {constraint["prefix"]!r}')
        elif kind == 'parenthetical_count':
            opens = text.count('(')
            closes = text.count(')')
            if opens != value or closes != value or opens != closes:
                violations.append(f'parenthetical_count: expected {value} balanced pairs, got {opens} open / {closes} close')
        elif kind == 'bullet_count':
            lines = [line for line in text.split('\n') if line.strip()]
            bullets = [line for line in lines if re.match(r'^\s*[-*•]\s+', line)]
            if len(bullets) != value:
                violations.append(f'bullet_count: expected {value}, got {len(bullets)}')
        elif kind == 'numbered_list_count':
            items = re.findall(r'^\s*\d+[.)]\s+', text, re.MULTILINE)
            if len(items) != value:
                violations.append(f'numbered_list_count: expected {value}, got {len(items)}')
        elif kind == 'item_prefix':
            items = re.findall(r'^\s*\d+\.\s+(.*)$', text, re.MULTILINE)
            index = constraint['index']
            if len(items) < index or not items[index - 1].startswith(constraint['prefix']):
                violations.append(f'item_prefix: item {index} must start with {constraint["prefix"]!r}')
        elif kind == 'item_contains':
            items = re.findall(r'^\s*\d+\.\s+(.*)$', text, re.MULTILINE)
            index = constraint['index']
            term = constraint['term']
            if len(items) < index or term not in items[index - 1]:
                violations.append(f'item_contains: item {index} missing {term!r}')
        elif kind == 'item_excludes_chars':
            items = re.findall(r'^\s*\d+\.\s+(.*)$', text, re.MULTILINE)
            index = constraint['index']
            if len(items) < index or any(char in items[index - 1] for char in constraint['chars']):
                violations.append(f'item_excludes_chars: item {index} contains a forbidden character')
        elif kind == 'item_sentence_count':
            items = re.findall(r'^\s*\d+\.\s+(.*)$', text, re.MULTILINE)
            required = int(constraint['value'])
            if 'index' not in constraint:
                # No index (e.g. Q017 "each step must be one sentence"):
                # every numbered item must hold exactly `value` sentences.
                if not items:
                    violations.append('item_sentence_count: no numbered items found')
                for position, item in enumerate(items, 1):
                    endings = len(re.findall(r'[.!?]+(?:\s|$)', item))
                    if endings != required:
                        violations.append(
                            f'item_sentence_count: item {position} has {endings} sentences, expected {required}'
                        )
            else:
                index = constraint['index']
                if len(items) < index:
                    violations.append(f'item_sentence_count: item {index} missing')
                else:
                    sentence_endings = len(re.findall(r'[.!?]+(?:\s|$)', items[index - 1]))
                    if sentence_endings != required:
                        violations.append(f'item_sentence_count: item {index} must be one sentence')
        elif kind == 'sentence_count':
            sentences = [s for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
            if len(sentences) != value:
                violations.append(f'sentence_count: expected {value}, got {len(sentences)}')
        elif kind == 'sentence_contains':
            sentences = [s for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
            index = constraint.get('index')
            terms = list(constraint.get('terms') or [])
            if 'term' in constraint:
                terms.append(constraint['term'])
            if constraint.get('all_sentences'):
                term = terms[0] if terms else ''
                missing = [i + 1 for i, s in enumerate(sentences) if term.lower() not in s.lower()]
                if missing:
                    violations.append(f'sentence_contains: sentences {missing} missing {term!r}')
            elif index:
                if len(sentences) < index:
                    violations.append(f'sentence_contains: sentence {index} missing')
                else:
                    for term in terms:
                        if term.lower() not in sentences[index - 1].lower():
                            violations.append(f'sentence_contains: sentence {index} missing {term!r}')
        elif kind == 'sentence_numeric_tokens':
            sentences = [s for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
            index = constraint['index']
            if len(sentences) < index:
                violations.append(f'sentence_numeric_tokens: sentence {index} missing')
            else:
                numbers = re.findall(r'\d+(?:\.\d+)?', sentences[index - 1])
                if not (constraint['min'] <= len(numbers) <= constraint['max']):
                    violations.append(f'sentence_numeric_tokens: sentence {index} has {len(numbers)} numeric tokens, allowed {constraint["min"]}-{constraint["max"]}')
        elif kind == 'sentence_is_question':
            sentences = [s for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
            index = constraint['index']
            if len(sentences) < index or not sentences[index - 1].rstrip().endswith('?'):
                violations.append(f'sentence_is_question: sentence {index} must end with ?')
        elif kind == 'markdown_table_shape':
            rows = [line for line in text.split('\n') if line.strip().startswith('|')]
            if len(rows) < 3:
                violations.append('markdown_table_shape: expected header + separator + data rows')
            else:
                header = [cell.strip() for cell in rows[0].strip('|').split('|')]
                if header != constraint['columns']:
                    violations.append(f'markdown_table_shape: header {header} != {constraint["columns"]}')
                data_rows = [row for row in rows[2:] if not re.match(r'^\s*\|?[\s:|-]+\|?\s*$', row)]
                if len(data_rows) != constraint['data_rows']:
                    violations.append(f'markdown_table_shape: expected {constraint["data_rows"]} data rows, got {len(data_rows)}')
        elif kind == 'trailing_literal':
            if not text.rstrip().endswith(constraint['literal']):
                violations.append(f'trailing_literal: must end with {constraint["literal"]!r}')
        elif kind == 'line_count':
            lines = [line for line in text.split('\n') if line.strip()]
            if len(lines) != value:
                violations.append(f'line_count: expected {value}, got {len(lines)}')
        elif kind == 'line_prefix_sequence':
            lines = [line for line in text.split('\n') if line.strip()]
            for index, prefix in enumerate(constraint['prefixes']):
                if index >= len(lines) or not lines[index].lstrip().startswith(prefix):
                    violations.append(f'line_prefix_sequence: line {index + 1} must start with {prefix!r}')
        elif kind == 'line_word_limit':
            lines = [line for line in text.split('\n') if line.strip()]
            for index, line in enumerate(lines, 1):
                count = len(words(line))
                if count > constraint['max_words']:
                    violations.append(f'line_word_limit: line {index} has {count} words > {constraint["max_words"]}')
        elif kind == 'line_contains':
            lines = [line for line in text.split('\n') if line.strip()]
            for label in constraint['lines']:
                line_index = ord(label) - ord('A')
                if line_index >= len(lines) or constraint['term'].lower() not in lines[line_index].lower():
                    violations.append(f'line_contains: line {label} missing {constraint["term"]!r}')
        elif kind == 'bullet_prefix':
            lines = [line for line in text.split('\n') if line.strip()]
            bullets = [line for line in lines if re.match(r'^\s*[-*•]\s+', line)]
            index = constraint['index']
            # index -1 addresses the last bullet (Q011); positive is 1-based.
            if not bullets:
                violations.append('bullet_prefix: no bullets found')
                continue
            target = bullets[-1] if index == -1 else (bullets[index - 1] if 0 < index <= len(bullets) else None)
            if target is None or not target.lstrip('-*• ').startswith(constraint['prefix']):
                which = 'last bullet' if index == -1 else f'bullet {index}'
                violations.append(f'bullet_prefix: {which} must start with {constraint["prefix"]!r}')
        elif kind == 'bullet_word_limit':
            lines = [line for line in text.split('\n') if line.strip()]
            bullets = [line for line in lines if re.match(r'^\s*[-*•]\s+', line)]
            for index, bullet in enumerate(bullets, 1):
                count = len(words(bullet.lstrip('-*• ')))
                if count >= constraint['max_words'] + 1:
                    violations.append(f'bullet_word_limit: bullet {index} has {count} words, max {constraint["max_words"]}')
        elif kind == 'ends_with':
            if not text.rstrip().endswith(constraint['suffix']):
                violations.append(f'ends_with: must end with {constraint["suffix"]!r}')
        elif kind == 'required_prefix':
            if not text.startswith(constraint['prefix']):
                violations.append(f'required_prefix: must start with {constraint["prefix"]!r}')
        else:
            violations.append(f'unknown constraint kind: {kind}')
    return GraderResult('constraints', not violations, 'all constraints satisfied' if not violations else 'violations', violations)


def grade_fact_check(output: str, grader: dict[str, Any]) -> GraderResult:
    text = normalize_text(output)
    lower = text.lower()
    violations: list[str] = []
    for fact in grader.get('required_facts', []):
        if fact.lower() not in lower:
            violations.append(f'missing fact: {fact!r}')
    for fact in grader.get('forbidden_facts', []):
        if fact.lower() in lower:
            violations.append(f'forbidden fact present: {fact!r}')
    if 'word_limit' in grader:
        count = len(words(text))
        if count > grader['word_limit']['max']:
            violations.append(f'word_limit: {count} > {grader["word_limit"]["max"]}')
    if 'word_range' in grader:
        count = len(words(text))
        if not (grader['word_range']['min'] <= count <= grader['word_range']['max']):
            violations.append(f'word_range: {count} outside {grader["word_range"]["min"]}-{grader["word_range"]["max"]}')
    if 'sentence_count' in grader:
        sentences = [s for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
        if len(sentences) != grader['sentence_count']['value']:
            violations.append(f'sentence_count: expected {grader["sentence_count"]["value"]}, got {len(sentences)}')
    if 'bullet_count' in grader:
        bullets = [line for line in text.split('\n') if re.match(r'^\s*[-*•]\s+', line)]
        if len(bullets) != grader['bullet_count']['value']:
            violations.append(f'bullet_count: expected {grader["bullet_count"]["value"]}, got {len(bullets)}')
    if 'bullet_word_limit' in grader:
        bullets = [line for line in text.split('\n') if re.match(r'^\s*[-*•]\s+', line)]
        for index, bullet in enumerate(bullets, 1):
            count = len(words(bullet.lstrip('-*• ')))
            if count > grader['bullet_word_limit']['max']:
                violations.append(f'bullet_word_limit: bullet {index} has {count} words > {grader["bullet_word_limit"]["max"]}')
    return GraderResult('fact_check', not violations, 'facts preserved' if not violations else 'violations', violations)


def grade_refusal(output: str, grader: dict[str, Any]) -> GraderResult:
    refusal_expected = grader['refusal_expected']
    normalized = normalize_text(output).lower()
    normalized = re.sub(r'[.!?]+$', '', normalized)
    refusal_markers = ('cannot determine', 'not established', 'unknown', 'insufficient information', 'cannot answer', 'unable to answer')
    is_refusal = any(re.fullmatch(re.escape(marker) + r'[\s_]*', normalized) or normalized == re.sub(r'[ ]', '_', marker) for marker in refusal_markers)
    is_refusal = is_refusal or normalized in ('unknown', 'insufficient_information', 'cannot_determine', 'not_established')
    if refusal_expected == is_refusal:
        return GraderResult('refusal', True, f'refusal_expected={refusal_expected}, output_is_refusal={is_refusal}')
    return GraderResult('refusal', False, f'refusal_expected={refusal_expected}, output_is_refusal={is_refusal}')


def grade_rubric_judge(output: str, grader: dict[str, Any]) -> GraderResult:
    criteria = grader.get('criteria')
    if not criteria:
        return GraderResult('rubric_judge', False, 'rubric judge requires criteria')
    return GraderResult('rubric_judge', True, f'deferrable to judge with {len(criteria)} criteria; deterministic criteria must not be included here')


_GRADER_FUNCS = {
    'exact': grade_exact,
    'numeric': grade_numeric,
    'time': grade_time,
    'structured': grade_structured,
    'tool_call': grade_tool_call,
    'constraints': grade_constraints,
    'fact_check': grade_fact_check,
    'refusal': grade_refusal,
    'rubric_judge': grade_rubric_judge,
}


def grade_output(output: str, graders: list[dict[str, Any]]) -> list[GraderResult]:
    return [_GRADER_FUNCS[cast(str, grader['type'])](output, grader) for grader in graders]
