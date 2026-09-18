"""Failure-mode taxonomy v1 + row-first analysis (post-hoc, diagnostic-only).

failure-modes-v1 is FROZEN: categories may not be added, merged, or
redefined based on later model behavior without a versioned amendment.
Q5 and all subsequent models are analyzed under these exact rules.

Categories: OUTPUT_CONTRACT / LEXICAL_CONSTRAINT / CONTENT_ERROR /
STRUCTURE_ERROR / MIXED / UNKNOWN.

- MIXED = one failed generation contains >=2 materially independent
  failure-mode categories. Cross-trial variation is reported as a
  distribution (multi_mode_across_trials), never auto-MIXED.
- UNKNOWN = no rule fires. Never forced into another category.
- Substantive assessment is human-authored (review section) and validated,
  never auto-filled. The analyzer populates auto_analysis only.
- Benchmark verdicts are re-asserted from persisted rows, never recomputed.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass, field
from typing import Any, Mapping

TAXONOMY_VERSION = 'failure-modes-v1'

OUTPUT_CONTRACT = 'OUTPUT_CONTRACT'
LEXICAL_CONSTRAINT = 'LEXICAL_CONSTRAINT'
CONTENT_ERROR = 'CONTENT_ERROR'
STRUCTURE_ERROR = 'STRUCTURE_ERROR'
MIXED = 'MIXED'
UNKNOWN = 'UNKNOWN'

CATEGORIES = (
    OUTPUT_CONTRACT, LEXICAL_CONSTRAINT, CONTENT_ERROR,
    STRUCTURE_ERROR, MIXED, UNKNOWN,
)

SUBSTANTIVE_LABELS = frozenset(
    {'CORRECT', 'LIKELY_CORRECT', 'INCORRECT', 'AMBIGUOUS', 'NOT_REVIEWED'}
)

# Committed signature -> mode rule table. Signature extraction is
# deterministic over persisted grader details; mapping is total over the
# rule table with UNKNOWN as the explicit fallback.
_GRADER_MODE_RULES: tuple[tuple[str, str, str], ...] = (
    # (grader_type, signature-substring, mode)
    ('numeric', 'not a bare number', OUTPUT_CONTRACT),
    ('time', 'could not parse', OUTPUT_CONTRACT),
    ('time', 'mismatch', OUTPUT_CONTRACT),
    ('exact', '', OUTPUT_CONTRACT),
    ('constraints', 'word_count', LEXICAL_CONSTRAINT),
    ('constraints', 'word_range', LEXICAL_CONSTRAINT),
    ('constraints', 'word_limit', LEXICAL_CONSTRAINT),
    ('constraints', 'term_occurrence', LEXICAL_CONSTRAINT),
    ('constraints', 'forbidden_terms', LEXICAL_CONSTRAINT),
    ('constraints', 'forbidden_chars', LEXICAL_CONSTRAINT),
    ('constraints', 'required_terms', LEXICAL_CONSTRAINT),
    ('constraints', 'sentence_contains', LEXICAL_CONSTRAINT),
    ('constraints', 'sentence_numeric_tokens', LEXICAL_CONSTRAINT),
    ('constraints', 'bullet_prefix', LEXICAL_CONSTRAINT),
    ('constraints', 'item_contains', LEXICAL_CONSTRAINT),
    ('constraints', 'item_excludes_chars', LEXICAL_CONSTRAINT),
    ('constraints', 'line_contains', LEXICAL_CONSTRAINT),
    ('constraints', 'sentence_count', LEXICAL_CONSTRAINT),
    ('constraints', 'allowed_punctuation', LEXICAL_CONSTRAINT),
    ('structured', 'expected', CONTENT_ERROR),
    ('structured', 'missing field', STRUCTURE_ERROR),
    ('structured', 'extra field', STRUCTURE_ERROR),
    ('structured', 'not valid JSON', STRUCTURE_ERROR),
    ('structured', 'schema', STRUCTURE_ERROR),
    ('fact_check', '', CONTENT_ERROR),
    ('tool_call', '', STRUCTURE_ERROR),
    ('constraints', 'item_sentence_count', LEXICAL_CONSTRAINT),
    ('constraints', 'sentence_is_question', LEXICAL_CONSTRAINT),
    ('constraints', 'first_word', LEXICAL_CONSTRAINT),
    ('constraints', 'last_word', LEXICAL_CONSTRAINT),
    ('constraints', 'first_prefix', LEXICAL_CONSTRAINT),
    ('constraints', 'required_prefix', LEXICAL_CONSTRAINT),
    ('constraints', 'ends_with', LEXICAL_CONSTRAINT),
    ('constraints', 'trailing_literal', LEXICAL_CONSTRAINT),
    ('constraints', 'line_count', LEXICAL_CONSTRAINT),
    ('constraints', 'line_prefix_sequence', LEXICAL_CONSTRAINT),
    ('constraints', 'line_word_limit', LEXICAL_CONSTRAINT),
    ('constraints', 'bullet_count', LEXICAL_CONSTRAINT),
    ('constraints', 'bullet_word_limit', LEXICAL_CONSTRAINT),
    ('constraints', 'numbered_list_count', LEXICAL_CONSTRAINT),
    ('constraints', 'item_prefix', LEXICAL_CONSTRAINT),
    ('constraints', 'parenthetical_count', LEXICAL_CONSTRAINT),
    ('constraints', 'markdown_table_shape', STRUCTURE_ERROR),
)


def taxonomy_hash() -> str:
    """sha256 over the frozen rule table (provenance)."""
    canonical = json.dumps(
        [[g, s, m] for g, s, m in _GRADER_MODE_RULES], separators=(',', ':')
    )
    return hashlib.sha256(canonical.encode('utf-8')).hexdigest()


def signature_for_detail(detail: Mapping[str, Any]) -> str:
    """Compact stable signature for one failed grader result."""
    grader_type = str(detail.get('grader_type', 'unknown'))
    violations = detail.get('violations') or []
    if violations:
        head = str(violations[0])
        # Normalize volatile values: keep the assertion shape, drop specifics.
        for prefix in (
            'term_occurrence(', 'sentence_contains:', 'sentence_numeric_tokens:',
            'word_count:', 'word_range:', 'word_limit:', 'forbidden_terms:',
            'required_terms:', 'item_contains:', 'line_contains:',
            'bullet_prefix:', 'line_prefix_sequence:', 'first_word:',
            'sentence_count:',
        ):
            if head.startswith(prefix):
                return f'{grader_type}:{prefix.rstrip(":")}'
        return f'{grader_type}:{head[:80]}'
    return f"{grader_type}:{str(detail.get('detail', 'fail'))[:80]}"


def mode_for_signature(grader_type: str, signature: str) -> str:
    """Map one signature to a mode via the frozen rule table."""
    body = signature.split(':', 1)[1] if ':' in signature else signature
    for rule_grader, fragment, mode in _GRADER_MODE_RULES:
        if rule_grader != grader_type:
            continue
        if not fragment or fragment in signature or fragment in body:
            return mode
    return UNKNOWN


def modes_for_failed_row(
    grader_details: list[dict[str, Any]], *, raw_output: str = ''
) -> tuple[list[str], list[str]]:
    """Return (signatures, modes) for the failed grader results in one row.

    Refinement: an exact-grader failure on a bare single-token output is a
    clean wrong answer (CONTENT_ERROR, e.g. Q036 NO vs UNKNOWN), while
    exact failures with appended prose are OUTPUT_CONTRACT. The rule table
    alone cannot see the output shape, hence this explicit split.
    """
    from evals.graders.engine import normalize_text

    signatures: list[str] = []
    modes: list[str] = []
    bare_single_token = bool(raw_output) and not any(
        ch.isspace() for ch in normalize_text(raw_output)
    )
    for detail in grader_details:
        if not isinstance(detail, dict) or detail.get('passed', True):
            continue
        signature = signature_for_detail(detail)
        signatures.append(signature)
        grader_type = str(detail.get('grader_type', ''))
        if grader_type == 'exact' and bare_single_token:
            mode = CONTENT_ERROR
        else:
            mode = mode_for_signature(grader_type, signature)
        if mode not in modes:
            modes.append(mode)
    return signatures, modes


def row_failure_mode(modes: list[str]) -> str:
    """One row's mode: single category, MIXED if >=2 independent, else UNKNOWN."""
    distinct = [m for m in modes if m != UNKNOWN]
    if not distinct:
        return UNKNOWN
    if len(distinct) == 1:
        return distinct[0]
    return MIXED


@dataclass
class FailedRowFeatures:
    task_id: str
    trial: int
    task_status: str
    benchmark_verdict: str
    signatures: list[str] = field(default_factory=list)
    failure_mode: str = UNKNOWN
    excerpt: str = ''


def extract_failed_rows(
    conn: sqlite3.Connection,
    execution_id: str,
    statuses: Mapping[str, str],
    *,
    excerpt_chars: int = 300,
) -> list[FailedRowFeatures]:
    """Objective pre-fill from persisted rows. Verdicts re-asserted, never
    recomputed. Unknown task IDs raise (same invariant as capability)."""
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            'SELECT task_id, trial, grader_verdict, grader_details_json,'
            ' raw_output FROM runs WHERE experiment_id=? AND is_warmup=0'
            " AND grader_verdict='FAIL' ORDER BY task_id, trial",
            (execution_id,),
        ).fetchall()
    finally:
        conn.row_factory = None
    features: list[FailedRowFeatures] = []
    for row in rows:
        task_id = str(row['task_id'])
        if task_id not in statuses:
            raise ValueError(
                f'task {task_id} has no frozen grading_status; refusing to infer'
            )
        try:
            details = json.loads(row['grader_details_json'] or '[]')
        except ValueError:
            details = []
        if not isinstance(details, list):
            details = []
        signatures, modes = modes_for_failed_row(
            [d for d in details if isinstance(d, dict)],
            raw_output=str(row['raw_output'] or ''),
        )
        raw = str(row['raw_output'] or '')
        features.append(
            FailedRowFeatures(
                task_id=task_id,
                trial=int(row['trial']),
                task_status=str(statuses[task_id]),
                benchmark_verdict=str(row['grader_verdict']),
                signatures=signatures,
                failure_mode=row_failure_mode(modes),
                excerpt=raw[:excerpt_chars],
            )
        )
    return features


def validate_review_entry(entry: Mapping[str, Any]) -> list[str]:
    """Validate one review entry's authorship boundary. Returns error list."""
    errors: list[str] = []
    task_id = entry.get('task_id', '?')
    auto = entry.get('auto_analysis')
    review = entry.get('review')
    if not isinstance(auto, dict):
        errors.append(f'{task_id}: auto_analysis section missing')
    if not isinstance(review, dict):
        errors.append(f'{task_id}: review section missing')
        return errors
    assessment = review.get('substantive_answer_assessment')
    if assessment not in SUBSTANTIVE_LABELS:
        errors.append(
            f'{task_id}: substantive assessment {assessment!r} not in vocabulary'
        )
    if not review.get('assessment_author'):
        errors.append(f'{task_id}: assessment_author missing (authorship required)')
    override = review.get('mode_override')
    if override is not None:
        if override not in CATEGORIES:
            errors.append(f'{task_id}: mode_override {override!r} unknown')
        if not review.get('override_reason'):
            errors.append(f'{task_id}: mode_override without override_reason')
    if 'failure_mode' in review and 'auto_analysis' in entry:
        errors.append(
            f'{task_id}: failure_mode belongs in auto_analysis, not review'
        )
    return errors


def validate_review_file(review: Mapping[str, Any]) -> list[str]:
    """Validate the full review file structure + approval gate value."""
    errors: list[str] = []
    if review.get('taxonomy_version') != TAXONOMY_VERSION:
        errors.append(
            f"taxonomy_version must be {TAXONOMY_VERSION!r}, "
            f"got {review.get('taxonomy_version')!r}"
        )
    if review.get('review_status') not in ('DRAFT', 'APPROVED'):
        errors.append('review_status must be DRAFT or APPROVED')
    entries = review.get('tasks')
    if not isinstance(entries, list) or not entries:
        errors.append('tasks must be a non-empty list')
        return errors
    for entry in entries:
        if isinstance(entry, dict):
            errors.extend(validate_review_entry(entry))
        else:
            errors.append('task entry must be a mapping')
    return errors


def review_generation_allowed(review: Mapping[str, Any]) -> bool:
    """Report generation requires an APPROVED, valid review file."""
    if review.get('review_status') != 'APPROVED':
        return False
    return not validate_review_file(review)
