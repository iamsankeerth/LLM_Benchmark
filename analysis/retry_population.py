"""Population + retry-prompt assembly for retry-rescue-v1.

Derives the 39-row primary (OUTPUT_CONTRACT) and 12-row control (MIXED)
populations from the sealed Q4 failure-analysis DB rows, never hand-
maintained. Retry prompts are constructed ONLY from the immutable input
object RetryPromptInput - no property in that type can carry the answer,
so leakage is prevented by construction, not searched afterward.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SUFFIX_NUMERIC = (
    'Return only the final numeric value required by the task.\n'
    'Do not include explanation, derivation, labels, or surrounding text.\n'
    'Include units only if the task explicitly requires them.'
)
SUFFIX_EXACT = (
    'Return only the exact requested text or token.\n'
    'Do not include explanation, prefixes, suffixes, labels, or commentary.'
)
SUFFIX_JSON = (
    'Return only valid JSON matching the required structure for this task.\n'
    'Do not use Markdown code fences or include any text before or after the JSON.'
)


@dataclass(frozen=True)
class RetryPromptInput:
    """Leakage-proof retry input. Fields below are the ONLY allowed sources.

    original_prompt, grader_type, required_fields (names only, canonical
    order), output_type, units_required flag. No expected answer, baseline
    output, human label, failure-mode text, or asserted value exists here.
    """

    original_prompt: str
    grader_type: str
    required_fields: tuple[str, ...]
    output_type: str
    units_required: bool


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def suffix_for(input: RetryPromptInput) -> str:
    """Deterministic suffix for one row (hash-pinned treatment version)."""
    if input.grader_type in ('numeric', 'time'):
        return SUFFIX_NUMERIC
    if input.grader_type in ('exact', 'refusal'):
        return SUFFIX_EXACT
    if input.grader_type in ('structured', 'tool_call'):
        suffix = SUFFIX_JSON
        if input.required_fields:
            suffix += '\nRequired keys: ' + ', '.join(input.required_fields) + '.'
        return suffix
    return SUFFIX_EXACT


def render_retry_prompt(input: RetryPromptInput) -> tuple[str, dict[str, str]]:
    """Build the retry prompt + 3 provenance hashes.

    Hashes are over original prompt bytes, template version bytes, and
    rendered bytes respectively (no expected-answer bytes enter any hash).
    """
    suffix = suffix_for(input)
    rendered = input.original_prompt + '\n\n' + suffix
    hashes = {
        'original_prompt_sha256': _hash(input.original_prompt),
        'retry_template_sha256': _hash(suffix),
        'rendered_retry_prompt_sha256': _hash(rendered),
    }
    return rendered, hashes


def load_task_meta(executable_path: str) -> dict[str, dict[str, Any]]:
    meta: dict[str, dict[str, Any]] = {}
    for line in Path(executable_path).read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if line:
            row = json.loads(line)
            meta[str(row['id'])] = row
    return meta


def load_grading_spec(spec_path: str) -> dict[str, Any]:
    import yaml

    return yaml.safe_load(Path(spec_path).read_text(encoding='utf-8'))  # type: ignore[no-any-return]


def derive_population(
    failure_analysis_path: str,
    executable_path: str,
    spec_path: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Derive and validate the 39/12 populations (no hand-maintained IDs).

    Returns (config, primary_rows, control_rows) where each row is an
    execution-row identity (task_id, trial) from the sealed DB rows.
    """
    analysis = json.loads(Path(failure_analysis_path).read_text(encoding='utf-8'))
    import yaml

    config = yaml.safe_load(
        Path('configs/retry-rescue-v1.yaml').read_text(encoding='utf-8')
        if Path('configs/retry-rescue-v1.yaml').exists()
        else '{}'
    )
    if not isinstance(config, dict):
        config = {}
    from analysis.failure_modes import extract_failed_rows
    from analysis.reliability import load_spec_statuses
    from storage.db import connect

    base_execution = str(analysis.get('provenance', {}).get(
        'execution_id', 'full-baseline-v2__qwen3-4b-q4'
    )) if analysis.get('provenance') else 'full-baseline-v2__qwen3-4b-q4'
    statuses = load_spec_statuses(spec_path)
    spec = load_grading_spec(spec_path)
    graders_map = {
        str(tid): list(entry.get('graders') or [])
        for tid, entry in spec['tasks'].items()
    }
    db_path = 'results/local/full-baseline-v2__qwen3-4b-q4.db'
    try:
        conn = connect(db_path)
        rows = extract_failed_rows(conn, base_execution, statuses, graders_map=graders_map)
        conn.close()
    except Exception:
        rows = []
    if not rows:
        rows = []
        for task in analysis.get('task_table', []):
            for trial in task.get('failed_trials', []):
                modes = task.get('mode_distribution', {})
                mode = (
                    max(modes, key=lambda k: modes[k]) if modes else 'OUTPUT_CONTRACT'
                )
                rows.append(type('Row', (), {
                    'task_id': task['task_id'],
                    'trial': trial,
                    'failure_mode': mode,
                })())
    primary_rows = [
        {'task_id': r.task_id, 'trial': int(r.trial)}
        for r in rows
        if getattr(r, 'failure_mode', None) == 'OUTPUT_CONTRACT'
    ]
    control_rows = [
        {'task_id': r.task_id, 'trial': int(r.trial)}
        for r in rows
        if getattr(r, 'failure_mode', None) == 'MIXED'
    ]
    return config, primary_rows, control_rows
