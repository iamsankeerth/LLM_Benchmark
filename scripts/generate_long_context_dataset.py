"""Generate the deterministic synthetic long-context-v1 dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ARCHETYPES = (
    'start_needle', 'middle_needle', 'end_needle', 'multi_needle',
    'state_recency', 'global_aggregate', 'grounded_abstention',
)
LENGTHS = (('S', 350), ('M', 700), ('L', 1400))


def _tokens(text: str) -> list[str]:
    return text.split()


def _make_words(
    *,
    target_words: int,
    records: list[tuple[float, list[str]]],
) -> list[str]:
    record_tokens = sum(len(tokens) for _, tokens in records)
    if record_tokens >= target_words:
        raise ValueError('record tokens exceed target length')
    words = [f'context_note_{index:05d}' for index in range(target_words - record_tokens)]
    for fraction, tokens in sorted(records, key=lambda item: item[0], reverse=True):
        index = min(target_words - len(tokens), int(target_words * fraction))
        words[index:index] = tokens
    if len(words) != target_words:
        raise AssertionError('document length invariant failed')
    return words


def _document(task_id: str, archetype: str, length_name: str, target_words: int) -> tuple[str, dict[str, Any]]:
    records: list[tuple[float, list[str]]] = []
    expected: Any
    grader: dict[str, Any]
    distractor_count = 0
    if archetype in {'start_needle', 'middle_needle', 'end_needle'}:
        fraction = {'start_needle': 0.08, 'middle_needle': 0.50, 'end_needle': 0.92}[archetype]
        records.append((fraction, _tokens('record ORION STATUS ACTIVE VALUE BLUE-17')))
        question = 'Return only the VALUE for ORION.'
        expected = 'BLUE-17'
        grader = {'type': 'exact', 'expected': expected}
        distractor_count = 12
        for index in range(distractor_count):
            records.append((min(0.95, max(0.02, fraction + (index - 6) * 0.055)), _tokens(f'record ORION-{index:02d} STATUS ACTIVE VALUE OTHER-{index:02d}')))
    elif archetype == 'multi_needle':
        records.extend([
            (0.18, _tokens('record alpha VALUE A7')),
            (0.50, _tokens('record beta VALUE B3')),
            (0.82, _tokens('record gamma VALUE C9')),
        ])
        question = 'Return only JSON with alpha, beta, and gamma values.'
        expected = {'alpha': 'A7', 'beta': 'B3', 'gamma': 'C9'}
        grader = {'type': 'structured', 'expected_output': expected, 'required_fields': ['alpha', 'beta', 'gamma'], 'allow_extra_fields': False}
        distractor_count = 15
        for index in range(distractor_count):
            fraction = 0.08 + (index % 5) * 0.18
            records.append((fraction, _tokens(f'record delta{index:02d} VALUE D{index:02d}')))
    elif archetype == 'state_recency':
        for index in range(8):
            status = 'SUPERSEDED' if index < 7 else 'APPROVED'
            fraction = 0.10 + index * 0.10
            records.append((fraction, _tokens(f'record ORION revision {index:02d} STATUS {status} VALUE STATE-{index:02d}')))
        question = 'Return only the VALUE from the latest APPROVED ORION record.'
        expected = 'STATE-07'
        grader = {'type': 'exact', 'expected': expected}
        distractor_count = 7
    elif archetype == 'global_aggregate':
        values = [3, 5, 8, 17, 12, 9, 14, 6]
        values += [4] * 24
        for index, value in enumerate(values):
            records.append((0.04 + index * 0.030, _tokens(f'record METRIC-{index:02d} VALUE {value}')))
        question = 'Return only the largest VALUE among all METRIC records.'
        expected = 17
        grader = {'type': 'numeric', 'expected': expected, 'absolute_tolerance': 0, 'relative_tolerance': 0}
        distractor_count = 31
    else:
        records.extend([
            (0.15, _tokens('record BATCH-07 STATUS COMPLETE VALUE VERIFIED')),
            (0.50, _tokens('record BATCH-08 STATUS COMPLETE VALUE VERIFIED')),
            (0.85, _tokens('record BATCH-09 STATUS COMPLETE VALUE VERIFIED')),
        ])
        question = 'Return only INSUFFICIENT_INFORMATION because the decisive approval condition is absent.'
        expected = 'INSUFFICIENT_INFORMATION'
        grader = {'type': 'refusal', 'refusal_expected': True}
        distractor_count = 18
        for index in range(distractor_count):
            records.append((0.05 + (index % 6) * 0.16, _tokens(f'record BATCH-{index:02d} STATUS COMPLETE VALUE VERIFIED')))
    words = _make_words(target_words=target_words, records=records)
    document = ' '.join(words)
    metadata = {
        'task_id': task_id,
        'archetype': archetype,
        'source_length': length_name,
        'target_words': target_words,
        'distractor_count': distractor_count,
        'question': question,
        'expected_output': expected,
        'grader': grader,
        'document_sha256': hashlib.sha256((document + '\n').encode('utf-8')).hexdigest(),
        'actual_words': len(words),
    }
    return document, metadata


def generate(root: Path, *, force: bool = False) -> dict[str, Any]:
    dataset_dir = root / 'evals/datasets/long-context-v1'
    documents_dir = dataset_dir / 'documents'
    dataset_dir.mkdir(parents=True, exist_ok=True)
    documents_dir.mkdir(parents=True, exist_ok=True)
    tasks_path = dataset_dir / 'tasks.jsonl'
    manifest_path = dataset_dir / 'dataset-manifest.json'
    if (tasks_path.exists() or manifest_path.exists()) and not force:
        raise FileExistsError('long-context-v1 dataset already exists; use --force')
    tasks: list[dict[str, Any]] = []
    for length_name, target_words in LENGTHS:
        for archetype in ARCHETYPES:
            task_id = f'LC{len(tasks) + 1:03d}'
            document, metadata = _document(task_id, archetype, length_name, target_words)
            document_name = f'{task_id}.md'
            (documents_dir / document_name).write_text(document + '\n', encoding='utf-8', newline='\n')
            prompt = (
                f'Long-context task {task_id}. {metadata["question"]}\n'
                'Return only the requested answer with no explanation.\n\n'
                f'DOCUMENT:\n{document}\n'
            )
            task = {
                'id': task_id,
                'suite': 'long-context-v1',
                'difficulty': 'medium' if length_name == 'S' else 'hard',
                'context_size': target_words,
                'archetype': archetype,
                'source_length': length_name,
                'document_path': f'documents/{document_name}',
                'document_sha256': metadata['document_sha256'],
                'prompt': prompt,
                'grading_status': 'READY_DETERMINISTIC',
                'graders': [metadata['grader']],
                'expected_output': metadata['expected_output'],
                'distractor_count': metadata['distractor_count'],
            }
            tasks.append(task)
    tasks_path.write_text(
        ''.join(json.dumps(task, sort_keys=True) + '\n' for task in tasks),
        encoding='utf-8', newline='\n',
    )
    manifest = {
        'schema_version': 'long-context-v1-dataset-v1',
        'suite_id': 'long-context-v1',
        'task_count': len(tasks),
        'lengths': {name: words for name, words in LENGTHS},
        'archetypes': list(ARCHETYPES),
        'tasks_sha256': hashlib.sha256(tasks_path.read_bytes()).hexdigest(),
        'documents': [
            {'task_id': task['id'], 'path': task['document_path'], 'sha256': task['document_sha256']}
            for task in tasks
        ],
        'source': 'original deterministic synthetic documents generated by scripts/generate_long_context_dataset.py',
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    return manifest


def main(argv: list[str] | None = None) -> int:
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description='Generate long-context-v1 dataset')
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args(argv)
    try:
        manifest = generate(root, force=args.force)
    except (OSError, ValueError) as exc:
        print(f'LONG-CONTEXT GENERATION REFUSED: {exc}', flush=True)
        return 2
    print(f"generated {manifest['task_count']} long-context tasks")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
