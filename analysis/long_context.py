"""Long-context-v1 dataset loading and report aggregation."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class LongContextTask:
    task_id: str
    source_length: str
    archetype: str
    context_size: int
    prompt: str
    expected_output: Any
    graders: tuple[dict[str, Any], ...]
    document_sha256: str
    distractor_count: int


def load_long_context_tasks(root: Path) -> list[LongContextTask]:
    dataset = root / 'evals/datasets/long-context-v1'
    tasks: list[LongContextTask] = []
    with open(dataset / 'tasks.jsonl', encoding='utf-8') as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            tasks.append(LongContextTask(
                task_id=str(row['id']),
                source_length=str(row['source_length']),
                archetype=str(row['archetype']),
                context_size=int(row['context_size']),
                prompt=str(row['prompt']),
                expected_output=row['expected_output'],
                graders=tuple(row['graders']),
                document_sha256=str(row['document_sha256']),
                distractor_count=int(row['distractor_count']),
            ))
    return tasks


def summarize_long_context(
    rows: Iterable[dict[str, Any]],
    tasks: list[LongContextTask],
) -> dict[str, Any]:
    task_map = {task.task_id: task for task in tasks}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row['task_id'])].append(row)
    by_length: dict[str, dict[str, Any]] = {}
    by_archetype: dict[str, dict[str, Any]] = {}
    all_trials = 0
    all_passes = 0
    for length in sorted({task.source_length for task in tasks}):
        selected = [row for task in tasks if task.source_length == length for row in grouped[task.task_id]]
        passes = sum(row.get('grader_verdict') == 'PASS' for row in selected)
        all_trials += len(selected)
        all_passes += passes
        by_length[length] = {'trials': len(selected), 'passes': passes, 'accuracy': passes / len(selected) if selected else None}
    for archetype in sorted({task.archetype for task in tasks}):
        selected = [row for task in tasks if task.archetype == archetype for row in grouped[task.task_id]]
        passes = sum(row.get('grader_verdict') == 'PASS' for row in selected)
        by_archetype[archetype] = {'trials': len(selected), 'passes': passes, 'accuracy': passes / len(selected) if selected else None}
    task_results = {
        task_id: {
            'trials': len(grouped[task_id]),
            'passes': sum(row.get('grader_verdict') == 'PASS' for row in grouped[task_id]),
            'all_pass': bool(grouped[task_id]) and all(row.get('grader_verdict') == 'PASS' for row in grouped[task_id]),
            'any_pass': any(row.get('grader_verdict') == 'PASS' for row in grouped[task_id]),
        }
        for task_id in sorted(grouped)
    }
    return {
        'task_count': len(tasks),
        'trial_count': all_trials,
        'passes': all_passes,
        'trial_accuracy': all_passes / all_trials if all_trials else None,
        'by_length': by_length,
        'by_archetype': by_archetype,
        'tasks': task_results,
        'task_id_coverage': sorted(grouped) == sorted(task_map),
    }
