import argparse
import hashlib
import json
import re
import zipfile
from collections import Counter
from pathlib import Path
from typing import cast
from xml.etree import ElementTree as ET

NS = {'m': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
REL = 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'
ROOT = Path(__file__).resolve().parents[1]

SheetRows = dict[int, dict[str, str]]
Workbook = dict[str, SheetRows]
SourceRecord = dict[str, object]
TaskRecord = dict[str, object]


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def workbook_rows(path: Path) -> Workbook:
    with zipfile.ZipFile(path) as archive:
        strings: list[str] = []
        if 'xl/sharedStrings.xml' in archive.namelist():
            strings = [''.join(node.itertext()) for node in ET.fromstring(archive.read('xl/sharedStrings.xml')).findall('m:si', NS)]
        relationships = {node.attrib['Id']: node.attrib['Target'] for node in ET.fromstring(archive.read('xl/_rels/workbook.xml.rels'))}
        result: Workbook = {}
        for sheet in ET.fromstring(archive.read('xl/workbook.xml')).findall('m:sheets/m:sheet', NS):
            target = relationships[sheet.attrib[f'{{{REL}}}id']]
            member = target.lstrip('/') if target.startswith('/') else 'xl/' + target
            rows: SheetRows = {}
            for row in ET.fromstring(archive.read(member)).findall('m:sheetData/m:row', NS):
                cells: dict[str, str] = {}
                for cell in row.findall('m:c', NS):
                    inline = cell.find('m:is', NS)
                    value = cell.find('m:v', NS)
                    text = ''.join(inline.itertext()) if inline is not None else (value.text or '' if value is not None else '')
                    if cell.get('t') == 's':
                        text = strings[int(text)]
                    if cell.find('m:f', NS) is not None:
                        raise ValueError('Formula cells require explicit review')
                    cells[re.sub(r'\d+', '', cell.attrib['r'])] = text
                rows[int(row.attrib['r'])] = cells
            result[sheet.attrib['name']] = rows
        return result


def records(rows: SheetRows, header_row: int) -> list[tuple[int, dict[str, str]]]:
    headers = rows[header_row]
    return [(number, {label: row.get(column, '') for column, label in headers.items() if label}) for number, row in rows.items() if number > header_row and row.get('A', '').strip()]


def slug(text: str) -> str:
    return re.sub(r'[^a-z0-9]+', '_', text.lower()).strip('_')


def convert(path: Path) -> tuple[list[SourceRecord], list[TaskRecord], Workbook]:
    sheets = workbook_rows(path)
    questions = records(sheets['Eval Questions'], 4)
    answers = records(sheets['Answer Key'], 4)
    answer_map: dict[str, tuple[int, dict[str, str]]] = {record['ID']: (row, record) for row, record in answers}
    ids = [record['ID'] for _, record in questions]
    expected_ids = {f'Q{i:03}' for i in range(1, 81)}
    if len(ids) != 80 or set(ids) != expected_ids or len(answer_map) != len(answers) or set(answer_map) != expected_ids:
        raise ValueError('Expected exactly 80 unique, matching question and answer IDs')
    source: list[SourceRecord] = []
    tasks: list[TaskRecord] = []
    for row, question in questions:
        task_id = question['ID']
        answer_row, answer = answer_map[task_id]
        if question['Primary Grader'] != answer['Primary Grader']:
            raise ValueError(f'{task_id}: grader labels disagree')
        if not question['Question / Prompt'] or not answer['Expected Answer / Required Criteria']:
            raise ValueError(f'{task_id}: missing prompt or answer')
        provenance = {'question_sheet': 'Eval Questions', 'question_row': row, 'answer_sheet': 'Answer Key', 'answer_row': answer_row}
        source.append({'id': task_id, 'question': question, 'answer_key': answer, 'provenance': provenance})
        coding = 27 <= int(task_id[1:]) <= 34
        suite = 'coding-v1' if coding else ('stress-v1' if question['Suite'] == 'Stress' else 'core-v1')
        raw_answer = answer['Expected Answer / Required Criteria']
        reasons: list[str] = ['Acceptance rules and grader configuration require human specification review']
        if coding:
            reasons.append('Executable coding fixtures have not been authored')
        output_type = slug(question['Expected Output'])
        category = slug(question['Category'])
        if category == 'long_context_retrieval':
            category = 'context_comprehension'
        tasks.append({'id': task_id, 'suite': suite, 'category': category, 'difficulty': question['Difficulty'].lower(), 'prompt': question['Question / Prompt'], 'output_type': output_type, 'expected_output': None, 'expected_output_source': raw_answer, 'graders': [], 'graders_source': question['Primary Grader'], 'grading_notes_source': answer['Grading Notes'], 'failure_modes': [], 'failure_modes_source': question['Target Failure Mode'], 'grading_status': 'PENDING_SPECIFICATION', 'pending_reasons': reasons, 'timeout_seconds': None, 'context_size': None, 'source_recommended_repeats': int(question['Repeat Runs']), 'source': provenance})
    validate(source, tasks)
    return source, tasks, sheets


def validate(source: list[SourceRecord], tasks: list[TaskRecord]) -> None:
    if len(source) != 80 or len(tasks) != 80:
        raise ValueError('Invalid total')
    source_map: dict[object, SourceRecord] = {record['id']: record for record in source}
    if len(source_map) != 80 or len({task['id'] for task in tasks}) != 80:
        raise ValueError('Duplicate IDs')
    if Counter(cast(str, task['suite']) for task in tasks) != {'core-v1': 52, 'stress-v1': 20, 'coding-v1': 8}:
        raise ValueError('Invalid suite split')
    for task in tasks:
        original = source_map[task['id']]
        question = cast(dict[str, str], original['question'])
        answer = cast(dict[str, str], original['answer_key'])
        pairs: list[tuple[object, object]] = [
            (task['prompt'], question['Question / Prompt']),
            (task['expected_output_source'], answer['Expected Answer / Required Criteria']),
            (task['failure_modes_source'], question['Target Failure Mode']),
            (task['difficulty'], question['Difficulty'].lower()),
            (task['source_recommended_repeats'], int(question['Repeat Runs'])),
        ]
        if any(a != b for a, b in pairs):
            raise ValueError(f"Source fidelity failed: {task['id']}")
        if task['grading_status'] == 'PENDING_SPECIFICATION' and not task['pending_reasons']:
            raise ValueError('Pending tasks need reasons')


def encode_json(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + '\n').encode('utf-8')


def encode_jsonl(records: list[dict[str, object]]) -> bytes:
    return ''.join(json.dumps(record, ensure_ascii=False, sort_keys=True) + '\n' for record in records).encode('utf-8')


def build(path: Path, output: Path, check: bool = False) -> dict[str, object]:
    original_bytes = path.read_bytes()
    source, tasks, sheets = convert(path)
    review: list[dict[str, object]] = []
    for task, record in zip(tasks, source):
        question = cast(dict[str, str], record['question'])
        answer = cast(dict[str, str], record['answer_key'])
        review.append({'id': task['id'], 'suite': task['suite'], 'grading_status': task['grading_status'], 'reasons': task['pending_reasons'], 'source_grader': question['Primary Grader'], 'source_answer': answer['Expected Answer / Required Criteria'], 'grading_notes': answer['Grading Notes']})
    artifacts: dict[str, bytes] = {
        'source-v1.jsonl': encode_jsonl(source),
        'executable-v1.jsonl': encode_jsonl(tasks),
        'grading-review.json': encode_json(review),
        'workbook-metadata.json': encode_json({name: rows for name, rows in sheets.items() if name not in ('Eval Questions', 'Answer Key')}),
    }
    schema_path = output / 'task.schema.json'
    if not schema_path.exists():
        raise ValueError('task.schema.json must exist before conversion')
    manifest: dict[str, object] = {
        'version': 'eval-v1',
        'release_status': 'DRAFT',
        'source_snapshot_status': 'HASH_PINNED',
        'source': {'filename': path.name, 'sha256': digest(original_bytes)},
        'task_count': len(tasks),
        'suite_counts': dict(Counter(cast(str, t['suite']) for t in tasks)),
        'difficulty_counts': {suite: dict(Counter(cast(str, t['difficulty']) for t in tasks if t['suite'] == suite)) for suite in ('core-v1', 'stress-v1', 'coding-v1')},
        'grading_status_counts': dict(Counter(cast(str, t['grading_status']) for t in tasks)),
        'artifacts': {name: digest(data) for name, data in artifacts.items()},
        'schema_sha256': digest(schema_path.read_bytes()),
        'freeze_blockers': ['Review and specify mandatory task grading before executable release', 'Grader configuration and judge prompts are not yet authored or frozen'],
    }
    artifacts['manifest.json'] = encode_json(manifest)
    if check:
        for name, data in artifacts.items():
            if not (output / name).exists() or (output / name).read_bytes() != data:
                raise ValueError(f'Artifact differs: {name}')
    else:
        output.mkdir(parents=True, exist_ok=True)
        if (output / 'manifest.json').exists():
            prior = cast(dict[str, object], json.loads((output / 'manifest.json').read_text(encoding='utf-8')))
            prior_source = cast(dict[str, object], prior['source'])
            if prior_source['sha256'] != digest(original_bytes) or prior['release_status'] != 'DRAFT':
                raise ValueError('Refusing to overwrite changed source or frozen release; use a new version directory')
            for name, data in artifacts.items():
                if (output / name).exists() and (output / name).read_bytes() != data:
                    raise ValueError(f'Refusing to overwrite edited artifact: {name}')
        for name, data in artifacts.items():
            (output / name).write_bytes(data)
    if path.read_bytes() != original_bytes:
        raise ValueError('Source workbook changed during conversion')
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, default=ROOT / 'LocalLLM_Eval_Questions.xlsx')
    parser.add_argument('--output', type=Path, default=ROOT / 'evals' / 'datasets' / 'eval-v1')
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    print(json.dumps(build(args.source, args.output, args.check), indent=2))


if __name__ == '__main__':
    main()
