"""Run the local Judge Suite panel after approved human calibration."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis.judge import (
    JudgePair,
    JudgeProtocolError,
    JudgeSourceItem,
    build_controlled_pairs,
    build_pair_prompt,
    build_rubric_prompt,
    load_judge_protocol,
    load_source_items,
    parse_pair_response,
    parse_rubric_response,
    select_real_pairs,
)
from inference.adapters import get_model_config, render_prompt
from inference.ollama_client import GenerationRequest, GenerationResult, generate
from storage.judge import open_judge_db


class JudgeRunError(ValueError):
    """The local judge run is not authorized or cannot preserve provenance."""


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


def _schema_for_rubric(criteria: dict[str, Any]) -> dict[str, Any]:
    return {
        'type': 'object',
        'properties': {
            'schema_version': {'const': 'judge-rubric-v1'},
            'item_id': {'type': 'string'},
            'criteria': {
                'type': 'object',
                'properties': {
                    key: {
                        'type': 'object',
                        'properties': {
                            'observed': {'type': 'boolean'},
                            'confidence': {'enum': ['low', 'medium', 'high']},
                            'evidence': {'type': 'array', 'items': {'type': 'string'}},
                        },
                        'required': ['observed', 'confidence', 'evidence'],
                    }
                    for key in criteria
                },
                'required': list(criteria),
            },
            'abstain': {'type': 'boolean'},
        },
        'required': ['schema_version', 'item_id', 'criteria', 'abstain'],
    }


def _schema_for_pair() -> dict[str, Any]:
    return {
        'type': 'object',
        'properties': {
            'schema_version': {'const': 'judge-pair-v1'},
            'pair_id': {'type': 'string'},
            'decision': {'enum': ['A', 'B', 'TIE', 'NEITHER']},
            'confidence': {'enum': ['low', 'medium', 'high']},
            'evidence': {'type': 'array'},
            'abstain': {'type': 'boolean'},
        },
        'required': ['schema_version', 'pair_id', 'decision', 'confidence', 'evidence', 'abstain'],
    }


def _request(model_id: str, prompt: str, schema: dict[str, Any]) -> GenerationRequest:
    config = get_model_config(model_id)
    return GenerationRequest(
        model=config.ollama_identifier,
        prompt=render_prompt(config, prompt),
        temperature=0.0,
        num_ctx=config.num_ctx,
        num_predict=768,
        stop=config.stop_tokens,
        raw=config.mode == 'raw',
        think=config.think,
        num_gpu=config.num_gpu,
        format=schema,
    )


def _calibration(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise JudgeRunError(f'human calibration is missing: {path}')
    data = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(data, dict):
        raise JudgeRunError('human calibration is not an object')
    if data.get('status') != 'APPROVED':
        raise JudgeRunError('human calibration is not APPROVED')
    if int(data.get('reviewer_count', 0)) < 2:
        raise JudgeRunError('human calibration requires at least two reviewers')
    if data.get('holdout_locked') is not True:
        raise JudgeRunError('human calibration holdout is not locked')
    if not isinstance(data.get('calibration_sha256'), str):
        raise JudgeRunError('human calibration has no content hash')
    return data


def _call(
    conn: sqlite3.Connection,
    *,
    protocol_id: str,
    item_id: str | None,
    pair_id: str | None,
    model_id: str,
    orientation: str,
    request: GenerationRequest,
    generate_fn: Callable[..., GenerationResult],
    base_url: str,
    timeout_s: float,
) -> tuple[str, GenerationResult] | None:
    call_id = _sha(f'{protocol_id}|{item_id}|{pair_id}|{model_id}|{orientation}')
    existing = conn.execute(
        'SELECT status,response_sha256 FROM judge_calls WHERE call_id=?', (call_id,)
    ).fetchone()
    if existing is not None and existing[0] == 'COMPLETE':
        return None
    started = _utcnow()
    request_hash = _sha(request.prompt)
    try:
        result = generate_fn(base_url, request, timeout_s=timeout_s)
        response_hash = _sha(result.text)
        conn.execute(
            'INSERT OR REPLACE INTO judge_calls'
            '(call_id,protocol_id,item_id,pair_id,judge_type,judge_model_config_id,'
            'judge_artifact_digest,orientation,request_sha256,response_sha256,status,'
            'parse_status,error_json,started_at_utc,ended_at_utc) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)',
            (
                call_id, protocol_id, item_id, pair_id, 'local', model_id,
                get_model_config(model_id).ollama_model_digest, orientation,
                request_hash, response_hash, 'CALL_COMPLETE', 'PENDING', None,
                started, _utcnow(),
            ),
        )
        conn.commit()
        return call_id, result
    except Exception as exc:
        conn.execute(
            'INSERT OR REPLACE INTO judge_calls'
            '(call_id,protocol_id,item_id,pair_id,judge_type,judge_model_config_id,'
            'judge_artifact_digest,orientation,request_sha256,response_sha256,status,'
            'parse_status,error_json,started_at_utc,ended_at_utc) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)',
            (
                call_id, protocol_id, item_id, pair_id, 'local', model_id,
                get_model_config(model_id).ollama_model_digest, orientation,
                request_hash, None, 'CALL_ERROR', 'ERROR',
                json.dumps({'type': type(exc).__name__, 'message': str(exc)}),
                started, _utcnow(),
            ),
        )
        conn.commit()
        return None


def _save_rubric_result(
    conn: sqlite3.Connection,
    *,
    call_id: str,
    item_id: str,
    rubric: dict[str, Any],
    payload: dict[str, Any],
) -> str:
    status = 'ADJUDICATION_REQUIRED' if payload['abstain'] else 'JUDGE_RESULT'
    for criterion_id, result in payload['criteria'].items():
        desired = bool(rubric[criterion_id]['desired'])
        passed = result['observed'] == desired
        if result['confidence'] == 'low':
            status = 'ADJUDICATION_REQUIRED'
        conn.execute(
            'INSERT OR REPLACE INTO judge_rubric_results'
            '(call_id,criterion_id,observed,desired,passed,confidence,evidence_json)'
            ' VALUES(?,?,?,?,?,?,?)',
            (
                call_id, criterion_id, int(result['observed']), int(desired),
                int(passed), result['confidence'], json.dumps(result['evidence']),
            ),
        )
    conn.execute(
        'UPDATE judge_calls SET status=?,parse_status=\'VALID\' WHERE call_id=?',
        (status, call_id),
    )
    conn.commit()
    return status


def _run_rubric_calls(
    conn: sqlite3.Connection,
    *,
    protocol_id: str,
    protocol: Any,
    items: list[JudgeSourceItem],
    model_ids: list[str],
    base_url: str,
    timeout_s: float,
    generate_fn: Callable[..., GenerationResult],
) -> dict[str, int]:
    counts = {'calls': 0, 'valid': 0, 'adjudication_required': 0, 'errors': 0}
    for item in items:
        rubric = protocol.rubrics[item.task_id]
        prompt = build_rubric_prompt(item, {'criteria': rubric})
        schema = _schema_for_rubric(rubric)
        for model_id in model_ids:
            call = _call(
                conn, protocol_id=protocol_id, item_id=item.source_item_id,
                pair_id=None, model_id=model_id, orientation='rubric',
                request=_request(model_id, prompt, schema), generate_fn=generate_fn,
                base_url=base_url, timeout_s=timeout_s,
            )
            if call is None:
                continue
            call_id, result = call
            counts['calls'] += 1
            try:
                payload = parse_rubric_response(
                    result.text, item_id=item.source_item_id,
                    rubric={'criteria': rubric}, candidate=item.raw_output,
                )
                status = _save_rubric_result(
                    conn, call_id=call_id, item_id=item.source_item_id,
                    rubric=rubric, payload=payload,
                )
                counts['valid'] += 1
                if status == 'ADJUDICATION_REQUIRED':
                    counts['adjudication_required'] += 1
            except JudgeProtocolError:
                counts['errors'] += 1
                conn.execute(
                    'UPDATE judge_calls SET status=\'PARSE_ERROR\',parse_status=\'INVALID\''
                    ' WHERE call_id=?', (call_id,),
                )
                conn.commit()
    return counts


def _run_pair_calls(
    conn: sqlite3.Connection,
    *,
    protocol_id: str,
    pairs: list[JudgePair],
    model_ids: list[str],
    base_url: str,
    timeout_s: float,
    generate_fn: Callable[..., GenerationResult],
) -> dict[str, int]:
    counts = {'calls': 0, 'valid': 0, 'errors': 0}
    schema = _schema_for_pair()
    for pair in pairs:
        conn.execute(
            'INSERT OR REPLACE INTO judge_pairs'
            '(pair_id,protocol_id,task_id,trial,left_source_item_id,'
            'right_source_item_id,population_hash) VALUES(?,?,?,?,?,?,?)',
            (
                pair.pair_id, protocol_id, pair.task_id, pair.trial,
                pair.left.source_item_id, pair.right.source_item_id,
                _sha(f'{pair.left.source_item_id}|{pair.right.source_item_id}'),
            ),
        )
        conn.commit()
        candidates = {'A': pair.left.raw_output, 'B': pair.right.raw_output}
        for model_id in model_ids:
            for orientation, left, right in (
                ('forward', pair.left, pair.right),
                ('inverse', pair.right, pair.left),
            ):
                oriented = JudgePair(
                    pair_id=pair.pair_id, task_id=pair.task_id,
                    trial=pair.trial, left=left, right=right,
                )
                call = _call(
                    conn, protocol_id=protocol_id, item_id=None,
                    pair_id=pair.pair_id, model_id=model_id,
                    orientation=orientation,
                    request=_request(model_id, build_pair_prompt(oriented), schema),
                    generate_fn=generate_fn, base_url=base_url, timeout_s=timeout_s,
                )
                if call is None:
                    continue
                call_id, result = call
                counts['calls'] += 1
                try:
                    payload = parse_pair_response(
                        result.text, pair_id=pair.pair_id, candidates=candidates,
                    )
                    conn.execute(
                        'INSERT OR REPLACE INTO judge_pair_results'
                        '(call_id,pair_id,orientation,decision,confidence,evidence_json,'
                        'parse_status) VALUES(?,?,?,?,?,?,?)',
                        (
                            call_id, pair.pair_id, orientation, payload['decision'],
                            payload['confidence'], json.dumps(payload['evidence']),
                            'VALID',
                        ),
                    )
                    conn.execute(
                        'UPDATE judge_calls SET status=\'PAIR_RESULT\',parse_status=\'VALID\''
                        ' WHERE call_id=?', (call_id,),
                    )
                    conn.commit()
                    counts['valid'] += 1
                except JudgeProtocolError:
                    counts['errors'] += 1
                    conn.execute(
                        'UPDATE judge_calls SET status=\'PARSE_ERROR\',parse_status=\'INVALID\''
                        ' WHERE call_id=?', (call_id,),
                    )
                    conn.commit()
    return counts


def run_local_panel(
    root: Path,
    *,
    calibration_path: Path,
    base_url: str = 'http://127.0.0.1:11434',
    timeout_s: float = 300.0,
    generate_fn: Callable[..., GenerationResult] = generate,
) -> dict[str, Any]:
    protocol = load_judge_protocol(root)
    if protocol.document.get('local_judges_enabled') is not False:
        raise JudgeRunError('local judges are disabled; use run_external_judge.py')
    if protocol.document.get('judge_mode') == 'external_openai_compatible':
        raise JudgeRunError('external-only protocol cannot use the local runner')
    _calibration(calibration_path)
    items = load_source_items(root, protocol)
    pending = [item for item in items if item.precheck_status == 'JUDGE_PENDING']
    real_pairs = select_real_pairs(items, per_task=10)
    controlled_pairs = build_controlled_pairs(items, per_task=8)
    conn = open_judge_db(root / 'results/local/judge-suite-v1.db')
    try:
        rubric_counts = _run_rubric_calls(
            conn, protocol_id='judge-suite-v1', protocol=protocol,
            items=pending, model_ids=list(protocol.document['local_panel']),
            base_url=base_url, timeout_s=timeout_s, generate_fn=generate_fn,
        )
        pair_counts = _run_pair_calls(
            conn, protocol_id='judge-suite-v1', pairs=real_pairs + controlled_pairs,
            model_ids=list(protocol.document['local_panel']), base_url=base_url,
            timeout_s=timeout_s, generate_fn=generate_fn,
        )
    finally:
        conn.close()
    return {
        'status': 'LOCAL_PANEL_COMPLETE_HUMAN_ADJUDICATION_REQUIRED',
        'source_items': len(items),
        'precheck_pass_items': len(pending),
        'real_pairs': len(real_pairs),
        'controlled_pairs': len(controlled_pairs),
        'rubric_calls': rubric_counts,
        'pair_calls': pair_counts,
    }


def main(argv: list[str] | None = None) -> int:
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description='Run the local Judge Suite panel')
    parser.add_argument('--calibration', default='results/local/judge-suite-v1-calibration.json')
    parser.add_argument('--base-url', default='http://127.0.0.1:11434')
    parser.add_argument('--timeout-s', type=float, default=300.0)
    args = parser.parse_args(argv)
    try:
        result = run_local_panel(
            root, calibration_path=root / args.calibration,
            base_url=args.base_url, timeout_s=args.timeout_s,
        )
    except (OSError, JudgeProtocolError, JudgeRunError, ValueError) as exc:
        print(f'JUDGE PANEL REFUSED: {exc}', flush=True)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
