"""Run the external-only Judge Suite through an OpenAI-compatible endpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

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
from inference.external_judge import ExternalJudgeClient, ExternalJudgeConfig, ExternalJudgeError
from storage.judge import open_judge_db, record_external_call, source_population_hash


class ExternalJudgeRunError(RuntimeError):
    """The external judge run cannot satisfy the frozen protocol."""


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


def _schema_for_rubric(criteria: Mapping[str, Any]) -> dict[str, Any]:
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


def _call_external(
    conn: sqlite3.Connection,
    client: ExternalJudgeClient,
    *,
    protocol_id: str,
    item_id: str | None,
    pair_id: str | None,
    orientation: str,
    prompt: str,
    schema: Mapping[str, Any],
) -> tuple[str, str]:
    call_id = _sha(f'{protocol_id}|{item_id}|{pair_id}|{orientation}')
    started = _utcnow()
    request_hash = _sha(prompt)
    try:
        result = client.judge(prompt, schema)
    except Exception as exc:
        external_error = exc if isinstance(exc, ExternalJudgeError) else None
        if external_error is None:
            request_id = None
            usage: dict[str, Any] = {}
            cost_usd = None
            response_sha256 = None
            attempts = client.config.max_attempts
        else:
            request_id = external_error.request_id
            usage = external_error.usage or {}
            cost_usd = external_error.estimated_cost_usd
            response_sha256 = external_error.response_sha256
            attempts = external_error.attempts
        record_external_call(
            conn, call_id=call_id, protocol_id=protocol_id, item_id=item_id,
            pair_id=pair_id, provider='openai_compatible',
            model_id=client.config.model_id, base_url_sha256=_sha(client.config.base_url),
            request_id=request_id, request_sha256=request_hash,
            response_sha256=response_sha256, usage=usage, cost_usd=cost_usd,
            attempts=attempts, status='CALL_ERROR', parse_status='ERROR',
            forced_label=False, confidence=None,
            error={'type': type(exc).__name__, 'message': str(exc)},
            started_at_utc=started, ended_at_utc=_utcnow(),
        )
        raise
    response_hash = _sha(result.text)
    record_external_call(
        conn, call_id=call_id, protocol_id=protocol_id, item_id=item_id,
        pair_id=pair_id, provider='openai_compatible', model_id=client.config.model_id,
        base_url_sha256=_sha(client.config.base_url), request_id=result.request_id,
        request_sha256=request_hash, response_sha256=response_hash,
        usage=result.usage, cost_usd=result.estimated_cost_usd,
        attempts=result.attempts, status='CALL_COMPLETE', parse_status='PENDING',
        forced_label=False, confidence=None, error=None,
        started_at_utc=started, ended_at_utc=_utcnow(),
    )
    return call_id, result.text


def _force_rubric(text: str, item: JudgeSourceItem, rubric: Mapping[str, Any]) -> tuple[dict[str, Any], bool]:
    payload = parse_rubric_response(
        text, item_id=item.source_item_id, rubric={'criteria': rubric},
        candidate=item.raw_output,
    )
    forced = bool(payload.get('abstain'))
    payload['abstain'] = False
    return payload, forced


def _force_pair(text: str, pair: JudgePair, candidates: Mapping[str, str]) -> tuple[dict[str, Any], bool]:
    payload = parse_pair_response(text, pair_id=pair.pair_id, candidates=candidates)
    forced = bool(payload.get('abstain'))
    payload['abstain'] = False
    if forced and payload['decision'] == 'NEITHER':
        payload['decision'] = 'TIE'
    return payload, forced


def _mark_parse_failure(
    conn: sqlite3.Connection,
    call_id: str,
    error: Exception,
) -> None:
    conn.execute(
        'UPDATE external_judge_calls SET status=\'PARSE_ERROR\',parse_status=\'INVALID\','
        'error_json=? WHERE call_id=?',
        (json.dumps({'type': type(error).__name__, 'message': str(error)}), call_id),
    )
    conn.commit()


def _record_pair_result(
    conn: sqlite3.Connection,
    *,
    call_id: str,
    pair_id: str,
    orientation: str,
    decision: str,
    confidence: str,
    evidence: object,
    parse_status: str = 'VALID',
) -> None:
    conn.execute(
        'INSERT OR REPLACE INTO judge_pair_results'
        '(call_id,pair_id,orientation,decision,confidence,evidence_json,parse_status)'
        ' VALUES(?,?,?,?,?,?,?)',
        (
            call_id, pair_id, orientation, decision, confidence,
            json.dumps(evidence, sort_keys=True), parse_status,
        ),
    )
    conn.commit()


def _record_aggregate_pair(
    conn: sqlite3.Connection,
    *,
    pair_id: str,
    decision: str,
    evidence: object,
    parse_status: str,
) -> None:
    aggregate_id = _sha(f'aggregate|{pair_id}')
    _record_pair_result(
        conn,
        call_id=aggregate_id,
        pair_id=pair_id,
        orientation='aggregate',
        decision=decision,
        confidence='low',
        evidence=evidence,
        parse_status=parse_status,
    )


def _normalize_pair(decision: str, orientation: str) -> str:
    if decision in {'TIE', 'NEITHER'}:
        return decision
    if orientation == 'forward':
        return 'LEFT' if decision == 'A' else 'RIGHT'
    return 'RIGHT' if decision == 'A' else 'LEFT'


def _run_rubric(
    conn: sqlite3.Connection,
    client: ExternalJudgeClient,
    *,
    protocol_id: str,
    protocol: Any,
    items: list[JudgeSourceItem],
) -> dict[str, int]:
    counts = {'calls': 0, 'forced': 0, 'errors': 0}
    for item in items:
        rubric = protocol.rubrics[item.task_id]
        call_id, text = _call_external(
            conn, client, protocol_id=protocol_id, item_id=item.source_item_id,
            pair_id=None, orientation='rubric',
            prompt=build_rubric_prompt(item, {'criteria': rubric}),
            schema=_schema_for_rubric(rubric),
        )
        counts['calls'] += 1
        try:
            payload, forced = _force_rubric(text, item, rubric)
        except JudgeProtocolError as exc:
            counts['errors'] += 1
            _mark_parse_failure(conn, call_id, exc)
            continue
        if forced:
            counts['forced'] += 1
        for criterion_id, result in payload['criteria'].items():
            desired = bool(rubric[criterion_id]['desired'])
            passed = bool(result['observed']) == desired
            conn.execute(
                'INSERT OR REPLACE INTO judge_rubric_results'
                '(call_id,criterion_id,observed,desired,passed,confidence,evidence_json)'
                ' VALUES(?,?,?,?,?,?,?)',
                (call_id, criterion_id, int(bool(result['observed'])), int(desired),
                 int(passed), result['confidence'], json.dumps(result['evidence'])),
            )
        conn.execute(
            'UPDATE external_judge_calls SET parse_status=\'VALID\',forced_label=?,confidence=?'
            ' WHERE call_id=?',
            (int(forced), _min_confidence(payload), call_id),
        )
        conn.commit()
    return counts


def _min_confidence(payload: Mapping[str, Any]) -> str:
    order = {'low': 0, 'medium': 1, 'high': 2}
    values = [
        result['confidence'] for result in payload.get('criteria', {}).values()
        if isinstance(result, Mapping)
    ]
    return min(values, key=lambda value: order.get(str(value), 0), default='low')


def _run_pairs(
    conn: sqlite3.Connection,
    client: ExternalJudgeClient,
    *,
    protocol_id: str,
    pairs: list[JudgePair],
) -> dict[str, int]:
    counts = {'pairs': 0, 'calls': 0, 'third_calls': 0, 'forced': 0, 'errors': 0}
    schema = _schema_for_pair()
    for pair in pairs:
        conn.execute(
            'INSERT OR REPLACE INTO judge_pairs'
            '(pair_id,protocol_id,task_id,trial,left_source_item_id,'
            'right_source_item_id,population_hash) VALUES(?,?,?,?,?,?,?)',
            (pair.pair_id, protocol_id, pair.task_id, pair.trial,
             pair.left.source_item_id, pair.right.source_item_id,
             _sha(f'{pair.left.source_item_id}|{pair.right.source_item_id}')),
        )
        conn.execute('DELETE FROM judge_pair_results WHERE pair_id=?', (pair.pair_id,))
        conn.commit()
        decisions: list[str] = []
        call_ids: list[str] = []
        valid_orientations = 0
        for orientation, left, right in (
            ('forward', pair.left, pair.right),
            ('inverse', pair.right, pair.left),
        ):
            oriented = JudgePair(pair.pair_id, pair.task_id, pair.trial, left, right)
            call_id, text = _call_external(
                conn, client, protocol_id=protocol_id, item_id=None,
                pair_id=pair.pair_id, orientation=orientation,
                prompt=build_pair_prompt(oriented), schema=schema,
            )
            counts['calls'] += 1
            call_ids.append(call_id)
            candidates = {'A': left.raw_output, 'B': right.raw_output}
            try:
                payload, forced = _force_pair(text, oriented, candidates)
            except JudgeProtocolError as exc:
                counts['errors'] += 1
                _mark_parse_failure(conn, call_id, exc)
                continue
            if forced:
                counts['forced'] += 1
            decision = _normalize_pair(payload['decision'], orientation)
            _record_pair_result(
                conn,
                call_id=call_id,
                pair_id=pair.pair_id,
                orientation=orientation,
                decision=decision,
                confidence=payload['confidence'],
                evidence=payload['evidence'],
            )
            conn.execute(
                'UPDATE external_judge_calls SET parse_status=\'VALID\',forced_label=?,confidence=?'
                ' WHERE call_id=?',
                (int(forced), payload['confidence'], call_id),
            )
            conn.commit()
            decisions.append(decision)
            valid_orientations += 1
        if valid_orientations < 2:
            _record_aggregate_pair(
                conn,
                pair_id=pair.pair_id,
                decision='INVALID',
                evidence={'normalized_decisions': decisions, 'call_ids': call_ids},
                parse_status='INCOMPLETE',
            )
            continue
        parse_status = 'VALID'
        if decisions[0] != decisions[1] and client.calls_made < client.config.max_calls:
            tie_pair = JudgePair(pair.pair_id, pair.task_id, pair.trial, pair.left, pair.right)
            call_id, text = _call_external(
                conn, client, protocol_id=protocol_id, item_id=None,
                pair_id=pair.pair_id, orientation='third_call',
                prompt=build_pair_prompt(tie_pair) + '\nTIE_BREAK: choose the closest label.',
                schema=schema,
            )
            counts['third_calls'] += 1
            call_ids.append(call_id)
            third_candidates = {'A': tie_pair.left.raw_output, 'B': tie_pair.right.raw_output}
            try:
                payload, forced = _force_pair(text, tie_pair, third_candidates)
            except JudgeProtocolError as exc:
                counts['errors'] += 1
                _mark_parse_failure(conn, call_id, exc)
                _record_aggregate_pair(
                    conn,
                    pair_id=pair.pair_id,
                    decision=decisions[0],
                    evidence={'normalized_decisions': decisions, 'call_ids': call_ids},
                    parse_status='FALLBACK_FORWARD_ERROR',
                )
                counts['pairs'] += 1
                continue
            if forced:
                counts['forced'] += 1
            third_decision = _normalize_pair(payload['decision'], 'forward')
            _record_pair_result(
                conn,
                call_id=call_id,
                pair_id=pair.pair_id,
                orientation='third_call',
                decision=third_decision,
                confidence=payload['confidence'],
                evidence=payload['evidence'],
            )
            conn.execute(
                'UPDATE external_judge_calls SET parse_status=\'VALID\',forced_label=?,confidence=?'
                ' WHERE call_id=?',
                (int(forced), payload['confidence'], call_id),
            )
            conn.commit()
            decisions.append(third_decision)
        elif decisions[0] != decisions[1]:
            parse_status = 'FALLBACK_FORWARD'
        if decisions.count(decisions[0]) >= 2:
            canonical = decisions[0]
        elif decisions.count(decisions[1]) >= 2:
            canonical = decisions[1]
        else:
            canonical = decisions[0]
            if parse_status == 'VALID':
                parse_status = 'FALLBACK_FORWARD'
        _record_aggregate_pair(
            conn,
            pair_id=pair.pair_id,
            decision=canonical,
            evidence={'normalized_decisions': decisions, 'call_ids': call_ids},
            parse_status=parse_status,
        )
        counts['pairs'] += 1
    return counts


def run_external_judge(
    root: Path,
    *,
    canary: bool = False,
    client: ExternalJudgeClient | None = None,
) -> dict[str, Any]:
    protocol = load_judge_protocol(root)
    if protocol.document.get('judge_mode') != 'external_openai_compatible':
        raise ExternalJudgeRunError('active protocol is not external-only')
    if protocol.document.get('local_judges_enabled') is not False:
        raise ExternalJudgeRunError('local judges must be disabled')
    if client is None:
        client = ExternalJudgeClient(
            ExternalJudgeConfig.from_environment(protocol.document)
        )
    items = load_source_items(root, protocol)
    pending = [item for item in items if item.precheck_status == 'JUDGE_PENDING']
    if canary:
        pending = pending[:3]
        pairs: list[JudgePair] = []
    else:
        real_pairs = select_real_pairs(items, per_task=10)
        pairs = real_pairs + build_controlled_pairs(items, per_task=8)
    protocol_id = str(protocol.document['study'])
    conn = open_judge_db(root / 'results/local/judge-suite-v1.db')
    try:
        protocol_row = conn.execute(
            'SELECT source_population_hash FROM judge_protocols WHERE protocol_id=?',
            (protocol_id,),
        ).fetchone()
        source_count = int(conn.execute(
            'SELECT COUNT(*) FROM judge_source_items WHERE protocol_id=?',
            (protocol_id,),
        ).fetchone()[0])
        if protocol_row is None or source_count != len(items):
            raise ExternalJudgeRunError('judge source preparation is missing or incomplete')
        if protocol_row[0] != source_population_hash(items):
            raise ExternalJudgeRunError('judge source population hash mismatch')
        conn.execute(
            'UPDATE judge_protocols SET status=? WHERE protocol_id=?',
            ('RUNNING', protocol_id),
        )
        conn.commit()
        try:
            rubric_counts = _run_rubric(
                conn, client, protocol_id=protocol_id, protocol=protocol,
                items=pending,
            )
            pair_counts = _run_pairs(
                conn, client, protocol_id=protocol_id, pairs=pairs,
            ) if pairs else {
                'pairs': 0, 'calls': 0, 'third_calls': 0, 'forced': 0, 'errors': 0,
            }
        except Exception:
            conn.execute(
                'UPDATE judge_protocols SET status=? WHERE protocol_id=?',
                ('FAILED', protocol_id),
            )
            conn.commit()
            raise
        run_status = (
            'CANARY_COMPLETE' if canary else 'EXTERNAL_JUDGE_COMPLETE'
        )
        if rubric_counts['errors'] or pair_counts['errors']:
            run_status = f'{run_status}_WITH_ERRORS'
        conn.execute(
            'UPDATE judge_protocols SET status=? WHERE protocol_id=?',
            (run_status, protocol_id),
        )
        conn.commit()
    finally:
        conn.close()
    return {
        'status': run_status,
        'study': 'judge-suite-v1',
        'model_id': client.config.model_id,
        'source_items': len(items),
        'precheck_pass_items': len(pending),
        'calls_made': client.calls_made,
        'total_cost_usd': client.total_cost_usd,
        'rubric': rubric_counts,
        'pairs': pair_counts,
        'human_involvement': False,
        'local_judges_used': False,
        'raw_text_in_public_report': False,
    }


def main(argv: list[str] | None = None) -> int:
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description='Run external Judge Suite')
    parser.add_argument('--canary', action='store_true')
    parser.add_argument('--out')
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args(argv)
    output_name = args.out or (
        'results/reports/judge-suite-v1-external-canary.json'
        if args.canary else 'results/reports/judge-suite-v1-external.json'
    )
    output = root / output_name
    if output.exists() and not args.force:
        print(f'EXTERNAL JUDGE REFUSED: report exists: {output}', flush=True)
        return 2
    try:
        result = run_external_judge(root, canary=args.canary)
    except (OSError, JudgeProtocolError, ExternalJudgeError, ExternalJudgeRunError, ValueError) as exc:
        print(f'EXTERNAL JUDGE REFUSED: {exc}', flush=True)
        return 2
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
