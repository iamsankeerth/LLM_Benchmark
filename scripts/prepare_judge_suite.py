"""Prepare the external-only Judge Suite V1 source population."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis.judge import JudgeProtocolError, load_judge_protocol, load_source_items
from storage.judge import (
    open_judge_db,
    record_protocol,
    record_source_item,
    source_population_hash,
)


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def _hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(root: Path, *, force: bool = False) -> dict[str, Any]:
    protocol = load_judge_protocol(root)
    items = load_source_items(root, protocol)
    population_hash = source_population_hash(items)
    rubric_hash = _hash_file(root / 'evals/specs/judge-suite-v1-rubrics.yaml')
    db_path = root / 'results/local/judge-suite-v1.db'
    conn = open_judge_db(db_path)
    try:
        record_protocol(
            conn, protocol=protocol, population_hash=population_hash,
            rubric_hash=rubric_hash, created_at_utc=_utcnow(),
        )
        for item in items:
            record_source_item(
                conn, protocol_id=str(protocol.document['study']), item=item,
            )
        existing = conn.execute(
            'SELECT COUNT(*) FROM judge_source_items WHERE protocol_id=?',
            (str(protocol.document['study']),),
        ).fetchone()[0]
    finally:
        conn.close()
    if int(existing) != len(items):
        raise JudgeProtocolError('judge source persistence count mismatch')
    report_path = root / 'results/reports/judge-suite-v1.json'
    if report_path.exists() and not force:
        raise JudgeProtocolError(f'refusing to overwrite {report_path}')
    precheck_counts: dict[str, int] = {}
    for item in items:
        precheck_counts[item.precheck_status] = precheck_counts.get(item.precheck_status, 0) + 1
    report = {
        'schema_version': 'judge-suite-report-v1',
        'study': 'judge-suite-v1',
        'status': 'PREPARED_EXTERNAL_ONLY',
        'protocol_version': protocol.document['protocol_version'],
        'source_population_hash': population_hash,
        'rubric_sha256': rubric_hash,
        'eval_spec_sha256': protocol.contract.hashes['spec_sha256'],
        'dataset_sha256': protocol.contract.hashes['dataset_sha256'],
        'source_items': len(items),
        'precheck_counts': precheck_counts,
        'judge_mode': protocol.document['judge_mode'],
        'external_judge': protocol.document['external_judge'],
        'local_judges_enabled': protocol.document['local_judges_enabled'],
        'human_involvement': False,
        'bias_pairs': protocol.document['bias_pairs'],
        'human_adjudication': protocol.document['human_adjudication'],
        'model_calls': 0,
        'raw_text_in_public_report': False,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    return report


def main(argv: list[str] | None = None) -> int:
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description='Prepare Judge Suite V1')
    parser.add_argument('--force', action='store_true', help='overwrite local preparation outputs')
    args = parser.parse_args(argv)
    try:
        report = prepare(root, force=args.force)
    except (OSError, JudgeProtocolError, ValueError) as exc:
        print(f'JUDGE PREPARE REFUSED: {exc}', flush=True)
        return 2
    print(f"prepared {report['source_items']} judge source items")
    print('model calls: 0; external-only API execution is not started by preparation')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
