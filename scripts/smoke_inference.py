"""3-task inference smoke verifier: runs smoke-3, then asserts the full path.

Checks raw output, timing fields, token counts, grader result, RAM/VRAM
capture and database persistence for one exact, one structured and one
constraint task.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.run_benchmark import run_experiment

EXPECTED_TASKS = ('Q071', 'Q001', 'Q011')


def verify(db_path: str, experiment_id: str = 'smoke-3') -> None:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        'SELECT * FROM runs WHERE experiment_id=? AND is_warmup=0 ORDER BY task_id',
        (experiment_id,),
    ).fetchall()
    found = [row['task_id'] for row in rows]
    assert found == sorted(EXPECTED_TASKS), f'tasks mismatch: {found}'
    for row in rows:
        task = row['task_id']
        assert row['status'] == 'COMPLETE', f'{task}: status {row["status"]}'
        assert row['raw_output'], f'{task}: empty raw output'
        assert row['ttft_ms'] is not None and row['ttft_ms'] > 0, f'{task}: TTFT missing'
        assert row['client_e2e_ms'] is not None and row['client_e2e_ms'] > 0
        assert row['eval_count'] is not None and row['eval_count'] > 0, (
            f'{task}: eval_count missing'
        )
        assert row['prompt_eval_count'] is not None and row['prompt_eval_count'] > 0
        assert row['decode_tok_s'] is not None and row['decode_tok_s'] > 0
        assert row['prefill_compute_tok_s'] is not None, f'{task}: prefill missing'
        details = json.loads(row['grader_details_json'])
        assert details, f'{task}: grader produced no results'
        assert row['grader_verdict'] in ('PASS', 'FAIL', 'NEEDS_JUDGE'), (
            f'{task}: unexpected verdict {row["grader_verdict"]}'
        )
        assert row['ram_baseline_mb'] is not None and row['ram_peak_mb'] is not None
        assert row['vram_baseline_mib'] is not None, f'{task}: VRAM capture missing'
        assert row['vram_peak_mib'] is not None, f'{task}: VRAM peak missing'
        assert row['eligibility_status'] == 'ELIGIBLE_GPU', (
            f'{task}: {row["eligibility_status"]}'
        )
        assert row['rendered_prompt_sha256'] and len(row['rendered_prompt_sha256']) == 64
        print(
            f"{task}: verdict={row['grader_verdict']} "
            f"chars={len(row['raw_output'])} "
            f"ttft={row['ttft_ms']:.0f}ms "
            f"decode={row['decode_tok_s']:.1f}tok/s "
            f"vram_peak={row['vram_peak_mib']:.0f}MiB"
        )
    warmups = conn.execute(
        "SELECT task_id, trial FROM runs WHERE experiment_id=? AND run_kind='WARMUP'"
        ' ORDER BY trial',
        (experiment_id,),
    ).fetchall()
    # 2 warmups per invocation with continuing trial numbers (resume-safe).
    assert len(warmups) >= 2 and len(warmups) % 2 == 0, (
        f'expected 2 warmups per invocation, found {len(warmups)}'
    )
    assert [w['trial'] for w in warmups] == list(range(1, len(warmups) + 1)), (
        'warmup trials must continue without gaps or collisions'
    )
    measured = conn.execute(
        'SELECT COUNT(*) FROM runs WHERE experiment_id=? AND is_warmup=0',
        (experiment_id,),
    ).fetchone()[0]
    assert measured == 3
    conn.close()
    print('SMOKE PASS: dataset -> inference -> profiler -> grader -> SQLite')


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Run + verify the 3-task smoke')
    parser.add_argument('--db', required=True)
    parser.add_argument('--base-url', default='http://127.0.0.1:11434')
    parser.add_argument('--model', default='qwen3-4b-q4')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parent.parent
    run_args = argparse.Namespace(
        config=str(root / 'configs/smoke-3.yaml'),
        model=args.model,
        db=args.db,
        base_url=args.base_url,
        resume=args.resume,
        num_predict=512,
        timeout_s=300.0,
        max_tasks=None,
    )
    code = run_experiment(run_args)
    if code != 0:
        return code
    verify(args.db)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
