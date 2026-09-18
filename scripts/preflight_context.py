"""Per-model context preflight: prove prompt headroom before benchmarking.

Renders all 72 mandatory prompts through the model's pinned adapter and
runs one num_predict=1 probe per task (in-memory only; nothing is written
to the runs table). Requires max_prompt_eval_count + num_predict <= num_ctx
for the SAME model/template whose counts were observed — Qwen evidence
never substitutes for another family.

Writes results/summaries/prompt-tokens-v2-<model>.json and exits nonzero
when the headroom gate fails. The runner's runtime guard remains as a
last-line invariant only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml

from inference.adapters import get_model_config, render_prompt
from inference.ollama_client import GenerationRequest, OllamaClientError, generate

NUM_CTX = 4096


def load_mandatory_prompts(root: Path) -> dict[str, str]:
    prompts: dict[str, str] = {}
    with open(
        root / 'evals/datasets/eval-v1/executable-v1.jsonl', encoding='utf-8'
    ) as handle:
        for line in handle:
            line = line.strip()
            if line:
                row = json.loads(line)
                prompts[str(row['id'])] = str(row['prompt'])
    return prompts


def probe_prompt_tokens(
    base_url: str,
    model_identifier: str,
    rendered: str,
    *,
    raw: bool,
    think: bool | None,
    timeout_s: float = 120.0,
) -> int:
    """One num_predict=1 probe of the RENDERED benchmark prompt.

    Uses the adapter's own mode (raw for Qwen): this measures the exact
    prompt the benchmark will send, not the default chat route (route
    selection is a separate derivation-stage decision).
    """
    result = generate(
        base_url,
        GenerationRequest(
            model=model_identifier, prompt=rendered, temperature=0.0,
            num_ctx=NUM_CTX, num_predict=1, raw=raw, think=think,
        ),
        timeout_s=timeout_s,
    )
    if result.prompt_eval_count is None:
        raise OllamaClientError('probe returned no prompt_eval_count')
    return result.prompt_eval_count


def run_preflight(
    base_url: str, model_config_id: str, num_predict: int, *, timeout_s: float = 120.0
) -> dict[str, object]:
    """Render + probe all 72 prompts; return the artifact document (no DB)."""
    root = Path(__file__).resolve().parent.parent
    config = get_model_config(model_config_id)
    prompts = load_mandatory_prompts(root)
    if len(prompts) != 80:
        raise ValueError(f'expected 80 executable tasks, found {len(prompts)}')
    spec = yaml.safe_load(open(root / 'evals/specs/eval-v1-grading.yaml', encoding='utf-8'))
    mandatory = sorted(
        tid for tid, entry in spec['tasks'].items()
        if entry['grading_status'] in ('READY_DETERMINISTIC', 'READY_JUDGE')
    )
    if len(mandatory) != 72:
        raise ValueError(f'expected 72 mandatory tasks, found {len(mandatory)}')
    counts: dict[str, int] = {}
    use_raw = config.mode == 'raw'
    for task_id in mandatory:
        rendered = render_prompt(config, prompts[task_id])
        counts[task_id] = probe_prompt_tokens(
            base_url, config.ollama_identifier, rendered,
            raw=use_raw, think=config.think, timeout_s=timeout_s,
        )
        print(f'{task_id}: {counts[task_id]} prompt tokens', flush=True)
    worst_task = max(counts, key=lambda tid: counts[tid])
    worst_count = counts[worst_task]
    headroom = NUM_CTX - worst_count - num_predict
    return {
        'model_config_id': model_config_id,
        'template_sha256': config.template_sha256,
        'num_ctx': NUM_CTX,
        'num_predict': num_predict,
        'task_prompt_tokens': counts,
        'max_prompt_eval_count': worst_count,
        'max_task_id': worst_task,
        'headroom_tokens': headroom,
        'preflight_pass': headroom >= 0,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Context preflight probe')
    parser.add_argument('--model', required=True, help='model config id')
    parser.add_argument('--num-predict', type=int, default=2048)
    parser.add_argument('--base-url', default='http://127.0.0.1:11434')
    parser.add_argument('--out', default=None)
    parser.add_argument('--timeout-s', type=float, default=120.0)
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parent.parent
    try:
        document = run_preflight(
            args.base_url, args.model, args.num_predict, timeout_s=args.timeout_s
        )
    except (OllamaClientError, ValueError) as exc:
        print(f'PREFLIGHT ERROR: {exc}', flush=True)
        return 2
    out_path = (
        Path(args.out)
        if args.out
        else root / 'results/summaries' / f'prompt-tokens-v2-{args.model}.json'
    )
    out_path.write_text(json.dumps(document, indent=2) + '\n', encoding='utf-8')
    print(
        f"preflight {args.model}: max={document['max_prompt_eval_count']} "
        f"({document['max_task_id']}) headroom={document['headroom_tokens']} "
        f"pass={document['preflight_pass']}",
        flush=True,
    )
    print(f'artifact: {out_path}')
    return 0 if document['preflight_pass'] else 3


if __name__ == '__main__':
    raise SystemExit(main())
