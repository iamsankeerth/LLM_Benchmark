"""Generate the canonical V2 sweep report (JSON canonical, MD/PNG derived)."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import yaml

from analysis.cross_model import (
    LIMITATIONS,
    METHODOLOGY_INCIDENTS,
    NEXT_EXPERIMENTS,
    build_registry_row,
    eligible_rows,
    pareto_skyline,
    summarize_quant_pairs,
)

REPORT_DEPS = {
    'matplotlib': '3.10.9',
    'pandas': '2.2.3',
    'numpy': '2.2.3',
    'pillow': '11.1.0',
}


def _read_json(path: Path) -> dict[str, object]:
    raw = json.loads(path.read_text(encoding='utf-8'))
    assert isinstance(raw, dict)
    return raw


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _execution_id(spec: str, model: str) -> str:
    return f'{spec}__{model}'


def pareto_plot(
    rows: list[dict[str, object]],
    *,
    x_key: str,
    y_key: str,
    y_bigger_better: bool,
    x_label: str,
    y_label: str,
    title: str,
    out_path: Path,
) -> None:
    points = eligible_rows(rows)
    members = set(
        pareto_skyline(points, x_key=x_key, y_key=y_key,
                       y_bigger_better=y_bigger_better)
    )
    figure, axes = plt.subplots(figsize=(10, 6))
    for row in points:
        mid = str(row['model_config_id'])
        axes.scatter([row[x_key]], [row[y_key]],
                     s=120 if mid in members else 45,
                     marker='*' if mid in members else 'o')
        axes.annotate(mid, (float(str(row[x_key])), float(str(row[y_key]))),
                      fontsize=7)
    axes.set_xlabel(x_label)
    axes.set_ylabel(y_label)
    axes.set_title(title + '  (* = Pareto skyline; ties co-members)')
    axes.grid(True, alpha=0.3)
    figure.tight_layout()
    figure.savefig(out_path, dpi=150)
    plt.close(figure)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Generate V2 sweep report')
    parser.add_argument('--registry', default='configs/models-v2.yaml')
    parser.add_argument('--spec', default='full-baseline-v2')
    parser.add_argument('--out-json', default=None)
    parser.add_argument('--out-md', default=None)
    parser.add_argument('--plots-dir', default=None)
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parent.parent
    summaries = root / 'results/summaries'
    manifests = root / 'results/experiment-manifests'
    registry = yaml.safe_load(
        (root / args.registry).read_text(encoding='utf-8')
    )

    import pandas  # noqa: E402  (locked version, provenance-recorded)

    dep_versions = {
        'matplotlib': matplotlib.__version__,
        'pandas': pandas.__version__,
    }
    import numpy

    dep_versions['numpy'] = numpy.__version__
    for name, pinned in REPORT_DEPS.items():
        actual = dep_versions.get(name)
        if actual is not None and actual != pinned:
            print(f'DEP MISMATCH: {name} {actual} != locked {pinned}')
            return 2

    rows: list[dict[str, Any]] = []
    hashed_paths: list[str] = [
        'evals/datasets/eval-v1/executable-v1.jsonl',
        'evals/specs/eval-v1-grading.yaml',
        'evals/specs/eval-v1-grading.freeze.json',
        f'configs/{Path(args.registry).name}',
        'configs/full-baseline-v2.yaml',
        'requirements-dev.lock',
    ]
    for entry in registry.get('models', []):
        mid = str(entry['model_config_id'])
        exe = _execution_id(args.spec, mid)
        ineligible_path = summaries / f'ineligible-{mid}.json'
        if ineligible_path.exists():
            evidence = _read_json(ineligible_path)
            evidence['artifact_path'] = str(ineligible_path.relative_to(root))
            hashed_paths.append(f'results/summaries/ineligible-{mid}.json')
            rows.append(build_registry_row(
                model_config_id=mid, family=str(entry.get('family', '')),
                quantization=str(entry.get('quantization', '')),
                capability=None, performance=None, failure_modes=None,
                manifest=None, ineligible_evidence=evidence,
            ))
            continue
        capability = _read_json(summaries / f'{exe}-capability.json')
        performance = _read_json(summaries / f'{exe}-performance.json')
        failure_modes = _read_json(summaries / f'{exe}-failure-modes.json')
        manifest = _read_json(manifests / f'{exe}.json')
        rows.append(build_registry_row(
            model_config_id=mid, family=str(entry.get('family', '')),
            quantization=str(entry.get('quantization', '')),
            capability=capability, performance=performance,
            failure_modes=failure_modes, manifest=manifest,
            ineligible_evidence=None,
        ))
        hashed_paths += [
            f'results/summaries/{exe}-capability.json',
            f'results/summaries/{exe}-performance.json',
            f'results/summaries/{exe}-failure-modes.json',
            f'results/experiment-manifests/{exe}.json',
        ]

    comparisons = {}
    for path in sorted(summaries.glob('*-vs-*.json')):
        document = _read_json(path)
        # Lineage guard: V1 comparisons (e.g. qwen-q4-vs-q5.json) must never
        # leak into the V2 report even though the filename pattern matches.
        if document.get('experiment_spec_id') != args.spec:
            print(f'skip non-{args.spec} comparison: {path.name}', flush=True)
            continue
        comparisons[path.stem] = document
        hashed_paths.append(f'results/summaries/{path.name}')
    quant_pairs = summarize_quant_pairs(comparisons)

    eligible = eligible_rows(rows)
    skyline_speed = pareto_skyline(
        eligible, x_key='deterministic_trial_accuracy',
        y_key='decode_median_tok_s', y_bigger_better=True,
    )
    skyline_vram = pareto_skyline(
        eligible, x_key='deterministic_trial_accuracy',
        y_key='peak_vram_mib', y_bigger_better=False,
    )

    plots_dir = Path(args.plots_dir) if args.plots_dir else (
        root / 'results/reports'
    )
    plots_dir.mkdir(parents=True, exist_ok=True)
    speed_png = plots_dir / 'pareto-accuracy-speed.png'
    vram_png = plots_dir / 'pareto-accuracy-vram.png'
    pareto_plot(
        rows, x_key='deterministic_trial_accuracy',
        y_key='decode_median_tok_s', y_bigger_better=True,
        x_label='deterministic trial accuracy',
        y_label='median decode tok/s',
        title='V2 accuracy vs speed (68 det tasks x 3, T=0)',
        out_path=speed_png,
    )
    pareto_plot(
        rows, x_key='deterministic_trial_accuracy', y_key='peak_vram_mib',
        y_bigger_better=False, x_label='deterministic trial accuracy',
        y_label='peak VRAM MiB',
        title='V2 accuracy vs VRAM (RTX 2050 4 GB)',
        out_path=vram_png,
    )

    import subprocess

    report_commit = "1d63305fadd416387f218ad8ba868e85a1345382"  # V2 sealed analytical lineage
    freeze_commit = subprocess.run(
        ['git', 'rev-parse', 'HEAD'], cwd=root, capture_output=True,
        text=True, timeout=15,
    ).stdout.strip()
    document = {
        'title': 'LocalLLM Lab V2 sweep report (canonical analysis artifact)',
        'experiment_spec_id': args.spec,
        'registry_table_14_rows': rows,
        'quant_pairs': quant_pairs,
        'pareto_accuracy_speed': skyline_speed,
        'pareto_accuracy_vram': skyline_vram,
        'methodology_incidents': METHODOLOGY_INCIDENTS,
        'limitations': LIMITATIONS,
        'next_experiments': NEXT_EXPERIMENTS,
        'human_substantive_review': {
            'qwen3-4b-q4': 'available / APPROVED '
                           '(results/reports/qwen3-4b-q4-failure-analysis.json)',
            'other_configs': 'NOT_REVIEWED (automatic taxonomy only)',
        },
        'licensed_claims': [
            'Observed quantization differences within the tested paired '
            'deterministic tasks were small and were not statistically '
            'distinguishable under the exact McNemar tests used here.',
            'Temperature-0 inference showed high repeatability under this '
            'benchmark, hardware, and runtime configuration.',
        ],
        'provenance': {
            'report_dependencies': dep_versions,
            'analysis_code_git_commit': report_commit,
        },
    }
    out_json = Path(args.out_json) if args.out_json else (
        root / 'results/reports/sweep-v2-report.json'
    )
    out_json.write_text(json.dumps(document, indent=2) + '\n', encoding='utf-8')

    lines = [
        '# V2 sweep report: 14 configurations attempted, 11 benchmarked',
        '',
        'Three Phi configs preserved as INELIGIBLE_RUNTIME_HEADROOM with '
        'evidence refs; null metrics never enter denominators or skylines.',
        '',
        '| model | acc | any3 | all3 | dec_med | dec_p95 | vram | status |',
        '|---|---|---|---|---|---|---|---|',
    ]

    def _fmt(value: object, digits: int = 3) -> str:
        return f'{value:.{digits}f}' if isinstance(value, float) else 'n/a'

    for row in rows:
        lines.append(
            f"| {row['model_config_id']} | {_fmt(row['deterministic_trial_accuracy'])} | "
            f"{_fmt(row['any_pass_3_rate'])} | {_fmt(row['all_pass_3_rate'])} | "
            f"{_fmt(row['decode_median_tok_s'], 1)} | {_fmt(row['decode_p95_tok_s'], 1)} | "
            f"{_fmt(row['peak_vram_mib'], 0)} | {row['eligibility_status']} |"
        )
    lines += [
        '',
        '## Licensed claims',
        '',
    ]
    claims_raw = document.get('licensed_claims', [])
    claims: list[object] = claims_raw if isinstance(claims_raw, list) else []
    for claim in claims:
        lines.append(f'- {claim}')
    lines += [
        '',
        '## Limitations',
        '',
    ]
    limitations_raw = document.get('limitations', LIMITATIONS)
    limitations: list[object] = (
        limitations_raw if isinstance(limitations_raw, list) else []
    )
    for limitation in limitations:
        lines.append(f'- {limitation}')
    out_md = Path(args.out_md) if args.out_md else (
        root / 'results/reports/sweep-v2-report.md'
    )
    out_md.write_text('\n'.join(lines) + '\n', encoding='utf-8')

    freeze_paths = sorted(set(hashed_paths + [
        'results/reports/sweep-v2-report.json',
        'results/reports/sweep-v2-report.md',
        'results/reports/pareto-accuracy-speed.png',
        'results/reports/pareto-accuracy-vram.png',
        'results/reports/qwen3-4b-q4-failure-analysis.json',
        'analysis/reviews/qwen-q4-v2-failure-review.yaml',
    ]))
    freeze = {
        'freeze_id': 'full-baseline-v2',
        'analysis_code_git_commit': freeze_commit,
        'artifacts': {
            path: hashlib.sha256(
                (root / path).read_bytes()).hexdigest()
            for path in freeze_paths
        },
    }
    freeze_path = root / 'results/reports/sweep-v2-freeze.json'
    freeze_path.write_text(json.dumps(freeze, indent=2) + '\n', encoding='utf-8')
    print(f'wrote {out_json}', flush=True)
    print(f'wrote {out_md}', flush=True)
    print(f'wrote {speed_png} {vram_png}', flush=True)
    print(f'wrote {freeze_path} ({len(freeze["artifacts"])} artifacts)', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
