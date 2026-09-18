"""Results path resolution with test hermeticity override.

Production code must never hard-code the results tree: every writer goes
through results_path(), which honors LLM_BENCH_RESULTS_ROOT when set
(tests point it at a temp dir). Without the variable, paths resolve under
the repo root exactly as before. This keeps full-gate test runs from
rewriting committed release artifacts.
"""

from __future__ import annotations

import os
from pathlib import Path


def results_root(repo_root: str | Path) -> Path:
    override = os.environ.get('LLM_BENCH_RESULTS_ROOT')
    if override:
        return Path(override)
    return Path(repo_root) / 'results'


def results_path(repo_root: str | Path, *parts: str) -> Path:
    return results_root(repo_root).joinpath(*parts)
