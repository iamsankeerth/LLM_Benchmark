# LLM_Benchmark

A reproducible, fully offline benchmark for small local language models. It measures
**output quality, latency, throughput, and memory** for 11 model configurations across
4 model families and 4 quantization levels, then compares those results under frozen,
hash-pinned conditions.

Everything runs locally through [Ollama](https://ollama.com). No model output is sent to
a third-party API, and no result is hand-written: every published number is copied from a
committed artifact whose dataset, prompt, grading-spec, model, and code hashes are recorded.

**Full technical report:** [`results/reports/benchmark-technical-report.md`](results/reports/benchmark-technical-report.md)

---

## Headline results

11 configurations, corrected `eval-v1.1` grading, 201 deterministic trials per model,
measured on one machine: RTX 2050 with 4 GB VRAM, Ollama 0.34.1, `num_ctx=4096`.

| Configuration | Accuracy | Median decode (tok/s) | Median TTFT (ms) | Peak VRAM (MiB) |
|---|---|---|---|---|
| `qwen3-4b-q5` | **55.22%** | 33.48 | 59.36 | 3626 |
| `qwen3-4b-q4` | 53.23% | 37.94 | 54.40 | 3252 |
| `llama3.2-3b-q6` | 38.81% | 37.70 | 53.71 | 3250 |
| `gemma-3n-e2b-q4` | 35.82% | 49.33 | 167.42 | 1858 |
| `llama3.2-3b-q4` | 35.82% | 47.96 | 48.27 | 2658 |
| `llama3.2-3b-q5` | 34.33% | 42.35 | 51.32 | 2946 |
| `smolm2-1.7b-q4` | 33.33% | **84.81** | **38.05** | 2056 |
| `smolm2-1.7b-q6` | 33.33% | 66.98 | 39.43 | 2388 |
| `gemma-3n-e2b-q5` | 31.34% | 44.13 | 172.78 | 2112 |
| `smolm2-1.7b-q5` | 29.35% | 74.91 | 48.89 | 2218 |
| `smolm2-1.7b-q8` | 26.87% | 54.04 | 44.71 | 2784 |

### What the data actually says

**1. Higher quantization did not reliably improve quality.** Seven paired comparisons were
run using exact McNemar tests. Not one reached significance, and four of the seven moved
accuracy *downward*:

| Comparison | Accuracy delta | McNemar exact p |
|---|---|---|
| `qwen3-4b-q4` → `q5` | +1.96 pp | 1.0 |
| `llama3.2-3b-q4` → `q5` | -0.98 pp | 1.0 |
| `llama3.2-3b-q4` → `q6` | +3.43 pp | 0.375 |
| `gemma-3n-e2b-q4` → `q5` | -4.41 pp | 0.375 |
| `smolm2-1.7b-q4` → `q5` | -2.45 pp | 1.0 |
| `smolm2-1.7b-q4` → `q6` | 0.00 pp | 1.0 |
| `smolm2-1.7b-q4` → `q8` | -4.90 pp | 0.5 |

**2. Higher quantization reliably cost speed and memory.** All seven steps up in
quantization increased peak VRAM and reduced median decode throughput, with no exceptions.
Q8 on SmolLM2 cost +728 MiB and dropped throughput from 84.81 to 54.04 tok/s.

**3. Temperature 0.0 buys repeatability, not proven accuracy.** Over 15 tasks × 5 trials:

| Metric | T = 0.0 | T = 0.7 |
|---|---|---|
| Passes | 29 | 26 |
| Trial accuracy | 38.67% | 34.67% |
| Distinct normalized outputs per task | 1.47 | 3.40 |
| McNemar exact p | 1.0 | |

The quality gap was not statistically significant, so the defensible claim is consistency.
Three trials of a single task accounted for the entire accuracy delta.

**4. Output formatting, not reasoning, is the dominant failure mode.** A human-reviewed
taxonomy of 97 failing `qwen3-4b-q4` rows found `OUTPUT_CONTRACT` was the largest class at
39 rows (40.21%), ahead of `CONTENT_ERROR` at 33. The separate substantive review rated 61
rows `CORRECT` and 6 `LIKELY_CORRECT`, so the strict output contract measurably costs
apparent capability.

---

## Quick start

Requires Python 3.12+ and a local [Ollama](https://ollama.com) installation.

```bash
git clone https://github.com/iamsankeerth/LLM_Benchmark.git
cd LLM_Benchmark

python -m venv .venv
# Windows:   .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate

pip install -r requirements-ci.txt
```

### Run the offline quality gate

This is the only command that needs no models and no GPU. It runs lint, strict type
checking, the full test suite, a workbook consistency check, and four suite audits.

```bash
python scripts/quality_gate.py
```

Expected final line: `QUALITY GATE PASS`

### Verify the sealed evidence

```bash
python scripts/audit_grading_specs.py --json     # grading-spec integrity
python scripts/audit_long_context.py --json      # long-context suite
python scripts/audit_coding_suite.py --json      # coding suite + pinned digests
```

### Reproduce a measurement

```bash
# 1. Start Ollama, then prove the model fits before trusting any number.
ollama serve
python scripts/preflight_context.py --model qwen3-4b-q4

# 2. Run one configuration.
python scripts/run_benchmark.py \
  --config configs/full-baseline-v2.yaml \
  --model qwen3-4b-q4 \
  --db results/local/full-baseline-v2__qwen3-4b-q4.db \
  --resume

# 3. Run the full registry sweep.
python scripts/run_model_sweep.py --experiment full-baseline-v2

# 4. Build the report and charts.
python scripts/generate_sweep_report.py

# 5. Compare two configurations with paired statistics.
python scripts/compare_models.py \
  --db-base results/local/full-baseline-v2__qwen3-4b-q4.db --base full-baseline-v2__qwen3-4b-q4 \
  --db-candidate results/local/full-baseline-v2__qwen3-4b-q5.db --candidate full-baseline-v2__qwen3-4b-q5
```

If `ollama` is not on `PATH`, set `OLLAMA_BIN` to its full path.

### External judge (optional)

The judge suite is the only component that calls a remote API, and it is configured purely
through environment variables so no credential is ever committed:

```bash
export JUDGE_API_BASE_URL="https://<endpoint>/v1"
export JUDGE_API_KEY="<key>"
python scripts/run_external_judge.py
```

---

## Suites

| Suite | Scope | Status |
|---|---|---|
| Baseline V2 (`eval-v1.1`) | 72 tasks, 3 trials, 11 configurations | SEALED |
| Temperature study | 15 tasks × 5 trials, T=0.0 vs 0.7 | SEALED |
| Reliability | 5 repetitions, variance analysis | SEALED |
| Long context V1 | 21 documents, S/M/L lengths, 11 configurations | FULL_SEALED |
| Coding V1 | 8 tasks, 264 rows, pinned OCI workers | LIVE_SEALED |
| External judge | 43 rubric items, 54 pairs, 164 calls | EXTERNAL_FULL_COMPLETE |

Release index: [`results/reports/core-release-v1.1-freeze.json`](results/reports/core-release-v1.1-freeze.json)

---

## Repository layout

```
analysis/     Metric derivation, paired statistics, failure taxonomy, temperature study
configs/      Frozen experiment contracts, model registry, per-model prompt adapters
evals/        Datasets, grading specs + freeze files, grader engine
inference/    Streaming Ollama client, profiler, retry policy, adapter rendering
storage/      SQLite schema, manifests, resume/provenance logic
scripts/      Runners and report generators (all CLI entry points)
tests/        373 offline unit tests, no network or GPU required
results/      Sealed reports, summaries, charts, and committed evidence databases
docs/         Design specifications
```

### Key entry points

| Path | Purpose |
|---|---|
| `inference/ollama_client.py` | Streaming client; always streams so TTFT is observable |
| `inference/profiler.py` | The single home of every derived metric formula |
| `inference/retry.py` | Transport-only fault classification and bounded retry |
| `evals/specs/eval-v1.1-grading.yaml` | Current frozen grading contract |
| `coding_worker/` | Pinned OCI workers for isolated code execution |
| `.github/workflows/quality.yml` | CI running the offline gate on Linux and Windows |

---

## Engineering approach

**Every number is traceable.** Each report is paired with a freeze file recording the dataset
hash, prompt-template hash, grading-spec hash, model artifact digest, runner code hash, and
the git commit. A claim cannot be published without the evidence needed to reproduce it.

**Unmeasurable is not estimated.** Missing server counters produce `null`, never a guess.
`null` values never enter denominators or ranking frontiers, and only one pooled decode
figure would be misleading, so per-token-bucket distributions are stored.

**Statistical honesty over marketing.** The headline quantization result is a *negative*
result: no significant quality difference. Reporting it as a win would be unsupported by the
data.

**Fail closed.** Semantic faults, such as config, hash, or dataset mismatches, stop
immediately rather than retrying. Retries apply only to transport faults, never change
benchmark configuration, and never persist partial output.

**Methodology incidents are first-class artifacts.** Six incidents are recorded with cause,
regression fixture, and the durable rule adopted, including an eligibility bug that condemned
a model for a misconfigured load, and a deletion bug where `ollama list` and `ollama ps`
disagreed. See section 9 of the technical report.

---

## Limitations

Read these before citing any number from this repository.

1. Quantization quality differences were **not** statistically distinguishable here.
2. **Recorded artifact digests verify on Windows only.** The digests stored in the sealed
   freeze files were captured from a Windows checkout, where git materializes CRLF line
   endings. A Linux or macOS checkout materializes LF for the same committed content, so
   the byte-level sha256 differs. The 6 tests that consume those digests are therefore
   scoped to Windows and report as **skipped** on POSIX, not as passing. This is a real
   limitation of the evidence trail, not a platform quirk: a reader on Linux cannot
   re-verify those digests from a fresh clone. See `tests/platform_scope.py`.
3. **7 tests skip on a fresh clone.** Besides the digest tests above,
   `test_real_pair_selector_handles_small_eligible_groups` needs one sealed database per
   completed model under `results/local/`, which project policy keeps out of git. It skips
   explicitly rather than erroring.
4. Temperature-0 repeatability holds for this benchmark, hardware, and runtime
   configuration only. No universal determinism claim is made.
5. Single machine, RTX 2050 / 4 GB VRAM. Feasibility findings are hardware-specific by
   construction. Three Phi-3.5-mini configurations were excluded with recorded evidence
   because full GPU residency alone did not guarantee runtime headroom.
6. `READY_JUDGE` tasks are excluded from all capability denominators.
7. Strict substring matchers for terms, dates, and units punish paraphrase, so contract
   accuracy understates reasoning capability by design.
8. Latency metrics are descriptive only, not causal accuracy evidence.
9. Three Phi configurations and the long-context M/L lengths scored 0.0; those zeros are
   reported as measured, not explained away.

---

## Scope

This is a **benchmark system**, not an end-user assistant application. It deliberately does
not include an interactive chat interface, a serving API, Pydantic response validation, a
JSON repair loop, RAG or retrieval, fine-tuning, or real-time multimodal pipelines. Section
11 of the technical report documents each boundary and what evidence the repository does
provide for such a system.

---

## Development

```bash
pip install -r requirements-ci.txt

python -m ruff check .
python -m mypy analysis evals inference scripts storage tests
python -m unittest discover -s tests -p "test_*.py"
```

The lint and type-check toolchain is pinned in `requirements-ci.txt` to match
`requirements-dev.lock`, so CI reproduces the recorded local toolchain exactly.

CI runs the full offline quality gate on both `ubuntu-latest` and `windows-latest`, so a
platform-specific regression fails the build. Note that the Windows-only digest assertions
report as *skipped* on Linux; the Linux job validates lint, types, functional behaviour, and
the suite audits, while the recorded-digest verification runs on the Windows job.

---

## License

No license file is currently present. All rights are reserved by default until one is added.
