# Local LLM Benchmark: Technical Report

Status: complete for the benchmarking scope. Every figure below is copied from a
committed, hash-pinned artifact. No number in this document is estimated.

- Repository: `LLM_Benchmark`
- Release: `core-v1.1`
- Final commit: `5ea3a0b`
- Grading specification: `eval-v1.1`
- Quality gate: 373 tests passing, Ruff PASS, mypy PASS

---

## 1. Executive Summary

A reproducible benchmark was built to measure small local language models running
entirely offline through Ollama. The system measures output quality, latency,
throughput, and memory for 11 model configurations across 4 model families and 4
quantization levels, then compares those results under controlled conditions.

Headline findings:

1. **Qwen3-4B is the strongest configuration on this contract**, scoring 53.23%
   deterministic trial accuracy under the corrected `eval-v1.1` grading, ahead of
   Llama 3.2 3B Q6 at 38.81%.
2. **Higher quantization did not reliably improve quality.** Seven paired
   quantization comparisons produced no statistically significant result under
   exact McNemar testing. Four of the seven moved accuracy downward.
3. **Higher quantization reliably cost speed and memory.** Every step up in
   quantization increased peak VRAM and reduced median decode throughput.
4. **SmolLM2 1.7B Q4 is the throughput leader** at 84.81 median tokens/second with
   38.05 ms median time to first token.
5. **Temperature 0.0 was more reliable than 0.7.** At temperature 0.0 the model
   produced 1.47 distinct normalized outputs per task versus 3.40 at 0.7, with
   slightly higher accuracy (38.67% vs 34.67%).
6. **Three Phi-3.5-mini configurations were excluded** with recorded evidence
   rather than silently dropped, because full GPU residency alone did not
   guarantee sufficient runtime headroom.

---

## 2. Local Inference System

### 2.1 Client

`inference/ollama_client.py` implements a streaming Ollama client.

- Requests always use `stream: true`, because time to first token is not
  observable on the non-streaming route.
- The client sends caller-rendered prompt text only. It never applies prompt
  templating itself; templates are resolved by `inference/adapters.py` from
  committed adapter overlays.
- Timing uses `time.perf_counter_ns`, a monotonic clock, so latency measurements
  are not distorted by wall-clock adjustments.
- Incremental UTF-8 decoding is used so the first NDJSON chunk is observed
  immediately rather than waiting for a buffer to fill. This preserves TTFT
  fidelity.
- Transport faults are typed and separated: HTTP status errors, timeouts,
  malformed NDJSON lines, and truncated streams each raise a distinct exception.
  Semantic faults are never retried.

### 2.2 Benchmark runner

`scripts/run_benchmark.py` is the measurement harness.

Pipeline per trial: frozen task prompt, frozen grading-spec entry, model adapter
rendering, streamed generation, metric derivation, grading, then SQLite
persistence. The runner commits after every generation so that an interrupted
sweep loses at most one trial. `--resume` skips exactly the completed
`(model, task, trial, config-hash)` tuples, so resuming cannot silently change
the measured population.

### 2.3 Measurement definitions

All derived metrics live in one function, `derive_metrics` in
`inference/profiler.py`. There are no duplicate formulas elsewhere in the
codebase.

| Metric | Definition |
|---|---|
| `ttft_ms` | Client-side: `first_token_ns - request_start_ns` |
| `client_e2e_ms` | Client-side: `request_end_ns - request_start_ns` |
| `decode_tok_s` | `eval_count / eval_duration` |
| `decode_ms_per_token` | `eval_duration_ms / eval_count` |
| `prefill_compute_tok_s` | Uncached prompt tokens / `prompt_eval_duration` |
| `prompt_cache_ratio` | `prompt_eval_cached_count / prompt_eval_count` |
| `client_overhead_ms` | `client_e2e_ms - server_total_duration_ms` |

Design decision: the naive `prompt_eval_count / prompt_eval_duration` ratio is
deliberately **not** computed, because the server duration covers uncached
tokens only. Computing it would overstate prefill throughput.

Unmeasurable quantities become `null`. They are never replaced with estimates,
and `null` metrics never enter denominators or ranking frontiers.

### 2.4 Sealed artifacts

Every report is paired with a freeze file recording dataset hashes, grading-spec
hashes, model artifact digests, runner code hashes, and the git commit used.
Examples: `results/reports/temperature-study-v1-freeze.json`,
`results/reports/coding-v1-full-freeze.json`,
`evals/specs/eval-v1.1-grading.freeze.json`.

---

## 3. Benchmark Methodology

### 3.1 Hardware and runtime

All measurements were taken on a single machine, so results are hardware-specific
by construction.

| Property | Value |
|---|---|
| Hardware ID | `Windows-AMD64-AMD64 Family 25 Model 80 Stepping 0, AuthenticAMD` |
| GPU | RTX 2050, 4 GB VRAM |
| Ollama version | 0.34.1 |
| Python version | 3.13.5 |
| Power mode | Balanced |
| Context window | 4096 |
| Output ceiling | 2048 tokens |
| Temperature | 0.0 for baseline runs |

### 3.2 Contract

`configs/full-baseline-v2.yaml` defines the canonical cross-model contract.

- 72 mandatory tasks: 68 `READY_DETERMINISTIC` plus 4 `READY_JUDGE`.
- 3 trials per task, giving 204 deterministic trials per configuration.
- Temperature 0.0, 2 warm-up generations excluded from measurement.
- Uniform `num_predict=2048` ceiling inside `num_ctx=4096`.
- A per-model preflight proves `max_prompt_tokens + 2048 <= 4096` before any
  measured generation, preventing silent context truncation.

The V1 lineage is archived read-only under a separate tag. No V1 row, review, or
summary is reused in V2.

### 3.3 Model configurations

14 configurations were attempted; 11 were benchmarked. Each configuration is
pinned to a model artifact digest.

| Configuration | Family | Quantization |
|---|---|---|
| `qwen3-4b-q4` | Qwen3 4B | Q4_K_M |
| `qwen3-4b-q5` | Qwen3 4B | Q5_K_M |
| `llama3.2-3b-q4` | Llama 3.2 3B | Q4_K_M |
| `llama3.2-3b-q5` | Llama 3.2 3B | Q5_K_M |
| `llama3.2-3b-q6` | Llama 3.2 3B | Q6_K |
| `gemma-3n-e2b-q4` | Gemma 3n E2B | Q4_K_M |
| `gemma-3n-e2b-q5` | Gemma 3n E2B | Q5_K_M |
| `smolm2-1.7b-q4` | SmolLM2 1.7B | Q4_K_M |
| `smolm2-1.7b-q5` | SmolLM2 1.7B | Q5_K_M |
| `smolm2-1.7b-q6` | SmolLM2 1.7B | Q6_K |
| `smolm2-1.7b-q8` | SmolLM2 1.7B | Q8_0 |

Excluded with evidence: `phi-3.5-mini-q4`, `phi-3.5-mini-q5`,
`phi-3.5-mini-q6`, all `COMPLETE_INELIGIBLE_RUNTIME_HEADROOM`.

### 3.4 Eligibility

A configuration is eligible only if it satisfies **both** conditions:

1. Full GPU residency, verified through `/api/ps`.
2. A successful operational canary generation.

The second condition exists because of a recorded incident. Phi Q4 was 100%
resident yet aborted a sustained generation with roughly 95 MiB free. Residency
is necessary but not sufficient. The rule was amended only after the failure was
observed and documented in `results/incidents/wrong-load-eligibility/incident.json`.

---

## 4. Performance Metrics

Corrected `eval-v1.1` figures, 67 deterministic tasks, 201 trials per model.
Source: `results/reports/eval-v1.1-companion-report.json`.

| Configuration | Accuracy | Median decode tok/s | Median TTFT (ms) | Peak VRAM (MiB) | Size (MiB) |
|---|---|---|---|---|---|
| `qwen3-4b-q5` | 55.22% | 33.48 | 59.36 | 3626 | 3241 |
| `qwen3-4b-q4` | 53.23% | 37.94 | 54.40 | 3252 | 2849 |
| `llama3.2-3b-q6` | 38.81% | 37.70 | 53.71 | 3250 | 2974 |
| `gemma-3n-e2b-q4` | 35.82% | 49.33 | 167.42 | 1858 | 1556 |
| `llama3.2-3b-q4` | 35.82% | 47.96 | 48.27 | 2658 | 2041 |
| `llama3.2-3b-q5` | 34.33% | 42.35 | 51.32 | 2946 | 2656 |
| `smolm2-1.7b-q4` | 33.33% | 84.81 | 38.05 | 2056 | 1162 |
| `smolm2-1.7b-q6` | 33.33% | 66.98 | 39.43 | 2388 | 1496 |
| `gemma-3n-e2b-q5` | 31.34% | 44.13 | 172.78 | 2112 | 1812 |
| `smolm2-1.7b-q5` | 29.35% | 74.91 | 48.89 | 2218 | 1324 |
| `smolm2-1.7b-q8` | 26.87% | 54.04 | 44.71 | 2784 | 1891 |

### 4.1 Latency distribution

Averages were deliberately avoided as the primary statistic, because averages
hide worst-case behavior. Both median and p95 are recorded for every
configuration. Example for `qwen3-4b-q4` over 216 measured rows:

| Metric | Median | p75 | p95 | Mean |
|---|---|---|---|---|
| TTFT (ms) | 54.40 | 85.22 | 125.27 | 66.94 |
| Decode (tok/s) | 37.94 | 42.45 | 74.69 | 43.33 |

The decode p95 of 74.69 tok/s sits far above the median of 37.94 tok/s. That
asymmetry comes from very short generations, where fixed overhead dominates.
Per-bucket breakdowns are stored for exactly this reason: a 1-4 token generation
has a median of 69.68 tok/s, while a 129+ token generation settles at 37.35 tok/s.
Reporting one pooled number would be misleading.

### 4.2 Pareto frontiers

- Accuracy versus speed: `gemma-3n-e2b-q4`, `qwen3-4b-q4`, `qwen3-4b-q5`,
  `smolm2-1.7b-q4`.
- Accuracy versus VRAM: `gemma-3n-e2b-q4`, `llama3.2-3b-q6`, `qwen3-4b-q4`,
  `qwen3-4b-q5`.

Chart artifacts: `results/reports/pareto-accuracy-speed.png` and
`results/reports/pareto-accuracy-vram.png`.

---

## 5. Temperature Study

Study: `temperature-study-v1`, comparing temperature 0.0 against 0.7 on
`qwen3-4b-q4`. Source: `results/reports/temperature-study-v1.json`.

### 5.1 Method

- 15 tasks, 5 trials each, giving 75 measured rows per arm.
- 2 warm-ups per arm plus 1 diagnostic call, for 78 requests per arm and 156
  total. Zero retries were required.
- Same prompts, same hardware, same model digest in both arms.

### 5.2 Results

| Metric | Temperature 0.0 | Temperature 0.7 |
|---|---|---|
| Passes | 29 | 26 |
| Trial accuracy | 38.67% | 34.67% |
| Mean distinct normalized outputs per task | 1.47 | 3.40 |
| Median TTFT (ms) | 56.11 | 68.29 |
| Median decode (tok/s) | 37.34 | 37.41 |
| McNemar exact p | 1.0 | |

### 5.3 Interpretation

Temperature 0.0 produced substantially more consistent output: 1.47 distinct
normalized outputs per task versus 3.40 at temperature 0.7. The quality
difference was not statistically significant under exact McNemar testing
(`p = 1.0`), so the defensible claim is about **repeatability**, not accuracy.

Three trials of task `Q011` passed at 0.0 and failed at 0.7. That single task
accounts for the entire accuracy delta.

---

## 6. Cross-Model Results

### 6.1 Quality ranking

Qwen3 4B leads by a wide margin: 55.22% for Q5 and 53.23% for Q4, against 38.81%
for the best Llama configuration. A 16.4 percentage-point gap separates Qwen3 Q5
from Llama 3.2 Q6.

### 6.2 Failure-mode decomposition

The `qwen3-4b-q4` failure analysis classified 97 failing rows into a
versioned taxonomy (`failure-modes-v1`, status `CONFIRMED_UNCHANGED`):

| Failure mode | Rows | Share |
|---|---|---|
| CONTENT_ERROR | 33 | 34.02% |
| OUTPUT_CONTRACT | 39 | 40.21% |
| LEXICAL_CONSTRAINT | 13 | 13.40% |
| MIXED | 12 | 12.37% |

The dominant failure class is `OUTPUT_CONTRACT` at 39 of 97 rows, or 40.21%.
In other words, the largest single source of lost points is output formatting
rather than reasoning. This finding directly motivated the retry-rescue work in
section 8, whose pre-registered recovery population is exactly these 39 rows.

A separate substantive review of the same 97 rows reached a more favorable
verdict on reasoning quality: 61 rows `CORRECT`, 6 `LIKELY_CORRECT`, and 30
`INCORRECT`. The gap between 40.21% contract failures and a 62.89% correct rate
is the practical measure of how much the strict output contract costs in
apparent capability.

Source: `results/reports/qwen3-4b-q4-failure-analysis.json`. Human substantive
review is `APPROVED` for `qwen3-4b-q4`; other configurations carry automatic
taxonomy only and are marked `NOT_REVIEWED`.

### 6.3 Long context

Suite: `long-context-v1`, 11 configurations, 693 measured trials.
Source: `results/reports/long-context-v1-full.json`.

Across every model, accuracy at the M and L lengths was 0.0, while S-length
tasks passed. The claim scope is explicit: this measures a configured
4096-token context window, not native 8K or 32K capability, and it deliberately
excludes RAG, embeddings, and chunking.

### 6.4 Isolated coding execution

Suite: `coding-v1`, 11 configurations, 264 rows, 8 tasks.
Source: `results/reports/coding-v1-full.json`.

- Trial accuracy: 64.77% (171 of 264).
- Sandbox errors: 0.
- Static failures: 33.
- Partial scores are diagnostic only.

Code executes inside pinned OCI workers with no network, a read-only root
filesystem, dropped capabilities, `no-new-privileges`, and PID, memory, CPU, file
descriptor, and tmpfs limits. Runtime digests are pinned in the fixture manifest.

---

## 7. Quantization Trade-offs

Seven paired comparisons were run. Each uses the exact McNemar test on paired
per-task outcomes. Source: `results/summaries/` and the `quant_pairs` section of
`results/reports/sweep-v2-report.json`.

| Comparison | Accuracy delta | All-pass delta | Net task gain | McNemar exact p |
|---|---|---|---|---|
| `qwen3-4b-q4` vs `q5` | +1.96 pp | +1 | +1 | 1.0 |
| `llama3.2-3b-q4` vs `q5` | -0.98 pp | 0 | 0 | 1.0 |
| `llama3.2-3b-q4` vs `q6` | +3.43 pp | +3 | +3 | 0.375 |
| `gemma-3n-e2b-q4` vs `q5` | -4.41 pp | -3 | -3 | 0.375 |
| `smolm2-1.7b-q4` vs `q5` | -2.45 pp | -1 | -1 | 1.0 |
| `smolm2-1.7b-q4` vs `q6` | 0.00 pp | +1 | +1 | 1.0 |
| `smolm2-1.7b-q4` vs `q8` | -4.90 pp | -2 | -2 | 0.5 |

### 7.1 Quality finding

No comparison reached statistical significance at the 0.05 level. The smallest
observed effect was 0.00 pp and the largest was 4.90 pp, but every exact p-value
was at or above 0.375. Four of seven comparisons moved accuracy downward as
quantization increased.

**Licensed claim:** observed quantization differences within the tested paired
deterministic tasks were small and were not statistically distinguishable under
the exact McNemar tests used here.

This is a negative result, and it is the useful one. A common portfolio claim
would be that higher quantization improves quality. This data does not support
that claim on this contract, hardware, or model set.

### 7.2 Speed and memory finding

The cost side is unambiguous and consistent. Taking Q4 as the baseline within
each family:

| Family | Step up | VRAM change | Median decode change |
|---|---|---|---|
| Qwen3 4B | Q4 to Q5 | +374 MiB | 37.94 to 33.48 tok/s |
| Llama 3.2 3B | Q4 to Q5 | +288 MiB | 47.96 to 42.35 tok/s |
| Llama 3.2 3B | Q4 to Q6 | +592 MiB | 47.96 to 37.70 tok/s |
| Gemma 3n E2B | Q4 to Q5 | +254 MiB | 49.33 to 44.13 tok/s |
| SmolLM2 1.7B | Q4 to Q5 | +162 MiB | 84.81 to 74.91 tok/s |
| SmolLM2 1.7B | Q4 to Q6 | +332 MiB | 84.81 to 66.98 tok/s |
| SmolLM2 1.7B | Q4 to Q8 | +728 MiB | 84.81 to 54.04 tok/s |

Every step up in quantization increased peak VRAM and reduced median decode
throughput, with no configuration breaking that pattern.

### 7.3 Practical recommendation

- **Quality-first:** `qwen3-4b-q5`, accepting roughly 12% lower throughput and
  374 MiB more VRAM for an unproven quality gain.
- **Balanced default:** `qwen3-4b-q4`, which is Pareto-optimal on both accuracy
  versus speed and accuracy versus VRAM.
- **Throughput-first:** `smolm2-1.7b-q4`, the fastest configuration measured.
- **Lowest memory:** `gemma-3n-e2b-q4`, at 1858 MiB peak VRAM, though with the
  worst median TTFT at 167 ms.
- **Avoid Q8 on this hardware.** `smolm2-1.7b-q8` was simultaneously the largest,
  the slowest within its family, and the lowest scoring.

---

## 8. Reliability and Reproducibility

### 8.1 Retry policy

`inference/retry.py` classifies faults before acting. Only transport-level
failures are retried: timeouts, resets, truncated streams, transient 5xx, and
interrupted model pulls. Everything semantic, including 4xx, malformed data,
config or hash mismatches, and persistence failures, fails closed immediately.
Retries never change benchmark configuration and never persist partial output.

Two additional robustness mechanisms are present:

- **Eviction re-warm.** A generation whose server load duration exceeds 1000 ms is
  treated as proof of a mid-run reload, triggering an automatic re-warm. The
  threshold is frozen and recorded in every manifest.
- **Verified deletion.** A successful model removal is not trusted on its own,
  because `ollama list` and `ollama ps` can disagree. A ghost-state regression
  test covers the case where removal reports success but the model is still
  present.

### 8.2 Retry-rescue investigation

Three follow-up studies were completed.

| Study | Scope | Outcome |
|---|---|---|
| `retry-rescue-v1` | 39 of 97 failing rows classified as pure `OUTPUT_CONTRACT` | Population derived programmatically, no hand-maintained task IDs |
| `retry-rescue-v2` | 18 rows whose contracts are expressible as constrained decoding | 0 of 18 recovered in every arm |
| `retry-rescue-v3` | Renderer-level analysis | Further rows marked `UNRENDERABLE` |

The v2 capability matrix is frozen and records eligibility per task: 39 source
rows, 18 eligible, 21 ineligible. Results apply only to the eligible subset, and
that restriction is stated in the report itself rather than buried.

### 8.3 External judge

Suite: `judge-suite-v1`, status `EXTERNAL_JUDGE_COMPLETE`.
Source: `results/reports/judge-suite-v1-external-full-final.json`.

- 164 API calls, total recorded cost 0.00 USD.
- 43 of 43 rubric items, 54 of 54 pairs.
- Zero final errors, zero forced verdicts.
- 39 evidence quotes pruned as unverifiable, giving 36 pruned in pairs and 3 in
  the rubric.
- No raw model text in the public report, and no local judges used.

### 8.4 Reproducibility controls

| Control | Implementation |
|---|---|
| Model identity | Pinned model artifact digest per configuration |
| Prompt identity | `rendered_prompt_sha256` and `template_sha256` per row |
| Grading identity | Frozen grading spec with recorded hash |
| Dataset identity | Dataset hash recorded in every manifest and freeze |
| Code identity | Runner and report code hashes plus git commit per freeze |
| Environment | Hardware ID, Ollama version, Python version, power mode recorded |
| Resume safety | Commit-per-generation, identity-keyed resume |
| Cleanup | Model removal verified against `ollama list` and `ollama ps` |

### 8.5 Test and CI coverage

`scripts/quality_gate.py` runs Ruff, mypy in strict mode across 118 source files,
the full offline test suite, a workbook consistency check, and both grading-spec
audits. Current state: **373 tests passing**, Ruff PASS, mypy PASS, workbook check
PASS, `eval-v1` audit PASS, `eval-v1.1` audit PASS.

CI runs the same gate on both `ubuntu-latest` and `windows-latest` via
`.github/workflows/quality.yml`, so a platform-specific regression fails the
build.

---

## 9. Methodology Incidents

Six incidents were recorded, each with cause, a regression fixture, and a
durable rule. Reproducing this list matters more than the headline numbers,
because each one is a case where a plausible-looking measurement would have been
wrong.

| Incident | Cause | Rule adopted |
|---|---|---|
| `wrong-canonical-load` | Eligibility measured a model loaded under default options, `num_ctx=512`, instead of the pinned config | Canonical load probe with exact pinned options; explicit unload and `ps`-empty check before any measured load |
| `stale-state-loop` | Main loop held a launch-time state copy, re-selecting a terminal model forever | State file reloaded every iteration; terminal states never transition back to work |
| `silent-gemma-metadata` | `/api/show` returned empty metadata for Gemma-3n although the model loads and generates | Committed overlay takes precedence; interrogation failure with no pin is `MANUAL_PIN_REQUIRED` |
| `first-token-trial` | A 1-token route trial falsely failed SmolLM2 Q6, whose first token can be non-text on a healthy route | Trials prove transport, not brevity; budget raised to 16 tokens |
| `phi-runtime-headroom` | Phi Q4 was fully resident yet aborted sustained generation with about 95 MiB free | Eligibility requires full residency **and** an operational canary |
| `ghost-deletion` | Removal success does not imply absence; `list` and `ps` can disagree | Verified deletion guard; `DELETION_FAILED` never advances |

---

## 10. Limitations

These constraints are part of the result, not footnotes.

1. Quantization quality differences were not statistically distinguishable under
   the exact McNemar tests used.
2. Temperature-0 repeatability was demonstrated for this benchmark, hardware, and
   runtime configuration only. No universal determinism claim is made.
3. Single machine, RTX 2050 with 4 GB VRAM. Feasibility findings are
   hardware-specific by construction.
4. `READY_JUDGE` tasks are excluded from all capability denominators, so
   qualitative outcomes remain deferred.
5. Strict substring matchers for terms, dates, and units punish paraphrase.
   Contract accuracy therefore understates reasoning capability by design.
6. Grader semantics are character-level while human interpretation is token-level.
   This is intentional and versioned, but cross-study comparisons must account
   for it.
7. `num_predict` truncation is recorded orthogonally as
   `TRUNCATED_AT_NUM_PREDICT` and never merged into failure modes, so truncated
   generations cannot establish what the model would have asserted with a larger
   budget.
8. Latency metrics are descriptive only and are not causal accuracy evidence.

---

## 11. Scope Boundaries

This repository is a **local model benchmarking system**. It is not an
end-user assistant application. Specifically, the following are out of scope and
not implemented here:

- An interactive chat CLI or a serving API wrapper. The `scripts/*.py` entry
  points are benchmark tools driven by frozen configurations, not a conversational
  interface.
- Pydantic-based validation of assistant responses. Pydantic appears in the
  benchmark dataset as a task *subject*, not as a response-validation dependency.
- A validate-repair-retry loop for malformed assistant JSON. The retry logic in
  `inference/retry.py` handles transport faults and deliberately excludes
  semantic response repair.
- RAG, embeddings, chunking, vector stores, or retrieval. The long-context suite
  explicitly excludes these, and its schema pins `retrieval_or_rag: false`.
- Fine-tuning, LoRA, QLoRA, or DPO training.
- Real-time multimodal pipelines such as ASR, TTS, or streaming video.

What the repository does provide is the measurement, grading, provenance, and
reproducibility layer that such an application would need in order to make
evidence-based model-selection decisions.

---

## 12. Evidence Index

| Claim | Artifact |
|---|---|
| Per-model accuracy, speed, VRAM, size | `results/reports/sweep-v2-report.json` |
| Corrected `eval-v1.1` figures | `results/reports/eval-v1.1-companion-report.json` |
| Human-readable sweep summary | `results/reports/sweep-v2-report.md` |
| Grading regrade overlay | `results/reports/eval-v1.1-regrade-overlay.json` |
| Paired quantization comparisons | `results/summaries/*-vs-*.json` |
| Per-model performance detail | `results/summaries/full-baseline-v2__*-performance.json` |
| Temperature study | `results/reports/temperature-study-v1.json` |
| Temperature study freeze | `results/reports/temperature-study-v1-freeze.json` |
| Failure-mode taxonomy | `results/reports/qwen3-4b-q4-failure-analysis.json` |
| Retry-rescue v1, v2, v3 | `results/reports/retry-rescue-v*.json` |
| Constrained-decoding capability matrix | `results/reports/retry-rescue-v2-capability-matrix.json` |
| Reliability analysis | `results/summaries/reliability-v1-reliability.json` |
| Long context, 11 models | `results/reports/long-context-v1-full.json` |
| Isolated coding execution | `results/reports/coding-v1-full.json` |
| External judge, sealed | `results/reports/judge-suite-v1-external-full-final.json` |
| Release index and test count | `results/reports/core-release-v1.1-freeze.json` |
| Methodology incidents | `results/reports/sweep-v2-report.json`, `results/incidents/` |
| Eligibility incident evidence | `results/incidents/wrong-load-eligibility/incident.json` |
| Pareto charts | `results/reports/pareto-accuracy-speed.png`, `results/reports/pareto-accuracy-vram.png` |
| Streaming client | `inference/ollama_client.py` |
| Metric definitions | `inference/profiler.py` |
| Retry policy | `inference/retry.py` |
| Benchmark runner | `scripts/run_benchmark.py` |
| Prompt and adapter resolution | `inference/adapters.py`, `configs/adapters/` |
| Baseline contract | `configs/full-baseline-v2.yaml` |
| Grading specification | `evals/specs/eval-v1.1-grading.yaml` |
| Quality gate | `scripts/quality_gate.py` |
| CI workflow | `.github/workflows/quality.yml` |
