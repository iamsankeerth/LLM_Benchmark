# V2 sweep report: 14 configurations attempted, 11 benchmarked

Three Phi configs preserved as INELIGIBLE_RUNTIME_HEADROOM with evidence refs; null metrics never enter denominators or skylines.

| model | acc | any3 | all3 | dec_med | dec_p95 | vram | status |
|---|---|---|---|---|---|---|---|
| qwen3-4b-q4 | 0.525 | 0.529 | 0.515 | 37.9 | 74.7 | 3252 | ELIGIBLE_GPU |
| qwen3-4b-q5 | 0.544 | 0.559 | 0.529 | 33.5 | 65.5 | 3626 | ELIGIBLE_GPU |
| llama3.2-3b-q4 | 0.348 | 0.353 | 0.338 | 48.0 | 94.0 | 2658 | ELIGIBLE_GPU |
| llama3.2-3b-q5 | 0.338 | 0.338 | 0.338 | 42.3 | 82.5 | 2946 | ELIGIBLE_GPU |
| llama3.2-3b-q6 | 0.382 | 0.382 | 0.382 | 37.7 | 73.3 | 3250 | ELIGIBLE_GPU |
| gemma-3n-e2b-q4 | 0.353 | 0.353 | 0.353 | 49.3 | 50.3 | 1858 | ELIGIBLE_GPU |
| gemma-3n-e2b-q5 | 0.309 | 0.309 | 0.309 | 44.1 | 44.8 | 2112 | ELIGIBLE_GPU |
| phi-3.5-mini-q4 | n/a | n/a | n/a | n/a | n/a | n/a | COMPLETE_INELIGIBLE_RUNTIME_HEADROOM |
| phi-3.5-mini-q5 | n/a | n/a | n/a | n/a | n/a | n/a | COMPLETE_INELIGIBLE_RUNTIME_HEADROOM |
| phi-3.5-mini-q6 | n/a | n/a | n/a | n/a | n/a | n/a | COMPLETE_INELIGIBLE_RUNTIME_HEADROOM |
| smolm2-1.7b-q4 | 0.314 | 0.324 | 0.294 | 84.8 | 158.0 | 2056 | ELIGIBLE_GPU |
| smolm2-1.7b-q5 | 0.289 | 0.294 | 0.279 | 74.9 | 113.2 | 2218 | ELIGIBLE_GPU |
| smolm2-1.7b-q6 | 0.314 | 0.324 | 0.309 | 67.0 | 101.2 | 2388 | ELIGIBLE_GPU |
| smolm2-1.7b-q8 | 0.265 | 0.265 | 0.265 | 54.0 | 94.5 | 2784 | ELIGIBLE_GPU |

## Licensed claims

- Observed quantization differences within the tested paired deterministic tasks were small and were not statistically distinguishable under the exact McNemar tests used here.
- Temperature-0 inference showed high repeatability under this benchmark, hardware, and runtime configuration.

## Limitations

- Observed quantization differences within the tested paired deterministic tasks were small and were not statistically distinguishable under the exact McNemar tests used here.
- Temperature-0 inference showed high repeatability under this benchmark, hardware, and runtime configuration; no universal determinism claim is made.
- Single machine (RTX 2050 4 GB VRAM); feasibility findings are hardware-specific by construction.
- READY_JUDGE tasks are excluded from all capability denominators; qualitative outcomes remain deferred.
- Temperature study (0 vs 0.7), long-context, and judge-bias suites are out of scope for this lineage.
- Strict substring matchers (terms, dates, units) punish paraphrase; contract accuracy understates reasoning capability by design.
