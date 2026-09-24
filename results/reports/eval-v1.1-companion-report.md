# Eval-v1.1 Corrected V2 Companion Report

Corrected grading-only view of the sealed V2 outputs. No model generation calls were made.

| Model | Deterministic passes | Accuracy | Any-pass | All-pass | Judge tasks |
|---|---:|---:|---:|---:|---:|
| gemma-3n-e2b-q4 | 72/201 | 0.358 | 24/67 | 24/67 | 5 |
| gemma-3n-e2b-q5 | 63/201 | 0.313 | 21/67 | 21/67 | 5 |
| llama3.2-3b-q4 | 72/201 | 0.358 | 25/67 | 23/67 | 5 |
| llama3.2-3b-q5 | 69/201 | 0.343 | 23/67 | 23/67 | 5 |
| llama3.2-3b-q6 | 78/201 | 0.388 | 26/67 | 26/67 | 5 |
| qwen3-4b-q4 | 107/201 | 0.532 | 36/67 | 35/67 | 5 |
| qwen3-4b-q5 | 111/201 | 0.552 | 38/67 | 36/67 | 5 |
| smolm2-1.7b-q4 | 67/201 | 0.333 | 23/67 | 21/67 | 5 |
| smolm2-1.7b-q5 | 59/201 | 0.294 | 20/67 | 19/67 | 5 |
| smolm2-1.7b-q6 | 67/201 | 0.333 | 23/67 | 22/67 | 5 |
| smolm2-1.7b-q8 | 54/201 | 0.269 | 18/67 | 18/67 | 5 |

Speed/accuracy Pareto: gemma-3n-e2b-q4, qwen3-4b-q4, qwen3-4b-q5, smolm2-1.7b-q4
VRAM/accuracy Pareto: gemma-3n-e2b-q4, llama3.2-3b-q6, qwen3-4b-q4, qwen3-4b-q5

Historical Eval-v1 reports remain unchanged. This companion is a separate Eval-v1.1 artifact.
