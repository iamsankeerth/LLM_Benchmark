# Long-Context V1 Design

## Objective

Measure full-context use of the configured 4,096-token window with controlled,
deterministic tasks. The suite does not claim native 8K/32K capability and does
not include RAG, embeddings, chunking, or retrieval.

## Dataset

Author 21 original Markdown documents under a new `long-context-v1` dataset.
Each document is synthetic, source-controlled, and hashed. Seven task archetypes
are represented at three source lengths:

- S: 350 whitespace words
- M: 700 whitespace words
- L: 1,400 whitespace words

Archetypes are start needle, middle needle, end needle, multi-needle binding,
state/recency resolution, global aggregation, and grounded abstention. Relevant
facts, distractor counts, and question requirements remain comparable across
lengths; only neutral material expands.

Each task freezes document hash, word/byte counts, needle offsets, position
fraction, distractor count, expected answer, and grader type. Observed
`prompt_eval_count`, not a generic tokenizer, is the authoritative runtime length.

## Execution

Use three fresh trials, temperature 0.0, output cap 256, and run kind
`LONG_CONTEXT`. The full document and question are sent in one request. A
four-model pilot covers Qwen Q4, Llama Q6, Gemma Q4, and SmolLM2 Q4. A passing
pilot expands to all 11 eligible V2 configurations. Ineligible configurations
remain null, never zero.

Before measured generation, verify model digest, canonical adapter load, GPU
residency, operational canary, actual prompt-token count, and:

`prompt_eval_count + 256 <= configured num_ctx`

A missing counter or insufficient headroom fails closed.

## Grading And Reports

Use exact, numeric, structured JSON, or fixed abstention graders only. Report
trial accuracy, strict all-pass, any-pass, accuracy by source length and
archetype, position behavior, abstention and false-refusal rates, TTFT, prefill,
cache state, decode speed, RAM/VRAM, and Wilson intervals. Length and position
comparisons are descriptive; the report must not claim isolated causal effects
from the small initial suite.

## Suite-Generalization Requirements

The runner, preflight, manifests, summaries, comparisons, and persistence
verification must accept an explicit suite/spec contract rather than hard-coded
Eval-v1 paths or 72-task counts. Existing Eval-v1 behavior remains unchanged.

## Verification And Release

Offline tests cover document and task hashes, exact 21-task population, length
ordering, needle positions, distractor integrity, full-prompt assembly,
canonical graders, preflight boundaries, SQLite persistence, resume identity,
report denominators, and freeze verification. Live rollout requires a separately
approved budget after the pilot and all integrity gates pass.

## Completion

The release is complete only with frozen documents/spec/schema, pilot and full
cohort reports when authorized, performance artifacts, provenance manifests, and
an explicit statement that the suite measures the configured 4K window rather
than native long-context capability.
