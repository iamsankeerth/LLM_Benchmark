# Retry-Rescue V3 Design

## Objective

Determine whether a deterministic renderer can recover contract failures from
already-produced, sealed baseline reasoning. V3 is a post-processing study, not
a model retry: it makes zero Ollama calls and never changes the baseline output.

## Population

V3 uses only the frozen numeric primary intersection X:

- Q019, Q020, Q021, Q022, Q024, and Q070
- Trials 1 through 3 for each task
- 18 identities total

Controls are excluded. Q016 remains excluded because its frozen structured
grader is internally unsatisfiable and cannot support a recovery claim.

## Inputs And Boundaries

The sealed `full-baseline-v2__qwen3-4b-q4` database is opened read-only. V3
binds its SHA-256, execution ID, selected population hash, grading-spec hash,
and renderer-code hash before reading any selected row.

The renderer receives only a baseline raw output and the task's numeric output
kind. It must not receive expected answers, grader details, failure-mode labels,
V1/V2 output, control rows, or human annotations.

## Extraction And Rendering

For each selected baseline output, the renderer accepts one candidate only when
it is found in an explicit final-answer marker or in a terminal standalone
numeric line. It rejects candidates with prose or units, non-finite values,
multiple distinct candidates, or no candidate.

An accepted finite number is serialized as the canonical bare numeric bytes.
All rejected inputs are recorded as `UNRENDERABLE`. The unchanged frozen grader
is the sole oracle for the rendered output.

## Execution Semantics

V3 transforms every selected A row once. It performs no model pull, eligibility
probe, canary, warmup, or inference request. A source hash, identity mismatch,
or config/parser hash mismatch aborts before producing results.

## Reporting

The report states:

- rendered recovery D / 18
- `UNRENDERABLE` / 18
- rendered-but-grader-failed / 18
- source provenance and all frozen hashes

A, B, and C are retained only as historical context. They are never pooled into
the V3 denominator or used by the renderer.

## Tests

Tests cover accepted/rejected forms, ambiguity and non-finite rejection, absence
of expected-answer inputs, read-only source access, exact 18-row selection,
source-drift refusal, and reproducible report output.

## Interpretation

V3 estimates recovery from deterministic serialization of existing A reasoning.
It does not estimate fresh-generation retry performance or establish that a
model can produce the same recoverable reasoning prospectively.
