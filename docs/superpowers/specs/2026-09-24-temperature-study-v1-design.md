# Temperature Study V1 Design

## Objective

Measure how Qwen Q4's deterministic benchmark behavior changes between
temperature 0.0 and 0.7. The primary comparison is over a fixed 15-task,
five-trial deterministic population; temperature is the only experimental
difference between arms.

## Frozen Population

The ordered study population contains these `READY_DETERMINISTIC` tasks:

- Structured: Q001, Q003, Q062, Q066
- Constraints: Q011, Q013, Q015, Q017
- Numeric/time: Q019, Q023, Q026
- Exact/time: Q035, Q039, Q042, Q071

Each task has five trials, for 75 measured rows per arm. Deferred judge tasks,
including Q014 and Q059, are excluded entirely.

## Arms And Budget

Two scalar-temperature executions use the same Qwen Q4 artifact, template,
prompt bytes, stop tokens, context, output cap, GPU policy, task order, trial
count, and frozen grader:

- `temperature-study-v1-t0`, temperature 0.0
- `temperature-study-v1-t07`, temperature 0.7

Each arm persists two warmups followed by 75 measured rows. The complete study
budget is 154 model calls: four warmups and 150 measured generations.

## Execution Integrity

One frozen study contract names the population and shared generation settings.
Two derived scalar runner configs must be identical except for their experiment
identity and temperature. Each execution validates model digest, template,
study population, five-trial completeness, and the temperature-sensitive
generation configuration before reporting results.

Recovery reloads must use the active arm temperature. A reload under another
temperature invalidates that arm and must fail closed.

## Analysis

The reducer requires exact 75-row task/trial identity equality across arms. It
reports deterministic trial accuracy, per-task all-pass and any-pass outcomes,
paired trial transitions, output diversity, and descriptive latency metrics.
It never creates a judge denominator or presents latency as an accuracy cause.

## Tests

Tests cover the frozen task set, deterministic-only eligibility, arm equality
except temperature, active-temperature reload propagation, temperature-aware
resume identity, five-trial completeness, source/model/template mismatch
rejection, and paired-report population mismatch rejection.

## Release Sequence

Commit code and frozen configs before the live study. After explicit approval,
run exactly the fixed 154-call budget, validate persistence and matching arms,
produce report and freeze artifacts, unload/remove the model, verify cleanup,
and commit artifacts separately.
