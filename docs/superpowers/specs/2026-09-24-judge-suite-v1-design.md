# Judge Suite V1 Design

## Objective

Create a reproducible qualitative adjudication and judge-bias suite for the five
Eval-v1.1 `READY_JUDGE` tasks without modifying the sealed model outputs or
mixing judge scores into deterministic capability denominators.

## Dependency And Population

The suite runs after Core Release V1.1 is frozen. It reuses the 11 eligible
Full Baseline V2 configurations, five judge tasks (Q014, Q047, Q051, Q059,
Q060), and three trials: 165 source rows total. The three runtime-ineligible
Phi configurations remain excluded and are not assigned zero scores.

Every source database is read-only. Source identity includes execution ID,
model, task, trial, raw-output hash, source database hash, and source manifest.

## Precheck And Panel

Corrected Eval-v1.1 deterministic graders run before semantic judging. A hard
precheck failure becomes `PRECHECK_FAIL` and receives no judge request. A
precheck pass becomes `JUDGE_PENDING` and is eligible for rubric evaluation.

The canonical local panel is:

- `qwen3-4b-q5`
- `llama3.2-3b-q6`

Both use pinned digests, templates, context, temperature, output policy, and
strict JSON response schemas. Candidate model identity, family, quantization,
trial, source verdict, answer key, and prior judgment are hidden from the judge.

Disagreements, abstentions, low confidence, and position instability require
blinded human adjudication. API judging is excluded from the canonical release.

## Rubric And Pair Protocols

Criterion definitions include desired polarity, positive and negative anchors,
criticality, and required evidence. A rubric response contains exact criterion
keys, observed booleans, confidence, evidence quotes, and an abstention flag.
Evidence quotes must be substrings of the candidate.

Pairwise judging uses opaque candidate IDs, injection-resistant instructions,
both A/B orientations, and `A`, `B`, `TIE`, `NEITHER`, or `INVALID` outcomes.
Malformed output is a parse failure, never a negative judgment.

## Human Calibration

Before judge inference, create a frozen human-labeled set:

- 15 semantic cases per task,
- pass and hard-fail precheck fixtures,
- 25 pairwise controls,
- a stratified 50-row real-output spot check.

Two blinded raters label criteria and disagreements; an adjudicator resolves
disagreements. Labels, evidence hashes, and adjudication records are frozen
before live judging. The holdout is not used to revise prompts.

## Bias Study

Evaluate 40 controlled pairs (8 per task) and 50 frozen stratified real-output
pairs (10 per task). Every pair is evaluated in both orientations by both local
judges. The sample is balanced across task, family, quality, output length, and
controlled nuisance factors.

Report position-flip rate, pairwise and tie accuracy, invalid/abstention rate,
self/family preference, quantization preference, verbosity/style/authority
effects, inter-judge agreement, adjudication rate, and post-adjudication
accuracy. Use pair-clustered uncertainty; repeated deterministic outputs are not
independent observations.

## Storage And Privacy

Use a dedicated local SQLite database with protocol, source item, judge call,
rubric result, pair, pair result, failure, and adjudication tables. Public
artifacts contain hashes, counts, confidence, and redacted evidence only. Raw
candidate and judge text remains local and restricted. No credentials or API
keys are stored.

## Verification And Release

Offline tests cover schema validation, evidence quotes, precheck exclusion,
blinding, deterministic pair IDs, A/B inverse mapping, abstention, invalid
responses, disagreement states, source immutability, and metric denominators.
Live judge calls require a separately approved budget after the frozen
population, gold labels, prompts, schemas, and source hashes pass the gate.

## Completion

The judge suite is complete only with a frozen protocol, source manifest, human
adjudication record, local panel report, bias report, database, freeze manifest,
and explicit separation from Eval-v1.1 deterministic results.
