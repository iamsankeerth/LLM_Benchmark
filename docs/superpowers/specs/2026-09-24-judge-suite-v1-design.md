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

The active protocol uses one external OpenAI-compatible judge pinned to
`stealth/space-bunny-alpha`, with local judges disabled. The runtime endpoint is
OpenRouter (`https://openrouter.ai/api/v1`) and credentials are supplied only
through environment variables. Candidate model identity, family, quantization,
trial, source verdict, answer key, and prior judgment are hidden from the judge.
Each independent rubric, pair-orientation, and tie-break request enables OpenRouter
reasoning with `{"enabled": true}`. Reasoning details are discarded after parsing
and are never forwarded to another judgment. No human adjudication is part of
this protocol.

## Rubric And Pair Protocols

Criterion definitions include desired polarity, positive and negative anchors,
criticality, and required evidence. A rubric response contains exact criterion
keys, observed booleans, confidence, evidence quotes, and an abstention flag.
Evidence quotes must be substrings of the candidate. Item and pair identifiers
are bound to the persisted call by the runner and are not required in model
output. Evidence that is not a verbatim candidate substring is discarded and
counted as pruned; it is never accepted as evidence.

Pairwise judging uses opaque candidate IDs, injection-resistant instructions,
both A/B orientations, and `A`, `B`, `TIE`, `NEITHER`, or `INVALID` outcomes.
Malformed output is a parse failure, never a negative judgment. The runner binds
the pair identity; the model does not need to echo it.

## External Calibration

Before the full external run, execute a canary against the frozen population,
prompts, schemas, and source hashes. The canary verifies the pinned model ID,
JSON parsing, evidence validation, call accounting, retry behavior, and call-cap
refusal. Malformed responses receive up to three fresh independent attempts;
unverifiable evidence is discarded and counted rather than accepted. A
successful canary is required before the full capped run. Low
confidence and abstentions are recorded as forced-label signals; they do not
create a human queue.

## Bias Study

Evaluate 40 controlled pairs (8 per task) and 50 frozen stratified real-output
pairs (10 per task). Every pair is evaluated in both orientations by the
external judge. Disagreements use one additional call with a forced label until
the configured call cap; a documented forward fallback is used after the cap.
The sample is balanced across task, family, quality, output length, and
controlled nuisance factors.

Report position-flip rate, pairwise and tie accuracy, invalid/forced-label
rate, self/family preference, quantization preference, verbosity/style/authority
effects, third-call rate, and fallback rate. Use pair-clustered uncertainty;
repeated deterministic outputs are not independent observations.

## Storage And Privacy

Use a dedicated local SQLite database with protocol, source item, judge call,
rubric result, pair, pair result, failure, and external-call tables. Public
artifacts contain hashes, counts, confidence, and redacted evidence only. Raw
candidate and judge text remains local and restricted; exact candidate text is
transmitted only to the configured external endpoint and is not stored in the
public report. No credentials or API keys are stored.

## Verification And Release

Offline tests cover schema validation, evidence quotes, precheck exclusion,
blinding, deterministic pair IDs, A/B inverse mapping, abstention, invalid
responses, disagreement states, source immutability, and metric denominators.
Live judge calls require runtime endpoint and credential configuration plus an
approved call cap after the frozen population, prompts, schemas, and source
hashes pass the gate. Space Bunny Alpha is currently free, so the active
protocol has no monetary cutoff; provider-reported usage cost is retained as
telemetry when available, and unknown cost remains unknown.

## Completion

The judge suite is complete only with a frozen protocol, source manifest,
external-call database, external report, and freeze manifest. Human calibration
and human adjudication records are intentionally absent, and judge scores
remain separate from Eval-v1.1 deterministic results.
