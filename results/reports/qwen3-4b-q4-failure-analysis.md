# Why Qwen3-4B Q4 Fails the Frozen Eval Contract

> Post-hoc diagnostic artifact. The 52.5% benchmark result is untouched; categories below never mutate verdicts, graders, or specs.

Central finding: Qwen Q4 frequently reaches the correct semantic answer but fails the required output contract.

## Table 1 — contract decomposition (97 deterministic FAIL rows)

- CONTENT_ERROR: 33 rows (34.0%)
- LEXICAL_CONSTRAINT: 13 rows (13.4%)
- MIXED: 12 rows (12.4%)
- OUTPUT_CONTRACT: 39 rows (40.2%)

## Table 2 — substantive assessment (97 rows inherit task labels)

- CORRECT: 61 rows (62.9%)
- INCORRECT: 30 rows (30.9%)
- LIKELY_CORRECT: 6 rows (6.2%)

## Table 3 — judge precheck failures (6 rows, descriptive only)

- Q014: substantive=LIKELY_CORRECT, trials=[1, 2, 3], deferrals tracked in capability summary
- Q051: substantive=CORRECT, trials=[1, 2, 3], deferrals tracked in capability summary

## Per-task failure summaries

### Q003 [READY_DETERMINISTIC] — INCORRECT
- trials: [1, 2, 3] modes: {'CONTENT_ERROR': 3} multi_mode: False flags: []
- notes: Arithmetic slip: 18500+500+3420=22420, model wrote 22020/12020. Wrong value.

### Q005 [READY_DETERMINISTIC] — CORRECT
- trials: [1, 2, 3] modes: {'CONTENT_ERROR': 3} multi_mode: False flags: []
- notes: Dates/times semantically identical (3 Oct 2026 = 2026-10-03; 21:10 = 1270 min). Failure is format (ISO date, minutes-integer time), not substance.

### Q006 [READY_DETERMINISTIC] — CORRECT
- trials: [1, 2, 3] modes: {'CONTENT_ERROR': 3} multi_mode: False flags: []
- notes: Only delta is 'Project Atlas' vs 'Atlas'; all dates/amounts exact.

### Q011 [READY_DETERMINISTIC] — CORRECT
- trials: [1] modes: {'LEXICAL_CONSTRAINT': 1} multi_mode: False flags: []
- notes: Four-bullet explanation is good; fails only on latency x2 vs x1. t2/t3 passed with the 277-char form.

### Q012 [READY_DETERMINISTIC] — CORRECT
- trials: [1, 2, 3] modes: {'LEXICAL_CONSTRAINT': 3} multi_mode: False flags: []
- notes: Three clean sentences, chlorophyll present, ends with question. Fails only because "Three" is not a digit token. t1-t3 are byte-identical with the single numeric-token violation; no plant/sunlight hit in any trial. (An earlier analyst remark about 'plant' matching inside 'plants' referred to a synthetic probe string, never to persisted rows.)

### Q014 [READY_JUDGE] — LIKELY_CORRECT
- trials: [1, 2, 3] modes: {'LEXICAL_CONSTRAINT': 3} multi_mode: False flags: []
- notes: Good recursion analogy + factorial example, but 35 words vs required 50-60 and contains forbidden "itself". Length shortfall leaves completeness uncertain.

### Q015 [READY_DETERMINISTIC] — INCORRECT
- trials: [1, 2, 3] modes: {'LEXICAL_CONSTRAINT': 3} multi_mode: False flags: []
- notes: Frozen matcher reports six `index` substring hits, including occurrences inside `indexing`; independent A.-F. marker-format violations also remain. Multiple substantive requirement misses, not envelope.

### Q016 [READY_DETERMINISTIC] — INCORRECT
- trials: [1, 2, 3] modes: {'MIXED': 3} multi_mode: False flags: []
- notes: Values are reasonable summaries but no value contains "latency" (required exactly once) and word-range flags trip. Explicit requirement missed.

### Q017 [READY_DETERMINISTIC] — INCORRECT
- trials: [1, 2, 3] modes: {'LEXICAL_CONSTRAINT': 3} multi_mode: False flags: []
- notes: Steps 1-3 satisfy their requirements; step 4 ("Use a tool to test call with low latencie.") still contains e/E despite evasion attempt. The intentionally hard constraint defeats it.

### Q018 [READY_DETERMINISTIC] — INCORRECT
- trials: [1, 2, 3] modes: {'LEXICAL_CONSTRAINT': 3} multi_mode: False flags: []
- notes: 65 words vs 70, no parenthetical phrase, ends "usage." not "tradeoff". Three independent requirement misses.

### Q019 [READY_DETERMINISTIC] — CORRECT
- trials: [1, 2, 3] modes: {'OUTPUT_CONTRACT': 3} multi_mode: False flags: []
- notes: Full derivation reaches 136 = expected. Failure is bare-number packaging only (891 chars of LaTeX reasoning).

### Q020 [READY_DETERMINISTIC] — CORRECT
- trials: [1, 2, 3] modes: {'OUTPUT_CONTRACT': 3} multi_mode: False flags: []
- notes: Computes 66080 = expected through correct discount+GST steps. Single asserted value, no contradiction; packaging only.

### Q021 [READY_DETERMINISTIC] — CORRECT
- trials: [1, 2, 3] modes: {'OUTPUT_CONTRACT': 3} multi_mode: False flags: []
- notes: Derives 2352 = expected (300x4 + 192x6). Packaging only.

### Q022 [READY_DETERMINISTIC] — CORRECT
- trials: [1, 2, 3] modes: {'OUTPUT_CONTRACT': 3} multi_mode: False flags: []
- notes: Derives 218.4 = expected (168 + 50.4). Packaging only.

### Q023 [READY_DETERMINISTIC] — INCORRECT
- trials: [1, 2, 3] modes: {'MIXED': 3} multi_mode: False flags: []
- notes: Genuine arithmetic slip: 252/156h = 96.92min but model writes 36.92min (dropped the hour), concluding 10:37 vs 11:37. Contract violation AND wrong value.

### Q024 [READY_DETERMINISTIC] — CORRECT
- trials: [1, 2, 3] modes: {'OUTPUT_CONTRACT': 3} multi_mode: False flags: []
- notes: Derives 978 = expected (378+240+360). Packaging only.

### Q025 [READY_DETERMINISTIC] — INCORRECT
- trials: [1, 2, 3] modes: {'MIXED': 3} multi_mode: False flags: ['TRUNCATED_AT_NUM_PREDICT']
- notes: V2 divergence: t1 completed (1656 chars, stop) asserting explicit wrong final 12:36 vs 11:36 (model skips 11:36 in its own 72-min enumeration AND mis-adds 10:24+72) -> MIXED. t2/t3 assert "answer is 12:36" explicitly mid-derivation (line 75) then ramble into truncation at 2048 -> MIXED + TRUNCATED_AT_NUM_PREDICT. Substantive INCORRECT in all trials.

### Q026 [READY_DETERMINISTIC] — INCORRECT
- trials: [1, 2, 3] modes: {'MIXED': 3} multi_mode: False flags: []
- notes: Genuine arithmetic slip at step 3: 0.92169x0.97 = 0.8940393, model writes 0.9000493, concluding 86.40 vs 85.83 (outside tolerance). Contract violation AND wrong value.

### Q036 [READY_DETERMINISTIC] — INCORRECT
- trials: [1, 2, 3] modes: {'CONTENT_ERROR': 3} multi_mode: False flags: []
- notes: Overconfident negation: answers NO where context supports only UNKNOWN (USB-C charging does not imply wireless). Clean wrong answer.

### Q039 [READY_DETERMINISTIC] — CORRECT
- trials: [1, 2, 3] modes: {'OUTPUT_CONTRACT': 3} multi_mode: False flags: []
- notes: Opens with CANNOT_DETERMINE = expected, with statistically sound reasoning. Failure is the appended explanation paragraph.

### Q040 [READY_DETERMINISTIC] — CORRECT
- trials: [1, 2, 3] modes: {'OUTPUT_CONTRACT': 3} multi_mode: False flags: []
- notes: Opens with NOT_ESTABLISHED = expected, reasoning sound. Failure is the appended explanation.

### Q044 [READY_DETERMINISTIC] — CORRECT
- trials: [1, 2, 3] modes: {'CONTENT_ERROR': 3} multi_mode: False flags: []
- notes: Value 25 exact; only the unit form ("s") missing. Unambiguous, no contradiction; presentational.

### Q045 [READY_DETERMINISTIC] — CORRECT
- trials: [1, 2, 3] modes: {'OUTPUT_CONTRACT': 3} multi_mode: False flags: []
- notes: Opens with CACHE = expected plus accurate evidence chain. Failure is the appended justification.

### Q047 [READY_DETERMINISTIC] — LIKELY_CORRECT
- trials: [1, 2, 3] modes: {'CONTENT_ERROR': 3} multi_mode: False flags: []
- notes: Meaning largely preserved but qualifiers dropped: "currently enrolled" lost, "September" abbreviated to "Sept", email refusal only implied ("portal only"). Precision loss keeps this below CORRECT.

### Q048 [READY_DETERMINISTIC] — CORRECT
- trials: [1, 2, 3] modes: {'CONTENT_ERROR': 3} multi_mode: False flags: []
- notes: Missing fact is "imports longer than two hours"; output says "imports over two hours" — same meaning, strict substring miss. Paraphrase penalized, not wrongness.

### Q049 [READY_DETERMINISTIC] — CORRECT
- trials: [1, 2, 3] modes: {'CONTENT_ERROR': 3} multi_mode: False flags: []
- notes: Faithful cause/impact/resolution summary; the "violation" is mentioning database health, which is source-faithful but spec-forbidden. Truthful output, scope failure.

### Q050 [READY_DETERMINISTIC] — LIKELY_CORRECT
- trials: [1, 2, 3] modes: {'CONTENT_ERROR': 3} multi_mode: False flags: []
- notes: Team-selection meaning present ("has not yet selected"), but "multi-step" genuinely absent (replaced by "task complexity"). One real omission.

### Q051 [READY_JUDGE] — CORRECT
- trials: [1, 2, 3] modes: {'CONTENT_ERROR': 3} multi_mode: False flags: []
- notes: All policy facts present; missing fact "carry" is satisfied by "carried forward" (morphology). Strict substring miss, same class as Q012-plants/Q048.

### Q054 [READY_DETERMINISTIC] — CORRECT
- trials: [1, 2, 3] modes: {'OUTPUT_CONTRACT': 3} multi_mode: False flags: []
- notes: Opens with NO = expected with valid modal-logic reasoning (possibility vs necessity). Appendix only.

### Q055 [READY_DETERMINISTIC] — CORRECT
- trials: [1, 2, 3] modes: {'OUTPUT_CONTRACT': 3} multi_mode: False flags: []
- notes: Reaches YES = expected via a valid proof by contradiction. Appendix only.

### Q066 [READY_DETERMINISTIC] — CORRECT
- trials: [1, 2, 3] modes: {'CONTENT_ERROR': 3} multi_mode: False flags: []
- notes: Owners/shares exact; project and milestone names carry source status words ("Project Nova", "Prototype complete"). Naming format, not substance.

### Q068 [READY_DETERMINISTIC] — CORRECT
- trials: [1, 2, 3] modes: {'OUTPUT_CONTRACT': 3} multi_mode: False flags: []
- notes: Derives and simplifies to 4/15 = expected (even excludes the yellow-ball distractor correctly). Boxed-prose packaging only.

### Q070 [READY_DETERMINISTIC] — CORRECT
- trials: [1, 2, 3] modes: {'OUTPUT_CONTRACT': 3} multi_mode: False flags: []
- notes: Derives 18 = expected (30 initial failures, 12 retried). Packaging only.

### Q071 [READY_DETERMINISTIC] — CORRECT
- trials: [1, 2, 3] modes: {'OUTPUT_CONTRACT': 3} multi_mode: False flags: []
- notes: Timeline reaches 00:55 = expected with all intermediate steps right. Strict HH:MM parse rejects the prose; packaging only.

### Q080 [READY_DETERMINISTIC] — INCORRECT
- trials: [1, 2, 3] modes: {'CONTENT_ERROR': 3} multi_mode: False flags: []
- notes: Miscounts consistency: outputs accuracy 1.0/true vs true 0.8/false (4 of 5 = 0.8). Genuine reasoning error on a meta-consistency question.

## Lineage context (V1 512-token vs V2 2048-token)

- Across all 216 measured rows, the persisted benchmark verdict was identical between V1 and V2.
- Within the 204 READY_DETERMINISTIC rows, the PASS/FAIL matrix was identical row-for-row.
- 34/35 failed tasks byte-identical across lineages; Q025 diverged (completed wrong final + mid-derivation assertions into truncation).

## Threats to validity

- Substring semantics are frozen for eval-v1: required/forbidden term matching is substring-based (e.g. "plant" matches inside "plants", "carried forward" does not satisfy "carry", "over two hours" does not satisfy "longer than two hours"). Human interpretation is token-level; the grader is character-level. This is intentional and versioned, not a defect to hotfix — but cross-study comparisons must account for it.
- Generation-cap entanglement: Q025 trials hitting num_predict truncation cannot establish what final the model would have asserted with a larger budget. Truncation is recorded orthogonally (TRUNCATED_AT_NUM_PREDICT) and never merged into failure modes.
- Strict output contracts (bare numbers, exact tokens, ISO dates) punish verbosity and paraphrase even when substance is correct. The 52.5% contract accuracy therefore understates reasoning capability by design; Table 2 quantifies the gap.
- Single-model scope: every substantive label below describes Qwen3-4B Q4 outputs only. The deterministic mode taxonomy is frozen for reuse; human substantive labels must never be auto-generated for other models.

## Cross-model decomposition shell

Qwen Q4 column live under frozen failure-modes-v1; Q5+ reserved.

## Provenance

```json
{
  "grading_spec_hash": "d1cff373d5216f9d1d86d863bd4d86468d05d1f6a43a4e2cab839e257947718f",
  "dataset_hash": "0cdfc4f5099b4822e0de1d01e8807b54441042407cde1293fd6f5d07492c315b",
  "taxonomy_version": "failure-modes-v1",
  "taxonomy_hash": "fc5515a1c568745ad05c566b5ac0448b03a6ad69669f9a98edce76e57335dc30",
  "review_file": "analysis\\reviews\\qwen-q4-v2-failure-review.yaml",
  "review_file_hash": "d0ddb4266521f6e1fc8ad246f21888cce1a4bd4fe91b71a1a5c64778dec82e9f",
  "review_status": "APPROVED",
  "reviewed_by": "iamsan",
  "reviewed_on": "2026-09-18",
  "analysis_code_git_commit": "b5382ec54d3ad2a8b30d7e1a3fea08042e51c8da",
  "analysis_worktree_dirty": false
}
```
