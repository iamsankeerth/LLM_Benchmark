# Retry-Rescue V2: constrained decoding

## Primary Intersection

The primary result applies only to the frozen, representable 18 of 39
`OUTPUT_CONTRACT` identities. A and B are sealed read-only evidence; C used the
unchanged original prompts with task-specific response-format constraints.

| Arm | Recovered | Denominator |
| --- | ---: | ---: |
| A: sealed baseline | 0 | 18 |
| B: V1 prompt retry | 0 | 18 |
| C: constrained decoding | 0 | 18 |

Paired B/C outcomes: both PASS `0`, B-only PASS `0`, C-only PASS `0`, neither
PASS `18`. Exact two-sided McNemar p-value: `1.0` (secondary/descriptive).

No C primary row recovered, so content-change buckets are all zero:
`contract_only=0`, `recovered_with_content_change=0`, `unknown=0`.

## Controls

Controls remain separate from the primary denominator:

| Arm | Recovered | Denominator |
| --- | ---: | ---: |
| B: V1 prompt retry | 0 | 6 |
| C: constrained decoding | 0 | 6 |

The 24 measured C rows used 21 `NUMBER` constraints and 3 `STRUCTURED_JSON`
constraints. Two persisted warmups are excluded from every outcome denominator.

## Scope

Results from retry-rescue-v2 apply to the **18 of 39 OUTPUT_CONTRACT rows whose
frozen output contracts are representable by the tested constrained-decoding
mechanism**. They do not estimate constrained-decoding recovery across all 39
OUTPUT_CONTRACT failures.
