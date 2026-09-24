# Retry-Rescue V3: Deterministic Numeric Renderer

V3 transformed the sealed A baseline output once for each identity in the frozen
18-row numeric primary intersection. It made zero Ollama calls and opened the
baseline database read-only.

| Outcome | Rows | Denominator |
| --- | ---: | ---: |
| Rendered and passed frozen grader | 2 | 18 |
| Rendered but failed frozen grader | 0 | 18 |
| Unrenderable | 16 | 18 |

The two recoveries are Q022 trials 2 and 3, rendered as bare `218.4`. Every
other source output was rejected by the frozen ambiguity-safe renderer rather
than guessing or using expected-answer information.

V3 estimates recovery from deterministic serialization of existing baseline
reasoning. It does not estimate a fresh-generation retry rate. A/B/C remain
historical context only and do not enter this denominator.
