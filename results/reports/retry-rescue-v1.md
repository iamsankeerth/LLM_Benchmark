# Retry-Rescue V1: Qwen Q4 contract-aware prompt retry

Primary recovery: 12 / 39 = 30.8%
Control recovery: 0 / 12 = 0.0%
Post-retry pass rate: 119 / 204 = 58.3% (baseline 52.5%, uplift +5.9%)

Content-change among recovered: contract_only=12, with_change=0, unknown=0

Baseline immutable; single retry per identity; frozen grader exactly.
