# Coding V1 Design

## Objective

Add a separately versioned, production-quality coding suite for Q027–Q034
without activating unsafe host execution or changing the mandatory Eval-v1.1
denominator.

## Contract

Reuse the exact eight source prompts unchanged. Coding-v1 has its own frozen
fixture manifest, grading schema, extraction policy, resource policy, runner
contract, experiment ID, report, and freeze. Q027–Q030 and Q032–Q034 use a
pinned CPython runtime; Q031 uses a pinned Node runtime.

The suite uses three trials, temperature 0.0, the existing model eligibility
contract, and a 2,048-token generation cap. Each eligible model produces 24
measured rows. Coding results remain separate from mandatory capability totals.

## Fixture And Extraction Boundary

Fixtures are authored and reviewed before any model output is inspected. Hidden
tests cover correctness, immutability, edge cases, Unicode, deterministic
complexity, retry counts, and explicit forbidden constructs. Ambiguous edge
cases are fixed in the coding spec before fixture execution.

Accept only the entire visible response as source or exactly one whole-document
Python/JavaScript Markdown fence. Reject empty, oversized, NUL-containing,
truncated, prose-wrapped, or multi-fence responses. Preserve raw and extracted
hashes, extraction method, and truncation state.

## Grader

Static AST/syntax checks enforce prompt-specific restrictions. Functional tests
run in pinned isolated containers. Headline task success is binary: every
required functional and static test must pass. Separate diagnostics report
functional/static test pass rates, parse/entrypoint status, immutability,
complexity, timeout, and resource failures.

## Sandbox

Model-generated code never executes on the Windows benchmark host or through
host `exec`, `eval`, Node `vm`, or an unisolated subprocess. An isolated Linux
OCI worker runs pinned Python and Node images with no network, read-only root,
no host mounts, no Docker socket, dropped capabilities, unprivileged user,
CPU/wall/PID/memory/file/descriptor/output limits, small tmpfs, and guaranteed
process-tree cleanup.

A sandbox startup or infrastructure failure is `ERROR`; candidate timeout,
OOM, or prohibited resource behavior is task `FAIL`.

## Persistence And Reporting

A dedicated code-grader database persists raw/extracted hashes, fixture and
image digests, resource policy, parse/entrypoint results, every test case,
partial rates, failure kind, exit/signal, timing, and memory evidence. Candidate
sources are stored as content-addressed local artifacts. Public reports contain
hashes, counts, rates, and limitations, not unrestricted raw code.

The suite is not merged into the current failure taxonomy or mandatory task
population. A future composite report may reference both Eval-v1.1 and
Coding-v1 hashes explicitly.

## Verification And Rollout

Before model calls, run golden solutions, targeted mutants, extraction fixtures,
sandbox security tests in the isolated worker, and fake-model integration tests.
The full offline quality gate must pass before a separately approved live run.
Results are frozen only after exact population, source, fixture, image, and
resource-policy hashes verify.

## Completion

Coding-v1 is complete only with approved fixtures, pinned runtime evidence,
passing security tests, committed reports/freeze, explicit partial-score
limitations, and cleanup of temporary candidate artifacts and model weights.
