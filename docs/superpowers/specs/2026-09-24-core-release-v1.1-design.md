# Core Release V1.1 Design

## Objective

Close the core release gaps identified after the canonical V2 sweep without
rewriting historical Eval-v1 evidence. The release makes grading semantics
satisfiable, binds future provenance to actual bytes, hardens CI, and runs the
fixed-budget Qwen temperature study under the corrected current contract.

## Version Boundaries

Eval-v1 remains immutable. Its YAML, database outputs, manifests, summaries,
comparisons, reports, and freeze records remain historical evidence.

Eval-v1.1 is a new versioned grading contract:

- Q016 remains `READY_DETERMINISTIC`, with an empty fixed-value map, explicit
  open-value constraints, the `stale` lexical root, and Q016-only word-boundary
  matching for `latency`.
- Q047 becomes `READY_JUDGE` / `JUDGE_REQUIRED`, with deterministic fact
  prechecks and a required unsupported-claims rubric.
- The corrected mandatory population is 67 deterministic tasks and 5 judge
  tasks: 201 deterministic trials and 15 judge precheck trials at three trials.

The new freeze pins the grading YAML, schemas, grader engine, verdict reducer,
regrade code, executable dataset, workbook lineage, and release metadata.

## Append-Only Regrade

Existing raw outputs are not regenerated or modified. A regrade overlay records
source execution/model/task/trial identity, raw-output hash, old and new spec
hashes, grader-engine hash, old verdict/details, and new verdict/details.

The overlay is idempotent and refuses source hash or population drift. Historical
reports remain untouched; corrected capability, comparison, Pareto, and sweep
artifacts use distinct Eval-v1.1 names.

## Actual-Byte Contract

A shared contract loader reads the exact selected grading YAML, executable JSONL,
and freeze record, hashes their current bytes, compares them to the freeze, and
fails before database creation, model loading, or output writes.

Manifest schema v2 records separate actual hashes for the grading spec, dataset,
freeze record, adapter/template, and Git lineage. Future summaries may not stamp
a new grading hash over old persisted verdicts; corrected views must consume the
regrade overlay.

The immutable workbook conversion manifest remains a `DRAFT` conversion
snapshot. A machine-readable current-release index distinguishes conversion
status, frozen dataset-content status, authoritative grading status, and the
current release version.

## Strict Temperature Execution

The temperature arm path is separate from the generic recovery path:

- It rejects resume, task truncation, output-cap overrides, and noncanonical
  execution IDs before any model request.
- It refuses any existing execution ID before the load probe.
- It performs one canonical initial load with the active arm temperature,
  adapter mode/template, context, GPU policy, thinking policy, and stops.
- It records one diagnostic request, two warmups, and 75 measured requests per
  arm: 78 requests and 77 persisted generations per arm, or 156 requests and
  154 persisted generations across the two-arm study.
- Any probe, warmup, measured, grading, persistence, or reload-evidence fault
  stops immediately without retry, reload, re-warm, or synthetic error rows.
- Failure preserves the partial database, writes a typed sidecar, closes SQLite,
  and unloads the model without automatic deletion or resume.

Successful arms require exactly two warmups and 75 complete measured rows. The
paired report independently validates warmups, measured identities, source and
grading hashes, model digests, templates, run hashes, manifests, and
non-temperature setting equality.

## Release Hardening

Three hash-pinned critical SQLite evidence files become tracked release
artifacts; ordinary working databases remain ignored. A platform-neutral quality
gate runs Ruff, strict mypy, offline tests, workbook reproducibility, grading
audits, and portable frozen-artifact checks. A Windows/Linux CI workflow runs
the same gate; hosted activation is not performed because no remote is
configured.

Historical report-freeze hashes are not rewritten for this release. Their
platform-specific status is explicitly reported rather than falsely certified
as portable.

## Verification And Live Gate

Offline tests cover satisfiability, Q016 boundaries, Q047 status, regrade
idempotence, actual-byte tamper refusal, manifest provenance, strict execution
ordering, all failure paths, exact row counts, report rejection, and CI command
composition.

After the implementation commit and complete offline gate, obtain explicit live
approval. Run `t0` and `t07` in order, generate and freeze the paired report,
unload/remove Qwen, verify cleanup, and commit result artifacts separately.

## Completion

Core completion requires the Eval-v1.1 companion release, tracked evidence,
passing quality gate, committed CI workflow, successful fixed-budget temperature
artifacts, and cleanup evidence. Optional suites are separate release trains and
do not alter the mandatory Eval-v1.1 denominators.
