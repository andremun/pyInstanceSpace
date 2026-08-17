# Reliability and TRACE3 remediation plan

## Baseline

- Python base: `codex/open-issue-big-rocks` at `d175048`
- Work branch: `codex/validation-serialization-trace3`
- MATLAB reference: `andremun/InstanceSpace` at `34c0129`
- Open issues in scope: #278, #310, and #313
- Audit findings in scope: input validation, output integrity, and warning cleanup

## Delivery sequence

### 1. Validate inputs and options

Implement metadata schema and shape validation without breaking feature-only explore data.
Normalize manual selection names and reject fail-open selection. Reject nonviable data after
washing.

Replace weak option checks with strict finite-real, integer, Boolean, enum, range, array,
and cross-field checks. Add stage-context validation for PILOT arrays, PYTHIA parameters,
and cross-validation folds.

Tests cover every rejected schema class, one-row data, valid `NaN` data, direct versus file
loading, manual selections, washed-empty data, every active option group, and contextual
array shapes.

### 2. Repair serialization and numerical contracts

Make table exports idempotent and model-preserving. Add footprint CSV v2, compound geometry
plots, safe unique path stems, structured archives, and contextual write failures.

Centralize finite-aware normalization. Preserve undefined PYTHIA metrics as `NaN` and keep
selection stable. Replace the warning-producing alpha-shape dependency path with a local
array-based primitive shared by legacy TRACE and TRACE3.

Tests deep-compare models before and after repeated exports. They cover components, holes,
unsafe and colliding labels, constant and missing data, write failures, duplicate basenames,
undefined metrics, local alpha-shape geometry, and warnings promoted to errors.

### 3. Establish fixture provenance

Correct the MATLAB exporter before moving fixtures. Add a versioned manifest, input and
output hashes, clean-source requirements, complete resolved options, raw TRACE metrics,
region-aware geometry, explicit empties, and atomic publication.

Add a Python verifier with synthetic valid and invalid fixture bundles. Create a complete
inventory that classifies current files as MATLAB verified, Python regression, or
legacy-unknown. Add a migration map for issue #310.

Run the exporter on clean MATLAB R2025a+ before marking issue #278 complete. Only then move
verified fixtures into the unified layout and update readers in one commit. Until that run,
the verifier and migration tooling are complete but the provenance gate remains open.

### 4. Port TRACE3

Add method-aware options and dispatch. Implement the two-dimensional radius-unit alpha-shape
engine, MATLAB construction loop, stateful region threshold, parallel contract, raw metrics,
and trained-geometry rescoring.

Keep the legacy path stable. Keep the Python default on legacy. Add focused geometry,
orchestration, metric, option, sequential and parallel, rescore, and explore tests. Add a
frozen Python regression baseline only when useful, and label it as such.

### 5. Verify and document results

Run:

1. focused validation tests;
2. focused serializer and warning-as-error tests;
3. legacy and TRACE3 tests;
4. fixture verifier tests;
5. build and explore integration tests;
6. the complete pytest suite;
7. Ruff on production and changed tests;
8. Black check;
9. strict mypy on production and changed tests;
10. `git diff --check` and repository status checks.

Record exact commands, pass counts, warning changes, compatibility changes, provenance
status, and remaining blockers in `docs/implemented_fixes.md` and
`docs/pending_issue_backlog.md`.

## Acceptance criteria

### Validation

- Malformed or ambiguous metadata cannot reach a stage.
- Standard builds have viable dimensions.
- Manual selections cannot silently widen an experiment.
- Every active option has an eager, named validation failure.
- Direct and file-loaded options follow one contract.

### Serialization and warnings

- Repeated export leaves the model unchanged and produces deterministic content.
- Geometry output preserves every polygon part and hole.
- No label can escape the output root or overwrite another label's file.
- ZIP members preserve relative paths and are unique.
- Constant, empty, missing, and undefined numerical cases do not warn.
- Legacy TRACE geometry remains within its existing regression contract.

### Fixture provenance

- The manifest verifier rejects missing, altered, stale, dirty, or ambiguous bundles.
- Every committed fixture has one explicit trust class.
- No legacy-unknown or Python-regression file is called a MATLAB oracle.
- Issue #310 moves only data that passed a real current-MATLAB provenance run.

### TRACE3

- Legacy TRACE remains behaviorally stable.
- TRACE3 uses truth, predictions, and portfolio indices at their correct boundaries.
- Supporting-point, area, purity, radius, threshold, and fallback rules match current MATLAB
  source.
- Parallel and sequential execution agree.
- Explore rescoring never rebuilds trained geometry.
- The public empty-footprint and summary contracts remain consistent.

## Known external gate

The discovered MATLAB R2024a installation is below the gold repository's R2025a requirement.
Its first batch probe failed before the user refreshed the MATLAB login, so it will be
tested again. A clean R2025a+ environment with the required toolboxes and license remains
the required source for verified reference data. This is the only authorized basis for
closing #278 and completing the fixture move in #310.
