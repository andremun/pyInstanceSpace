# Reliability and TRACE3 remediation plan

> **Superseded historical plan (2026-08-20).** This records the branch-two design and
> its former 2D/current-gold limits. The implemented branch-three contract includes
> native 3D PILOT, TRACE3, plotting, and serialization, backed by the installed
> 423-file R2026a Update 4 v2 oracle (84 provenance tests; 36 current readers). See
> `docs/architecture.md` and `docs/implemented_fixes.md` for current status.

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
output hashes, clean-source requirements, complete post-validation/default option trees,
raw TRACE metrics, region-aware geometry, explicit build and explore inputs, explicit
empties, and atomic publication.

Add a Python verifier with synthetic valid and invalid fixture bundles. Its strict
reference profile must reject missing stages or variants even when a file and its manifest
entry are removed together. Create a complete inventory that classifies current files as
MATLAB verified, Python regression, or legacy-unknown. Install issue #310 data only at
`tests/fixtures/matlab/current/{shared_inputs,resolved_options,build_data,explore_data}`.

The exporter ran from clean source under MATLAB R2026a before any fixture moved. The
strictly verified bundle was then installed in the unified layout with current readers;
#278 and #310 remain open only for maintainer review, not for missing local evidence.

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
- Every variant links a complete effective MATLAB option tree, and build/explore parity
  artifacts include the inputs needed to reproduce their stage result.
- The reference profile rejects an internally consistent partial manifest.
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

## Completed current-gold run

After Financial Toolbox was installed, verified mode completed under MATLAB R2026a Update
4 from clean MATLAB `34c0129` and generator `b87179f`. The strict verifier accepted all 229
manifest-listed files and the bundle was installed atomically at
`tests/fixtures/matlab/current/`. Current-layout numerical readers cover every exported
build and explore stage. TRACE3 build parity is within floating precision for both
variants; two default-explore memberships are pinned as export-precision boundary
ambiguities. Switching Python's TRACE default remains a separate maintainer decision.
