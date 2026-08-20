# Implemented fixes

## Scope

- Branch: `codex/validation-serialization-trace3`
- Base: `d175048` from `codex/open-issue-big-rocks`
- Reference: MATLAB InstanceSpace at `34c0129`
- Review date: 2026-08-18

This pass extends the earlier correctness work with strict validation, safe
serialization, fixture provenance tooling, warning cleanup, and an opt-in
two-dimensional TRACE3 port. Three-dimensional TRACE3 and a default-method switch are
deliberately deferred.

## GitHub issues

| Issue | Result |
|---|---|
| [#302](https://github.com/andremun/pyInstanceSpace/issues/302) | Repaired legacy TRACE masks, contradiction refinement, empty geometry, zero-support triangles, no-cluster output, 1D distances, and DBSCAN dtypes. |
| [#314](https://github.com/andremun/pyInstanceSpace/issues/314) | Single-algorithm PYTHIA portfolios now select index `0`; `-1` remains the no-selection value. |
| [#317](https://github.com/andremun/pyInstanceSpace/issues/317) | PYTHIA summary accuracy and precision now map to the correct columns. |
| [#315](https://github.com/andremun/pyInstanceSpace/issues/315) | No requested code change. MATLAB `polyshape.isinterior` includes boundary points, so Python retains inclusive membership and adds a regression test. |
| [#313](https://github.com/andremun/pyInstanceSpace/issues/313) | Ported MATLAB TRACE3 for two-dimensional spaces behind `method="trace3"`, including trained-geometry explore rescoring. Legacy remains the Python default. |
| [#278](https://github.com/andremun/pyInstanceSpace/issues/278) | Generated and independently verified the complete 229-file profile from clean source under MATLAB R2026a Update 4 with all required toolboxes. |
| [#310](https://github.com/andremun/pyInstanceSpace/issues/310) | Installed the reviewed bundle atomically at `tests/fixtures/matlab/current/`, classified it as `matlab-verified`, and added current-layout numerical readers. |

## Audit fixes

- Convert one-based PRELIM `p` only at TRACE and plotting boundaries; preserve
  zero-based PYTHIA `selection0`.
- Validate portfolio indices before TRACE and fix portfolio labels and masks.
- Normalize PRELIM's derived performance matrix, including sparse NaNs.
- Honor preprocessing, SIFTED, and parallel disable flags.
- Validate one-based subset files and retain the final valid instance index.
- Apply SIFTED density filtering after every enabled selection path.
- Preserve PRELIM's dense data separately in saved models.
- Isolate runner snapshots, persist successful overrides, and make
  `run_until_stage()` inclusive.
- Invalidate stale models after staged reruns and finalize generators only after
  complete execution.
- Avoid creating TRACE worker pools when parallel execution is disabled.
- Normalize empty TRACE footprints to `polygon=None`.
- Reject malformed metadata, invalid active options, and nonviable build/explore
  dimensions before stage execution.
- Make serialization model-preserving, path-safe, region-aware, and structurally safe
  for archives.
- Classify every fixture by trust level and verify hashes, schemas, source state, complete
  effective options, canonical stage coverage, and exact file sets before installation.
- Export explicit build and explore inputs so stage parity, membership, and rescoring are
  reproducible without rerunning MATLAB preprocessing.
- Replace legacy TRACE's warning-producing alpha dependency with a local Delaunay
  implementation while preserving its regression output.
- Remove the unused `alphashape` package and its orphaned lockfile dependencies; update
  the live demo dependency description.
- Implement TRACE3's true-label/prediction contract, all-points alpha radius, exact
  100-step tightening loop, shared-vertex region threshold, parallel parity, and
  fixed-geometry explore rescoring.
- Translate MATLAB SVM `KernelScale` to scikit-learn `gamma` at the estimator boundary,
  preserve MATLAB units in public output, and validate contextual PYTHIA parameters.
- Replace permissive, provenance-free Bayesian metric comparisons with deterministic
  integration and estimator-unit contracts. The old CSVs remain `legacy-unknown` data.
- Correct numerical PILOT's MATLAB-to-Python slice so its reported `C` matrix retains
  every algorithm column and reconstructs the complete fitted response.
- Allow MATLAB-supported sparse-class cross-validation with a named warning while still
  rejecting splits that leave an unusable training fold. This unblocks the verified
  default KNN variant without weakening impossible-layout checks.

## Compatibility notes

- Correct inclusive boundary membership can materially change TRACE geometry on
  boundary-heavy data.
- `correct_results_simulation.csv` is a Python regression baseline, not a verified
  MATLAB oracle.
- The R2024a bundle remains diagnostic evidence. The committed current oracle was generated
  under R2026a Update 4 from clean MATLAB and Python commits with all five required
  toolboxes, then passed the strict verifier before installation.
- TRACE3 is opt-in and two-dimensional. Legacy remains the default; 3D support and a
  default switch require separate review.
- The public mixed index contract is intentional and documented in
  `docs/architecture.md`.

## Verification

- Repository-wide integration checkpoint: **787 passed** with normal process
  permissions; no failures or exclusions.
- Both R2024a TRACE3 variants matched every exported good, best, hard, and space
  geometry and raw metric to floating-point precision.
- Focused TRACE3, alpha-shape, build, explore, viability, and provenance checks:
  **98 passed**; Ruff, Black, and strict mypy are clean on the affected files.
- Fixture provenance and strict-profile checks: **42 passed**, including mutations that
  remove both required files and their manifest entries.
- MATLAB R2026a `checkcode` reports no exporter findings. Its first diagnostic run stopped
  at PRELIM because Financial Toolbox was absent; after installation, verified mode
  completed all three variants in 54.4 seconds and published 229 manifest-listed files.
- Current-gold TRACE3 build geometry and raw metrics match both exported variants to
  about `1.3e-13`. Explore fallback is exact; the default variant has two explicitly
  pinned CSV-boundary ambiguities and exact agreement for every off-boundary membership.
- Current-gold stage readers: **16 passed** across PRELIM, SIFTED, PILOT, CLOISTER,
  PYTHIA build/explore, and TRACE build/explore. They account explicitly for randomized
  PRELIM ties, equivalent PILOT rotations/reflections, and tied-neighbour KNN scores.
- Ruff, Black, strict mypy, lock consistency, and patch-whitespace checks pass across all
  36 changed Python files. The full suite reports 70 known third-party warnings, recorded
  in `docs/pending_issue_backlog.md`.
