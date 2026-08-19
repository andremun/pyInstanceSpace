# Implemented fixes

## Scope

- Branch: `codex/open-issue-big-rocks`
- Base: `e12e92f` from `v0.9.0/development-branch-QSF`
- Reference: local MATLAB InstanceSpace v0.9.0 source
- Review date: 2026-08-17

This pass addresses verified correctness and lifecycle failures. It does not port
TRACE3, add 3D support, or redesign the build/explore API.

## GitHub issues

| Issue | Result |
|---|---|
| [#302](https://github.com/andremun/pyInstanceSpace/issues/302) | Repaired legacy TRACE masks, contradiction refinement, empty geometry, zero-support triangles, no-cluster output, 1D distances, and DBSCAN dtypes. |
| [#314](https://github.com/andremun/pyInstanceSpace/issues/314) | Single-algorithm PYTHIA portfolios now select index `0`; `-1` remains the no-selection value. |
| [#317](https://github.com/andremun/pyInstanceSpace/issues/317) | PYTHIA summary accuracy and precision now map to the correct columns. |
| [#315](https://github.com/andremun/pyInstanceSpace/issues/315) | No requested code change. MATLAB `polyshape.isinterior` includes boundary points, so Python retains inclusive membership and adds a regression test. |

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

## Compatibility notes

- Correct inclusive boundary membership can materially change TRACE geometry on
  boundary-heavy data.
- `correct_results_simulation.csv` is a Python regression baseline, not a verified
  MATLAB oracle. MATLAB export provenance remains tracked by issue #278.
- The public mixed index contract is intentional and documented in
  `docs/architecture.md`.

## Verification

- Behavioral tests: **512 passed**.
  - 505 passed inside the managed sandbox.
  - Seven process-pool tests blocked by sandbox semaphore access then passed with
    normal process permissions.
- Production source: Ruff clean.
- Production source: Black clean.
- Production source: strict mypy clean across 25 files.
- Patch whitespace: clean.

Repository-wide legacy test lint, formatting, typing, and warning debt is recorded
in `docs/pending_issue_backlog.md`.
