# Implemented fixes

## Scope

- Branch: `codex/matlab-parity-next-wave`
- Integration base: `v0.9.0/development-branch-QSF` at `3a7f21a`
- Branch 1: `codex/open-issue-big-rocks` at `d175048`, merged by PR #319 as
  `324830c`
- Branch 2: `codex/validation-serialization-trace3` at `67c73de`
- Branch 3 integration merge: `030937d`
- Gold implementation: MATLAB InstanceSpace v0.9.1 at `98a01ac`, run with R2026a
  Update 4
- Installed oracle: verified `reference-export/v2`, 423 files
- Review date: 2026-08-21

This branch carries both predecessor branches and completes the largest remaining 3D,
TRACE3, output, and current-MATLAB evidence gaps. GitHub reports are audit leads; MATLAB
source and reproduced R2026a behavior decide parity.

## GitHub issue outcomes

| Issue | Local result |
|---|---|
| [#320](https://github.com/andremun/pyInstanceSpace/issues/320) | Removed the unused polygon-region filter. TRACE3 retains the active vertex-connected simplex-region implementation. |
| [#321](https://github.com/andremun/pyInstanceSpace/issues/321) | Validation and loading now share one `casefold()` JSON-key canonicalizer and reject casefold-equivalent conflicts. |
| [#262](https://github.com/andremun/pyInstanceSpace/issues/262) | Completed 2D/3D PILOT across analytic, numerical, and SIMPLS paths, including SIFTED propagation, MATLAB-order solver contracts, restart defaults, and persisted grouped viewpoints. **[Behavior-changing]** The omitted-value restart default changed from Python's former 5 to MATLAB R2026a's 10; set `n_tries=5` to retain the prior Python budget/output selection. |
| [#265](https://github.com/andremun/pyInstanceSpace/issues/265) | Added native 3D projections, camera-aware plots, TRACE meshes, and versioned numerical serialization without changing the 2D geometry schema. |
| [#313](https://github.com/andremun/pyInstanceSpace/issues/313) | Completed native 2D/3D TRACE3 construction, membership, metrics, parallel execution, and fixed-geometry explore rescoring. Legacy remains selectable. |
| [#272](https://github.com/andremun/pyInstanceSpace/issues/272) | Superseded by R2026a evidence: the all-points alpha can intentionally contain multiple regions, so Python preserves them and adds no single-region retry. |
| [#278](https://github.com/andremun/pyInstanceSpace/issues/278) | Extended the clean-source R2026a provenance profile to 423 files with PILOT and 3D TRACE evidence. |
| [#310](https://github.com/andremun/pyInstanceSpace/issues/310) | Installed the verified v2 bundle atomically at `tests/fixtures/matlab/current/`; historical data remains separately classified. |
| [#304](https://github.com/andremun/pyInstanceSpace/issues/304) | Corrected the proven per-algorithm RNG boundary. The reported convergence premise used unverified rounded metrics, so optimizer-trace evidence remains pending and defaults did not change. |
| [#316](https://github.com/andremun/pyInstanceSpace/issues/316) | Added a typed `PredictiveStage` contract for PRELIM, SIFTED, PILOT, PYTHIA, and TRACE. `InstanceSpace` now orchestrates stage-owned inference without changing `StageRunner` or persisted model schemas. |

The stacked predecessor work also resolved #302, #314, and #317 and rejected #315's
boundary-exclusive proposal. Those four issues are already closed upstream.

## Additional corrections

- Preserve KNN's MATLAB-facing search range of 1--25 while capping neighbours at each
  fold or final fit; reported parameters retain the requested value.
- Derive PYTHIA folds and classifier/search randomness from one-based `seed + i` and
  retain the actual splitter per algorithm.
- **[Behavior-changing]** Match MATLAB PILOT's ten default restarts (formerly five in
  Python), seeded stage-local MT19937 starts, valid `precalcAlpha` precedence, column-major
  packing, loss axes, and rank fallback. The default and explicit five-restart override
  are pinned in option tests; the ten generated starts match the verified R2026a export.
- **[Behavior-changing]** Match v0.9.1's seeded random PRELIM tie-breaking and new
  `sifted.seed`/`pilot.seed` options. Both stage seeds inherit `general.seed` when omitted.
- Persist 3D viewpoint matrices and radian angles; an empty group list means one global
  view and overlapping groups remain valid.
- Add a tetrahedral alpha complex with strict volume thresholds, vertex-connected
  regions, outward boundary faces, volume and surface metrics, and descending spectra.
- Resolve near-face 3D membership with exact predicates over the stored IEEE-754 values.
  Boundaries are inclusive without admitting an exterior tolerance shell.
- Label 3D TRACE summaries with `Volume_*` and use MATLAB's three-decimal,
  half-away-from-zero rounding for build and explore.
- Serialize 3D footprints as `pyinstancespace.trace-mesh/v1`: a manifest plus one-based
  vertex, tetrahedron, and boundary-face tables with explicit empty records.
- Render native 3D scatter and mesh plots with persisted group cameras or MATLAB's
  `view(3)` fallback. Footprint overlays use experimental truth regardless of
  `trace.use_sim`.
- Preserve trained TRACE geometry during explore and rescore only membership and raw
  evidence metrics.

## Carried-forward reliability work

- Strictly validate metadata, active options, selection files, stage viability, and
  dimensional contracts before execution.
- Keep PRELIM's one-based portfolio and PYTHIA's zero-based `selection0` boundaries
  explicit; `-1` remains no selection.
- Isolate stage-runner snapshots, invalidate stale models, and honor disabled
  preprocessing, SIFTED, and parallel paths.
- Keep serialization model-preserving, path-safe, region-aware, deterministic, and safe
  for structured archives.
- Preserve empty footprints canonically, all 2D polygon parts and holes, and trained
  classifiers through model save/load.
- Keep legacy and historical fixtures explicitly separate from manifest-verified MATLAB
  evidence.

## Intentional compatibility boundaries

- Python keeps `trace.method="legacy"` as its default; MATLAB defaults to TRACE3. A
  default switch requires a separate versioned decision. A 3D legacy request warns and
  dispatches to TRACE3 because legacy geometry is two-dimensional.
- MATLAB uses expected-improvement-plus for Bayesian tuning. skopt plain EI is the
  closest available base acquisition and lacks MATLAB's anti-overexploitation loop.
- MATLAB and sklearn stratifiers may choose different folds even after matching the
  `seed + i` boundary.
- Python rejects a malformed explicit `precalcAlpha`; MATLAB can silently fall through
  to `X0`.
- `CloisterOptions.hull_dims="all"` keeps a native n-dimensional hull. Set it to `2` for
  MATLAB's legacy first-two-coordinate hull.
- Python preserves hole cycles that MATLAB v0.9.1's CSV helper still omits; both preserve
  every disconnected region.
- Python retains a useful whole-space `good_elements` count in memory; MATLAB leaves the
  corresponding exported field unset.
- `Footprint.area` remains the compatibility field for 2D area or 3D volume;
  `Footprint.measure` is the dimension-neutral alias.

## Verification

- The installed v2 oracle contains 423 manifest-listed files generated from clean MATLAB
  v0.9.1 at `98a01ac` and clean Python generator `4816b8c` under R2026a Update 4.
- The run records MATLAB plus Statistics and Machine Learning, Optimization, Global
  Optimization, and Financial Toolbox.
- Exporter identity:
  `d11293556b12beb63e3320094a2340ba3f7f8b7a58677ff404f20c0ba3b7350c`.
- Provenance, strict-profile, identity, and semantic mutation tests: **86 passed**.
- Current-gold scientific readers: **41 passed**, including native 3D TRACE build,
  topology, spectra, membership, and rescore at **3/3 passed**.
- Full-suite accounting: the local CI-equivalent gate passed all **1,046 collected**
  tests. Branch coverage was **92.08%**, with no uncaught warnings under `-W error`.
