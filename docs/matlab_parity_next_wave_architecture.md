# MATLAB parity next-wave architecture

## Authority and branch stack

- Integration base: `v0.9.0/development-branch-QSF` at `3a7f21a`.
- Branch 1: `codex/open-issue-big-rocks` at `d175048`, merged by PR #319 as
  `324830c`.
- Branch 2: `codex/validation-serialization-trace3` at `67c73de`.
- Branch 3: `codex/matlab-parity-next-wave`, integrating both lines at `030937d`.
- Gold implementation: MATLAB InstanceSpace at `34c0129`, executed with R2026a
  Update 4.
- Installed evidence: verified `reference-export/v2` under
  `tests/fixtures/matlab/current/`, containing 423 files.

GitHub issues are audit leads, not specifications. MATLAB source and reproduced R2026a
behavior are authoritative unless a deliberate Python safety or compatibility boundary is
recorded below.

## Corrected boundaries

### Reviewer reports

Issue #320 identified a genuinely unused polygon-region helper. It is removed; TRACE3's
active simplex filter remains because MATLAB joins alpha-shape simplices that share even one
vertex. Issue #321 is resolved by one `casefold()` JSON-key canonicalizer shared by validation
and loading, including conflict detection for casefold-equivalent keys.

### PYTHIA

MATLAB keeps KNN's nominal neighbour range at 1--25 and caps `NumNeighbors` at each fold
or final fit. Python applies the same per-fit cap while retaining the requested value for
cloning and reports. This covers Sobol, Bayesian, and precalculated parameters.

Each algorithm derives folds and classifier/search randomness from one-based `seed + i`.
Python retains the actual splitter per algorithm; a degenerate or skipped algorithm stores
`None`. Matching the seed boundary does not make MATLAB and sklearn stratifiers choose the
same rows.

Issue #304's earlier convergence claim used `legacy-unknown` rounded final metrics, not an
optimizer trace. R2026a uses expected-improvement-plus and four seed points. Python retains
the shared evaluation budget, four seed points, and plain EI as skopt's closest base
acquisition. No verified Bayesian trace was added to v2, so further convergence changes need
equal-budget, repeated-seed candidate and objective traces.

### Multi-region alpha shapes

R2026a disproves issue #272's retry premise. Its all-points alpha can intentionally produce
multiple regions. On the pinned two-cluster cloud, MATLAB and Python use radius `sqrt(0.5)`,
retain two unit-area regions, and include all six points. Python preserves this topology.
MATLAB's legacy CSV helper traces only the first boundary cycle; Python does not reproduce
that output-only defect.

## Three-dimensional pipeline

### PILOT and viewpoints

`PilotOptions.dims` is restricted to 2 or 3 and is the aggregate pipeline's dimensionality
source for SIFTED and PILOT. Standard analytic, numerical, and SIMPLS paths support both
dimensions. Numerical vectors use MATLAB column-major packing; the loss averages instances
before columns; rank-deficient analytic input falls back to the numerical solver. MATLAB's
ten default restarts and stage-local MT19937 stream are explicit contracts.

Every configured zero-based algorithm group produces one `2 x 3` view matrix and
azimuth/elevation in radians. An empty group list resolves to one global group; overlapping
groups are valid and the first match wins. Viewpoints are persisted with the model. Public
PLS explore retains MATLAB's uncentred `Z = X @ A.T` behavior even though its build projection
is centred.

The v2 oracle verifies five PILOT variants on fixed default-SIFTED inputs. This proves PILOT,
viewpoint, and public explore projection behavior; it is not a separate end-to-end
SIFTED-3D dataset.

### TRACE3 geometry

TRACE3 consumes true `Ybin`, true one-based portfolio `P`, optional PYTHIA predictions,
difficulty labels, and 2D or 3D `Z`. Predictions filter algorithm support when available;
they do not replace truth. The hard footprint is never prediction-filtered.

Two-dimensional footprints retain Shapely polygon or multipolygon geometry. Three-dimensional
footprints retain an immutable `TetrahedralMesh` containing vertices, tetrahedra, outward
boundary faces, alpha, region threshold, region count, volume, and surface area. Both engines:

1. use the all-points critical alpha;
2. group simplices by shared-vertex connectivity;
3. keep regions whose summed area or volume is strictly greater than the threshold;
4. apply MATLAB's exact 100-radius tightening loop and stateful threshold;
5. preserve the trained geometry during explore rescoring.

Three-dimensional membership has a bounded vectorized barycentric fast path. Points near a
face are resolved by exact rational orientation predicates over their stored IEEE-754 values.
This includes true boundaries without creating an exterior tolerance shell.

TRACE summaries use `Area_*` in 2D and `Volume_*` in 3D. Both build and explore use MATLAB's
three-decimal, half-away-from-zero rounding. `Footprint.area` remains the legacy storage name
for area or volume; `Footprint.measure` is the dimension-neutral alias.

### Output boundary

Two-dimensional footprint CSV v2 remains stable. Three-dimensional output uses
`pyinstancespace.trace-mesh/v1` with `footprint_meshes.json` and one-based vertex,
tetrahedron, and boundary-face CSVs for every good, best, and hard footprint. Empty meshes
remain explicit and statistics are lossless.

Plots use native 3D axes and `Poly3DCollection` boundary faces. Algorithm plots use their
first matching group camera; global or uncovered plots use the first view. Models without a
persisted viewpoint use MATLAB `view(3)`: azimuth -37.5 degrees and elevation 30 degrees.
Footprint image overlays always use experimental `Ybin` and `P`, independent of
`trace.use_sim`.

## Fixture trust boundary

The installed 423-file v2 oracle was generated from clean MATLAB `34c0129` and clean Python
generator `cf3cde0` under R2026a Update 4. The exporter identity is pinned to
`d11293556b12beb63e3320094a2340ba3f7f8b7a58677ff404f20c0ba3b7350c`.
The manifest records MATLAB plus Statistics and Machine Learning, Optimization, Global
Optimization, and Financial Toolbox.

The profile records eight complete effective option artifacts, explicit stage inputs,
full-precision 3D coordinates, meshes, spectra, raw metrics, membership, and rescored
summaries. The verifier recomputes topology, orientation, geometry, membership, and summary
semantics without trusting row order or manifest descriptions. The frozen 229-file v1
profile remains readable; diagnostic and historical bundles remain non-oracles.

## Intentional compatibility boundaries

- `trace.method="legacy"` remains Python's default; MATLAB defaults to TRACE3. A 3D legacy
  request warns and dispatches to TRACE3 because legacy geometry is two-dimensional.
- skopt plain EI lacks MATLAB expected-improvement-plus's anti-overexploitation loop.
- MATLAB and sklearn cross-validation folds may differ after matching their seed boundary.
- Python rejects malformed explicit `precalcAlpha`; MATLAB can silently fall through to
  `X0`.
- `CloisterOptions.hull_dims="all"` keeps a native n-dimensional hull; `2` selects MATLAB's
  legacy first-two-coordinate hull.
- Python preserves every multi-region CSV component instead of MATLAB's first-cycle helper
  defect.
- Python retains a useful whole-space `good_elements` count; MATLAB leaves the exported field
  unset.

## Acceptance evidence

- Strict v2 provenance and semantic verification: 423 files.
- Provenance tests: 84 passed.
- Current-gold readers: 36 passed, including native 3D TRACE at 3/3 passed.
- Full-suite accounting: 988 passed with 63 documented warnings. The sandbox run reached
  987 passed and one macOS semaphore-permission failure; that exact test passed 1/1
  outside the sandbox.
