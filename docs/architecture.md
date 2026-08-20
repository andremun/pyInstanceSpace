# ISA architecture and parity contract

## Purpose

Instance Space Analysis explains where algorithms perform well and why.
It relates instance features to algorithm performance in a low-dimensional space.
The result supports benchmarking, algorithm selection, and benchmark coverage analysis.

The MATLAB repository is the behavioral authority.
Its current source and verified R2026a execution take priority over issue descriptions,
reviews, papers, old option names, and archived plans. A Python divergence is accepted
only when it is explicit, tested, and documented below.

## Pipeline

| Stage | Responsibility | Main trained output |
|---|---|---|
| PREPROCESSING | Load, clean, and optionally subset metadata | Dense and selected input data |
| PRELIM | Define good performance, clip outliers, transform, and scale | Labels, bounds, and transform parameters |
| SIFTED | Select predictive and non-redundant features | Selected feature indices |
| PILOT | Project selected features into a 2D or 3D instance space | Projection, reconstruction, and optional viewpoints |
| CLOISTER | Estimate feasible projected bounds | Empirical and correlation-aware hulls |
| PYTHIA | Train one good-performance classifier per algorithm | Classifiers, predictions, metrics, and selections |
| TRACE | Build good, best, hard, and full-space footprints | 2D polygons or 3D tetrahedral meshes and metrics |

Python runs PYTHIA and CLOISTER after PILOT.
TRACE depends on PYTHIA but not on CLOISTER.
The built-in order is fixed in `instancespace/instance_space.py`.

## Build and explore

`InstanceSpace.build()` fits all selected stages through `StageRunner`.
The final `Model` stores processed data, stage outputs, options, classifiers, footprint
geometry, and any optimized PILOT viewpoints.

`InstanceSpace.explore()` applies the trained model to new metadata.
It must not refit feature selection, projection, classifiers, or footprints.
It applies the stored PRELIM parameters, selected features, PILOT matrix, PYTHIA models,
and TRACE polygons or meshes. TRACE membership is rescored against the new instances;
the trained geometry is not rebuilt.

Known algorithms align by name without case sensitivity.
New test algorithms receive placeholder prediction and footprint columns.
Ground-truth performance enables evaluation metrics but does not retrain the model.

## Reliability invariants

- PRELIM `p` keeps MATLAB's one-based algorithm indices.
- PYTHIA `selection0` uses zero-based algorithm indices.
- `-1` means that PYTHIA made no selection.
- A single-algorithm portfolio can only select index `0`.
- PILOT supports 2D and 3D standard analytic/numerical projection and R2026a SIMPLS.
- A 3D PILOT result stores one global viewpoint or the configured algorithm-group views.
- Public PLS explore projection remains uncentred (`Z = X @ A.T`), matching MATLAB's
  observable `exploreIS` behavior even though the trained SIMPLS coordinates are centred.
- Empty TRACE footprints use `polygon=None`, retain their dimension, and have zero metrics.
- TRACE membership is boundary-inclusive during build and explore, matching MATLAB
  `polyshape.isinterior` and `alphaShape.inShape`.
- Ambiguous 3D face cases use an exact tetrahedral orientation predicate; tolerances
  select the fallback path and never create an inclusion shell.
- TRACE3 geometry and summaries are dimension-aware: area in 2D, volume in 3D.
- A 3D footprint is a tetrahedral mesh with outward boundary faces; serialized indices
  are one-based while Python's in-memory indices remain zero-based.
- Summary columns must match their named raw metrics.
- Build and explore must use the same fitted transformations and selection rules.
- Disabled preprocessing, feature-selection, and parallel flags must be no-ops.
- Stage rollback must restore an isolated snapshot and invalidate derived models.
- Stage output shapes and dtypes must remain stable across serializing and loading.

## MATLAB parity boundaries

Python keeps its public API and typed stage architecture.
It does not need a line-by-line MATLAB translation.
Parity means matching observable contracts, formulas, edge cases, and trained-model behavior.

Current supported parity includes:

- PILOT standard analytic/numerical and SIMPLS projection in 2D and 3D;
- global and grouped 3D viewpoint optimization;
- TRACE3 2D polygons and 3D tetrahedral meshes, including build metrics, membership,
  alpha selection, and explore rescoring; and
- native 3D plotting and versioned mesh serialization.

The deliberate compatibility choices are:

- MATLAB defaults to TRACE3; Python retains legacy TRACE as the 2D default. An explicit
  `method="trace3"` selects TRACE3, and 3D always warns and dispatches to it.
- Python option indices, including PILOT viewpoint groups, are zero-based. MATLAB fixture
  indices are translated at the boundary.
- Python's `cloister.hull_dims="all"` default preserves the complete projected hull;
  `hull_dims=2` selects MATLAB's first-two-coordinate behavior.
- Explore features are matched by name rather than requiring MATLAB's input column order.
- `adjust_rotation` is an optional Python 2D visualization aid and is not a MATLAB parity
  requirement.

Build logic remains in stage classes. PRELIM, SIFTED, PILOT, PYTHIA, and TRACE also own
their fitted inference through the typed `PredictiveStage.predict()` contract implemented
for #316. `InstanceSpace` retains explore orchestration and compatibility wrappers; see
`docs/stage_inference_architecture.md`.

## Verified MATLAB oracle

`tests/fixtures/matlab/current/` contains the installed 423-file
`reference-export/v2` oracle from clean MATLAB R2026a Update 4 and pinned clean source
commits. Its manifest binds resolved options, canonical inputs, exporter identity, file
hashes, numeric shapes, and explicit empty artifacts.

The verifier independently recomputes PILOT solver selection and reconstruction plus
TRACE geometry, topology, spectra, metrics, exact membership, and rescoring before a
bundle can be installed. Historical data in `tests/matlab_reference/` remains
unverified and cannot establish parity.

## Source map

- MATLAB orchestration: `InstanceSpace.m`
- MATLAB stages: `core/PRELIM.m` through `core/TRACE.m`, including `core/PILOT.m`
  and `core/PILOTviewpoint.m`
- Python orchestration: `instancespace/instance_space.py`
- Python execution: `instancespace/stage_runner.py`
- Python stages: `instancespace/stages/`
- Python model contracts: `instancespace/data/model.py` and `instancespace/model.py`
- Verified MATLAB parity fixtures: `tests/fixtures/matlab/current/`
- Historical unverified snapshots: `tests/matlab_reference/`
