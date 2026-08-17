# ISA architecture and parity contract

## Purpose

Instance Space Analysis explains where algorithms perform well and why.
It relates instance features to algorithm performance in a low-dimensional space.
The result supports benchmarking, algorithm selection, and benchmark coverage analysis.

The MATLAB repository is the behavioral reference.
The current source takes priority over papers, old option names, and archived plans.
Known MATLAB defects are not parity targets.

## Pipeline

| Stage | Responsibility | Main trained output |
|---|---|---|
| PREPROCESSING | Load, clean, and optionally subset metadata | Dense and selected input data |
| PRELIM | Define good performance, clip outliers, transform, and scale | Labels, bounds, and transform parameters |
| SIFTED | Select predictive and non-redundant features | Selected feature indices |
| PILOT | Project selected features into the instance space | Projection matrix and coordinates |
| CLOISTER | Estimate feasible projected bounds | Empirical and correlation-aware hulls |
| PYTHIA | Train one good-performance classifier per algorithm | Classifiers, predictions, metrics, and selections |
| TRACE | Build good, best, hard, and full-space footprints | Polygons and footprint metrics |

Python runs PYTHIA and CLOISTER after PILOT.
TRACE depends on PYTHIA but not on CLOISTER.
The built-in order is fixed in `instancespace/instance_space.py`.

## Build and explore

`InstanceSpace.build()` fits all selected stages through `StageRunner`.
The final `Model` stores processed data, stage outputs, options, classifiers, and polygons.

`InstanceSpace.explore()` applies the trained model to new metadata.
It must not refit feature selection, projection, classifiers, or footprints.
It applies the stored PRELIM parameters, selected features, PILOT matrix, PYTHIA models, and TRACE polygons.

Known algorithms align by name without case sensitivity.
New test algorithms receive placeholder prediction and footprint columns.
Ground-truth performance enables evaluation metrics but does not retrain the model.

## Reliability invariants

- PRELIM `p` keeps MATLAB's one-based algorithm indices.
- PYTHIA `selection0` uses zero-based algorithm indices.
- `-1` means that PYTHIA made no selection.
- A single-algorithm portfolio can only select index `0`.
- Empty TRACE footprints use `polygon=None` and zero metrics.
- TRACE legacy membership includes exact polygon-boundary points, matching MATLAB
  `polyshape.isinterior`.
- Summary columns must match their named raw metrics.
- Build and explore must use the same fitted transformations and selection rules.
- Disabled preprocessing, feature-selection, and parallel flags must be no-ops.
- Stage rollback must restore an isolated snapshot and invalidate derived models.
- Stage output shapes and dtypes must remain stable across serializing and loading.

## MATLAB parity boundaries

Python keeps its public API and typed stage architecture.
It does not need a line-by-line MATLAB translation.
Parity means matching observable contracts, formulas, edge cases, and trained-model behavior.

The main known gaps are:

- Python implements legacy TRACE only. MATLAB defaults to TRACE3.
- Python PILOT supports PLS but remains 2D and has no viewpoint groups.
- Python output and plotting paths remain 2D.
- MATLAB reference fixtures lack verified generation provenance.
- Build logic lives in stage classes, while explore logic lives in `InstanceSpace` methods.

These gaps need separate designs and acceptance tests.
They are not part of the current correctness pass.

## Source map

- MATLAB orchestration: `InstanceSpace.m`
- MATLAB stages: `core/PRELIM.m` through `core/TRACE.m`
- Python orchestration: `instancespace/instance_space.py`
- Python execution: `instancespace/stage_runner.py`
- Python stages: `instancespace/stages/`
- Python model contracts: `instancespace/data/model.py` and `instancespace/model.py`
- MATLAB parity fixtures: `tests/matlab_reference/`
