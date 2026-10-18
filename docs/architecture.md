# Architecture and scientific contracts

## Pipeline ownership

`InstanceSpace` coordinates metadata, immutable options, execution, model state
and exploration. `StageRunner` schedules the built-in stages in this order:

PREPROCESSING → PRELIM → SIFTED → PILOT → (PYTHIA, CLOISTER) → TRACE.

PYTHIA and CLOISTER share a scheduling wave. TRACE depends on PYTHIA, not on
CLOISTER. A custom stage attaches through explicit `RunBefore`/`RunAfter` markers;
matching field types alone do not define its position. Runtime NamedTuple input
annotations must be concrete types, not unresolved strings.

| Stage | Responsibility |
|---|---|
| PREPROCESSING | Select metadata columns, wash missing input data, record retained feature order. |
| PRELIM | Define good performance and difficulty, prune never-good training algorithms, refit preprocessing from the retained raw portfolio. |
| SIFTED | Correlation screening/clustering and PILOT/KNN genetic feature selection. |
| PILOT | Standard analytic/numerical or SIMPLS projection in 2D/3D, plus optional viewpoints. |
| PYTHIA | Fit per-algorithm good-performance classifiers and a training selection policy. |
| CLOISTER | Project feature-range corners constrained by correlations, or use an observed-data hull above the feature cap. |
| TRACE | Build and score good, best and hard footprints with dimension-aware geometry. |

Low-level `PrelimStage.prelim()` computes numerical preprocessing without pipeline
portfolio pruning. `sifted.n_trees` is a compatibility setting with no effect on
current PILOT/KNN fitness. Output flags are preferences for callers; explicit
`Model` export methods write files.

## Fitted inference

Build-only plugins implement `Stage[Input, Output]`. PRELIM, SIFTED, PILOT, PYTHIA
and TRACE additionally implement `PredictiveStage[Input, Fitted, Output]`.
Their `predict()` methods consume persisted artifacts without fitting or mutating
them. Preprocessing and CLOISTER have no explore-time predictive stage.

`InstanceSpace.explore()` reconciles metadata and algorithms, then applies PRELIM,
SIFTED, PILOT, PYTHIA and TRACE. It conditionally evaluates observed test outcomes
and rescores TRACE evidence against fixed trained geometry. The `_explore_*`
wrappers remain compatibility entry points.

- Replay fitted retained-feature order and preprocessing parameters.
- Standard PILOT inference is `X @ A.T`. PLS inference is
  `(X - x_mean) @ A.T`, using the persisted training mean.
- Reuse trained classifier normalization and the saved selection policy.
- Outcomes score predictions; changing or omitting test labels must not change
  coordinates, recommendations, predictions or trained-footprint membership.
- New test algorithms receive prediction/membership placeholders and cannot be
  recommended without a trained classifier. Unobserved outcomes are excluded
  from metrics; wholly undefined rates are `NaN`.
- `TraceStage.rescore()` updates evidence metrics without changing geometry or
  its area/volume. It uses the training-space normalization.

## Indices, geometry and persistence

PRELIM winner `p` is one-based. PYTHIA selectors and Python viewpoint groups are
zero-based; `selection0=-1` means abstention. Make conversions only at explicit
interfaces. Mesh export conventions differ: interpret each manifest instead of
assuming a single index base.

TRACE boundaries are closed. `boundary_tolerance=0.0` uses exact membership;
a positive Euclidean distance in projection units admits nearby edge/face points,
including hole boundaries. Persist and reuse that value independently of the
query batch. It changes membership, not geometry or area/volume. Sequential
legacy contradiction trimming can amplify small coordinate changes.

TRACE3 supports holes, disconnected regions and native 3D tetrahedral geometry.
A search can terminate below the requested purity; inspect the returned metrics.
Python does not currently persist MATLAB's acceptance/termination fields.

`Model.save/load` preserves fitted results. `InstanceSpace.save/load` preserves
runner state for checkpoints and exploration. Native persistence is runtime
specific and trusted-producer-only; MAT export is not a MATLAB classifier-loading
format. Stage rollback restores isolated state and invalidates derived models.

## MATLAB comparison boundaries

Use current MATLAB source and verified execution for behavioral comparisons.
The approved oracle is a separately pinned snapshot; see
[fixture provenance](fixture-provenance.md). Never turn an unknown-provenance
snapshot or a Python-generated result into a claimed MATLAB oracle.

Python retains its typed architecture and public interfaces. Defaults differ in
performance mode/epsilon, seed, classifier and TRACE method; choose these
explicitly before comparing results. See the
[maintained compatibility reference](../docs_site/src/matlab-compatibility.md).

Python provides optional canonical 2D PILOT rotation; MATLAB does not currently
implement it. Independent solvers, classifiers, calibration and search algorithms
can differ under matched nominal settings. The
[controlled Bayesian comparison](research/bayes-304/report.md) supports retaining
20 evaluations, four initial points and plain EI, without asserting trajectory
identity with MATLAB EI-plus.

## Source map

- Orchestration: `instancespace/instance_space.py`.
- Scheduling and plugins: `instancespace/stage_runner.py`, `instancespace/stages/stage.py`.
- Scientific implementations and fitted inference: `instancespace/stages/`.
- Results and geometry: `instancespace/data/model.py`.
- Persistence and exports: `instancespace/model.py`, `instancespace/_serialisers.py`.
- Configuration: `instancespace/data/options.py`, `instancespace/data/default_options.py`.
- Classifiers: `instancespace/utils/get_classifier_fcn.py`.
