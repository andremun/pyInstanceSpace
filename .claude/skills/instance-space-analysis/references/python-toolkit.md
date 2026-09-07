# Python `pyInstanceSpace` Package -- Operational Reference

Verified against `andremun/pyInstanceSpace` (cloned directly from GitHub
for this reference, `main` branch) as of 2026-09-07. Package name on
PyPI/in `pyproject.toml` is `instancespace`, version **0.3.0** at time of
writing -- check for updates, this package moves quickly (TRACE3 and 3D
PILOT support, for example, were added after an earlier revision of this
reference incorrectly said they were absent -- see below). This is a
from-scratch, more modern re-architecture (stage-pipeline pattern, a
`PredictiveStage` typed contract for stage-owned fitted inference, a
flat test suite with a manifest-verified MATLAB oracle) rather than a
line-for-line port of the MATLAB toolkit -- expect method availability
and defaults to diverge from `InstanceSpace`; check both if a specific
behaviour is load-bearing. The MATLAB reference oracle currently pinned
in this package's tests is `InstanceSpace` v0.9.1
(`98a01ac0513c0dd0f8a9bd91ed2926c871334d7b`).

## Package layout

```
instancespace/
  instance_space.py    InstanceSpace class: build()/explore()/explore_stage_iter()
                        (recommended entry point); hardcodes the built-in
                        7-stage execution order (_BUILTIN_STAGE_ORDER) and
                        defines the ExploreStage enum (PRELIM/SIFTED/PILOT/
                        PYTHIA/TRACE/EVALUATION)
  model.py              Model class: holds/saves all stage outputs
  stage_runner.py       StageRunner (execution/rollback engine) and
                         build_stage_runner() for attaching extra/plugin
                         stages via RunBefore/RunAfter
  _serialisers.py        CSV/graph/mesh/MAT output writers
  data/
    options.py            Options dataclasses (frozen, __post_init__-validated)
    default_options.py     DEFAULT_* constants consumed by data/options.py
    metadata.py             Metadata dataclass, CSV/JSON loaders
    model.py                 Data/PilotOut/TraceOut/FeatSel/Footprint
                              output dataclasses
  stages/               preprocessing.py, prelim.py, sifted.py, pilot.py,
                         pilot_viewpoint.py (3D per-group viewpoint
                         optimisation), pythia.py, cloister.py, trace.py,
                         stage.py (Stage/PredictiveStage base classes)
  utils/                alpha_shape.py (AlphaShape2D/3D, TetrahedralMesh --
                         native 3D footprint geometry for TRACE3), 
                         numerics.py (shared helpers incl. matlab_round),
                         get_classifier_fcn.py (PYTHIA's six-classifier
                         registry), filter.py, print_options.py
tests/                 Flat suite (no per-stage subdirectories); see
                        "Testing conventions" below.
```

No `stage_builder.py` and no `scripting/` package exist in the current
layout -- both were removed in prior cleanup passes (stage wiring was
folded into `stage_runner.py`; the old CLI scaffolding under `scripting/`
was deleted as dead code). If you find either referenced in an older
document, it predates that cleanup.

## Recommended Python API usage

```python
from instancespace import InstanceSpace
from instancespace.data import metadata, options

metadata_object = metadata.from_csv_file("metadata.csv")   # same feature_/algo_ CSV convention as MATLAB
options_object = options.from_json_file("options.json")     # same schema as MATLAB options.json

instance_space = InstanceSpace(metadata_object, options_object)
instance_space.build()          # runs Preprocessing -> Prelim -> Sifted -> Pilot -> Pythia -> Cloister -> Trace

instance_space.model.save_to_csv("output/")
instance_space.model.save_graphs("output/")
```

See `integration_demo.py` (repo root) for a complete, runnable version of
this, including the explicit stage list, and `example_plugin.py` for how
to add a custom `Stage` to the pipeline. `liveDemoIS.ipynb` is the
Python counterpart of the MATLAB live demo and walks through both
`build()` and `explore()`/`explore_stage_iter()` stage by stage -- read
it as the primary usage guide, more so than this reference.

Notes:
- Default stage order is `[Preprocessing, Prelim, Sifted, Pilot, Pythia,
  Cloister, Trace]` (`_BUILTIN_STAGE_ORDER` in `instance_space.py`) --
  PYTHIA runs *before* CLOISTER here. The MATLAB `buildIS.m` calls them
  PILOT -> CLOISTER -> PYTHIA -> TRACE; the two orderings are logically
  equivalent since CLOISTER only needs PILOT's `A` and PYTHIA only needs
  PILOT's `Z`.
- `instance_space.explore_stage_iter(test_metadata)` is the
  `explore()`-time counterpart of `build()`'s own `run_iter()`: it yields
  an `(ExploreStage, output)` pair after each inference stage
  (`ExploreStage.PRELIM`/`SIFTED`/`PILOT`/`PYTHIA`/`TRACE`, then
  `EVALUATION` if a ground truth is available) instead of returning
  everything from one `explore()` call. Use it to inspect intermediate
  output (e.g. SIFTED's selected features, PILOT's projected `Z`) while
  applying a trained model to new data. (There is a separate, analogous
  `run_iter()` used during `build()`, yielding `AnnotatedStageOutput` --
  do not conflate the two; both exist, for build-time and explore-time
  respectively.)
- `explore()` works directly on the model `build()` produced: PRELIM,
  SIFTED, PILOT and TRACE pass their stored parameters through unchanged
  (no re-fit), and the trained PYTHIA classifiers (fitted scikit-learn
  estimators) are called via their own `predict`/`predict_proba` -- no
  intermediate flattened representation or conversion step.
- `stages: list[StageClass]` in the `InstanceSpace` constructor can be
  overridden to run a subset or custom pipeline; `build_stage_runner()`
  in `stage_runner.py` supports attaching extra/plugin stages via
  `RunBefore`/`RunAfter` markers (see `example_plugin.py`).
- Stages that own fitted inference (currently PYTHIA) implement the
  `PredictiveStage` typed contract in `stages/stage.py`, so `explore()`
  can call back into stage-specific prediction logic rather than the
  `InstanceSpace` class re-implementing per-stage inference itself.

## Key option fields worth knowing (not exhaustive -- read `data/options.py`)

- `PythiaOptions.classifier`: one of `"svm"` (default), `"knn"`,
  `"tree"`, `"nb"`, `"linear"`, `"ensemble"` -- see
  `utils/get_classifier_fcn.py` for the registry and each classifier's
  tunable parameters.
- `PythiaOptions.skip` (default `False`): bypasses classifier training
  entirely, matching MATLAB's `core/PYTHIA.m` `opts.skip`. Legacy TRACE
  needs true-label footprints when this is set (no `Yhat` exists to
  build footprints from); TRACE3 can also fall back to true labels
  without changing `trace.use_sim`.
- `PilotOptions.n_tries` (default `10`, matches the current MATLAB
  default in `ISAdefaults.m`, both differing from the paper's 30).
- `PilotOptions.method`: `"standard"` (analytic/numeric, gated by
  `PilotOptions.analytic`) or `"pls"` (Partial Least Squares, ignores
  `analytic` entirely).
- `PilotOptions.dims` (default `2`, may be `3`) and
  `PilotOptions.view_groups` (tuple of index-tuples; empty tuple = one
  global viewpoint) -- 3D projection and per-algorithm-group viewpoint
  selection, matching MATLAB's `ISA3D` branches. See
  `stages/pilot_viewpoint.py` for the viewpoint-optimisation logic.
- `TraceOptions.method`: `"legacy"` (default) or `"trace3"`. Purity
  default is method-aware: `DEFAULT_TRACE_PURITY = 0.55` for legacy,
  `DEFAULT_TRACE3_PURITY = 0.60` for trace3 (resolved automatically if
  `purity` is left `None`). Also has `min_instances` and `min_area_frac`
  floor parameters. TRACE3 with `PilotOptions.dims=3` gives native 3D
  volumes via `utils/alpha_shape.py`'s `TetrahedralMesh`.

## No CLI

There is currently no installed console script (`[tool.poetry.scripts]`
is absent from `pyproject.toml`) -- the package is used as a library, not
run from the command line. `CLIDocs.txt` at the repo root documents a
CLI that has not been built yet (the README explicitly calls it "notes on
the (not yet built) command-line interface"); do not assume a `matilda`
(or similarly named) command exists, and do not follow `CLIDocs.txt`'s
invocations literally without first checking `pyproject.toml` for a
`[tool.poetry.scripts]` section. The demonstrated usage pattern
throughout the repo is the direct API shown above, via
`integration_demo.py`, `example_plugin.py`, and `liveDemoIS.ipynb`.

## Testing conventions in this repo

Tests live flat under `tests/` (no per-stage subdirectories), split by
filename prefix rather than directory:

| Prefix | Covers | Example |
|---|---|---|
| `test_build_<stage>.py` | `build()`-time (training) behaviour | `test_build_pilot.py` |
| `test_explore_<stage>.py` | `explore()`-time inference, unit + MATLAB validation merged into one file | `test_explore_pilot.py` |

Stages without an `explore()`-time counterpart (CLOISTER, preprocessing --
neither appears in `ExploreStage`) only have a `test_build_*` file. Files
spanning more than one stage (e.g. `test_build_pilot_pythia.py`) keep a
`test_build_`/`test_explore_` prefix but are not folded into either
stage's own file. Cross-cutting infrastructure (executor pooling, model
save/load, option validation, plotting, progress reporting, etc.) keeps
its own descriptive name with no single-stage distinction. Full
naming/trust-convention writeup: `tests/README.md`.

Two data tiers, not one:
- `tests/fixtures/matlab/current/` is the **manifest-verified** current
  oracle -- a `reference-export/v2` bundle generated from a clean MATLAB
  run against the pinned `InstanceSpace` commit (currently v0.9.1). This
  is the trustworthy source for asserting exact MATLAB parity.
- `tests/matlab_reference/` holds older, **unverified** ("legacy-unknown")
  regression snapshots -- useful for catching an unintended behaviour
  change, but not proof of MATLAB parity on its own.

These replace an earlier convention of one `test_<stage>.py` file per
stage plus a `tests/test_integration/` directory -- that layout no longer
exists; do not look for it.

Run from the repository root (tests locate reference data via paths
relative to it):

```bash
poetry run pytest -v
poetry run pytest --cov=instancespace --cov-report=term-missing --cov-report=xml   # matches CI's `poe test_pytest`
```

## Verified: TRACE3 and 3D PILOT are implemented

An earlier revision of this reference stated TRACE3 was MATLAB-only and
absent from `pyInstanceSpace`. That is no longer true (verified 2026-09
against `main`, `TraceOptions`/`PilotOptions` in `data/options.py`,
`stages/pilot_viewpoint.py`, and `utils/alpha_shape.py`; also exercised
by `tests/test_trace3.py`, `tests/test_pilot_viewpoint.py`,
`tests/test_current_matlab_trace3_3d_parity.py`, and
`tests/test_current_matlab_pilot_3d_parity.py`):

- `TraceOptions.method` selects `"legacy"` (default) or `"trace3"`.
- `PilotOptions.dims` selects 2D (default) or 3D projection, with
  `PilotOptions.view_groups` for per-algorithm-group viewpoint reduction
  when 3D.
- `utils/alpha_shape.py` provides `AlphaShape2D`, `AlphaShape3D`, and
  `TetrahedralMesh` for native 3D footprint geometry (volumes, not just
  areas).
- Legacy TRACE remains the *default* method in this package (unlike the
  MATLAB toolkit, where `trace3` is the default in `options.json`) --
  callers must opt in to `TraceOptions.method="trace3"` explicitly.

If a specific TRACE3/3D-PILOT behaviour still looks unimplemented or
different from the MATLAB toolkit when you check, re-verify directly
against `data/options.py`, `stages/trace.py`, and `stages/pilot.py`
rather than trusting this note -- both toolkits are under active
development.
