# Test Suite

Tests live flat under `tests/` (no per-stage subdirectories), disambiguated by filename
prefix rather than directory: `test_build_<stage>.py` for a stage's training-time
behaviour, `test_explore_<stage>.py` for its `explore()`-time inference counterpart.
See the roadmap's T7 section (`docs/pyIS_docs_quality_roadmap.md`) for the full decision
and file-by-file mapping.

## Naming Convention

| Prefix | Covers | Example |
|---|---|---|
| `test_build_<stage>.py` | `build()`-time stage behaviour (training) | `test_build_pilot.py` |
| `test_explore_<stage>.py` | `explore()`-time inference for that stage, unit + MATLAB validation merged into one file | `test_explore_pilot.py` |

Stages without an `explore()`-time counterpart (CLOISTER, preprocessing — neither appears
in `ExploreStage`) only have a `test_build_*` file. Files spanning more than one stage
(e.g. `test_build_pilot_pythia.py`, a PILOT+PYTHIA build-time integration test) keep a
`test_build_`/`test_explore_` prefix but aren't folded into either stage's own file.
Everything else (cross-cutting infrastructure: executor pooling, model save/load, option
validation, plotting, etc.) has no single-stage distinction to make and keeps its
existing name.

## Test Types Within `test_explore_<stage>.py`

### Validation tests

Historical stage tests feed stored training artifacts and previous-stage outputs into
each Python stage. Data under `tests/matlab_reference/` is `legacy-unknown`: useful for
regression detection, but not a verified MATLAB oracle. Current parity tests use the
manifest-verified bundle under `tests/fixtures/matlab/current/`. Run with `-s` to see
comparison diagnostics.

### Unit tests

Edge cases and behavioural guarantees for each stage method, with synthetic inputs and
no reference data: output shapes, single-instance and NaN inputs, bounding behaviour,
input preservation, determinism, and stage-specific arithmetic (kernel evaluation,
selection rules, polygon membership).

## Running Tests

Run from the repository root — the tests locate reference data through paths relative
to it.

```bash
# Full suite
poetry run pytest -v

# One stage's explore()-time tests, with validation statistics printed
poetry run pytest tests/test_explore_trace.py -v -s
```

## Validation Criteria

Every threshold is documented, with its rationale, in the docstring of the test that
asserts it:

| Stage  | Criterion | Rationale |
|--------|-----------|-----------|
| PRELIM | max relative error < 1% | deterministic bounding → Box-Cox → z-score with stored parameters; expected to match to floating-point precision |
| SIFTED | exact match | pure column indexing |
| PILOT  | max relative error < 1% | single matrix product with the stored projection matrix |
| PYTHIA | binary agreement ≥ 99%; mean probability Pearson \|r\| ≥ 0.99 | SVM evaluation with stored parameters; small margin for thresholded predictions near the decision boundary |
| TRACE  | per-column boolean agreement ≥ 99% | boundary-inclusive membership matching MATLAB `polyshape.isinterior`; the 1% budget covers floating-point boundary edge cases after the CSV round-trip |
