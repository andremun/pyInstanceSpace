# Test Suite

Tests live flat under `tests/` (no per-stage subdirectories), disambiguated by filename
prefix rather than directory: `test_build_<stage>.py` for a stage's training-time
behaviour, `test_explore_<stage>.py` for its `explore()`-time inference counterpart.
See the roadmap's T7 section (`docs/pyIS_docs_quality_roadmap.md`) for the full decision
and file-by-file mapping.

## Current MATLAB oracle

`tests/fixtures/matlab/current/` is the canonical `matlab-verified` oracle: 423 files
under `reference-export/v2`, generated with MATLAB R2026a Update 4 from gold source
`34c01293fef99b4eabd53323c393cb184cc95a8e` and Python generator
`cf3cde0da5a3067300bd94a48d4d09ff5cf20b0c`. Its exporter SHA-256 is
`d11293556b12beb63e3320094a2340ba3f7f8b7a58677ff404f20c0ba3b7350c`.
Collection contains 86 provenance tests and 40 current scientific readers. The
CI-equivalent Linux gate passed all 1,039 collected tests with 92.00% branch coverage
and no uncaught warnings under `-W error`. The frozen 229-file v1 format remains readable,
but is not the installed oracle.

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
manifest-verified v2 bundle under `tests/fixtures/matlab/current/`. Run with `-s` to see
comparison diagnostics. No legacy fixture can establish MATLAB parity.

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

# Strict provenance and all current-gold scientific readers
poetry run pytest tests/test_fixture_provenance.py -q
poetry run pytest tests/test_current_matlab_*.py -q
```

## Validation criteria

Current-gold readers validate the manifest and use exact comparisons or narrowly scoped
numeric tolerances documented beside each assertion. They include standard and SIMPLS
PILOT in 2D/3D, viewpoints, TRACE/TRACE3 geometry, topology, membership, and rescoring.

The table below records the older `legacy-unknown` regression thresholds. Passing these
tests detects drift but is not a current MATLAB parity claim.

Every threshold is documented, with its rationale, in the docstring of the test that
asserts it:

| Stage  | Criterion | Rationale |
|--------|-----------|-----------|
| PRELIM | max relative error < 1% | deterministic bounding → Box-Cox → z-score with stored parameters; expected to match to floating-point precision |
| SIFTED | exact match | pure column indexing |
| PILOT  | max relative error < 1% | single matrix product with the stored projection matrix |
| PYTHIA | exact binary outputs; probabilities within `1e-13` absolute error | direct replay of the stored historical SVM artifacts; correlation is not used because it accepts inverted or shifted probabilities |
| TRACE  | per-column boolean agreement ≥ 99% | boundary-inclusive membership matching MATLAB `polyshape.isinterior`; the 1% budget covers floating-point boundary edge cases after the CSV round-trip |
