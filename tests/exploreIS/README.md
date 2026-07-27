# exploreIS Test Suite

Validation and unit tests for the `explore()` pipeline implementation.

## Directory Structure

```
tests/exploreIS/
├── README.md
├── __init__.py
├── prelim/
│   ├── __init__.py
│   ├── test_prelim_unit.py
│   └── test_prelim_validation.py
├── sifted/
│   ├── __init__.py
│   ├── test_sifted_unit.py
│   └── test_sifted_validation.py
├── pilot/
│   ├── __init__.py
│   ├── test_pilot_unit.py
│   └── test_pilot_validation.py
├── pythia/
│   ├── __init__.py
│   ├── test_pythia_unit.py
│   └── test_pythia_validation.py
└── trace/
    ├── __init__.py
    ├── test_trace_unit.py
    └── test_trace_validation.py
```

## Test Types

### Validation Tests (`test_<stage>_validation.py`)

Each stage is fed MATLAB's own inputs — the trained-model artifacts plus the previous
stage's MATLAB output — and its result is compared against the MATLAB reference output
for that stage. This isolates per-stage port fidelity from error accumulated along the
pipeline. Reference data lives in `tests/matlab_reference/` (see its README for the
file inventory). Run with `-s` to see the comparison statistics each test prints.

### Unit Tests (`test_<stage>_unit.py`)

Edge cases and behavioural guarantees for each stage method, with synthetic inputs and
no reference data: output shapes, single-instance and NaN inputs, bounding behaviour,
input preservation, determinism, and stage-specific arithmetic (kernel evaluation,
selection rules, polygon membership).

## Running Tests

Run from the repository root — the tests locate reference data through paths relative
to it.

```bash
# Full suite
uv run pytest tests/exploreIS/ -v

# One stage, with validation statistics printed
uv run pytest tests/exploreIS/trace/ -v -s
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
| TRACE  | per-column boolean agreement ≥ 99% | boundary-inclusive membership matching MATLAB `inpolygon`; the 1% budget covers floating-point boundary edge cases after the CSV round-trip |
