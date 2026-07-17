# build_explore_adapter Test Suite

Unit tests for the build() → explore() adapter
(`instancespace/build_explore_adapter.py`) and for the model-shape detection in
`InstanceSpace._ensure_explore_model`.

## Directory Structure

```
tests/build_explore_adapter/
├── README.md
├── __init__.py
└── test_adapter.py
```

## What is covered

All tests train a tiny scikit-learn SVM inline (no full `build()`), so the suite runs
in milliseconds.

`test_adapter.py`:

- **Decision-function transfer** — the flattened record (support vectors, signed
  coefficients, bias, kernel scale `gamma**-0.5`) reproduces scikit-learn's
  `decision_function` to `1e-9`. The tolerance is machine precision headroom: the two
  computations are algebraically identical, so any real conversion error would exceed
  it by orders of magnitude.
- **Kernel-scale identity** — `exp(-‖x−y‖²/s²) = exp(-γ‖x−y‖²)` requires
  `s = γ^(-1/2)` exactly.
- **Posterior ranking** — `_explore_pythia` on the flattened record correlates with
  scikit-learn's `predict_proba` at `|r| > 0.99`; the small residual is the clean
  Platt sigmoid (MATLAB convention) versus scikit-learn's internal posterior, not a
  conversion error.
- **Unsupported-kernel guard** — a polynomial-kernel `SVC` raises
  `NotImplementedError`.
- **Pass-through** — PRELIM, SIFTED, PILOT and TRACE objects survive
  `adapt_for_explore` untouched, and the original feature labels are restored.
- **Shape detection** — `_ensure_explore_model` converts a Python-built model (SVMs
  without an `alphas` field) exactly once, leaves the scikit-learn model object
  untouched, and consumes an already-flattened model as-is.

Run with:

```
pytest tests/build_explore_adapter/
```
