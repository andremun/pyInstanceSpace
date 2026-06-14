# build() → explore() adapter

## Why an adapter is needed

`explore()` is an inference-only port of MATLAB `exploreIS.m`. Its 1:1 fidelity is
proved by feeding it a model **MATLAB trained** (exported as CSV artifacts) and
checking that every stage reproduces MATLAB's output exactly. That validation must
not change, so `explore()` is left untouched.

A model trained by the Python `build()` cannot be fed into `explore()` directly,
because the two were written against different in-memory representations. The
mismatch is **not** in the inference maths — it is in how the trained parameters are
*stored*. An adapter (middleware) translates one representation into the other.

## Stage-by-stage format comparison

For each inference stage, what `explore()` reads vs. what `build()` produces:

| Stage | `explore()` reads | `build()` produces | Match? |
|-------|-------------------|--------------------|--------|
| PRELIM | `model.prelim.{lo_bound, hi_bound, min_x, lambda_x, mu_x, sigma_x}` — arrays of length *n_features* | `PrelimOut` with exactly those fields | ✅ pass through |
| SIFTED | `model.sifted.selvars` — selected feature indices | `SiftedOut.selvars` | ✅ pass through |
| PILOT | `model.pilot.a` — the 2×*n* projection matrix | `PilotOut.a` | ✅ pass through |
| PYTHIA | `model.pythia.mu`, `.sigma`, `.precision`, and per algorithm `.svm[i].{support_vectors, alphas, bias, kernel_fn, kernel_param, platt_A, platt_B}` | `mu`/`sigma` stored **post-normalisation** (≈0 / ≈1); each `.svm[i]` is a fitted scikit-learn **`SVC`** (`.support_vectors_`, `.dual_coef_`, `.intercept_`, `._gamma`, `.probA_`, `.probB_`) | ❌ **translate** |
| TRACE | `model.trace.{good,best}[i].polygon` — shapely polygons | `TraceOut.{good,best}[i]` are `Footprint` objects exposing `.polygon` | ✅ pass through |
| (feature extraction) | `model.data.feat_labels` — the **original** feature labels, so PRELIM's *n* parameters line up | `data.feat_labels` is the **post-SIFTED** subset | ❌ **restore** |

So four of five stages already match. The adapter only has real work in PYTHIA, plus
restoring two metadata items.

## The PYTHIA translation (the only conversion)

### 1. SVM record: scikit-learn `SVC` → artifact form

| Artifact field `explore()` needs | Source in the fitted `SVC` | Note |
|----------------------------------|----------------------------|------|
| `support_vectors` | `svc.support_vectors_` | identical |
| `alphas` | `svc.dual_coef_.ravel()` | already signed (`α_i · y_i`) |
| `bias` | `svc.intercept_[0]` | identical |
| `kernel_fn` | from `svc.kernel` (`rbf`→`gaussian`) | |
| `kernel_param` (Gaussian scale *s*) | `svc._gamma ** -0.5` | so `exp(-‖x-y‖²/s²) = exp(-γ‖x-y‖²)` |
| `platt_A`, `platt_B` | `svc.probA_[0]`, `svc.probB_[0]` | Platt sigmoid coefficients |

With these substitutions the artifact decision function
`Σ αᵢ·exp(-‖z-svᵢ‖²/s²) + bias` equals scikit-learn's `decision_function` exactly
(verified to `1e-9`).

### 2. Normalisation constants: recomputed, not copied

`build()` z-scores the PILOT coordinates before fitting the SVMs but then stores the
mean/std *of the already-normalised* coordinates (`mu ≈ 0`, `sigma ≈ [1, 1]`). Using
those at inference would skip normalisation entirely and feed raw coordinates to SVMs
trained on normalised ones. The adapter therefore recomputes the constants from the
trained projection: `mu = mean(pilot.z)`, `sigma = std(pilot.z, ddof=1)` (measured
`sigma ≈ [0.884, 0.781]`).

### 3. Feature labels: restored

The adapter sets `data.feat_labels` back to the original training feature names so
`explore()`'s feature extraction selects all *n* columns PRELIM expects, before SIFTED
reduces them.

## What the adapter does *not* touch

The cold-start Python `build()` lands in a different — but equally valid — instance
space than MATLAB (PILOT's objective has infinitely many equivalent optima). The
adapter does not align the two spaces; it makes a Python-built model
*self-consistent* under `explore()`, not numerically equal to MATLAB. Comparing a
Python-built model to MATLAB's reference numbers remains the job of the validated
artifact path (`liveDemoExploreIS.ipynb`, Part 1).

## Verification

Measured on the trial dataset (212 training instances, 10 algorithms), comparing
`explore()`-via-adapter against the `build()` model's own training-time predictions:

| Check | Result |
|-------|--------|
| Binary `y_hat` agreement | **98.96 %** (residual = scikit-learn `predict` decision-sign vs. Platt-posterior ≥ 0.5 boundary inconsistency) |
| Probability ranking, mean \|Pearson r\| | **0.9992** (sign is flipped by the class-label convention, as in the inference port; ranking is preserved) |
| Public `explore()` end-to-end on 235 test instances | runs; `z` (235×2), `y_hat` (235×10), `in_good`/`in_best` (235×10) |
| PILOT projection self-consistency | max relative error **9.9e-15** |

Unit tests in `tests/build_explore_adapter/test_adapter.py` cover the conversion in
isolation (decision function exact to `1e-9`, kernel-scale identity, posterior ranking
`|r| > 0.99`, unsupported-kernel guard, pass-through).

## Usage

```python
from instancespace.build_explore_adapter import adapt_for_explore

space = InstanceSpace(train_metadata, options)
space.build()
space._model = adapt_for_explore(space.model, train_metadata.feature_names)
result = space.explore(test_metadata)
```
