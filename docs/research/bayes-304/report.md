# Controlled Bayesian SVM comparison

This maintainer record resolves the convergence investigation in #304 by
separating optimizer behavior from differences in the classifier objectives.
It is not part of the user website or an approved MATLAB fixture replacement.

## Decision

Retain the production default of 20 evaluations, four initial points and plain
expected improvement in Python. The experiment does not establish a uniform
Python convergence deficit or justify a default budget increase. MATLAB uses
`expected-improvement-plus`; Python uses scikit-optimize's `EI`. Numerical
identity between those optimizers or the underlying SVM implementations is not
promised. No production numerical setting changes in this work.

## Design and provenance

- Python baseline: `0159ed32538f7e33a0d8afc0b0200451bdef096a` (v0.4.0).
- MATLAB baseline: `5f8cb25682202a8243bd2a2481ab8feb40c8c4ce` (v0.9.2).
- MATLAB R2026a Update 5; Python 3.12.14, NumPy 1.26.4,
  scikit-learn 1.5.2 and scikit-optimize 0.10.2. Full versions accompany traces.
- The current approved `legacy_svm` PYTHIA input supplies 212 rows. Python's
  production PYTHIA coordinate normalization is frozen for both backends.
  Input CSV and reference-manifest hashes are in `inputs.json`.
- The two algorithm labels closest to balanced are selected mechanically:
  `RandF` (zero-based column 9) and `L_SVM` (column 6). Both are prediction
  targets for Gaussian/RBF SVMs, not the classifier types used in this study.
- Seeds 7, 42 and 101, five shared stratified folds per target/seed,
  evaluation budgets 20 and 40, uniform weights. These are explicit algorithm
  seeds, without portfolio-index seed offsets.
- Both searches receive the same four starting pairs, sampled uniformly in
  log2 box constraint and kernel scale on [-10, 4]. Python's gamma is
  `1 / kernel_scale**2`. Maximum observed initial-point relative rounding
  error is 3.91e-16. No tolerance is applied to the objective results.
- Each budget is an independent optimizer run. There are 12 searches and
  360 candidate evaluations per backend. All six 20-evaluation traces match
  the first 20 evaluations of their corresponding 40-evaluation runs in
  each backend.
- Python runs the production estimator factory, search space and optimizer
  settings, with `PredefinedSplit` and a fixed initial-point generator.
  MATLAB runs `bayesopt` and verbatim local PYTHIA fitting, calibration and
  cross-validation helpers from the pinned source, using `isacompat.Partition`
  masks. Production files are not modified.
- Both winners from every search pair are cross-scored in both classifiers
  (24 pairs). A separate 3×3 grid at box/scale values 2^-7, 2^-1 and 2^3 for
  every target/seed adds 54 pairs: 78 cross-evaluations in total.

## Results

Own-objective minima are classification error proportions; lower is better.
Python uses the unweighted mean of fold errors. MATLAB uses the error across
all held-out rows. The table should not be read as a common-objective ranking.

| Target | Seed | Python 20 | Python 40 | MATLAB 20 | MATLAB 40 |
|---|---:|---:|---:|---:|---:|
| RandF | 7 | 0.396346 | 0.344297 | 0.396226 | 0.349057 |
| RandF | 42 | 0.344297 | 0.344297 | 0.349057 | 0.349057 |
| RandF | 101 | 0.362901 | 0.325360 | 0.334906 | 0.334906 |
| L_SVM | 7 | 0.235659 | 0.235659 | 0.235849 | 0.235849 |
| L_SVM | 42 | 0.254707 | 0.235659 | 0.235849 | 0.235849 |
| L_SVM | 101 | 0.245183 | 0.245183 | 0.245283 | 0.245283 |

Increasing the budget improves Python's own objective in 3/6 target/seed pairs
and MATLAB's in 1/6. Cross-scoring exposes classifier differences even at the
same physical parameters: global error is identical in 29/78 records, with a
maximum difference of 6.132 percentage points and a mean absolute difference
of 1.113 points. The maximum Python fold-mean versus row-weighted aggregation
difference is only 0.0653 points. Consequently, acquisition-function behavior
alone cannot explain the observed gap.

The native MATLAB searches emitted 12 `glmfit` iteration-limit warning events
in posterior calibration. All requested candidate evaluations completed with
finite objectives. Cross-evaluation emitted no warnings. Warnings are disclosed
rather than suppressed or presented as evidence of numerical identity.

`summary.csv` includes each backend's winning parameters scored by both
classifiers using the common row-weighted error. `cross-results.json` retains
fold mistake counts and sizes, allowing either aggregation to be recomputed.
Each search trace retains every candidate, objective and running minimum.
`manifest.json` checksums all evidence files and records derived statistics.
The native console transcript removes control characters and trailing spaces;
the original console digest is also retained in the manifest.

## Limits

This is a bounded diagnostic study of one frozen coordinate dataset, two
prediction targets, three seeds and Gaussian SVMs. It does not establish global
convergence, optimizer superiority, equivalence across other classifier
families, or unbiased out-of-sample performance. The folds are tuning folds,
not an independent outer test set. Shared starting points deliberately isolate
later search behavior and differ from each backend's native random design.
Calibration and solver differences remain part of each production objective.
Any further acquisition-function change needs a separately designed study
against a common evaluator and multiple datasets.

## Reproduce

Use the repository's locked Python 3.12 development environment and an empty
output directory. Export the exact MATLAB revision outside its working tree:

```sh
mkdir -p .cache/bayes-reproduction/matlab-source
git -C ../InstanceSpace archive 5f8cb25682202a8243bd2a2481ab8feb40c8c4ce \
  | tar -x -C .cache/bayes-reproduction/matlab-source
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 poetry run python -m tools.compare_bayes \
  prepare --output .cache/bayes-reproduction/run \
  --matlab-source .cache/bayes-reproduction/matlab-source
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 poetry run python -m tools.compare_bayes \
  python --output .cache/bayes-reproduction/run
```

From MATLAB, substitute absolute paths for these repository-relative examples:

```matlab
addpath('.cache/bayes-reproduction/matlab-source/utils');
addpath('.cache/bayes-reproduction/run');
pyis_bayes_comparison('.cache/bayes-reproduction/run/inputs.json', ...
    '.cache/bayes-reproduction/run', 'search');
```

Then run Python cross-scoring and MATLAB cross-scoring:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 poetry run python -m tools.compare_bayes \
  cross --output .cache/bayes-reproduction/run
```

```matlab
pyis_bayes_comparison('.cache/bayes-reproduction/run/inputs.json', ...
    '.cache/bayes-reproduction/run', 'cross');
```

Preparation checks pinned MATLAB source hashes. The tools refuse to overwrite
existing evidence. Reproduce in a new directory rather than mixing partial
runs, environments or inputs. Tests validate the committed evidence without
requiring MATLAB or rerunning a costly stochastic search in CI.
