# MATLAB Reference Data for the exploreIS Pipeline

Ground-truth inputs, trained-model artifacts and stage outputs from the MATLAB
implementation of the exploreIS pipeline, consumed by the validation tests in
`tests/exploreIS/`. All files were exported from a single MATLAB run of the reference
toolkit (https://github.com/andremun/InstanceSpace) on the trial dataset, so the
trained artifacts and the reference outputs are mutually consistent.

## Directory Structure

```
tests/matlab_reference/
├── README.md
├── validation_summary.csv      # row/column inventory of the explore_outputs files
├── input/
│   ├── metadata.csv            # training data (212 instances)
│   └── metadata_test.csv       # test data (235 instances)
├── training_artifacts/
│   ├── prelim/
│   │   └── prelim_params.csv   # outlier bounds, Box-Cox lambda, z-score mu/sigma
│   ├── sifted/
│   │   └── sifted_indices.csv  # selected feature indices (1-based, MATLAB convention)
│   ├── pilot/
│   │   └── pilot_matrix.csv    # 2D projection matrix A
│   ├── pythia/
│   │   ├── zscore.csv          # mu/sigma normalising the 2D coordinates
│   │   ├── precision.csv       # per-algorithm selection weights; defines algorithm order
│   │   └── svm_<algo>.csv      # one per algorithm: support vectors and SVM scalars
│   └── trace/
│       ├── good_<algo>.csv     # good-footprint polygon vertices, one per algorithm
│       └── best_CART.csv       # best-footprint vertices (only CART's is non-empty)
└── explore_outputs/
    ├── step1_after_prelim.csv          # test data after bounding + Box-Cox + z-score
    ├── step2_after_sifted.csv          # after feature selection
    ├── step3_after_pilot.csv           # 2D coordinates (z1, z2)
    ├── step4_pythia_predictions.csv    # binary good/bad predictions
    ├── step4_pythia_probabilities.csv  # posterior probability of the bad class
    └── step5_trace_membership.csv      # footprint membership flags
```

## File Descriptions

### input/

Both metadata files follow the repository's metadata format: an `Instances` identifier
column, ten `feature_` columns and ten `algo_` columns. The test set re-uses all 212
training instances and adds 23 new ones.

### training_artifacts/

One subdirectory per pipeline stage, holding the parameters that MATLAB's `buildIS`
learned at training time.

- **prelim/prelim_params.csv** — columns `feature_name, min_x, lambda_x, mu_x,
  sigma_x, lo_bound, hi_bound`. Per-feature parameters for outlier bounding and
  Box-Cox + z-score normalisation, re-applied unchanged to test data.
- **sifted/sifted_indices.csv** — columns `original_index, feature_name`. Indices of
  the features selected at build time; 1-based (MATLAB), so subtract 1 in Python.
- **pilot/pilot_matrix.csv** — columns `feature_name, z1_coef, z2_coef`. The
  projection matrix A stored row-per-feature; coordinates are Z = X × Aᵀ.
- **pythia/zscore.csv** — columns `mu_z1, mu_z2, sigma_z1, sigma_z2`. Normalisation of
  the 2D coordinates before SVM evaluation.
- **pythia/precision.csv** — columns `algo, precision`. Per-algorithm cross-validated
  precision used to weight PYTHIA's algorithm selection. Its row order defines the
  algorithm order for all per-algorithm files and reference columns.
- **pythia/svm_<algo>.csv** — one row per support vector (`sv_z1, sv_z2, alpha`); the
  first row additionally carries the per-SVM scalars `kernel_fn, kernel_param, bias,
  platt_A, platt_B`. Alphas are exported pre-signed (`Alpha .* SupportVectorLabels`).
- **trace/good_<algo>.csv, trace/best_CART.csv** — footprint polygon vertices
  (`x, y`). MATLAB polyshapes with several regions are exported as one vertex list
  with NaN rows delimiting the regions. A missing file is an empty footprint; only
  CART has a non-empty best footprint in this trained model.

### explore_outputs/

MATLAB's stage-by-stage outputs on the 235-instance test set. Every file carries an
`instance_id` column.

- **step1_after_prelim.csv** — 235 × 10 features after bounding and Box-Cox + z-score.
- **step2_after_sifted.csv** — 235 × 10 selected features (all ten survive selection
  on this dataset).
- **step3_after_pilot.csv** — 235 × 2 projected coordinates (`z1, z2`).
- **step4_pythia_predictions.csv** — 235 × 10 binary values; 1 = good performance
  predicted.
- **step4_pythia_probabilities.csv** — 235 × 10 posterior probabilities of the bad
  class.
- **step5_trace_membership.csv** — 235 × 21 boolean flags: `in_space`,
  `in_good_<algo>` × 10 and `in_best_<algo>` × 10. The `in_space` column comes from
  CLOISTER, a build-time stage outside the inference port's scope, and is not
  validated.

### validation_summary.csv

Row/column inventory of the six `explore_outputs` files.

## Validation Criteria

Per-stage thresholds and their rationale are documented in the validation tests under
`tests/exploreIS/` — see that suite's README.

## Dataset Statistics

- Training instances: 212
- Test instances: 235 (212 re-used training instances + 23 new)
- Features: 10 (all selected by SIFTED)
- Algorithms: 10 (NB, LDA, QDA, CART, J48, KNN, L_SVM, poly_SVM, RBF_SVM, RandF)

## References

- Smith-Miles, K., & Muñoz, M. A. (2023). Instance Space Analysis for Algorithm
  Testing: Methodology and Software Tools. *ACM Computing Surveys*, 55(12), 1-31.
  DOI: 10.1145/3572895
- MATLAB toolkit: https://github.com/andremun/InstanceSpace
