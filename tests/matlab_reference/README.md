# MATLAB Reference Data for exploreIS Pipeline

## Purpose

This dataset provides ground truth outputs from the MATLAB implementation of the exploreIS pipeline.
Use these files to validate the Python implementation of Instance Space Analysis.

## Directory Structure

```
matlab-reference-data/
├── README.md
├── validation_summary.csv              # Quick overview of all files
├── input/
│   ├── metadata_train.csv             # Training data (212 instances)
│   └── metadata_test.csv              # Test data (235 instances)
├── training_artifacts/
│   ├── prelim_params.csv              # Normalization & bounding params
│   ├── sifted_indices.csv             # Selected feature indices
│   ├── pilot_matrix.csv               # 2D projection matrix A
│   └── pythia_params.csv              # Z-coordinate normalization
└── explore_outputs/
    ├── step1_after_prelim.csv         # After bounding + normalization
    ├── step2_after_sifted.csv         # After feature selection
    ├── step3_after_pilot.csv          # After 2D projection
    ├── step4_pythia_predictions.csv   # Binary predictions (0/1)
    ├── step4_pythia_probabilities.csv # Probability estimates [0,1]
    └── step5_trace_membership.csv     # Footprint membership flags
```

## File Descriptions

### Training Artifacts

#### prelim_params.csv
- **Columns:** `feature_name, min_x, lambda_x, mu_x, sigma_x, lo_bound, hi_bound`
- **Purpose:** Parameters for Box-Cox + Z-score normalization and outlier bounding
- **Use in Python:** Apply same transformation to test data

#### sifted_indices.csv
- **Columns:** `original_index, feature_name`
- **Purpose:** Which features were selected from the original set
- **Use in Python:** Index test data to keep only these features

#### pilot_matrix.csv
- **Columns:** `feature_name, z1_coef, z2_coef`
- **Purpose:** Projection matrix A where Z = X × A^T
- **Use in Python:** Project test features to 2D coordinates

#### pythia_params.csv
- **Columns:** `coordinate, mu, sigma`
- **Purpose:** Normalization of Z coordinates before SVM prediction
- **Use in Python:** Normalize Z before feeding to SVM models

### Explore Outputs

#### step1_after_prelim.csv
- **Shape:** 235 instances × 10 features
- **Purpose:** Test data after bounding and Box-Cox + Z-score transformation
- **Validate:** `explore()` method PRELIM stage

#### step2_after_sifted.csv
- **Shape:** 235 instances × 10 selected features
- **Purpose:** Test data with only selected features retained
- **Validate:** `explore()` method SIFTED stage

#### step3_after_pilot.csv
- **Shape:** 235 instances × 2 (z1, z2)
- **Purpose:** Test data projected to 2D instance space
- **Formula:** Z = X_selected × A^T
- **Validate:** `explore()` method PILOT stage

#### step4_pythia_predictions.csv
- **Shape:** 235 instances × 10 algorithms
- **Values:** 0 = bad performance, 1 = good performance
- **Validate:** `explore()` method PYTHIA predictions

#### step4_pythia_probabilities.csv
- **Shape:** 235 instances × 10 algorithms
- **Values:** Probability estimates in [0, 1]
- **Validate:** `explore()` method PYTHIA probability outputs

#### step5_trace_membership.csv
- **Shape:** 235 instances × 21 columns
- **Columns:** `in_space, in_good_*, in_best_*` (boolean flags)
- **Purpose:** Which footprints each test instance belongs to
- **Validate:** `explore()` method TRACE stage

## Validation Guidelines

1. **Numerical Tolerance:** Allow floating-point differences ≤ 1e-6
2. **Shape Matching:** All arrays must have identical dimensions
3. **Column Names:** Must match exactly (order can differ)
4. **NaN Handling:** NaN values must appear in identical positions

## Dataset Statistics

- **Training instances:** 212
- **Test instances:** 235
- **Original features:** 10
- **Selected features:** 10
- **Algorithms:** 10

## References

- Smith-Miles, K., & Muñoz, M. A. (2023). Instance Space Analysis for Algorithm Testing.
  *ACM Computing Surveys*, 55(12), 1-31. DOI: 10.1145/3572895
- MATLAB Toolkit: https://github.com/andremun/InstanceSpace