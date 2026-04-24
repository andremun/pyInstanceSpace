#!/usr/bin/env python3
"""Validation test for PRELIM stage against MATLAB reference data.

This test validates that the Python _explore_prelim() implementation
produces outputs that match MATLAB within 5% tolerance.
"""

import sys
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from instancespace.data.model import PrelimOut
from instancespace.instance_space import InstanceSpace


def print_banner(text, char="=", width=80):
    """Print a banner with centered text."""
    print()
    print(char * width)
    print(f"{text:^{width}}")
    print(char * width)
    print()


def print_pass_banner():
    """Print a large PASSED banner."""
    banner = r"""
    ██████╗  █████╗ ███████╗███████╗███████╗██████╗
    ██╔══██╗██╔══██╗██╔════╝██╔════╝██╔════╝██╔══██╗
    ██████╔╝███████║███████╗███████╗█████╗  ██║  ██║
    ██╔═══╝ ██╔══██║╚════██║╚════██║██╔══╝  ██║  ██║
    ██║     ██║  ██║███████║███████║███████╗██████╔╝
    ╚═╝     ╚═╝  ╚═╝╚══════╝╚══════╝╚══════╝╚═════╝
    """
    print(banner)


def load_prelim_params():
    """Load PRELIM parameters from MATLAB training artifacts."""
    df = pd.read_csv("tests/matlab_reference/training_artifacts/prelim/prelim_params.csv")

    return PrelimOut(
        med_val=np.zeros(len(df)),
        iq_range=np.zeros(len(df)),
        hi_bound=df["hi_bound"].values,
        lo_bound=df["lo_bound"].values,
        min_x=df["min_x"].values,
        lambda_x=df["lambda_x"].values,
        mu_x=df["mu_x"].values,
        sigma_x=df["sigma_x"].values,
        min_y=0.0,
        lambda_y=np.array([]),
        sigma_y=np.array([]),
        mu_y=np.array([])
    )


def load_test_features():
    """Load raw test features."""
    df = pd.read_csv("tests/matlab_reference/input/metadata_test.csv")

    instance_labels = df.iloc[:, 0].values
    features = df.iloc[:, 1:11].values
    feature_cols = df.columns[1:11]
    feature_names = [col.replace('feature_', '') for col in feature_cols]

    return instance_labels, features, feature_names


def test_prelim_validation():
    """Validate PRELIM implementation against MATLAB reference."""
    print_banner("PRELIM Stage Validation Test", "=", 80)

    # Step 1: Load input data
    print("[Step 1/4] Loading Input Data...")
    inst_labels, x_raw, feature_names = load_test_features()
    prelim_params = load_prelim_params()

    print(f"  INPUT:")
    print(f"    - Test instances: {len(inst_labels)}")
    print(f"    - Features: {len(feature_names)}")
    print(f"    - Data shape: {x_raw.shape}")
    print(f"    - Data source: tests/matlab_reference/input/metadata_test.csv")
    print()
    print(f"  PARAMETERS:")
    print(f"    - Transformation: Bounding -> Box-Cox -> Z-score")
    print(f"    - Source: tests/matlab_reference/training_artifacts/prelim/prelim_params.csv")

    # Step 2: Run Python implementation
    print("\n[Step 2/4] Running Python PRELIM Implementation...")
    mock_instance_space = Mock(spec=InstanceSpace)
    mock_instance_space._model = Mock()
    mock_instance_space._model.prelim = prelim_params

    x_prelim_python = InstanceSpace._explore_prelim(mock_instance_space, x_raw)

    print(f"  OUTPUT:")
    print(f"    - Transformed shape: {x_prelim_python.shape}")
    print(f"    - Sample values (first instance, first 3 features):")
    for i in range(min(3, len(feature_names))):
        print(f"      {feature_names[i]:<40} {x_prelim_python[0, i]:>12.6f}")

    # Step 3: Load MATLAB reference
    print("\n[Step 3/4] Loading MATLAB Reference Output...")
    matlab_prelim = pd.read_csv(
        "tests/matlab_reference/explore_outputs/step1_after_prelim.csv",
        index_col="instance_id"
    )
    python_prelim = pd.DataFrame(x_prelim_python, index=inst_labels, columns=feature_names)

    print(f"  REFERENCE:")
    print(f"    - Shape: {matlab_prelim.shape}")
    print(f"    - Source: tests/matlab_reference/explore_outputs/step1_after_prelim.csv")

    # Step 4: Compare and validate
    print("\n[Step 4/4] Comparing Python vs MATLAB Outputs...")

    # Calculate differences
    diff = python_prelim.values - matlab_prelim.values
    abs_diff = np.abs(diff)
    rel_diff = abs_diff / (np.abs(matlab_prelim.values) + 1e-10)

    max_abs_diff = np.max(abs_diff)
    max_rel_diff = np.max(rel_diff)
    mean_rel_diff = np.mean(rel_diff)

    print(f"\n  DIFFERENCE STATISTICS:")
    print(f"    Max Absolute Error:  {max_abs_diff:.6e}")
    print(f"    Max Relative Error:  {max_rel_diff*100:.4f}%")
    print(f"    Mean Relative Error: {mean_rel_diff*100:.6f}%")

    # Detailed comparison table
    print(f"\n  DETAILED COMPARISON (first 5 instances, first 3 features):")
    print(f"  {'Instance':<15} {'Feature':<40} {'Python':>12} {'MATLAB':>12} {'Diff%':>10}")
    print(f"  {'-'*95}")

    for i in range(min(5, len(inst_labels))):
        for j in range(min(3, len(feature_names))):
            inst = inst_labels[i]
            feat = feature_names[j]
            py_val = python_prelim.iloc[i, j]
            ml_val = matlab_prelim.iloc[i, j]
            diff_pct = ((py_val - ml_val) / (abs(ml_val) + 1e-10)) * 100
            print(f"  {inst:<15} {feat:<40} {py_val:>12.6f} {ml_val:>12.6f} {diff_pct:>9.4f}%")
        if i < min(5, len(inst_labels)) - 1:
            print(f"  {'-'*95}")

    # Validation check
    print("\n  VALIDATION CHECK:")
    try:
        np.testing.assert_allclose(
            python_prelim.values,
            matlab_prelim.values,
            rtol=0.05,
            err_msg="Values exceed 5% tolerance"
        )

        print(f"    Tolerance Threshold: 5%")
        print(f"    Actual Max Error:    {max_rel_diff*100:.4f}%")
        print(f"    Status:              WITHIN TOLERANCE")

        # Print success banner
        print_pass_banner()
        print_banner("PRELIM Validation: PASSED", "=", 80)
        print(f"  Python implementation matches MATLAB within {max_rel_diff*100:.4f}% error")
        print(f"  (Required: <5%, Achieved: {max_rel_diff*100:.6f}%)")
        print("=" * 80)

        return True

    except AssertionError as e:
        print(f"\n  [FAIL] Validation failed: {e}")
        print_banner("PRELIM Validation: FAILED", "!", 80)
        return False


if __name__ == "__main__":
    success = test_prelim_validation()
    sys.exit(0 if success else 1)
