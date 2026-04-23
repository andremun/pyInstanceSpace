#!/usr/bin/env python3
"""Validation test for SIFTED stage against MATLAB reference data.

This test validates that the Python _explore_sifted() implementation
produces outputs that match MATLAB within 5% tolerance.
"""

import sys
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from instancespace.data.model import SiftedOut
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


def load_sifted_indices():
    """Load SIFTED indices from MATLAB training artifacts."""
    df = pd.read_csv("tests/matlab_reference/training_artifacts/sifted_indices.csv")
    indices_0based = df["original_index"].values - 1  # Convert from 1-based to 0-based

    return SiftedOut(
        selvars=indices_0based.astype(np.intc),
        idx=indices_0based.astype(np.intc),
        rho=None,
        pval=None,
        silhouette_scores=None,
        clust=None
    )


def test_sifted_validation():
    """Validate SIFTED implementation against MATLAB reference."""
    print_banner("SIFTED Stage Validation Test", "=", 80)

    # Step 1: Load input data (PRELIM output)
    print("[Step 1/4] Loading Input Data (PRELIM Output)...")
    matlab_prelim = pd.read_csv(
        "tests/matlab_reference/explore_outputs/step1_after_prelim.csv",
        index_col="instance_id"
    )
    x_prelim = matlab_prelim.values
    inst_labels = matlab_prelim.index.values
    feature_names = list(matlab_prelim.columns)

    sifted_params = load_sifted_indices()

    print(f"  INPUT:")
    print(f"    - Data shape: {x_prelim.shape}")
    print(f"    - Features: {len(feature_names)}")
    print(f"    - Source: tests/matlab_reference/explore_outputs/step1_after_prelim.csv")
    print()
    print(f"  SELECTION PARAMETERS:")
    print(f"    - Selected indices: {list(sifted_params.selvars)}")
    print(f"    - Selected features: {len(sifted_params.selvars)}")
    print(f"    - Source: tests/matlab_reference/training_artifacts/sifted_indices.csv")

    # Step 2: Run Python implementation
    print("\n[Step 2/4] Running Python SIFTED Implementation...")
    mock_instance_space = Mock(spec=InstanceSpace)
    mock_instance_space._model = Mock()
    mock_instance_space._model.sifted = sifted_params

    x_sifted_python = InstanceSpace._explore_sifted(mock_instance_space, x_prelim)

    selected_feature_names = [feature_names[i] for i in sifted_params.selvars]

    print(f"  OUTPUT:")
    print(f"    - Output shape: {x_sifted_python.shape}")
    print(f"    - Selected features: {selected_feature_names}")
    print(f"    - Sample values (first instance, first 3 selected features):")
    for i in range(min(3, len(selected_feature_names))):
        print(f"      {selected_feature_names[i]:<40} {x_sifted_python[0, i]:>12.6f}")

    # Step 3: Load MATLAB reference
    print("\n[Step 3/4] Loading MATLAB Reference Output...")
    matlab_sifted = pd.read_csv(
        "tests/matlab_reference/explore_outputs/step2_after_sifted.csv",
        index_col="instance_id"
    )
    python_sifted = pd.DataFrame(
        x_sifted_python,
        index=inst_labels,
        columns=selected_feature_names
    )

    print(f"  REFERENCE:")
    print(f"    - Shape: {matlab_sifted.shape}")
    print(f"    - Source: tests/matlab_reference/explore_outputs/step2_after_sifted.csv")

    # Step 4: Compare and validate
    print("\n[Step 4/4] Comparing Python vs MATLAB Outputs...")

    # Calculate differences
    diff = python_sifted.values - matlab_sifted.values
    abs_diff = np.abs(diff)
    rel_diff = abs_diff / (np.abs(matlab_sifted.values) + 1e-10)

    max_abs_diff = np.max(abs_diff)
    max_rel_diff = np.max(rel_diff)
    mean_rel_diff = np.mean(rel_diff)

    print(f"\n  DIFFERENCE STATISTICS:")
    print(f"    Max Absolute Error:  {max_abs_diff:.6e}")
    print(f"    Max Relative Error:  {max_rel_diff*100:.4f}%")
    print(f"    Mean Relative Error: {mean_rel_diff*100:.6f}%")

    # Detailed comparison table
    print(f"\n  DETAILED COMPARISON (first 5 instances, all selected features):")
    print(f"  {'Instance':<15} {'Feature':<40} {'Python':>12} {'MATLAB':>12} {'Diff%':>10}")
    print(f"  {'-'*95}")

    for i in range(min(5, len(inst_labels))):
        for j in range(len(selected_feature_names)):
            inst = inst_labels[i]
            feat = selected_feature_names[j]
            py_val = python_sifted.iloc[i, j]
            ml_val = matlab_sifted.iloc[i, j]
            diff_pct = ((py_val - ml_val) / (abs(ml_val) + 1e-10)) * 100
            print(f"  {inst:<15} {feat:<40} {py_val:>12.6f} {ml_val:>12.6f} {diff_pct:>9.4f}%")
        if i < min(5, len(inst_labels)) - 1:
            print(f"  {'-'*95}")

    # Validation check
    print("\n  VALIDATION CHECK:")
    try:
        np.testing.assert_allclose(
            python_sifted.values,
            matlab_sifted.values,
            rtol=0.05,
            err_msg="Values exceed 5% tolerance"
        )

        print(f"    Tolerance Threshold: 5%")
        print(f"    Actual Max Error:    {max_rel_diff*100:.4f}%")
        print(f"    Status:              WITHIN TOLERANCE")

        # Print success banner
        print_pass_banner()
        print_banner("SIFTED Validation: PASSED", "=", 80)
        print(f"  Python implementation matches MATLAB within {max_rel_diff*100:.4f}% error")
        print(f"  (Required: <5%, Achieved: {max_rel_diff*100:.6f}%)")
        print("=" * 80)

        return True

    except AssertionError as e:
        print(f"\n  [FAIL] Validation failed: {e}")
        print_banner("SIFTED Validation: FAILED", "!", 80)
        return False


if __name__ == "__main__":
    success = test_sifted_validation()
    sys.exit(0 if success else 1)
