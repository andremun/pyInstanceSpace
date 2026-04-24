"""Validation test for PILOT stage against MATLAB reference data.

Validates that _explore_pilot() produces outputs matching MATLAB within 5% tolerance.
"""

from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from instancespace.data.model import PilotOut
from instancespace.instance_space import InstanceSpace

REFERENCE_DIR = Path("tests/matlab_reference")
ARTIFACTS_DIR = REFERENCE_DIR / "training_artifacts"
OUTPUTS_DIR = REFERENCE_DIR / "explore_outputs"


def build_mock_model() -> Mock:
    """Construct mock model with PILOT projection matrix from MATLAB training artifacts."""
    df = pd.read_csv(ARTIFACTS_DIR / "pilot_matrix.csv")
    # a has shape (2, n_features): row 0 = z1 coefficients, row 1 = z2 coefficients
    a = df[["z1_coef", "z2_coef"]].values.T

    pilot = Mock(spec=PilotOut)
    pilot.a = a

    model = Mock()
    model.pilot = pilot
    return model


def test_pilot_matches_matlab():
    """PILOT output must match MATLAB step3_after_pilot.csv within 5% tolerance."""
    # Load input: features after SIFTED (step2)
    x_input = pd.read_csv(OUTPUTS_DIR / "step2_after_sifted.csv", index_col=0).values

    # Load expected output
    expected = pd.read_csv(OUTPUTS_DIR / "step3_after_pilot.csv", index_col=0).values

    # Build mock and run
    instance_space = Mock(spec=InstanceSpace)
    instance_space._model = build_mock_model()
    result = InstanceSpace._explore_pilot(instance_space, x_input)

    # Validate shape
    assert result.shape == expected.shape, (
        f"Shape mismatch: got {result.shape}, expected {expected.shape}"
    )

    # Compute errors
    mask = ~np.isnan(expected) & ~np.isnan(result)
    abs_err = np.abs(result[mask] - expected[mask])
    rel_err = abs_err / (np.abs(expected[mask]) + 1e-10)
    max_rel_err = rel_err.max() * 100

    # Report
    print(f"\nInput:    {x_input.shape[0]} instances × {x_input.shape[1]} features")
    print(f"Output:   {result.shape[0]} instances × {result.shape[1]} coordinates (z1, z2)")
    print(f"Max relative error: {max_rel_err:.4f}%")
    print(f"Mean relative error: {rel_err.mean() * 100:.4f}%")

    # Sample comparison
    print("\nSample (first 5 rows):")
    print(f"{'z1 python':>15} {'z1 matlab':>15} {'z2 python':>15} {'z2 matlab':>15}")
    for i in range(min(5, result.shape[0])):
        print(
            f"{result[i, 0]:15.6f} {expected[i, 0]:15.6f} "
            f"{result[i, 1]:15.6f} {expected[i, 1]:15.6f}"
        )

    assert max_rel_err < 5.0, f"Max relative error {max_rel_err:.4f}% exceeds 5% tolerance"
    print(f"\n[PASS] PILOT validation: {max_rel_err:.4f}% max error (threshold: 5%)")
