"""Validation test for PILOT stage against MATLAB reference data.

Loads MATLAB-trained PILOT projection matrix A and verifies that
_explore_pilot reproduces MATLAB's 2D projection on the test set.

Threshold: max relative error < 1%. PILOT inference is a pure linear
projection z = x @ A.T; when fed MATLAB's trained A, Python should match
MATLAB to floating-point precision.
"""

from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pandas as pd

from instancespace.data.model import PilotOut
from instancespace.instance_space import InstanceSpace

REFERENCE_DIR = Path("tests/matlab_reference")
ARTIFACTS_DIR = REFERENCE_DIR / "training_artifacts"
OUTPUTS_DIR = REFERENCE_DIR / "explore_outputs"


def load_pilot_matrix() -> Mock:
    df = pd.read_csv(ARTIFACTS_DIR / "pilot" / "pilot_matrix.csv")
    a = df[["z1_coef", "z2_coef"]].to_numpy(dtype=np.double).T

    pilot = Mock(spec=PilotOut)
    pilot.a = a
    model = Mock()
    model.pilot = pilot
    return model


def test_pilot_matches_matlab() -> None:
    """PILOT max relative error < 1% against MATLAB step3."""
    x_input = pd.read_csv(
        OUTPUTS_DIR / "step2_after_sifted.csv", index_col="instance_id",
    ).to_numpy(dtype=np.double)

    instance_space = Mock(spec=InstanceSpace)
    instance_space._model = load_pilot_matrix()
    instance_space._require_model = Mock(return_value=instance_space._model)

    result = InstanceSpace._explore_pilot(instance_space, x_input)

    expected = pd.read_csv(
        OUTPUTS_DIR / "step3_after_pilot.csv", index_col=0,
    ).to_numpy(dtype=np.double)

    assert result.shape == expected.shape
    rel_err = np.abs(result - expected) / (np.abs(expected) + 1e-12)
    max_err = float(rel_err.max())
    mean_err = float(rel_err.mean())

    print(f"\nInput:    {x_input.shape[0]} instances x {x_input.shape[1]} features")
    print(f"Max relative error: {max_err * 100:.4f}%")
    print(f"Mean relative error: {mean_err * 100:.6f}%")

    assert max_err < 0.01, (
        f"PILOT max relative error {max_err * 100:.4f}% >= 1% threshold"
    )
    print(f"[PASS] PILOT validation: {max_err * 100:.4f}% max error")
