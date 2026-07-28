"""Validation test for PRELIM stage against MATLAB reference data.

Loads MATLAB-trained PRELIM parameters (bounds, Box-Cox lambda, z-score
mu/sigma) and verifies that _explore_prelim reproduces MATLAB's per-feature
transformations on the 235-instance test set.

Threshold: max relative error < 1%. PRELIM is a deterministic pipeline
(bounding -> Box-Cox -> z-score) — when fed the same parameters as MATLAB,
Python should match MATLAB to floating-point precision.
"""

from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pandas as pd

from instancespace.data.model import PrelimOut
from instancespace.instance_space import InstanceSpace

REFERENCE_DIR = Path("tests/matlab_reference")
ARTIFACTS_DIR = REFERENCE_DIR / "training_artifacts"
OUTPUTS_DIR = REFERENCE_DIR / "explore_outputs"
INPUT_DIR = REFERENCE_DIR / "input"


def load_prelim_params() -> PrelimOut:
    df = pd.read_csv(ARTIFACTS_DIR / "prelim" / "prelim_params.csv")
    n = len(df)
    return PrelimOut(
        med_val=np.zeros(n),
        iq_range=np.zeros(n),
        hi_bound=df["hi_bound"].to_numpy(),
        lo_bound=df["lo_bound"].to_numpy(),
        min_x=df["min_x"].to_numpy(),
        lambda_x=df["lambda_x"].to_numpy(),
        mu_x=df["mu_x"].to_numpy(),
        sigma_x=df["sigma_x"].to_numpy(),
        min_y=0.0,
        lambda_y=np.array([]),
        sigma_y=np.array([]),
        mu_y=np.array([]),
    )


def test_prelim_matches_matlab():
    """PRELIM max relative error < 1% against MATLAB step1."""
    test_df = pd.read_csv(INPUT_DIR / "metadata_test.csv")
    x_raw = test_df.iloc[:, 1:11].to_numpy(dtype=np.double)

    instance_space = Mock(spec=InstanceSpace)
    instance_space._model = Mock()
    instance_space._model.prelim = load_prelim_params()
    instance_space._require_model = Mock(return_value=instance_space._model)

    result = InstanceSpace._explore_prelim(instance_space, x_raw)

    expected = pd.read_csv(
        OUTPUTS_DIR / "step1_after_prelim.csv", index_col="instance_id",
    ).to_numpy(dtype=np.double)

    assert result.shape == expected.shape
    rel_err = np.abs(result - expected) / (np.abs(expected) + 1e-12)
    max_err = float(rel_err.max())
    mean_err = float(rel_err.mean())

    print(f"\nInput:    {x_raw.shape[0]} instances x {x_raw.shape[1]} features")
    print(f"Max relative error: {max_err * 100:.4f}%")
    print(f"Mean relative error: {mean_err * 100:.6f}%")

    assert max_err < 0.01, (
        f"PRELIM max relative error {max_err * 100:.4f}% >= 1% threshold"
    )
    print(f"[PASS] PRELIM validation: {max_err * 100:.4f}% max error")
