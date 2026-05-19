"""Validation test for SIFTED stage against MATLAB reference data.

Loads MATLAB-trained SIFTED feature indices and verifies that
_explore_sifted reproduces MATLAB's column selection on the test set.

Threshold: exact match. Feature selection is pure indexing — no
numerical operations, so any deviation indicates a port bug.
"""

from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pandas as pd

from instancespace.data.model import SiftedOut
from instancespace.instance_space import InstanceSpace

REFERENCE_DIR = Path("tests/matlab_reference")
ARTIFACTS_DIR = REFERENCE_DIR / "training_artifacts"
OUTPUTS_DIR = REFERENCE_DIR / "explore_outputs"


def load_sifted_indices() -> SiftedOut:
    df = pd.read_csv(ARTIFACTS_DIR / "sifted" / "sifted_indices.csv")
    selvars = (df["original_index"].to_numpy(dtype=np.intc) - 1).astype(np.intc)
    return SiftedOut(
        selvars=selvars,
        idx=selvars,
        rho=None,
        pval=None,
        silhouette_scores=None,
        clust=None,
    )


def test_sifted_matches_matlab():
    """SIFTED output exactly matches MATLAB step2."""
    x_input = pd.read_csv(
        OUTPUTS_DIR / "step1_after_prelim.csv", index_col="instance_id",
    ).to_numpy(dtype=np.double)

    instance_space = Mock(spec=InstanceSpace)
    instance_space._model = Mock()
    instance_space._model.sifted = load_sifted_indices()

    result = InstanceSpace._explore_sifted(instance_space, x_input)

    expected = pd.read_csv(
        OUTPUTS_DIR / "step2_after_sifted.csv", index_col="instance_id",
    ).to_numpy(dtype=np.double)

    assert result.shape == expected.shape
    max_abs_err = float(np.max(np.abs(result - expected)))

    print(f"\nInput:    {x_input.shape[0]} instances x {x_input.shape[1]} features")
    print(f"Output:   {result.shape[0]} instances x {result.shape[1]} selected features")
    print(f"Max absolute error: {max_abs_err:.2e}")

    np.testing.assert_array_equal(result, expected)
    print(f"[PASS] SIFTED validation: exact match")
