# ruff: noqa: D103, PLR2004, SLF001
"""Tests for PILOT stage's explore()-time inference (_explore_pilot).

Unit tests exercise _explore_pilot() with mocked/stubbed dependencies, independent
of MATLAB reference data. PILOT inference is the dimension-generic linear projection
``z = x @ A.T`` used by MATLAB explore, including its deliberate lack of the
centering used by the PLS build projection.
"""

from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from instancespace.data.model import PilotOut
from instancespace.instance_space import InstanceSpace

REFERENCE_DIR = Path("tests/matlab_reference")
ARTIFACTS_DIR = REFERENCE_DIR / "training_artifacts"
OUTPUTS_DIR = REFERENCE_DIR / "explore_outputs"

_rng = np.random.default_rng()


def make_instance_space(a: NDArray[np.double]) -> InstanceSpace:
    pilot = Mock(spec=PilotOut)
    pilot.a = a
    model = Mock()
    model.pilot = pilot
    instance_space = Mock(spec=InstanceSpace)
    instance_space._model = model
    instance_space._require_model = Mock(return_value=model)
    return instance_space


def test_pilot_output_shape() -> None:
    a = np.eye(2, 5)  # (2, 5): projects 5 features → 2 dimensions
    x = _rng.random((10, 5))
    result = InstanceSpace._explore_pilot(make_instance_space(a), x)
    assert result.shape == (10, 2)


def test_pilot_correct_projection() -> None:
    # Z = X @ A.T — verify with known values
    a = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])  # (2, 3)
    x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    result = InstanceSpace._explore_pilot(make_instance_space(a), x)
    expected = np.array([[1.0, 2.0], [4.0, 5.0]])
    np.testing.assert_array_almost_equal(result, expected)


def test_pilot_3d_projection_preserves_matlab_explore_centering_asymmetry() -> None:
    """Explore uses exact uncentred X @ A.T even when a PLS build was centred."""
    a = np.array(
        [
            [1.0, 0.0, 0.0, 0.5],
            [0.0, 1.0, -1.0, 0.0],
            [0.25, 0.0, 0.5, 1.0],
        ],
    )
    x = np.array(
        [
            [2.0, 4.0, 6.0, 8.0],
            [4.0, 8.0, 12.0, 16.0],
            [6.0, 12.0, 18.0, 24.0],
        ],
    )

    result = InstanceSpace._explore_pilot(make_instance_space(a), x)
    uncentred = x @ a.T
    centred = (x - np.mean(x, axis=0)) @ a.T

    assert result.shape == (x.shape[0], a.shape[0])
    np.testing.assert_array_equal(result, uncentred)
    assert not np.array_equal(result, centred)


def test_pilot_single_instance() -> None:
    a = _rng.random((2, 4))
    x = _rng.random((1, 4))
    result = InstanceSpace._explore_pilot(make_instance_space(a), x)
    assert result.shape == (1, 2)


def test_pilot_preserves_input() -> None:
    a = _rng.random((2, 3))
    x = _rng.random((5, 3))
    x_copy = x.copy()
    InstanceSpace._explore_pilot(make_instance_space(a), x)
    np.testing.assert_array_equal(x, x_copy)


def test_pilot_deterministic() -> None:
    a = _rng.random((2, 6))
    x = _rng.random((20, 6))
    r1 = InstanceSpace._explore_pilot(make_instance_space(a), x)
    r2 = InstanceSpace._explore_pilot(make_instance_space(a), x)
    np.testing.assert_array_equal(r1, r2)


def test_pilot_zero_matrix() -> None:
    a = np.zeros((2, 4))
    x = _rng.random((5, 4))
    result = InstanceSpace._explore_pilot(make_instance_space(a), x)
    np.testing.assert_array_equal(result, np.zeros((5, 2)))


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
        OUTPUTS_DIR / "step2_after_sifted.csv",
        index_col="instance_id",
    ).to_numpy(dtype=np.double)

    instance_space = Mock(spec=InstanceSpace)
    instance_space._model = load_pilot_matrix()
    instance_space._require_model = Mock(return_value=instance_space._model)

    result = InstanceSpace._explore_pilot(instance_space, x_input)

    expected = pd.read_csv(
        OUTPUTS_DIR / "step3_after_pilot.csv",
        index_col=0,
    ).to_numpy(dtype=np.double)

    assert result.shape == expected.shape
    rel_err = np.abs(result - expected) / (np.abs(expected) + 1e-12)
    max_err = float(rel_err.max())
    mean_err = float(rel_err.mean())

    print(f"\nInput:    {x_input.shape[0]} instances x {x_input.shape[1]} features")
    print(f"Max relative error: {max_err * 100:.4f}%")
    print(f"Mean relative error: {mean_err * 100:.6f}%")

    assert (
        max_err < 0.01
    ), f"PILOT max relative error {max_err * 100:.4f}% >= 1% threshold"
    print(f"[PASS] PILOT validation: {max_err * 100:.4f}% max error")
