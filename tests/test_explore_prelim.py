"""Tests for PRELIM stage's explore()-time inference (_explore_prelim).

Unit tests verify _explore_prelim() with various edge cases and error conditions,
independent of MATLAB reference data. The validation test loads MATLAB-trained
PRELIM parameters (bounds, Box-Cox lambda, z-score mu/sigma) and verifies that
_explore_prelim reproduces MATLAB's per-feature transformations on the 235-instance
test set (threshold: max relative error < 1% - PRELIM is a deterministic pipeline
[bounding -> Box-Cox -> z-score], so Python should match MATLAB to floating-point
precision when fed the same parameters).
"""

from collections.abc import Callable
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from instancespace.data.model import PrelimOut
from instancespace.instance_space import InstanceSpace

REFERENCE_DIR = Path("tests/matlab_reference")
ARTIFACTS_DIR = REFERENCE_DIR / "training_artifacts"
OUTPUTS_DIR = REFERENCE_DIR / "explore_outputs"
INPUT_DIR = REFERENCE_DIR / "input"


@pytest.fixture
def mock_prelim_params() -> PrelimOut:
    """Create mock PRELIM parameters for testing."""
    n_features = 3
    return PrelimOut(
        med_val=np.zeros(n_features),
        iq_range=np.zeros(n_features),
        hi_bound=np.array([10.0, 10.0, 10.0]),
        lo_bound=np.array([0.0, 0.0, 0.0]),
        min_x=np.array([0.0, 0.0, 0.0]),
        lambda_x=np.array([1.0, 1.0, 1.0]),  # No Box-Cox transform
        mu_x=np.array([5.0, 5.0, 5.0]),
        sigma_x=np.array([2.0, 2.0, 2.0]),
        min_y=0.0,
        lambda_y=np.array([]),
        sigma_y=np.array([]),
        mu_y=np.array([])
    )


@pytest.fixture
def mock_instance_space(mock_prelim_params: PrelimOut) -> InstanceSpace:
    """Create mock InstanceSpace with PRELIM parameters."""
    mock_is = Mock(spec=InstanceSpace)
    mock_is._model = Mock()
    mock_is._model.prelim = mock_prelim_params
    mock_is._require_model = Mock(return_value=mock_is._model)
    return mock_is


def test_prelim_basic_functionality(mock_instance_space: InstanceSpace) -> None:
    """Test basic PRELIM transformation."""
    # Input data: 5 instances, 3 features
    x_raw = np.array([
        [5.0, 5.0, 5.0],
        [1.0, 2.0, 3.0],
        [9.0, 8.0, 7.0],
        [0.5, 5.5, 10.5],
        [5.0, 5.0, 5.0]
    ])

    x_transformed = InstanceSpace._explore_prelim(mock_instance_space, x_raw)

    # Check output shape
    assert x_transformed.shape == x_raw.shape
    assert x_transformed.shape == (5, 3)

    # Check output is float array
    assert x_transformed.dtype == np.float64


def test_prelim_preserves_input(mock_instance_space: InstanceSpace) -> None:
    """Test that PRELIM doesn't modify input array."""
    x_raw = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    x_raw_copy = x_raw.copy()

    InstanceSpace._explore_prelim(mock_instance_space, x_raw)

    # Original input should be unchanged
    np.testing.assert_array_equal(x_raw, x_raw_copy)


def test_prelim_single_instance(mock_instance_space: InstanceSpace) -> None:
    """Test PRELIM with single instance."""
    x_raw = np.array([[5.0, 5.0, 5.0]])

    x_transformed = InstanceSpace._explore_prelim(mock_instance_space, x_raw)

    assert x_transformed.shape == (1, 3)


def test_prelim_bounding(mock_instance_space: InstanceSpace) -> None:
    """Test that PRELIM applies bounding correctly."""
    # Values outside bounds [0, 10]
    x_raw = np.array([
        [-5.0, 15.0, 5.0],  # Out of bounds
        [5.0, 5.0, 5.0]     # Within bounds
    ])

    x_transformed = InstanceSpace._explore_prelim(mock_instance_space, x_raw)

    # All values should be finite (no inf/nan from out-of-bounds)
    assert np.all(np.isfinite(x_transformed))


def test_prelim_handles_nan_input(mock_instance_space: InstanceSpace) -> None:
    """Test PRELIM handles NaN values in input."""
    x_raw = np.array([
        [5.0, np.nan, 5.0],
        [1.0, 2.0, np.nan]
    ])

    x_transformed = InstanceSpace._explore_prelim(mock_instance_space, x_raw)

    # Output shape should be preserved
    assert x_transformed.shape == (2, 3)

    # NaN values should remain NaN (not transformed)
    assert np.isnan(x_transformed[0, 1])
    assert np.isnan(x_transformed[1, 2])


def test_prelim_all_nan_feature(mock_instance_space: InstanceSpace) -> None:
    """Test PRELIM with a feature that is all NaN."""
    x_raw = np.array([
        [5.0, np.nan, 5.0],
        [1.0, np.nan, 3.0],
        [9.0, np.nan, 7.0]
    ])

    x_transformed = InstanceSpace._explore_prelim(mock_instance_space, x_raw)

    # Output should have same NaN pattern
    assert np.all(np.isnan(x_transformed[:, 1]))
    # Other features should be transformed
    assert np.all(np.isfinite(x_transformed[:, 0]))
    assert np.all(np.isfinite(x_transformed[:, 2]))


def test_prelim_dimension_consistency(mock_instance_space: InstanceSpace) -> None:
    """Test PRELIM with various input dimensions."""
    # Test with different numbers of instances
    for n_instances in [1, 10, 100]:
        x_raw = np.random.default_rng().random((n_instances, 3)) * 10

        x_transformed = InstanceSpace._explore_prelim(mock_instance_space, x_raw)

        assert x_transformed.shape == (n_instances, 3)


def test_prelim_deterministic(mock_instance_space: InstanceSpace) -> None:
    """Test that PRELIM is deterministic (same input → same output)."""
    x_raw = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    result1 = InstanceSpace._explore_prelim(mock_instance_space, x_raw)
    result2 = InstanceSpace._explore_prelim(mock_instance_space, x_raw)

    np.testing.assert_array_equal(result1, result2)


def test_prelim_numerical_stability(mock_instance_space: InstanceSpace) -> None:
    """Test PRELIM with very small and very large values."""
    x_raw = np.array([
        [1e-10, 1e10, 5.0],
        [5.0, 5.0, 5.0]
    ])

    # Should not raise errors
    x_transformed = InstanceSpace._explore_prelim(mock_instance_space, x_raw)

    # After bounding, values should be clipped to [0, 10]
    # After transformation, should be finite
    assert np.all(np.isfinite(x_transformed))


def _collect_warnings(fn: Callable[..., object], *args: object) -> list[str]:
    """Run fn(*args) and return the loguru WARNING-level messages it emitted."""
    from loguru import logger

    messages: list[str] = []
    sink_id = logger.add(lambda msg: messages.append(msg.record["message"]), level="WARNING")
    try:
        fn(*args)
    finally:
        logger.remove(sink_id)
    return messages


def test_prelim_ood_warning_fires_above_threshold(mock_instance_space: InstanceSpace) -> None:
    """Regression test for #250: explore() warns when >5% of instances are clipped."""
    # Bounds are [0, 10] on all 3 features (mock_prelim_params). 2 of 10 instances
    # (20%) have a feature outside bounds -> above the 5% threshold.
    x_raw = np.vstack([
        np.full((8, 3), 5.0),
        np.array([[-5.0, 5.0, 5.0], [5.0, 15.0, 5.0]]),
    ])

    messages = _collect_warnings(InstanceSpace._explore_prelim, mock_instance_space, x_raw)

    assert any("clipped" in m for m in messages)


def test_prelim_ood_warning_silent_below_threshold(mock_instance_space: InstanceSpace) -> None:
    """No warning when no instance needs clipping."""
    x_raw = np.full((10, 3), 5.0)  # well within [0, 10] on all features

    messages = _collect_warnings(InstanceSpace._explore_prelim, mock_instance_space, x_raw)

    assert messages == []


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


def test_prelim_matches_matlab() -> None:
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
