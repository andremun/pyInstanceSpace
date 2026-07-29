#!/usr/bin/env python3
"""Unit tests for PRELIM stage of explore() pipeline.

These tests verify the _explore_prelim() implementation with various
edge cases and error conditions, independent of MATLAB reference data.
"""

import sys
from collections.abc import Callable
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from instancespace.data.model import PrelimOut
from instancespace.instance_space import InstanceSpace


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
        x_raw = np.random.rand(n_instances, 3) * 10

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


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
