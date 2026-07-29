#!/usr/bin/env python3
"""Unit tests for SIFTED stage of explore() pipeline.

These tests verify the _explore_sifted() implementation with various
edge cases and error conditions, independent of MATLAB reference data.
"""

import sys
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from instancespace.data.model import SiftedOut
from instancespace.instance_space import InstanceSpace


@pytest.fixture
def mock_sifted_params_all() -> SiftedOut:
    """Create mock SIFTED parameters selecting all features."""
    # Select all 5 features (0-based indexing)
    indices = np.array([0, 1, 2, 3, 4], dtype=np.intc)
    return SiftedOut(
        selvars=indices,
        rho=None,
        pval=None,
        silhouette_scores=None,
        clust=None
    )


@pytest.fixture
def mock_sifted_params_subset() -> SiftedOut:
    """Create mock SIFTED parameters selecting subset of features."""
    # Select features 0, 2, 4 (skip 1 and 3)
    indices = np.array([0, 2, 4], dtype=np.intc)
    return SiftedOut(
        selvars=indices,
        rho=None,
        pval=None,
        silhouette_scores=None,
        clust=None
    )


@pytest.fixture
def mock_instance_space_all(mock_sifted_params_all: SiftedOut) -> InstanceSpace:
    """Create mock InstanceSpace selecting all features."""
    mock_is = Mock(spec=InstanceSpace)
    mock_is._model = Mock()
    mock_is._model.sifted = mock_sifted_params_all
    mock_is._require_model = Mock(return_value=mock_is._model)
    return mock_is


@pytest.fixture
def mock_instance_space_subset(mock_sifted_params_subset: SiftedOut) -> InstanceSpace:
    """Create mock InstanceSpace selecting subset of features."""
    mock_is = Mock(spec=InstanceSpace)
    mock_is._model = Mock()
    mock_is._model.sifted = mock_sifted_params_subset
    mock_is._require_model = Mock(return_value=mock_is._model)
    return mock_is


def test_sifted_select_all_features(mock_instance_space_all: InstanceSpace) -> None:
    """Test SIFTED when all features are selected."""
    x_prelim = np.array([
        [1.0, 2.0, 3.0, 4.0, 5.0],
        [6.0, 7.0, 8.0, 9.0, 10.0]
    ])

    x_sifted = InstanceSpace._explore_sifted(mock_instance_space_all, x_prelim)

    # All features selected, output should equal input
    np.testing.assert_array_equal(x_sifted, x_prelim)
    assert x_sifted.shape == (2, 5)


def test_sifted_select_subset(mock_instance_space_subset: InstanceSpace) -> None:
    """Test SIFTED selecting subset of features."""
    x_prelim = np.array([
        [1.0, 2.0, 3.0, 4.0, 5.0],
        [6.0, 7.0, 8.0, 9.0, 10.0]
    ])

    x_sifted = InstanceSpace._explore_sifted(mock_instance_space_subset, x_prelim)

    # Should select features 0, 2, 4
    expected = np.array([
        [1.0, 3.0, 5.0],
        [6.0, 8.0, 10.0]
    ])

    np.testing.assert_array_equal(x_sifted, expected)
    assert x_sifted.shape == (2, 3)


def test_sifted_preserves_input(mock_instance_space_subset: InstanceSpace) -> None:
    """Test that SIFTED doesn't modify input array."""
    x_prelim = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]])
    x_prelim_copy = x_prelim.copy()

    InstanceSpace._explore_sifted(mock_instance_space_subset, x_prelim)

    # Original input should be unchanged
    np.testing.assert_array_equal(x_prelim, x_prelim_copy)


def test_sifted_single_instance(mock_instance_space_subset: InstanceSpace) -> None:
    """Test SIFTED with single instance."""
    x_prelim = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]])

    x_sifted = InstanceSpace._explore_sifted(mock_instance_space_subset, x_prelim)

    assert x_sifted.shape == (1, 3)
    np.testing.assert_array_equal(x_sifted, np.array([[1.0, 3.0, 5.0]]))


def test_sifted_single_feature() -> None:
    """Test SIFTED selecting only one feature."""
    # Select only feature 2
    indices = np.array([2], dtype=np.intc)
    params = SiftedOut(
        selvars=indices,
        rho=None,
        pval=None,
        silhouette_scores=None,
        clust=None
    )

    mock_is = Mock(spec=InstanceSpace)
    mock_is._model = Mock()
    mock_is._model.sifted = params
    mock_is._require_model = Mock(return_value=mock_is._model)

    x_prelim = np.array([
        [1.0, 2.0, 3.0, 4.0, 5.0],
        [6.0, 7.0, 8.0, 9.0, 10.0]
    ])

    x_sifted = InstanceSpace._explore_sifted(mock_is, x_prelim)

    # Should select only feature 2
    expected = np.array([[3.0], [8.0]])

    np.testing.assert_array_equal(x_sifted, expected)
    assert x_sifted.shape == (2, 1)


def test_sifted_preserves_order(mock_instance_space_subset: InstanceSpace) -> None:
    """Test that SIFTED preserves feature order from selvars."""
    x_prelim = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]])

    x_sifted = InstanceSpace._explore_sifted(mock_instance_space_subset, x_prelim)

    # Features should be in order: 0, 2, 4
    expected = np.array([[1.0, 3.0, 5.0]])
    np.testing.assert_array_equal(x_sifted, expected)


def test_sifted_handles_nan(mock_instance_space_subset: InstanceSpace) -> None:
    """Test SIFTED preserves NaN values."""
    x_prelim = np.array([
        [1.0, np.nan, 3.0, 4.0, np.nan],
        [6.0, 7.0, np.nan, 9.0, 10.0]
    ])

    x_sifted = InstanceSpace._explore_sifted(mock_instance_space_subset, x_prelim)

    # Should select features 0, 2, 4
    # Feature 0: [1.0, 6.0], Feature 2: [3.0, nan], Feature 4: [nan, 10.0]
    assert x_sifted.shape == (2, 3)
    assert x_sifted[0, 0] == 1.0
    assert x_sifted[0, 1] == 3.0
    assert np.isnan(x_sifted[0, 2])
    assert np.isnan(x_sifted[1, 1])


def test_sifted_different_instance_counts(mock_instance_space_subset: InstanceSpace) -> None:
    """Test SIFTED with various numbers of instances."""
    for n_instances in [1, 10, 100]:
        x_prelim = np.random.default_rng().random((n_instances, 5))

        x_sifted = InstanceSpace._explore_sifted(mock_instance_space_subset, x_prelim)

        # Should maintain instance count, reduce features to 3
        assert x_sifted.shape == (n_instances, 3)


def test_sifted_deterministic(mock_instance_space_subset: InstanceSpace) -> None:
    """Test that SIFTED is deterministic."""
    x_prelim = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]])

    result1 = InstanceSpace._explore_sifted(mock_instance_space_subset, x_prelim)
    result2 = InstanceSpace._explore_sifted(mock_instance_space_subset, x_prelim)

    np.testing.assert_array_equal(result1, result2)


def test_sifted_maintains_dtype(mock_instance_space_subset: InstanceSpace) -> None:
    """Test that SIFTED maintains float64 dtype."""
    x_prelim = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]], dtype=np.float64)

    x_sifted = InstanceSpace._explore_sifted(mock_instance_space_subset, x_prelim)

    assert x_sifted.dtype == np.float64


def test_sifted_reverse_order() -> None:
    """Test SIFTED with features in reverse order."""
    # Select features in reverse order: 4, 3, 2, 1, 0
    indices = np.array([4, 3, 2, 1, 0], dtype=np.intc)
    params = SiftedOut(
        selvars=indices,
        rho=None,
        pval=None,
        silhouette_scores=None,
        clust=None
    )

    mock_is = Mock(spec=InstanceSpace)
    mock_is._model = Mock()
    mock_is._model.sifted = params
    mock_is._require_model = Mock(return_value=mock_is._model)

    x_prelim = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]])

    x_sifted = InstanceSpace._explore_sifted(mock_is, x_prelim)

    # Should be in reverse order
    expected = np.array([[5.0, 4.0, 3.0, 2.0, 1.0]])
    np.testing.assert_array_equal(x_sifted, expected)


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
