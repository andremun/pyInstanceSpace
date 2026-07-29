"""Tests for SIFTED stage's explore()-time inference (_explore_sifted).

Unit tests verify _explore_sifted() with various edge cases and error conditions,
independent of MATLAB reference data. The validation test loads MATLAB-trained
SIFTED feature indices and verifies that _explore_sifted reproduces MATLAB's
column selection on the test set (threshold: exact match - feature selection is
pure indexing, no numerical operations, so any deviation indicates a port bug).
"""

from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from instancespace.data.model import SiftedOut
from instancespace.instance_space import InstanceSpace

REFERENCE_DIR = Path("tests/matlab_reference")
ARTIFACTS_DIR = REFERENCE_DIR / "training_artifacts"
OUTPUTS_DIR = REFERENCE_DIR / "explore_outputs"


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


def load_sifted_indices() -> SiftedOut:
    df = pd.read_csv(ARTIFACTS_DIR / "sifted" / "sifted_indices.csv")
    selvars = (df["original_index"].to_numpy(dtype=np.intc) - 1).astype(np.intc)
    return SiftedOut(
        selvars=selvars,
        rho=None,
        pval=None,
        silhouette_scores=None,
        clust=None,
    )


def test_sifted_matches_matlab() -> None:
    """SIFTED output exactly matches MATLAB step2."""
    x_input = pd.read_csv(
        OUTPUTS_DIR / "step1_after_prelim.csv", index_col="instance_id",
    ).to_numpy(dtype=np.double)

    instance_space = Mock(spec=InstanceSpace)
    instance_space._model = Mock()
    instance_space._model.sifted = load_sifted_indices()
    instance_space._require_model = Mock(return_value=instance_space._model)

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
