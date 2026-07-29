"""Unit tests for PILOT stage (_explore_pilot)."""

from unittest.mock import Mock

import numpy as np
import pytest
from numpy.typing import NDArray

from instancespace.data.model import PilotOut
from instancespace.instance_space import InstanceSpace


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
    x = np.random.rand(10, 5)
    result = InstanceSpace._explore_pilot(make_instance_space(a), x)
    assert result.shape == (10, 2)


def test_pilot_correct_projection() -> None:
    # Z = X @ A.T — verify with known values
    a = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])  # (2, 3)
    x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    result = InstanceSpace._explore_pilot(make_instance_space(a), x)
    expected = np.array([[1.0, 2.0], [4.0, 5.0]])
    np.testing.assert_array_almost_equal(result, expected)


def test_pilot_single_instance() -> None:
    a = np.random.rand(2, 4)
    x = np.random.rand(1, 4)
    result = InstanceSpace._explore_pilot(make_instance_space(a), x)
    assert result.shape == (1, 2)


def test_pilot_preserves_input() -> None:
    a = np.random.rand(2, 3)
    x = np.random.rand(5, 3)
    x_copy = x.copy()
    InstanceSpace._explore_pilot(make_instance_space(a), x)
    np.testing.assert_array_equal(x, x_copy)


def test_pilot_deterministic() -> None:
    a = np.random.rand(2, 6)
    x = np.random.rand(20, 6)
    r1 = InstanceSpace._explore_pilot(make_instance_space(a), x)
    r2 = InstanceSpace._explore_pilot(make_instance_space(a), x)
    np.testing.assert_array_equal(r1, r2)


def test_pilot_zero_matrix() -> None:
    a = np.zeros((2, 4))
    x = np.random.rand(5, 4)
    result = InstanceSpace._explore_pilot(make_instance_space(a), x)
    np.testing.assert_array_equal(result, np.zeros((5, 2)))
