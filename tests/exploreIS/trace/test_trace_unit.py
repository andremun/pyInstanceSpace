"""Unit tests for TRACE stage (_explore_trace)."""

from types import SimpleNamespace
from typing import cast
from unittest.mock import Mock

import numpy as np
from shapely.geometry import Polygon

from instancespace.data.model import Footprint, TraceOut
from instancespace.instance_space import InstanceSpace


def make_footprint(polygon: Polygon | None) -> Footprint:
    return cast(Footprint, SimpleNamespace(polygon=polygon))


def make_instance_space(
    good_polys: list[Polygon | None],
    best_polys: list[Polygon | None],
) -> InstanceSpace:
    trace = Mock(spec=TraceOut)
    trace.good = [make_footprint(p) for p in good_polys]
    trace.best = [make_footprint(p) for p in best_polys]
    model = Mock()
    model.trace = trace
    instance_space = Mock(spec=InstanceSpace)
    instance_space._model = model
    instance_space._require_model = Mock(return_value=model)
    return instance_space


def test_trace_output_shapes() -> None:
    square = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    space = make_instance_space([square, square, square], [square, square, square])
    z = np.random.rand(5, 2)
    in_good, in_best = InstanceSpace._explore_trace(space, z)
    assert in_good.shape == (5, 3)
    assert in_best.shape == (5, 3)
    assert in_good.dtype == np.bool_
    assert in_best.dtype == np.bool_


def test_trace_inside_outside() -> None:
    square = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    space = make_instance_space([square], [square])
    z = np.array([[0.5, 0.5], [2.0, 2.0], [0.0, 0.0]])
    in_good, in_best = InstanceSpace._explore_trace(space, z)
    # (0.5, 0.5) inside; (2, 2) outside; (0, 0) on boundary -> inside (covers)
    assert in_good[0, 0] and not in_good[1, 0] and in_good[2, 0]
    assert in_best[0, 0] and not in_best[1, 0] and in_best[2, 0]


def test_trace_none_polygon_returns_false() -> None:
    space = make_instance_space([None, None], [None, None])
    z = np.array([[0.5, 0.5], [1.0, 1.0]])
    in_good, in_best = InstanceSpace._explore_trace(space, z)
    assert not in_good.any()
    assert not in_best.any()


def test_trace_per_algo_independent() -> None:
    # Two algos with disjoint polygons: instance inside one is outside the other.
    left = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    right = Polygon([(2, 0), (3, 0), (3, 1), (2, 1)])
    space = make_instance_space([left, right], [left, right])
    z = np.array([[0.5, 0.5], [2.5, 0.5]])
    in_good, in_best = InstanceSpace._explore_trace(space, z)
    assert in_good[0, 0] and not in_good[0, 1]
    assert not in_good[1, 0] and in_good[1, 1]
    assert in_best[0, 0] and not in_best[0, 1]
    assert not in_best[1, 0] and in_best[1, 1]


def test_trace_good_and_best_independent() -> None:
    # good polygon contains point, best polygon does not.
    good = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
    best = Polygon([(5, 5), (6, 5), (6, 6), (5, 6)])
    space = make_instance_space([good], [best])
    z = np.array([[1.0, 1.0]])
    in_good, in_best = InstanceSpace._explore_trace(space, z)
    assert in_good[0, 0]
    assert not in_best[0, 0]
