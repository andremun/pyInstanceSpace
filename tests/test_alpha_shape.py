"""Focused invariants for the local two-dimensional alpha-shape engine."""

import warnings

import numpy as np
import pytest
from shapely.geometry import MultiPoint, MultiPolygon

from instancespace.data.model import pointwise_covers
from instancespace.utils.alpha_shape import (
    AlphaShape2D,
    legacy_alpha_shape,
)

EXPECTED_COMPONENTS = 2


def test_default_radius_is_smallest_radius_covering_every_point() -> None:
    """The default all-points radius includes every input point."""
    points = np.array(
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
        dtype=np.double,
    )

    shape = AlphaShape2D.from_points(points)

    assert shape is not None
    assert shape.critical_radius == pytest.approx(np.sqrt(0.5))
    geometry = shape.geometry(shape.critical_radius)
    assert geometry is not None
    assert geometry.area == pytest.approx(1.0)
    assert pointwise_covers(geometry, points).all()


def test_radius_units_and_region_threshold_preserve_components() -> None:
    """Radius and area thresholds retain independent connected regions."""
    points = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [5.0, 0.0],
            [6.0, 0.0],
            [5.0, 1.0],
        ],
        dtype=np.double,
    )
    shape = AlphaShape2D.from_points(points)

    assert shape is not None
    geometry = shape.geometry(0.8)
    assert isinstance(geometry, MultiPolygon)
    assert len(geometry.geoms) == EXPECTED_COMPONENTS
    assert geometry.area == pytest.approx(1.0)
    assert shape.geometry(0.8, region_threshold=0.499999) is not None
    assert shape.geometry(0.8, region_threshold=0.5) is None


def test_region_threshold_groups_triangles_that_touch_at_one_vertex() -> None:
    """MATLAB considers point-connected alpha simplices one threshold region."""
    points = np.array(
        [
            [0.0, 0.0],
            [-1.0, 0.0],
            [-1.0, 0.1],
            [1.0, 0.0],
            [1.0, -0.1],
        ],
        dtype=np.double,
    )
    shape = AlphaShape2D.from_points(points)

    assert shape is not None
    geometry = shape.geometry(0.6, region_threshold=0.075)

    assert isinstance(geometry, MultiPolygon)
    assert len(geometry.geoms) == EXPECTED_COMPONENTS
    assert geometry.area == pytest.approx(0.1)


def test_degenerate_point_cloud_has_no_alpha_shape() -> None:
    """Collinear points produce the canonical no-shape result."""
    collinear = np.array(
        [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [2.0, 0.0]],
        dtype=np.double,
    )

    assert AlphaShape2D.from_points(collinear) is None


def test_legacy_inverse_alpha_path_is_local_and_warning_free() -> None:
    """Legacy inverse-alpha behavior avoids the former matrix warning."""
    points = np.array(
        [[0.0, 0.0], [0.2, 0.0], [0.2, 0.2], [0.0, 0.2]],
        dtype=np.double,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error", PendingDeprecationWarning)
        geometry = legacy_alpha_shape(points, 2.15)

    assert geometry.area == pytest.approx(MultiPoint(points).convex_hull.area)
