"""Focused invariants for the local two-dimensional alpha-shape engine."""

import warnings

import numpy as np
import pytest
from shapely.geometry import MultiPoint, MultiPolygon

from instancespace.data.model import pointwise_covers
from instancespace.utils.alpha_shape import (
    BARYCENTRIC_TOLERANCE,
    AlphaShape2D,
    AlphaShape3D,
    TetrahedralMesh,
    legacy_alpha_shape,
)

EXPECTED_COMPONENTS = 2
UNIT_TETRAHEDRON_RADIUS = np.sqrt(3.0) / 2.0
UNIT_TETRAHEDRON_VOLUME = 1.0 / 6.0


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


def test_default_radius_and_region_threshold_preserve_components() -> None:
    """Match R2026a's all-points radius without merging valid regions."""
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
    assert shape.critical_radius == pytest.approx(np.sqrt(0.5))
    geometry = shape.geometry(shape.critical_radius)
    assert isinstance(geometry, MultiPolygon)
    assert len(geometry.geoms) == EXPECTED_COMPONENTS
    assert geometry.area == pytest.approx(1.0)
    assert pointwise_covers(geometry, points).all()
    assert shape.geometry(shape.critical_radius, region_threshold=0.499999) is not None
    assert shape.geometry(shape.critical_radius, region_threshold=0.5) is None


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


def _unit_tetrahedron() -> np.ndarray:  # type: ignore[type-arg]
    return np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.double,
    )


def test_3d_unit_tetrahedron_measure_boundary_and_membership() -> None:
    """Match R2026a's audited unit-tetrahedron alpha-shape contract."""
    points = _unit_tetrahedron()

    shape = AlphaShape3D.from_points(points)

    assert shape is not None
    assert shape.critical_radius == pytest.approx(UNIT_TETRAHEDRON_RADIUS)
    np.testing.assert_allclose(shape.spectrum, [UNIT_TETRAHEDRON_RADIUS])
    mesh = shape.geometry(shape.critical_radius)
    assert isinstance(mesh, TetrahedralMesh)
    assert mesh.volume == pytest.approx(UNIT_TETRAHEDRON_VOLUME)
    assert mesh.surface_area == pytest.approx(1.5 + np.sqrt(3.0) / 2.0)
    assert mesh.region_count == 1
    assert mesh.tetrahedra.shape == (1, 4)
    assert mesh.boundary_faces.shape == (4, 3)

    probes = np.array(
        [
            [0.1, 0.1, 0.1],
            [0.5, 0.0, 0.0],
            [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
            [0.0, 0.0, 0.0],
            [0.34, 0.34, 0.34],
        ],
        dtype=np.double,
    )
    np.testing.assert_array_equal(
        mesh.covers(probes),
        [True, True, True, True, False],
    )

    centroid = np.mean(points, axis=0)
    for face in mesh.boundary_faces:
        first, second, third = mesh.vertices[face]
        outward = np.cross(second - first, third - first)
        assert np.dot(outward, centroid - first) < 0


def test_3d_membership_tolerance_distinguishes_boundary_roundoff() -> None:
    """The 1e-12 barycentric shell is inclusive but remains tightly bounded."""
    shape = AlphaShape3D.from_points(_unit_tetrahedron())
    assert shape is not None
    mesh = shape.geometry(shape.critical_radius)
    assert mesh is not None
    coordinate = 1.0 / 3.0
    probes = np.array(
        [
            [coordinate, coordinate, coordinate - 4 * BARYCENTRIC_TOLERANCE],
            [coordinate, coordinate, coordinate],
            [coordinate, coordinate, coordinate + BARYCENTRIC_TOLERANCE / 4],
            [coordinate, coordinate, coordinate + 4 * BARYCENTRIC_TOLERANCE],
        ],
        dtype=np.double,
    )

    np.testing.assert_array_equal(
        mesh.covers(probes),
        [True, True, True, False],
    )


def test_3d_cube_has_six_tetrahedra_and_twelve_boundary_faces() -> None:
    """SciPy's retained cube topology matches the audited MATLAB topology."""
    points = np.array(
        [[x, y, z] for x in (0.0, 1.0) for y in (0.0, 1.0) for z in (0.0, 1.0)],
        dtype=np.double,
    )

    shape = AlphaShape3D.from_points(points)

    assert shape is not None
    mesh = shape.geometry(shape.critical_radius)
    assert mesh is not None
    assert mesh.tetrahedra.shape == (6, 4)
    assert mesh.boundary_faces.shape == (12, 3)
    assert mesh.volume == pytest.approx(1.0)
    assert mesh.surface_area == pytest.approx(6.0)
    assert mesh.covers(points).all()


def test_3d_region_threshold_is_strict_and_preserves_two_regions() -> None:
    """Equal disconnected tetrahedra survive only below their exact volume."""
    first = _unit_tetrahedron()
    second = first + np.array([5.0, 0.0, 0.0], dtype=np.double)
    points = np.vstack((first, second))
    shape = AlphaShape3D.from_points(points)

    assert shape is not None
    retained = shape.geometry(
        UNIT_TETRAHEDRON_RADIUS,
        region_threshold=UNIT_TETRAHEDRON_VOLUME - 1e-12,
    )
    rejected = shape.geometry(
        UNIT_TETRAHEDRON_RADIUS,
        region_threshold=UNIT_TETRAHEDRON_VOLUME,
    )

    assert retained is not None
    assert retained.region_count == EXPECTED_COMPONENTS
    assert retained.volume == pytest.approx(2.0 * UNIT_TETRAHEDRON_VOLUME)
    assert rejected is None


def test_3d_regions_share_connectivity_at_one_vertex() -> None:
    """MATLAB groups tetrahedra that share only a vertex before filtering."""
    first = _unit_tetrahedron()
    second = np.array(
        [[0.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]],
        dtype=np.double,
    )
    points = np.unique(np.vstack((first, second)), axis=0)
    origin = int(np.flatnonzero(np.all(points == 0.0, axis=1))[0])
    positive = np.flatnonzero(np.any(points > 0.0, axis=1))
    negative = np.flatnonzero(np.any(points < 0.0, axis=1))
    simplices = np.array(
        [[origin, *positive.tolist()], [origin, *negative.tolist()]],
        dtype=np.int_,
    )
    shape = AlphaShape3D(
        points=points,
        simplices=simplices,
        circumradii=np.full(2, UNIT_TETRAHEDRON_RADIUS, dtype=np.double),
        simplex_volumes=np.full(2, UNIT_TETRAHEDRON_VOLUME, dtype=np.double),
        spectrum=np.array([UNIT_TETRAHEDRON_RADIUS], dtype=np.double),
        critical_radius=UNIT_TETRAHEDRON_RADIUS,
    )

    mesh = shape.geometry(UNIT_TETRAHEDRON_RADIUS, region_threshold=0.2)

    assert mesh is not None
    assert mesh.region_count == 1
    assert mesh.volume == pytest.approx(2.0 * UNIT_TETRAHEDRON_VOLUME)


def test_3d_radius_is_inclusive_and_retained_mesh_is_immutable() -> None:
    """Radius equality retains tetrahedra and callers cannot mutate mesh arrays."""
    shape = AlphaShape3D.from_points(_unit_tetrahedron())
    assert shape is not None

    inclusive = shape.geometry(shape.critical_radius)
    exclusive = shape.geometry(shape.critical_radius, inclusive=False)

    assert inclusive is not None
    assert exclusive is None
    assert not inclusive.vertices.flags.writeable
    assert not inclusive.tetrahedra.flags.writeable
    assert not inclusive.boundary_faces.flags.writeable
    with pytest.raises(ValueError, match="read-only"):
        inclusive.vertices[0, 0] = 1.0


def test_degenerate_3d_point_cloud_has_no_alpha_shape() -> None:
    """Coplanar support produces TRACE3's canonical no-shape result."""
    coplanar = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]],
        dtype=np.double,
    )

    assert AlphaShape3D.from_points(coplanar) is None
