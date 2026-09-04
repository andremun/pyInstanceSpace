# SPDX-License-Identifier: LicenseRef-PolyForm-Noncommercial-1.0.0
# Copyright (c) 2024-2026 Mario Andrés Muñoz
"""Local two- and three-dimensional alpha-shape geometry primitives."""

from __future__ import annotations

import itertools
from collections.abc import Iterable
from dataclasses import dataclass
from fractions import Fraction
from typing import cast

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import Delaunay, QhullError
from shapely.geometry import MultiLineString, MultiPoint, MultiPolygon, Polygon
from shapely.ops import polygonize, unary_union

type Polygon2D = Polygon | MultiPolygon
TWO_DIMENSIONS = 2
THREE_DIMENSIONS = 3
TRIANGLE_POINT_COUNT = 3
TETRAHEDRON_POINT_COUNT = 4
LEGACY_CONVEX_HULL_LIMIT = 4
# Barycentric coordinates are dimensionless. Values within this ambiguity band
# are resolved with exact predicates; the band is not an inclusion tolerance.
BARYCENTRIC_TOLERANCE = 1e-10
MAX_BARYCENTRIC_EVALUATIONS = 262_144
ORIENTATION_ROUNDOFF_FACTOR = 64.0

type ExactPoint3D = tuple[Fraction, Fraction, Fraction]


def _triangle_circumradii(
    points: NDArray[np.double],
    simplices: NDArray[np.int_],
) -> NDArray[np.double]:
    """Return the circumradius of each two-dimensional Delaunay triangle."""
    triangles = points[simplices]
    ab = triangles[:, 1] - triangles[:, 0]
    ac = triangles[:, 2] - triangles[:, 0]
    bc = triangles[:, 2] - triangles[:, 1]
    twice_area = np.abs(ab[:, 0] * ac[:, 1] - ab[:, 1] * ac[:, 0])
    side_product = (
        np.linalg.norm(ab, axis=1)
        * np.linalg.norm(ac, axis=1)
        * np.linalg.norm(bc, axis=1)
    )
    radii = np.full(simplices.shape[0], np.inf, dtype=np.double)
    nondegenerate = twice_area > 0
    radii[nondegenerate] = side_product[nondegenerate] / (
        2.0 * twice_area[nondegenerate]
    )
    return radii


def _polygonal_geometry(geometry: object) -> Polygon2D | None:
    """Return only polygonal parts from a Shapely operation."""
    if isinstance(geometry, Polygon | MultiPolygon):
        return None if geometry.is_empty else cast(Polygon2D, geometry)

    parts = cast(Iterable[object], getattr(geometry, "geoms", ()))
    polygons = [
        part for part in parts if isinstance(part, Polygon) and not part.is_empty
    ]
    if not polygons:
        return None
    merged = unary_union(polygons)
    if isinstance(merged, Polygon | MultiPolygon) and not merged.is_empty:
        return cast(Polygon2D, merged)
    return None


def _alpha_region_roots(
    simplices: NDArray[np.int_],
) -> NDArray[np.int_]:
    """Label vertex-connected simplex regions with union-find roots."""
    n_simplices = simplices.shape[0]
    parents = np.arange(n_simplices, dtype=np.int_)

    def find(index: int) -> int:
        while int(parents[index]) != index:
            parents[index] = parents[int(parents[index])]
            index = int(parents[index])
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parents[right_root] = left_root

    vertex_owner: dict[int, int] = {}
    for simplex_index, simplex in enumerate(simplices):
        for vertex_value in simplex:
            vertex = int(vertex_value)
            owner = vertex_owner.setdefault(vertex, simplex_index)
            union(simplex_index, owner)

    return np.asarray(
        [find(simplex_index) for simplex_index in range(n_simplices)],
        dtype=np.int_,
    )


def _alpha_region_mask(
    simplices: NDArray[np.int_],
    simplex_measures: NDArray[np.double],
    region_threshold: float,
) -> NDArray[np.bool_]:
    """Retain vertex-connected regions whose summed measure clears the threshold.

    MATLAB alpha shapes treat simplices that share even one vertex as belonging to
    the same region. Polygon libraries represent point-touching interiors as separate
    polygon parts, so filtering an already-unioned ``MultiPolygon`` would discard
    regions that MATLAB retains. Grouping selected Delaunay simplices first preserves
    the same semantics for both triangles and tetrahedra.
    """
    roots = _alpha_region_roots(simplices)

    region_measures: dict[int, float] = {}
    for root_value, simplex_measure in zip(roots, simplex_measures, strict=True):
        root = int(root_value)
        region_measures[root] = region_measures.get(root, 0.0) + float(
            simplex_measure,
        )

    return np.asarray(
        [region_measures[int(root)] > region_threshold for root in roots],
        dtype=np.bool_,
    )


def _immutable_array[T: np.generic](
    values: NDArray[T],
    *,
    dtype: np.dtype[T],
) -> NDArray[T]:
    """Copy an array and make the retained mesh state truly immutable."""
    result = np.array(values, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


def _tetrahedron_circumradii(
    points: NDArray[np.double],
    simplices: NDArray[np.int_],
) -> NDArray[np.double]:
    """Return circumsphere radii for three-dimensional Delaunay tetrahedra."""
    tetrahedra = points[simplices]
    radii = np.full(simplices.shape[0], np.inf, dtype=np.double)
    for index, tetrahedron in enumerate(tetrahedra):
        offsets = tetrahedron[1:] - tetrahedron[0]
        matrix = 2.0 * offsets
        right_hand_side = np.einsum("ij,ij->i", offsets, offsets)
        try:
            centre_offset = np.linalg.solve(matrix, right_hand_side)
        except np.linalg.LinAlgError:
            continue
        radius = float(np.linalg.norm(centre_offset))
        if np.isfinite(radius):
            radii[index] = radius
    return radii


def _tetrahedron_volumes(
    points: NDArray[np.double],
    simplices: NDArray[np.int_],
) -> NDArray[np.double]:
    """Return unsigned tetrahedron volumes."""
    tetrahedra = points[simplices]
    edge_matrices = tetrahedra[:, 1:] - tetrahedra[:, :1]
    return np.asarray(
        np.abs(np.linalg.det(edge_matrices)) / 6.0,
        dtype=np.double,
    )


def _exact_point(point: NDArray[np.double]) -> ExactPoint3D:
    """Convert one stored IEEE-754 point to its exact rational value."""
    return (
        Fraction.from_float(float(point[0])),
        Fraction.from_float(float(point[1])),
        Fraction.from_float(float(point[2])),
    )


def _exact_orientation(
    first: ExactPoint3D,
    second: ExactPoint3D,
    third: ExactPoint3D,
    fourth: ExactPoint3D,
) -> Fraction:
    """Return the exact signed volume determinant of four stored points."""
    ab_x = second[0] - first[0]
    ab_y = second[1] - first[1]
    ab_z = second[2] - first[2]
    ac_x = third[0] - first[0]
    ac_y = third[1] - first[1]
    ac_z = third[2] - first[2]
    ad_x = fourth[0] - first[0]
    ad_y = fourth[1] - first[1]
    ad_z = fourth[2] - first[2]
    return (
        ab_x * (ac_y * ad_z - ac_z * ad_y)
        - ab_y * (ac_x * ad_z - ac_z * ad_x)
        + ab_z * (ac_x * ad_y - ac_y * ad_x)
    )


def _exact_tetrahedron_covers(
    point: NDArray[np.double],
    tetrahedron: NDArray[np.double],
) -> bool:
    """Test inclusive tetrahedron membership using exact float predicates."""
    if not np.all(np.isfinite(point)) or not np.all(np.isfinite(tetrahedron)):
        return False
    query = _exact_point(point)
    first, second, third, fourth = (_exact_point(vertex) for vertex in tetrahedron)
    denominator = _exact_orientation(first, second, third, fourth)
    if denominator == 0:
        return False

    numerators = (
        _exact_orientation(query, second, third, fourth),
        _exact_orientation(first, query, third, fourth),
        _exact_orientation(first, second, query, fourth),
        _exact_orientation(first, second, third, query),
    )
    if denominator > 0:
        return all(numerator >= 0 for numerator in numerators)
    return all(numerator <= 0 for numerator in numerators)


def _outward_boundary_faces(
    points: NDArray[np.double],
    tetrahedra: NDArray[np.int_],
) -> NDArray[np.int_]:
    """Return exposed triangular facets, consistently oriented outwards."""
    exposed: dict[tuple[int, int, int], tuple[int, int, int]] = {}
    for tetrahedron in tetrahedra:
        for opposite_index in range(TETRAHEDRON_POINT_COUNT):
            face_values = np.delete(tetrahedron, opposite_index)
            face = (
                int(face_values[0]),
                int(face_values[1]),
                int(face_values[2]),
            )
            key_values = sorted(face)
            key = (key_values[0], key_values[1], key_values[2])
            if key in exposed:
                del exposed[key]
                continue

            first, second, third = points[np.asarray(face, dtype=np.int_)]
            opposite = points[int(tetrahedron[opposite_index])]
            normal = np.cross(second - first, third - first)
            if float(np.dot(normal, opposite - first)) > 0:
                face = (face[0], face[2], face[1])
            exposed[key] = face

    if not exposed:
        return np.empty((0, TRIANGLE_POINT_COUNT), dtype=np.int_)
    return np.asarray(list(exposed.values()), dtype=np.int_)


@dataclass(frozen=True, eq=False)
class TetrahedralMesh:
    """Immutable retained three-dimensional alpha-complex geometry."""

    vertices: NDArray[np.double]
    tetrahedra: NDArray[np.int_]
    boundary_faces: NDArray[np.int_]
    alpha: float
    region_threshold: float
    region_count: int
    volume: float
    surface_area: float

    def __post_init__(self) -> None:
        """Validate and freeze all retained array state."""
        vertices = np.asarray(self.vertices, dtype=np.double)
        tetrahedra = np.asarray(self.tetrahedra, dtype=np.int_)
        boundary_faces = np.asarray(self.boundary_faces, dtype=np.int_)
        if vertices.ndim != TWO_DIMENSIONS or vertices.shape[1] != THREE_DIMENSIONS:
            msg = "A tetrahedral mesh requires an (n, 3) vertex matrix."
            raise ValueError(msg)
        if (
            tetrahedra.ndim != TWO_DIMENSIONS
            or tetrahedra.shape[1] != TETRAHEDRON_POINT_COUNT
        ):
            msg = "A tetrahedral mesh requires an (m, 4) tetrahedron matrix."
            raise ValueError(msg)
        if (
            boundary_faces.ndim != TWO_DIMENSIONS
            or boundary_faces.shape[1] != TRIANGLE_POINT_COUNT
        ):
            msg = "A tetrahedral mesh requires a (k, 3) boundary-face matrix."
            raise ValueError(msg)
        if tetrahedra.size and (
            np.min(tetrahedra) < 0 or np.max(tetrahedra) >= vertices.shape[0]
        ):
            msg = "Tetrahedron indices must reference retained mesh vertices."
            raise ValueError(msg)
        if boundary_faces.size and (
            np.min(boundary_faces) < 0 or np.max(boundary_faces) >= vertices.shape[0]
        ):
            msg = "Boundary-face indices must reference retained mesh vertices."
            raise ValueError(msg)
        if not np.all(np.isfinite(vertices)):
            msg = "Tetrahedral mesh vertices must contain only finite values."
            raise ValueError(msg)
        exact_vertices = tuple(_exact_point(vertex) for vertex in vertices)
        if any(
            _exact_orientation(
                exact_vertices[int(tetrahedron[0])],
                exact_vertices[int(tetrahedron[1])],
                exact_vertices[int(tetrahedron[2])],
                exact_vertices[int(tetrahedron[3])],
            )
            == 0
            for tetrahedron in tetrahedra
        ):
            msg = "Tetrahedra must have nonzero exact signed volume."
            raise ValueError(msg)

        object.__setattr__(
            self,
            "vertices",
            _immutable_array(vertices, dtype=np.dtype(np.double)),
        )
        object.__setattr__(
            self,
            "tetrahedra",
            _immutable_array(tetrahedra, dtype=np.dtype(np.int_)),
        )
        object.__setattr__(
            self,
            "boundary_faces",
            _immutable_array(boundary_faces, dtype=np.dtype(np.int_)),
        )

    def __setstate__(self, state: dict[str, object]) -> None:
        """Restore frozen arrays after pickle/joblib bypasses ``__post_init__``."""
        for name, value in state.items():
            object.__setattr__(self, name, value)
        self.__post_init__()

    @property
    def is_empty(self) -> bool:
        """Match the geometry interface used by two-dimensional footprints."""
        return self.tetrahedra.shape[0] == 0

    @property
    def measure(self) -> float:
        """Return the dimension-neutral footprint measure."""
        return self.volume

    @property
    def dimension(self) -> int:
        """Return the coordinate count represented by the mesh."""
        return THREE_DIMENSIONS

    def covers(
        self,
        points: NDArray[np.double],
        *,
        tolerance: float = BARYCENTRIC_TOLERANCE,
    ) -> NDArray[np.bool_]:
        """Return robust inclusive membership using bounded vectorized batches.

        The fast path classifies unambiguous barycentric coordinates. Values near
        a face are resolved from the exact rational values of the stored floats,
        so true faces remain included without admitting an outside tolerance shell.
        ``tolerance`` controls only the minimum width of that exact-fallback band.
        """
        query = np.asarray(points, dtype=np.double)
        if query.ndim != TWO_DIMENSIONS or query.shape[1] != THREE_DIMENSIONS:
            msg = "Three-dimensional mesh membership requires an (n, 3) matrix."
            raise ValueError(msg)
        if not np.isfinite(tolerance) or tolerance < 0:
            msg = "The barycentric ambiguity tolerance must be finite and nonnegative."
            raise ValueError(msg)
        covered = np.zeros(query.shape[0], dtype=np.bool_)
        if self.is_empty or query.shape[0] == 0:
            return covered
        finite_query = np.all(np.isfinite(query), axis=1)

        tetrahedron_count = self.tetrahedra.shape[0]
        tetrahedron_batch_size = min(tetrahedron_count, 1024)
        for tetrahedron_start in range(0, tetrahedron_count, tetrahedron_batch_size):
            tetrahedron_stop = min(
                tetrahedron_start + tetrahedron_batch_size,
                tetrahedron_count,
            )
            indices = self.tetrahedra[tetrahedron_start:tetrahedron_stop]
            tetrahedron_vertices = self.vertices[indices]
            origins = tetrahedron_vertices[:, 0]
            edge_matrices = np.transpose(
                tetrahedron_vertices[:, 1:] - tetrahedron_vertices[:, :1],
                (0, 2, 1),
            )
            try:
                inverse_matrices = np.linalg.inv(edge_matrices)
            except np.linalg.LinAlgError:
                inverse_matrices = np.asarray(
                    [np.linalg.pinv(matrix) for matrix in edge_matrices],
                    dtype=np.double,
                )
            matrix_norms = np.linalg.norm(
                edge_matrices,
                ord=np.inf,
                axis=(1, 2),
            )
            inverse_norms = np.linalg.norm(
                inverse_matrices,
                ord=np.inf,
                axis=(1, 2),
            )
            condition_estimates = matrix_norms * inverse_norms

            active = np.flatnonzero(np.logical_and(~covered, finite_query))
            if active.size == 0:
                break
            point_batch_size = max(
                1,
                MAX_BARYCENTRIC_EVALUATIONS // indices.shape[0],
            )
            for point_start in range(0, active.size, point_batch_size):
                point_indices = active[point_start : point_start + point_batch_size]
                offsets = query[point_indices, None, :] - origins[None, :, :]
                coordinates = np.einsum(
                    "tij,btj->bti",
                    inverse_matrices,
                    offsets,
                )
                first_coordinate = 1.0 - np.sum(coordinates, axis=2)
                minimum_coordinate = np.minimum(
                    np.min(coordinates, axis=2),
                    first_coordinate,
                )
                coordinate_scale = np.maximum(
                    1.0,
                    np.maximum(
                        np.max(np.abs(coordinates), axis=2),
                        np.abs(first_coordinate),
                    ),
                )
                roundoff_band = (
                    ORIENTATION_ROUNDOFF_FACTOR
                    * np.finfo(np.double).eps
                    * condition_estimates[None, :]
                    * coordinate_scale
                )
                ambiguity_band = np.maximum(tolerance, roundoff_band)
                clear_inside = minimum_coordinate > ambiguity_band
                covered_in_batch = np.any(clear_inside, axis=1)
                covered[point_indices[covered_in_batch]] = True

                ambiguous = np.logical_or(
                    ~np.isfinite(minimum_coordinate),
                    np.logical_and(
                        minimum_coordinate >= -ambiguity_band,
                        minimum_coordinate <= ambiguity_band,
                    ),
                )
                ambiguous[covered_in_batch] = False
                for point_offset, tetrahedron_offset in np.argwhere(ambiguous):
                    point_index = int(point_indices[point_offset])
                    if covered[point_index]:
                        continue
                    if _exact_tetrahedron_covers(
                        query[point_index],
                        tetrahedron_vertices[tetrahedron_offset],
                    ):
                        covered[point_index] = True
        return covered

    def __eq__(self, other: object) -> bool:
        """Compare retained mesh state without ambiguous NumPy truth values."""
        return bool(
            isinstance(other, TetrahedralMesh)
            and np.array_equal(self.vertices, other.vertices)
            and np.array_equal(self.tetrahedra, other.tetrahedra)
            and np.array_equal(self.boundary_faces, other.boundary_faces)
            and self.alpha == other.alpha
            and self.region_threshold == other.region_threshold
            and self.region_count == other.region_count
            and self.volume == other.volume
            and self.surface_area == other.surface_area,
        )


@dataclass(frozen=True)
class AlphaShape2D:
    """Delaunay data needed to evaluate a point cloud at multiple alpha radii."""

    points: NDArray[np.double]
    simplices: NDArray[np.int_]
    circumradii: NDArray[np.double]
    spectrum: NDArray[np.double]
    critical_radius: float

    @classmethod
    def from_points(
        cls: type[AlphaShape2D],
        points: NDArray[np.double],
    ) -> AlphaShape2D | None:
        """Triangulate unique finite 2D points and find all-points critical alpha."""
        unique_points = np.unique(np.asarray(points, dtype=np.double), axis=0)
        if (
            unique_points.ndim != TWO_DIMENSIONS
            or unique_points.shape[1] != TWO_DIMENSIONS
            or unique_points.shape[0] < TRIANGLE_POINT_COUNT
            or not np.all(np.isfinite(unique_points))
        ):
            return None

        try:
            triangulation = Delaunay(unique_points)
        except QhullError:
            return None

        simplices = np.asarray(triangulation.simplices, dtype=np.int_)
        if simplices.size == 0:
            return None
        radii = _triangle_circumradii(unique_points, simplices)
        finite_radii = radii[np.isfinite(radii)]
        if finite_radii.size == 0:
            return None

        incident_radius = np.full(unique_points.shape[0], np.inf, dtype=np.double)
        for column in range(simplices.shape[1]):
            np.minimum.at(incident_radius, simplices[:, column], radii)
        if not np.all(np.isfinite(incident_radius)):
            return None

        spectrum = np.unique(finite_radii)[::-1].astype(np.double, copy=False)
        return cls(
            points=unique_points,
            simplices=simplices,
            circumradii=radii,
            spectrum=spectrum,
            critical_radius=float(np.max(incident_radius)),
        )

    def geometry(
        self,
        radius: float,
        *,
        region_threshold: float = 0.0,
        inclusive: bool = True,
    ) -> Polygon2D | None:
        """Build the alpha complex at a radius and filter its connected regions."""
        selected = (
            self.circumradii <= radius if inclusive else self.circumradii < radius
        )
        if not np.any(selected):
            return None

        selected_simplices = self.simplices[selected]
        triangles = [
            Polygon(self.points[simplex]) for simplex in self.simplices[selected]
        ]
        if region_threshold > 0:
            triangle_areas = np.asarray(
                [triangle.area for triangle in triangles],
                dtype=np.double,
            )
            retained = _alpha_region_mask(
                selected_simplices,
                triangle_areas,
                region_threshold,
            )
            triangles = [
                triangle
                for triangle, keep in zip(triangles, retained, strict=True)
                if keep
            ]
        if not triangles:
            return None
        return _polygonal_geometry(unary_union(triangles))


@dataclass(frozen=True)
class AlphaShape3D:
    """Delaunay data needed to evaluate a 3D point cloud at many alpha radii."""

    points: NDArray[np.double]
    simplices: NDArray[np.int_]
    circumradii: NDArray[np.double]
    simplex_volumes: NDArray[np.double]
    spectrum: NDArray[np.double]
    critical_radius: float

    @classmethod
    def from_points(
        cls: type[AlphaShape3D],
        points: NDArray[np.double],
    ) -> AlphaShape3D | None:
        """Triangulate finite 3D points and find MATLAB's all-points alpha."""
        unique_points = np.unique(np.asarray(points, dtype=np.double), axis=0)
        if (
            unique_points.ndim != TWO_DIMENSIONS
            or unique_points.shape[1] != THREE_DIMENSIONS
            or unique_points.shape[0] < TETRAHEDRON_POINT_COUNT
            or not np.all(np.isfinite(unique_points))
        ):
            return None

        try:
            triangulation = Delaunay(unique_points)
        except QhullError:
            return None

        simplices = np.asarray(triangulation.simplices, dtype=np.int_)
        if simplices.size == 0:
            return None
        radii = _tetrahedron_circumradii(unique_points, simplices)
        volumes = _tetrahedron_volumes(unique_points, simplices)
        finite = np.logical_and(np.isfinite(radii), volumes > 0)
        finite_radii = radii[finite]
        if finite_radii.size == 0:
            return None

        incident_radius = np.full(unique_points.shape[0], np.inf, dtype=np.double)
        for column in range(simplices.shape[1]):
            np.minimum.at(incident_radius, simplices[:, column], radii)
        if not np.all(np.isfinite(incident_radius)):
            return None

        spectrum = np.unique(finite_radii)[::-1].astype(np.double, copy=False)
        return cls(
            points=unique_points,
            simplices=simplices,
            circumradii=radii,
            simplex_volumes=volumes,
            spectrum=spectrum,
            critical_radius=float(np.max(incident_radius)),
        )

    def geometry(
        self,
        radius: float,
        *,
        region_threshold: float = 0.0,
        inclusive: bool = True,
    ) -> TetrahedralMesh | None:
        """Build a retained tetrahedral mesh at the requested alpha radius."""
        selected = np.logical_and(
            self.circumradii <= radius if inclusive else self.circumradii < radius,
            self.simplex_volumes > 0,
        )
        if not np.any(selected):
            return None

        selected_simplices = self.simplices[selected]
        selected_volumes = self.simplex_volumes[selected]
        retained = _alpha_region_mask(
            selected_simplices,
            selected_volumes,
            region_threshold,
        )
        retained_simplices = selected_simplices[retained]
        retained_volumes = selected_volumes[retained]
        if retained_simplices.shape[0] == 0:
            return None

        region_count = int(np.unique(_alpha_region_roots(retained_simplices)).size)

        boundary_faces = _outward_boundary_faces(self.points, retained_simplices)
        boundary_triangles = self.points[boundary_faces]
        cross_products = np.cross(
            boundary_triangles[:, 1] - boundary_triangles[:, 0],
            boundary_triangles[:, 2] - boundary_triangles[:, 0],
        )
        surface_area = float(np.sum(np.linalg.norm(cross_products, axis=1)) / 2.0)
        return TetrahedralMesh(
            vertices=self.points,
            tetrahedra=retained_simplices,
            boundary_faces=boundary_faces,
            alpha=float(radius),
            region_threshold=float(region_threshold),
            region_count=region_count,
            volume=float(np.sum(retained_volumes)),
            surface_area=surface_area,
        )


def legacy_alpha_shape(
    points: NDArray[np.double],
    alpha: float,
) -> Polygon2D:
    """Match the third-party inverse-alpha convention used by legacy TRACE."""
    unique_points = np.unique(np.asarray(points, dtype=np.double), axis=0)
    if unique_points.shape[0] < LEGACY_CONVEX_HULL_LIMIT or alpha <= 0:
        hull = MultiPoint(unique_points).convex_hull
        return (
            cast(Polygon2D, hull)
            if isinstance(hull, Polygon | MultiPolygon)
            else Polygon()
        )

    shape = AlphaShape2D.from_points(unique_points)
    if shape is None:
        return Polygon()

    edges: set[tuple[int, int]] = set()
    perimeter_edges: set[tuple[int, int]] = set()
    for simplex, circumradius in zip(
        shape.simplices,
        shape.circumradii,
        strict=True,
    ):
        if circumradius >= 1.0 / alpha:
            continue
        for edge_values in itertools.combinations(simplex, 2):
            edge = (int(edge_values[0]), int(edge_values[1]))
            if edge not in edges:
                edges.add(edge)
                perimeter_edges.add(edge)
            else:
                perimeter_edges.discard(edge)

    linework = MultiLineString(
        [unique_points[np.asarray(edge, dtype=np.int_)] for edge in perimeter_edges],
    )
    geometry = _polygonal_geometry(unary_union(list(polygonize(linework))))
    return geometry if geometry is not None else Polygon()
