# SPDX-License-Identifier: LicenseRef-PolyForm-Noncommercial-1.0.0
# Copyright (c) 2024-2026 Mario Andrés Muñoz
"""Local two-dimensional alpha-shape geometry primitives."""

from __future__ import annotations

import itertools
from collections.abc import Iterable
from dataclasses import dataclass
from typing import cast

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import Delaunay, QhullError
from shapely.geometry import MultiLineString, MultiPoint, MultiPolygon, Polygon
from shapely.ops import polygonize, unary_union

type Polygon2D = Polygon | MultiPolygon
DIMENSIONS = 2
SIMPLEX_POINT_COUNT = 3
LEGACY_CONVEX_HULL_LIMIT = 4


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


def _alpha_region_mask(
    simplices: NDArray[np.int_],
    triangle_areas: NDArray[np.double],
    region_threshold: float,
) -> NDArray[np.bool_]:
    """Retain alpha-complex regions whose combined area clears the threshold.

    MATLAB alpha shapes treat triangles that share even one vertex as belonging to
    the same region. Polygon libraries represent point-touching interiors as separate
    polygon parts, so filtering an already-unioned ``MultiPolygon`` would discard
    regions that MATLAB retains. Grouping the selected Delaunay simplices first keeps
    those semantics while still returning ordinary Shapely geometry.
    """
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

    region_areas: dict[int, float] = {}
    for simplex_index, triangle_area in enumerate(triangle_areas):
        root = find(simplex_index)
        region_areas[root] = region_areas.get(root, 0.0) + float(triangle_area)

    return np.asarray(
        [
            region_areas[find(simplex_index)] > region_threshold
            for simplex_index in range(n_simplices)
        ],
        dtype=np.bool_,
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
            unique_points.ndim != DIMENSIONS
            or unique_points.shape[1] != DIMENSIONS
            or unique_points.shape[0] < SIMPLEX_POINT_COUNT
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
