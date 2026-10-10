# SPDX-License-Identifier: LicenseRef-PolyForm-Noncommercial-1.0.0
# Copyright (c) 2024-2026 Mario Andrés Muñoz
"""Closed footprint membership with an explicit Euclidean boundary tolerance."""

import numbers

import numpy as np
from numpy.typing import NDArray
from shapely import covers, distance
from shapely import points as make_points
from shapely.geometry import MultiPolygon, Polygon

from instancespace.utils.alpha_shape import TetrahedralMesh


def _near_facets(
    queries: NDArray[np.double],
    facets: NDArray[np.double],
    tolerance: float,
) -> NDArray[np.bool_]:
    """Measure Euclidean distance to segments or triangles, without buffering."""
    near = np.zeros(len(queries), dtype=np.bool_)
    for facet in facets:
        selected = np.flatnonzero(
            ~near
            & np.all(queries >= facet.min(axis=0) - tolerance, axis=1)
            & np.all(queries <= facet.max(axis=0) + tolerance, axis=1),
        )
        if not selected.size:
            continue
        q = queries[selected]
        distances = np.full(len(q), np.inf)
        for first, second in zip(facet, np.roll(facet, -1, axis=0), strict=True):
            edge = second - first
            denominator = float(edge @ edge)
            if denominator == 0:
                candidate = np.linalg.norm(q - first, axis=1)
            else:
                fraction = np.clip((q - first) @ edge / denominator, 0, 1)
                candidate = np.linalg.norm(q - first - fraction[:, None] * edge, axis=1)
            distances = np.minimum(distances, candidate)
        if facet.shape == (3, 3):
            first, second, third = facet
            u, v = second - first, third - first
            normal = np.cross(u, v)
            norm = float(np.linalg.norm(normal))
            if norm > 0:
                normal /= norm
                signed = (q - first) @ normal
                projected = q - first - signed[:, None] * normal
                # Cross products avoid subtracting nearly equal Gram products.
                s = np.cross(projected, v) @ normal / norm
                t = np.cross(u, projected) @ normal / norm
                on_triangle = (s >= 0) & (t >= 0) & (s + t <= 1)
                distances = np.minimum(
                    distances,
                    np.where(on_triangle, abs(signed), np.inf),
                )
        near[selected] = distances <= tolerance
    return near


def footprint_covers(
    geometry: Polygon | MultiPolygon | TetrahedralMesh,
    queries: NDArray[np.double],
    tolerance: float = 0.0,
) -> NDArray[np.bool_]:
    """Include the interior and points within an explicit boundary distance.

    ``tolerance`` is a finite nonnegative Euclidean distance in projection units.
    Zero preserves exact closed-boundary semantics. The caller owns the error
    budget; neither the fitted geometry nor the query batch chooses it implicitly.
    Hole interiors beyond that distance remain excluded.
    """
    if (
        isinstance(tolerance, bool)
        or not isinstance(tolerance, numbers.Real)
        or not np.isfinite(tolerance)
        or tolerance < 0
    ):
        raise ValueError("Boundary tolerance must be finite and nonnegative.")
    queries = np.asarray(queries, dtype=np.double)
    dims = 3 if isinstance(geometry, TetrahedralMesh) else 2
    if queries.ndim != 2 or queries.shape[1] != dims:  # noqa: PLR2004
        raise ValueError(f"Expected query coordinates with shape (n, {dims}).")
    result = np.zeros(len(queries), dtype=np.bool_)
    finite = np.flatnonzero(np.isfinite(queries).all(axis=1))
    if not finite.size:
        return result
    if isinstance(geometry, TetrahedralMesh):
        facets = geometry.vertices[geometry.boundary_faces]
        if not facets.size:
            return result
        inside = geometry.covers(queries[finite])
        outside = np.flatnonzero(~inside)
        if tolerance > 0:
            inside[outside] = _near_facets(queries[finite[outside]], facets, tolerance)
    else:
        if geometry.is_empty:
            return result
        boundary = geometry.boundary
        q = make_points(queries[finite])
        inside = np.asarray(covers(geometry, q))
        if tolerance > 0:
            inside |= distance(boundary, q) <= tolerance
    result[finite] = inside
    return result
