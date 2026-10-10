# SPDX-License-Identifier: LicenseRef-PolyForm-Noncommercial-1.0.0
# Copyright (c) 2024-2026 Mario Andrés Muñoz
"""Triangulation of full-dimensional and planar 3D CLOISTER boundaries."""

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import ConvexHull

SPACE_DIMENSIONS = 3
PLANE_DIMENSIONS = 2


def boundary_faces(vertices: NDArray[np.double]) -> NDArray[np.int_]:
    """Return zero-based triangles into boundary vertices; 2D has no faces.

    Planar 3D boundaries are fan-triangulated in their own plane. Faces for
    volumetric hulls point outward. Vertex order and face diagonals are not
    cross-runtime invariants.
    """
    if vertices.ndim != PLANE_DIMENSIONS or vertices.shape[1] != SPACE_DIMENSIONS:
        return np.empty((0, SPACE_DIMENSIONS), dtype=np.int_)
    centered = vertices - vertices.mean(axis=0)
    rank = np.linalg.matrix_rank(centered) if vertices.size else 0
    if rank < PLANE_DIMENSIONS:
        raise ValueError("CLOISTER boundary spans fewer than two dimensions")
    if rank == PLANE_DIMENSIONS:
        _, _, basis = np.linalg.svd(centered, full_matrices=False)
        ids = ConvexHull(centered @ basis[:PLANE_DIMENSIONS].T).vertices
        return np.column_stack(
            (np.full(len(ids) - 2, ids[0]), ids[1:-1], ids[2:]),
        ).astype(np.int_)
    hull = ConvexHull(vertices)
    faces = hull.simplices.copy()
    triangles = vertices[faces]
    normals = np.cross(
        triangles[:, 1] - triangles[:, 0],
        triangles[:, 2] - triangles[:, 0],
    )
    inward = np.sum(normals * hull.equations[:, :SPACE_DIMENSIONS], axis=1) < 0
    faces[inward] = faces[inward][:, [0, 2, 1]]
    return np.asarray(faces, dtype=np.int_)
