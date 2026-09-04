# SPDX-License-Identifier: LicenseRef-PolyForm-Noncommercial-1.0.0
# Copyright (c) 2024-2026 Mario Andrés Muñoz
"""Thin matplotlib wrappers around a trained ``Model``.

Mirrors MATLAB's ``InstanceSpace.plot('sources' | 'portfolio' | 'good' | 'footprint',
algoIdx)``. Two-dimensional calls draw onto the current matplotlib axes by default,
so repeated calls compose in a notebook cell. Three-dimensional calls create native
3D axes when none are supplied. Pass an explicit dimension-compatible ``ax`` to draw
somewhere else instead.

These take a ``Model`` (or, for testing, anything duck-typing the same shape) rather
than an ``InstanceSpace``.
"""

from __future__ import annotations

from typing import Any, NamedTuple, cast

import numpy as np
from matplotlib.axes import Axes
from matplotlib.pyplot import figure, gca
from mpl_toolkits.mplot3d import Axes3D  # type: ignore[import-untyped]
from mpl_toolkits.mplot3d.art3d import (  # type: ignore[import-untyped]
    Poly3DCollection,
)
from shapely.geometry import MultiPolygon, Polygon

from instancespace.stages.pilot_viewpoint import PilotViewpointResult
from instancespace.utils.alpha_shape import TetrahedralMesh

_PROJECTION_ARRAY_DIMENSIONS = 2
_TWO_DIMENSIONAL = 2
_THREE_DIMENSIONAL = 3
# MATLAB ``view(3)`` is the fixed default view(azimuth=-37.5, elevation=30).
_MATLAB_DEFAULT_3D_AZIMUTH_DEGREES = -37.5
_MATLAB_DEFAULT_3D_ELEVATION_DEGREES = 30.0


class _ViewAngle(NamedTuple):
    """A camera angle in matplotlib's named argument order."""

    azimuth: float
    elevation: float


def _projection_dimensions(z: Any) -> int:  # noqa: ANN401
    """Validate and return a projection's supported coordinate count."""
    projection = np.asarray(z)
    if projection.ndim != _PROJECTION_ARRAY_DIMENSIONS or projection.shape[1] not in {
        _TWO_DIMENSIONAL,
        _THREE_DIMENSIONAL,
    }:
        raise ValueError("PILOT Z must be a two-dimensional 2D or 3D projection.")
    return int(projection.shape[1])


def _resolve_view_angle(
    viewpoint: PilotViewpointResult | None,
    algorithm_index: int | None,
) -> _ViewAngle:
    """Resolve MATLAB's global/per-algorithm camera rule in degrees.

    The first viewpoint is global and is also the fallback for an uncovered
    algorithm. When groups overlap, their first match wins. A missing viewpoint
    uses MATLAB ``view(3)``: azimuth -37.5 degrees, elevation 30 degrees.
    """
    if viewpoint is None:
        return _ViewAngle(
            _MATLAB_DEFAULT_3D_AZIMUTH_DEGREES,
            _MATLAB_DEFAULT_3D_ELEVATION_DEGREES,
        )

    group_index = 0
    if algorithm_index is not None:
        group_index = next(
            (
                index
                for index, group in enumerate(viewpoint.groups)
                if algorithm_index in group
            ),
            0,
        )
    return _ViewAngle(
        float(np.rad2deg(viewpoint.azimuth[group_index])),
        float(np.rad2deg(viewpoint.elevation[group_index])),
    )


def _prepare_axis(z: Any, ax: Axes | None) -> Axes:  # noqa: ANN401
    """Return native axes for ``z`` and reject a supplied dimension mismatch."""
    dimensions = _projection_dimensions(z)
    if ax is None:
        if dimensions == _THREE_DIMENSIONAL:
            return figure().add_subplot(projection="3d")
        ax = gca()

    is_three_dimensional_axis = isinstance(ax, Axes3D)
    if dimensions == _THREE_DIMENSIONAL and not is_three_dimensional_axis:
        raise ValueError(
            "A 3D projection requires a three-dimensional matplotlib axis.",
        )
    if dimensions == _TWO_DIMENSIONAL and is_three_dimensional_axis:
        raise ValueError("A 2D projection requires a two-dimensional matplotlib axis.")
    return ax


def _scatter_projection(
    ax: Axes,
    z: Any,  # noqa: ANN401
    *args: Any,  # noqa: ANN401
    **kwargs: Any,  # noqa: ANN401
) -> Any:  # noqa: ANN401
    """Scatter every coordinate of a validated 2D or 3D projection."""
    projection = np.asarray(z)
    dimensions = _projection_dimensions(projection)
    _prepare_axis(projection, ax)
    if dimensions == _THREE_DIMENSIONAL:
        return ax.scatter(
            projection[:, 0],
            projection[:, 1],
            projection[:, 2],
            *args,
            **kwargs,
        )
    return ax.scatter(projection[:, 0], projection[:, 1], *args, **kwargs)


def _label_projection_axes(ax: Axes, z: Any) -> None:  # noqa: ANN401
    """Apply coordinate labels for a 2D or 3D PILOT projection."""
    dimensions = _projection_dimensions(z)
    ax.set_xlabel("$z_1$")
    ax.set_ylabel("$z_2$")
    if dimensions == _THREE_DIMENSIONAL:
        ax.set_zlabel("$z_3$")  # type: ignore[attr-defined]


def _apply_view_angle(ax: Axes, z: Any, angle: _ViewAngle) -> None:  # noqa: ANN401
    """Apply ``angle`` to native 3D axes; leave a 2D view unchanged."""
    if _projection_dimensions(z) != _THREE_DIMENSIONAL:
        return
    if not isinstance(ax, Axes3D):
        raise ValueError(
            "A 3D projection requires a three-dimensional matplotlib axis.",
        )
    ax.view_init(elev=angle.elevation, azim=angle.azimuth)


def _model_view_angle(
    model: Any,  # noqa: ANN401
    algorithm_index: int | None,
) -> _ViewAngle:
    """Resolve a model's optional persisted viewpoint."""
    return _resolve_view_angle(getattr(model.pilot, "viewpoint", None), algorithm_index)


def _draw_tetrahedral_mesh(
    ax: Axes,
    mesh: TetrahedralMesh,
    *,
    facecolor: Any,  # noqa: ANN401
    edgecolor: Any,  # noqa: ANN401
    alpha: float,
) -> Poly3DCollection | None:
    """Draw every outward boundary face on native 3D axes."""
    if not isinstance(ax, Axes3D):
        raise ValueError("A TetrahedralMesh requires a three-dimensional axis.")
    if mesh.is_empty:
        return None

    faces = mesh.vertices[mesh.boundary_faces]
    collection = Poly3DCollection(
        faces,
        facecolors=facecolor,
        edgecolors=edgecolor,
        linewidths=1.5,
        alpha=alpha,
    )
    ax.add_collection3d(collection)
    ax.auto_scale_xyz(
        mesh.vertices[:, 0],
        mesh.vertices[:, 1],
        mesh.vertices[:, 2],
    )
    return collection


def _resolve_algo_index(model: Any, algo: str | int) -> int:  # noqa: ANN401
    if isinstance(algo, int):
        return algo
    try:
        return int(model.data.algo_labels.index(algo))
    except ValueError as exc:
        raise ValueError(
            f"Unknown algorithm {algo!r}; expected one of {model.data.algo_labels}",
        ) from exc


def plot_sources(model: Any, ax: Axes | None = None) -> Axes:  # noqa: ANN401
    """Scatter training instances in the instance space, coloured by source.

    Requires ``model.data.s`` (the metadata's optional ``source`` column); raises if
    it wasn't provided at build time, matching MATLAB's equivalent requirement.
    """
    if model.data.s is None:
        raise ValueError(
            "plot_sources() requires a 'source' column in the training metadata; "
            "model.data.s is None.",
        )
    z = np.asarray(model.pilot.z)
    ax = _prepare_axis(z, ax)
    sources = model.data.s.astype("category")
    scatter = _scatter_projection(
        ax,
        z,
        c=sources.cat.codes,
        cmap="tab10",
        s=20,
    )
    _label_projection_axes(ax, z)
    ax.set_title("Instance space, coloured by source")
    handles, _ = scatter.legend_elements()
    ax.legend(handles, sources.cat.categories, title="source", loc="upper left")
    _apply_view_angle(ax, z, _model_view_angle(model, None))
    return ax


def plot_portfolio(model: Any, ax: Axes | None = None) -> Axes:  # noqa: ANN401
    """Scatter instances coloured by their best-performing algorithm (``data.p``)."""
    z = np.asarray(model.pilot.z)
    ax = _prepare_axis(z, ax)
    scatter = _scatter_projection(ax, z, c=model.data.p, cmap="tab10", s=20)
    _label_projection_axes(ax, z)
    ax.set_title("Instance space, coloured by best-performing algorithm")
    fig = ax.get_figure()
    if fig is not None:
        fig.colorbar(scatter, ax=ax, label="algorithm index")
    _apply_view_angle(ax, z, _model_view_angle(model, None))
    return ax


def plot_good(
    model: Any,  # noqa: ANN401
    algo: str | int,
    ax: Axes | None = None,
) -> Axes:
    """Scatter instances coloured by PYTHIA's good/bad prediction for one algorithm."""
    j = _resolve_algo_index(model, algo)
    algo_name = model.data.algo_labels[j]
    z = np.asarray(model.pilot.z)
    ax = _prepare_axis(z, ax)
    good = np.asarray(model.pythia.y_hat)[:, j]
    _scatter_projection(
        ax,
        z[~good],
        c="tab:orange",
        s=20,
        label="predicted BAD",
    )
    _scatter_projection(
        ax,
        z[good],
        c="tab:blue",
        s=20,
        label="predicted GOOD",
    )
    _label_projection_axes(ax, z)
    ax.set_title(f"PYTHIA predictions for {algo_name}")
    ax.legend(loc="upper left")
    _apply_view_angle(ax, z, _model_view_angle(model, j))
    return ax


def plot_footprint(
    model: Any,  # noqa: ANN401
    algo: str | int,
    kind: str = "good",
    ax: Axes | None = None,
) -> Axes:
    """Draw one algorithm's native 2D polygon or 3D tetrahedral footprint.

    ``kind`` is ``"good"`` (region where good performance is statistically inferred)
    or ``"best"`` (region where this algorithm is expected to dominate), matching
    ``model.trace.good``/``model.trace.best``.
    """
    if kind not in ("good", "best"):
        raise ValueError(f"kind must be 'good' or 'best', got {kind!r}")
    j = _resolve_algo_index(model, algo)
    z = np.asarray(model.pilot.z)
    dimensions = _projection_dimensions(z)
    algo_name = model.data.algo_labels[j]
    footprints = model.trace.good if kind == "good" else model.trace.best
    footprint = footprints[j]
    footprint_dimension = getattr(footprint, "dimension", dimensions)
    if footprint_dimension != dimensions:
        raise ValueError(
            "Footprint metadata and PILOT projection dimensions do not match.",
        )
    geometry = footprint.polygon
    if (
        dimensions == _THREE_DIMENSIONAL
        and geometry is not None
        and not isinstance(
            geometry,
            TetrahedralMesh,
        )
    ):
        raise ValueError(
            "A 3D projection requires TetrahedralMesh footprint geometry; "
            "Shapely polygons cannot be projected into 3D.",
        )
    if dimensions == _TWO_DIMENSIONAL and isinstance(geometry, TetrahedralMesh):
        raise ValueError("A 2D projection cannot draw TetrahedralMesh geometry.")
    if (
        dimensions == _TWO_DIMENSIONAL
        and geometry is not None
        and not isinstance(
            geometry,
            Polygon | MultiPolygon,
        )
    ):
        raise ValueError("A 2D projection requires Shapely polygon geometry.")

    ax = _prepare_axis(z, ax)
    _scatter_projection(ax, z, c="0.8", s=20, label="training instances")
    if dimensions == _THREE_DIMENSIONAL and isinstance(geometry, TetrahedralMesh):
        _draw_tetrahedral_mesh(
            ax,
            geometry,
            facecolor="tab:blue",
            edgecolor="tab:blue",
            alpha=0.15,
        )
        if geometry.vertices.size:
            combined = np.vstack((z, geometry.vertices))
            axis_3d = cast(Any, ax)
            axis_3d.auto_scale_xyz(combined[:, 0], combined[:, 1], combined[:, 2])
    elif isinstance(geometry, Polygon | MultiPolygon):
        is_multi = isinstance(geometry, MultiPolygon)
        regions = geometry.geoms if is_multi else [geometry]
        for region in regions:
            xs, ys = region.exterior.xy
            ax.fill(
                xs,
                ys,
                facecolor="tab:blue",
                alpha=0.15,
                edgecolor="tab:blue",
                linewidth=1.5,
            )
    _label_projection_axes(ax, z)
    ax.set_title(f"{kind.capitalize()} footprint of {algo_name}")
    ax.legend(loc="upper left")
    _apply_view_angle(ax, z, _model_view_angle(model, j))
    return ax
