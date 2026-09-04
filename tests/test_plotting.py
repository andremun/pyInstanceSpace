"""Tests for the plot_*() convenience wrappers (Q7).

Uses lightweight ``SimpleNamespace`` fakes rather than a full ``Model``, since these
are thin matplotlib wrappers and don't need real stage output to exercise.

Each test passes its own fresh ``ax`` explicitly rather than relying on the global
"current axes" plt.gca() falls back to -- matplotlib figure state otherwise leaks
across tests run in the same process.
"""

# Existing convenience-wrapper tests intentionally use descriptive names instead of
# repeating one-line docstrings for each case.
# ruff: noqa: D103

from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any

import matplotlib as mpl
import numpy as np
import pandas as pd
import pytest

mpl.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from mpl_toolkits.mplot3d import Axes3D  # type: ignore[import-untyped]
from mpl_toolkits.mplot3d.art3d import (  # type: ignore[import-untyped]
    Poly3DCollection,
)
from shapely.geometry import Polygon

from instancespace import plotting
from instancespace.stages.pilot_viewpoint import PilotViewpointResult
from instancespace.utils.alpha_shape import TetrahedralMesh

_THREE_DIMENSIONS = 3


@pytest.fixture()
def ax() -> Iterator[Axes]:
    """Yield an isolated two-dimensional axis."""
    fig, axes = plt.subplots()
    yield axes
    plt.close(fig)


def _fake_model(
    *,
    source: "pd.Series[str] | None" = None,
    dimensions: int = 2,
    viewpoint: PilotViewpointResult | None = None,
) -> Any:
    z = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    if dimensions == _THREE_DIMENSIONS:
        z = np.column_stack((z, np.array([10.0, 11.0, 12.0, 13.0])))
    algo_labels = ["CART", "KNN"]
    y_hat = np.array([[True, False], [False, True], [True, True], [False, False]])
    good_footprint = SimpleNamespace(polygon=None)
    best_footprint = SimpleNamespace(polygon=None)
    return SimpleNamespace(
        pilot=SimpleNamespace(z=z, viewpoint=viewpoint),
        data=SimpleNamespace(
            algo_labels=algo_labels,
            p=np.array([0, 1, 0, 1]),
            s=source,
        ),
        pythia=SimpleNamespace(y_hat=y_hat),
        trace=SimpleNamespace(
            good=[good_footprint, good_footprint],
            best=[best_footprint, best_footprint],
        ),
    )


def _viewpoint(
    groups: tuple[tuple[int, ...], ...] = ((0,), (1,)),
) -> PilotViewpointResult:
    return PilotViewpointResult(
        groups=groups,
        a=(
            np.eye(2, 3, dtype=np.float64),
            np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
        ),
        azimuth=(np.deg2rad(15.0), np.deg2rad(75.0)),
        elevation=(np.deg2rad(25.0), np.deg2rad(45.0)),
    )


def _unit_tetrahedral_mesh() -> TetrahedralMesh:
    return TetrahedralMesh(
        vertices=np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.double,
        ),
        tetrahedra=np.array([[0, 1, 2, 3]], dtype=np.int_),
        boundary_faces=np.array(
            [[1, 2, 3], [0, 3, 2], [0, 1, 3], [0, 2, 1]],
            dtype=np.int_,
        ),
        alpha=float(np.sqrt(3) / 2),
        region_threshold=0.01,
        region_count=1,
        volume=1 / 6,
        surface_area=float(1.5 + np.sqrt(3) / 2),
    )


def test_viewpoint_resolution_matches_matlab_group_rules() -> None:
    """Global, uncovered, and overlapping groups all use MATLAB's first rule."""
    viewpoint = _viewpoint(groups=((0, 1), (1, 2)))

    fallback = plotting._resolve_view_angle(None, None)
    global_view = plotting._resolve_view_angle(viewpoint, None)
    uncovered = plotting._resolve_view_angle(viewpoint, 99)
    overlapping = plotting._resolve_view_angle(viewpoint, 1)
    second_group = plotting._resolve_view_angle(viewpoint, 2)

    assert fallback == pytest.approx((-37.5, 30.0))
    assert global_view == pytest.approx((15.0, 25.0))
    assert uncovered == pytest.approx((15.0, 25.0))
    assert overlapping == pytest.approx((15.0, 25.0))
    assert second_group == pytest.approx((75.0, 45.0))


def test_3d_global_plot_uses_native_axis_camera_and_z_coordinates() -> None:
    """Global plots use group zero and pass z3 directly to matplotlib."""
    model = _fake_model(dimensions=3, viewpoint=_viewpoint())

    axis = plotting.plot_portfolio(model)
    try:
        assert isinstance(axis, Axes3D)
        assert getattr(axis, "azim") == pytest.approx(15.0)
        assert getattr(axis, "elev") == pytest.approx(25.0)
        _, _, z_coordinates = getattr(axis.collections[0], "_offsets3d")
        np.testing.assert_array_equal(z_coordinates, model.pilot.z[:, 2])
        assert axis.get_zlabel() == "$z_3$"
    finally:
        plt.close(axis.get_figure())


def test_3d_algorithm_plot_uses_matching_group_camera() -> None:
    """Per-algorithm plots use the first group containing its zero-based index."""
    model = _fake_model(dimensions=3, viewpoint=_viewpoint())
    fig = plt.figure()
    axis = fig.add_subplot(projection="3d")
    try:
        returned = plotting.plot_good(model, 1, ax=axis)
        assert returned is axis
        assert getattr(axis, "azim") == pytest.approx(75.0)
        assert getattr(axis, "elev") == pytest.approx(45.0)
        plotted_z = np.concatenate(
            [
                np.asarray(getattr(collection, "_offsets3d")[2])
                for collection in axis.collections
            ],
        )
        np.testing.assert_array_equal(np.sort(plotted_z), model.pilot.z[:, 2])
    finally:
        plt.close(fig)


def test_3d_without_viewpoint_uses_matlab_view_3_default() -> None:
    """A legacy model gets MATLAB view(3)'s explicit azimuth/elevation."""
    axis = plotting.plot_portfolio(_fake_model(dimensions=3, viewpoint=None))
    try:
        assert getattr(axis, "azim") == pytest.approx(-37.5)
        assert getattr(axis, "elev") == pytest.approx(30.0)
    finally:
        plt.close(axis.get_figure())


def test_3d_projection_rejects_user_supplied_2d_axis(ax: Axes) -> None:
    """A 3D projection cannot silently discard z3 on a supplied 2D axis."""
    with pytest.raises(ValueError, match="3D projection.*axis"):
        plotting.plot_portfolio(_fake_model(dimensions=3), ax=ax)


def test_3d_footprint_plot_draws_native_mesh_with_algorithm_camera() -> None:
    """Every boundary face is native 3D and uses the algorithm's first group."""
    model = _fake_model(dimensions=3, viewpoint=_viewpoint())
    mesh = _unit_tetrahedral_mesh()
    model.trace.good[1].polygon = mesh

    axis = plotting.plot_footprint(model, 1)
    try:
        assert isinstance(axis, Axes3D)
        assert getattr(axis, "azim") == pytest.approx(75.0)
        assert getattr(axis, "elev") == pytest.approx(45.0)
        mesh_collections = [
            collection
            for collection in axis.collections
            if isinstance(collection, Poly3DCollection)
        ]
        assert len(mesh_collections) == 1
        axis.get_figure().canvas.draw()
        assert len(mesh_collections[0].get_paths()) == len(mesh.boundary_faces)
        _, _, plotted_z = getattr(axis.collections[0], "_offsets3d")
        np.testing.assert_array_equal(plotted_z, model.pilot.z[:, 2])
        z_lower, z_upper = axis.get_zlim()
        assert z_lower <= float(np.min(mesh.vertices[:, 2]))
        assert z_upper >= float(np.max(model.pilot.z[:, 2]))
    finally:
        plt.close(axis.get_figure())


def test_3d_footprint_rejects_shapely_geometry() -> None:
    model = _fake_model(dimensions=3)
    model.trace.good[0].polygon = Polygon([(0, 0), (1, 0), (0, 1)])

    with pytest.raises(ValueError, match="TetrahedralMesh.*Shapely"):
        plotting.plot_footprint(model, 0)


def test_2d_footprint_rejects_tetrahedral_geometry(ax: Axes) -> None:
    model = _fake_model()
    model.trace.good[0].polygon = _unit_tetrahedral_mesh()

    with pytest.raises(ValueError, match="2D projection.*TetrahedralMesh"):
        plotting.plot_footprint(model, 0, ax=ax)


def test_3d_footprint_rejects_user_supplied_2d_axis(ax: Axes) -> None:
    model = _fake_model(dimensions=3)
    model.trace.good[0].polygon = _unit_tetrahedral_mesh()

    with pytest.raises(ValueError, match="3D projection.*axis"):
        plotting.plot_footprint(model, 0, ax=ax)


def test_3d_footprint_rejects_2d_footprint_metadata() -> None:
    model = _fake_model(dimensions=3)
    model.trace.good[0].polygon = _unit_tetrahedral_mesh()
    model.trace.good[0].dimension = 2

    with pytest.raises(ValueError, match="metadata.*projection dimensions"):
        plotting.plot_footprint(model, 0)


def test_plot_sources_raises_without_source_column(ax: Axes) -> None:
    model = _fake_model(source=None)
    with pytest.raises(ValueError, match="source"):
        plotting.plot_sources(model, ax=ax)


def test_plot_sources_scatters_when_source_present(ax: Axes) -> None:
    model = _fake_model(source=pd.Series(["a", "b", "a", "b"]))
    plotting.plot_sources(model, ax=ax)
    assert len(ax.collections) == 1
    assert np.asarray(ax.collections[0].get_offsets()).shape == (4, 2)


def test_plot_portfolio_scatters_all_instances(ax: Axes) -> None:
    model = _fake_model()
    plotting.plot_portfolio(model, ax=ax)
    assert len(ax.collections) == 1
    assert np.asarray(ax.collections[0].get_offsets()).shape == (4, 2)


def test_plot_good_splits_by_prediction(ax: Axes) -> None:
    model = _fake_model()
    plotting.plot_good(model, "CART", ax=ax)
    # 2 good + 2 bad -> two separate scatter calls.
    expected_collections = 2
    assert len(ax.collections) == expected_collections
    sizes = sorted(np.asarray(c.get_offsets()).shape[0] for c in ax.collections)
    assert sizes == [2, 2]


def test_plot_good_resolves_algorithm_by_name_and_index() -> None:
    model = _fake_model()
    fig1, ax_by_name = plt.subplots()
    fig2, ax_by_index = plt.subplots()
    try:
        plotting.plot_good(model, "KNN", ax=ax_by_name)
        plotting.plot_good(model, 1, ax=ax_by_index)
        assert ax_by_name.get_title() == ax_by_index.get_title()
    finally:
        plt.close(fig1)
        plt.close(fig2)


def test_plot_good_unknown_algorithm_raises(ax: Axes) -> None:
    model = _fake_model()
    with pytest.raises(ValueError, match="Unknown algorithm"):
        plotting.plot_good(model, "not-an-algorithm", ax=ax)


def test_plot_footprint_invalid_kind_raises(ax: Axes) -> None:
    model = _fake_model()
    with pytest.raises(ValueError, match="kind must be"):
        plotting.plot_footprint(model, "CART", kind="bad", ax=ax)


def test_plot_footprint_draws_training_instances(ax: Axes) -> None:
    model = _fake_model()
    plotting.plot_footprint(model, "CART", ax=ax)
    assert len(ax.collections) == 1
    assert np.asarray(ax.collections[0].get_offsets()).shape == (4, 2)
