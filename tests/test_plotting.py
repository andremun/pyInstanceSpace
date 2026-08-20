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
from shapely.geometry import Polygon

from instancespace import plotting
from instancespace.stages.pilot_viewpoint import PilotViewpointResult

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
) -> Any:  # noqa: ANN401
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


def test_viewpoint_resolution_matches_matlab_group_rules() -> None:
    """Global, uncovered, and overlapping groups all use MATLAB's first rule."""
    viewpoint = _viewpoint(groups=((0, 1), (1, 2)))

    fallback = plotting._resolve_view_angle(None, None)  # noqa: SLF001
    global_view = plotting._resolve_view_angle(viewpoint, None)  # noqa: SLF001
    uncovered = plotting._resolve_view_angle(viewpoint, 99)  # noqa: SLF001
    overlapping = plotting._resolve_view_angle(viewpoint, 1)  # noqa: SLF001
    second_group = plotting._resolve_view_angle(viewpoint, 2)  # noqa: SLF001

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


def test_3d_footprint_plot_is_unavailable_until_native_geometry_exists() -> None:
    """A Shapely 2D polygon is never presented as a 3D footprint."""
    model = _fake_model(dimensions=3, viewpoint=_viewpoint())
    model.trace.good[0].polygon = Polygon([(0, 0), (1, 0), (0, 1)])

    with pytest.raises(NotImplementedError, match="3D footprint meshes"):
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
