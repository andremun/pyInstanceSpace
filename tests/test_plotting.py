"""Tests for the plot_*() convenience wrappers (Q7).

Uses lightweight ``SimpleNamespace`` fakes (matching the pattern already used in
``tests/build_explore_adapter/test_adapter.py``) rather than a full ``Model``, since
these are thin matplotlib wrappers and don't need real stage output to exercise.

Each test passes its own fresh ``ax`` explicitly rather than relying on the global
"current axes" plt.gca() falls back to -- matplotlib figure state otherwise leaks
across tests run in the same process.
"""

from types import SimpleNamespace

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from instancespace import plotting  # noqa: E402


@pytest.fixture
def ax():
    fig, axes = plt.subplots()
    yield axes
    plt.close(fig)


def _fake_model(*, source=None):
    z = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    algo_labels = ["CART", "KNN"]
    y_hat = np.array([[True, False], [False, True], [True, True], [False, False]])
    good_footprint = SimpleNamespace(polygon=None)
    best_footprint = SimpleNamespace(polygon=None)
    return SimpleNamespace(
        pilot=SimpleNamespace(z=z),
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


def test_plot_sources_raises_without_source_column(ax):
    model = _fake_model(source=None)
    with pytest.raises(ValueError, match="source"):
        plotting.plot_sources(model, ax=ax)


def test_plot_sources_scatters_when_source_present(ax):
    model = _fake_model(source=pd.Series(["a", "b", "a", "b"]))
    plotting.plot_sources(model, ax=ax)
    assert len(ax.collections) == 1
    assert ax.collections[0].get_offsets().shape == (4, 2)


def test_plot_portfolio_scatters_all_instances(ax):
    model = _fake_model()
    plotting.plot_portfolio(model, ax=ax)
    assert len(ax.collections) == 1
    assert ax.collections[0].get_offsets().shape == (4, 2)


def test_plot_good_splits_by_prediction(ax):
    model = _fake_model()
    plotting.plot_good(model, "CART", ax=ax)
    # 2 good + 2 bad -> two separate scatter calls.
    assert len(ax.collections) == 2
    sizes = sorted(c.get_offsets().shape[0] for c in ax.collections)
    assert sizes == [2, 2]


def test_plot_good_resolves_algorithm_by_name_and_index():
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


def test_plot_good_unknown_algorithm_raises(ax):
    model = _fake_model()
    with pytest.raises(ValueError, match="Unknown algorithm"):
        plotting.plot_good(model, "not-an-algorithm", ax=ax)


def test_plot_footprint_invalid_kind_raises(ax):
    model = _fake_model()
    with pytest.raises(ValueError, match="kind must be"):
        plotting.plot_footprint(model, "CART", kind="bad", ax=ax)


def test_plot_footprint_draws_training_instances(ax):
    model = _fake_model()
    plotting.plot_footprint(model, "CART", ax=ax)
    assert len(ax.collections) == 1
    assert ax.collections[0].get_offsets().shape == (4, 2)
