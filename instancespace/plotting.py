# SPDX-License-Identifier: LicenseRef-PolyForm-Noncommercial-1.0.0
# Copyright (c) 2024-2026 Mario Andrés Muñoz
"""Thin matplotlib wrappers around a trained ``Model``.

Mirrors MATLAB's ``InstanceSpace.plot('sources' | 'portfolio' | 'good' | 'footprint',
algoIdx)``. Each function draws onto the current matplotlib axes (``plt.gca()``) by
default, the
same way MATLAB's ``plot()`` draws onto the current figure, so repeated calls compose
in a notebook cell the way MATLAB's does rather than each silently opening a new
figure. Pass an explicit ``ax`` to draw somewhere else instead.

These take a ``Model`` (or, for testing, anything duck-typing the same shape) rather
than an ``InstanceSpace``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from matplotlib.axes import Axes
from matplotlib.pyplot import gca
from shapely.geometry import MultiPolygon


def _resolve_algo_index(model: Any, algo: str | int) -> int:  # noqa: ANN401
    if isinstance(algo, int):
        return algo
    try:
        return model.data.algo_labels.index(algo)
    except ValueError as exc:
        raise ValueError(
            f"Unknown algorithm {algo!r}; expected one of {model.data.algo_labels}",
        ) from exc


def plot_sources(model: Any, ax: Axes | None = None) -> Axes:  # noqa: ANN401
    """Scatter training instances in the 2D instance space, coloured by source.

    Requires ``model.data.s`` (the metadata's optional ``source`` column); raises if
    it wasn't provided at build time, matching MATLAB's equivalent requirement.
    """
    if model.data.s is None:
        raise ValueError(
            "plot_sources() requires a 'source' column in the training metadata; "
            "model.data.s is None.",
        )
    ax = ax if ax is not None else gca()
    z = np.asarray(model.pilot.z)
    sources = model.data.s.astype("category")
    scatter = ax.scatter(z[:, 0], z[:, 1], c=sources.cat.codes, cmap="tab10", s=20)
    ax.set_xlabel("$z_1$")
    ax.set_ylabel("$z_2$")
    ax.set_title("Instance space, coloured by source")
    handles, _ = scatter.legend_elements()
    ax.legend(handles, sources.cat.categories, title="source", loc="upper left")
    return ax


def plot_portfolio(model: Any, ax: Axes | None = None) -> Axes:  # noqa: ANN401
    """Scatter instances coloured by their best-performing algorithm (``data.p``)."""
    ax = ax if ax is not None else gca()
    z = np.asarray(model.pilot.z)
    scatter = ax.scatter(z[:, 0], z[:, 1], c=model.data.p, cmap="tab10", s=20)
    ax.set_xlabel("$z_1$")
    ax.set_ylabel("$z_2$")
    ax.set_title("Instance space, coloured by best-performing algorithm")
    fig = ax.get_figure()
    if fig is not None:
        fig.colorbar(scatter, ax=ax, label="algorithm index")
    return ax


def plot_good(
    model: Any,  # noqa: ANN401
    algo: str | int,
    ax: Axes | None = None,
) -> Axes:
    """Scatter instances coloured by PYTHIA's good/bad prediction for one algorithm."""
    j = _resolve_algo_index(model, algo)
    algo_name = model.data.algo_labels[j]
    ax = ax if ax is not None else gca()
    z = np.asarray(model.pilot.z)
    good = np.asarray(model.pythia.y_hat)[:, j]
    ax.scatter(z[~good, 0], z[~good, 1], c="tab:orange", s=20, label="predicted BAD")
    ax.scatter(z[good, 0], z[good, 1], c="tab:blue", s=20, label="predicted GOOD")
    ax.set_xlabel("$z_1$")
    ax.set_ylabel("$z_2$")
    ax.set_title(f"PYTHIA predictions for {algo_name}")
    ax.legend(loc="upper left")
    return ax


def plot_footprint(
    model: Any,  # noqa: ANN401
    algo: str | int,
    kind: str = "good",
    ax: Axes | None = None,
) -> Axes:
    """Draw one algorithm's trained footprint polygon(s) over the training instances.

    ``kind`` is ``"good"`` (region where good performance is statistically inferred)
    or ``"best"`` (region where this algorithm is expected to dominate), matching
    ``model.trace.good``/``model.trace.best``.
    """
    if kind not in ("good", "best"):
        raise ValueError(f"kind must be 'good' or 'best', got {kind!r}")
    j = _resolve_algo_index(model, algo)
    algo_name = model.data.algo_labels[j]
    footprints = model.trace.good if kind == "good" else model.trace.best
    footprint = footprints[j].polygon

    ax = ax if ax is not None else gca()
    z = np.asarray(model.pilot.z)
    ax.scatter(z[:, 0], z[:, 1], c="0.8", s=20, label="training instances")
    if footprint is not None:
        is_multi = isinstance(footprint, MultiPolygon)
        regions = footprint.geoms if is_multi else [footprint]
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
    ax.set_xlabel("$z_1$")
    ax.set_ylabel("$z_2$")
    ax.set_title(f"{kind.capitalize()} footprint of {algo_name}")
    ax.legend(loc="upper left")
    return ax
