# SPDX-License-Identifier: LicenseRef-PolyForm-Noncommercial-1.0.0
# Copyright (c) 2024-2026 Mario Andrés Muñoz
from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from matplotlib.patches import PathPatch
from matplotlib.path import Path as MatplotlibPath
from numpy.typing import NDArray
from scipy.io import savemat
from shapely import MultiPolygon, Polygon
from shapely.geometry.polygon import orient  # type: ignore[import-untyped]

from instancespace.data.model import (
    CloisterOut,
    Data,
    FeatSel,
    Footprint,
    PilotOut,
    PythiaOut,
    SiftedOut,
    TraceOut,
)
from instancespace.data.options import InstanceSpaceOptions

_FOOTPRINT_COLUMNS = ["Row", "Part", "Ring", "Vertex", "z_1", "z_2"]
_INVALID_STEM_CHARACTERS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_MAX_STEM_LENGTH = 80
_WINDOWS_RESERVED_STEMS = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}


class SerializationError(RuntimeError):
    """Report a failed output write with its operation and target path."""


def _portable_stems(labels: list[str], fallback: str) -> list[str]:
    """Return deterministic, unique filename stems for user labels."""
    stems: list[str] = []
    used: set[str] = set()

    for index, label in enumerate(labels, start=1):
        stem = unicodedata.normalize("NFKC", str(label))
        stem = _INVALID_STEM_CHARACTERS.sub("_", stem).strip(" .")
        if not stem or stem in {".", ".."}:
            stem = f"{fallback}_{index}"
        if stem.split(".", maxsplit=1)[0].upper() in _WINDOWS_RESERVED_STEMS:
            stem = f"_{stem}"

        stem = stem[:_MAX_STEM_LENGTH].rstrip(" .") or f"{fallback}_{index}"
        candidate = stem
        suffix_number = 2
        while candidate.casefold() in used:
            suffix = f"_{suffix_number}"
            candidate = f"{stem[: _MAX_STEM_LENGTH - len(suffix)]}{suffix}"
            suffix_number += 1

        used.add(candidate.casefold())
        stems.append(candidate)

    return stems


def _write_dataframe_to_csv(
    frame: pd.DataFrame,
    filename: Path,
    *,
    index: bool = False,
    index_label: str | None = None,
) -> None:
    """Write one CSV file and add the target path to write errors."""
    try:
        frame.to_csv(filename, index=index, index_label=index_label)
    except Exception as exc:
        raise SerializationError(f"Could not write CSV file '{filename}'.") from exc


def _footprint_boundary_frame(polygon: Polygon | MultiPolygon) -> pd.DataFrame:
    """Convert a polygon geometry to the lossless footprint CSV v2 schema."""
    records: list[dict[str, int | float | str]] = []
    parts = polygon.geoms if isinstance(polygon, MultiPolygon) else [polygon]

    for part_number, part in enumerate(parts, start=1):
        rings = [("exterior", part.exterior)]
        rings.extend(
            (f"hole_{hole_number}", ring)
            for hole_number, ring in enumerate(part.interiors, start=1)
        )
        for ring_name, ring in rings:
            coordinates = np.asarray(ring.coords, dtype=np.double)
            if coordinates.shape[0] > 1 and np.array_equal(
                coordinates[0],
                coordinates[-1],
            ):
                coordinates = coordinates[:-1]
            for vertex_number, (z_1, z_2) in enumerate(coordinates, start=1):
                records.append(
                    {
                        "Row": len(records) + 1,
                        "Part": part_number,
                        "Ring": ring_name,
                        "Vertex": vertex_number,
                        "z_1": float(z_1),
                        "z_2": float(z_2),
                    },
                )

    return pd.DataFrame.from_records(records, columns=_FOOTPRINT_COLUMNS)


def save_instance_space_to_csv(
    output_directory: Path,
    data: Data,
    sifted_out: SiftedOut,
    trace_out: TraceOut,
    pilot_out: PilotOut,
    cloister_out: CloisterOut,
    pythia_out: PythiaOut,
) -> None:
    if not output_directory.is_dir():
        raise ValueError("output_directory must be an existing directory.")

    num_algorithms = data.y.shape[1]
    algorithm_stems = _portable_stems(data.algo_labels, "algorithm")

    for i in range(num_algorithms):
        best = trace_out.best[i]
        if best is not None and best.polygon is not None and not best.polygon.is_empty:
            _write_dataframe_to_csv(
                _footprint_boundary_frame(best.polygon),
                output_directory / f"footprint_{algorithm_stems[i]}_best.csv",
            )

        good = trace_out.good[i]
        if good is not None and good.polygon is not None and not good.polygon.is_empty:
            _write_dataframe_to_csv(
                _footprint_boundary_frame(good.polygon),
                output_directory / f"footprint_{algorithm_stems[i]}_good.csv",
            )

    _write_array_to_csv(
        pilot_out.z,
        pd.Series(["z_1", "z_2"]),
        data.inst_labels,
        output_directory / "coordinates.csv",
    )

    if cloister_out is not None:
        _write_array_to_csv(
            cloister_out.z_edge,
            pd.Series(["z_1", "z_2"]),
            _make_bind_labels(cloister_out.z_edge),
            output_directory / "bounds.csv",
        )
        _write_array_to_csv(
            cloister_out.z_ecorr,
            pd.Series(["z_1", "z_2"]),
            _make_bind_labels(cloister_out.z_ecorr),
            output_directory / "bounds_prunned.csv",
        )

    _write_array_to_csv(
        data.x_raw[:, sifted_out.selvars],
        pd.Series(data.feat_labels),
        data.inst_labels,
        output_directory / "feature_raw.csv",
    )
    _write_array_to_csv(
        data.x,
        pd.Series(data.feat_labels),
        data.inst_labels,
        output_directory / "feature_process.csv",
    )
    _write_array_to_csv(
        data.y_raw,
        pd.Series(data.algo_labels),
        data.inst_labels,
        output_directory / "algorithm_raw.csv",
    )
    _write_array_to_csv(
        data.y,
        pd.Series(data.algo_labels),
        data.inst_labels,
        output_directory / "algorithm_process.csv",
    )
    _write_array_to_csv(
        data.y_bin,
        pd.Series(data.algo_labels),
        data.inst_labels,
        output_directory / "algorithm_bin.csv",
    )
    _write_array_to_csv(
        data.num_good_algos,
        pd.Series(["NumGoodAlgos"]),
        data.inst_labels,
        output_directory / "good_algos.csv",
    )
    _write_array_to_csv(
        data.beta,
        pd.Series(["IsBetaEasy"]),
        data.inst_labels,
        output_directory / "beta_easy.csv",
    )
    _write_array_to_csv(
        data.p,
        pd.Series(["Best_Algorithm"]),
        data.inst_labels,
        output_directory / "portfolio.csv",
    )
    _write_array_to_csv(
        pythia_out.y_hat,
        pd.Series(data.algo_labels),
        data.inst_labels,
        output_directory / "algorithm_svm.csv",
    )
    _write_array_to_csv(
        pythia_out.selection0,
        pd.Series(["Best_Algorithm"]),
        data.inst_labels,
        output_directory / "portfolio_svm.csv",
    )

    trace_summary = trace_out.summary.iloc[:, [0, 2, 4, 5, 7, 9, 10]].copy()
    trace_summary = trace_summary.rename(columns={"Algorithm": "Row"})
    _write_dataframe_to_csv(
        trace_summary,
        output_directory / "footprint_performance.csv",
    )

    if pilot_out.summary is not None:
        pilot_summary = pilot_out.summary.copy().rename(columns={0: "Row"})
        _write_dataframe_to_csv(
            pilot_summary,
            output_directory / "projection_matrix.csv",
        )

    pythia_summary = pythia_out.summary.copy().rename(columns={"Algorithms": "Row"})
    _write_dataframe_to_csv(
        pythia_summary,
        output_directory / "svm_table.csv",
    )


def save_instance_space_for_web(
    output_directory: Path,
    data: Data,
    feat_sel: FeatSel,
) -> None:
    if not output_directory.is_dir():
        raise ValueError("output_directory must be an existing directory.")

    colours = (
        np.array(
            mpl.colormaps["viridis"].resampled(256).__dict__["colors"],
        )[:, :3]
        * 255
    ).astype(np.int_)

    _write_dataframe_to_csv(
        pd.DataFrame(colours, columns=["R", "G", "B"]),
        output_directory / "color_table.csv",
    )

    _write_colour_array_to_csv(
        _colour_scale(data.x_raw[:, feat_sel.idx]),
        pd.Series(data.feat_labels),
        data.inst_labels,
        output_directory / "feature_raw_color.csv",
    )
    _write_colour_array_to_csv(
        _colour_scale(data.y_raw),
        pd.Series(data.algo_labels),
        data.inst_labels,
        output_directory / "algorithm_raw_single_color.csv",
    )
    _write_colour_array_to_csv(
        _colour_scale(data.x),
        pd.Series(data.feat_labels),
        data.inst_labels,
        output_directory / "feature_process_color.csv",
    )
    _write_colour_array_to_csv(
        _colour_scale(data.y),
        pd.Series(data.algo_labels),
        data.inst_labels,
        output_directory / "algorithm_process_single_color.csv",
    )
    _write_colour_array_to_csv(
        _colour_scale_g(data.y_raw),
        pd.Series(data.algo_labels),
        data.inst_labels,
        output_directory / "algorithm_raw_color.csv",
    )
    _write_colour_array_to_csv(
        _colour_scale_g(data.y),
        pd.Series(data.algo_labels),
        data.inst_labels,
        output_directory / "algorithm_process_color.csv",
    )
    _write_colour_array_to_csv(
        _colour_scale_g(data.num_good_algos),
        pd.Series(["NumGoodAlgos"]),
        data.inst_labels,
        output_directory / "good_algos_color.csv",
    )


def save_instance_space_graphs(
    output_directory: Path,
    data: Data,
    options: InstanceSpaceOptions,
    pythia: PythiaOut,
    pilot: PilotOut,
    trace: TraceOut,
) -> None:
    if not output_directory.is_dir():
        raise ValueError("output_directory must be an existing directory.")

    num_feats = data.x.shape[1]
    num_algorithms = data.y.shape[1]
    feature_stems = _portable_stems(data.feat_labels, "feature")
    algorithm_stems = _portable_stems(data.algo_labels, "algorithm")

    x_aux = _minmax_scale(data.x, axis=0)
    y_ind = _minmax_scale(data.y_raw, axis=0)
    y_log = np.full(data.y_raw.shape, np.nan, dtype=np.double)
    log_domain = np.isfinite(data.y_raw) & (data.y_raw > -1)
    y_log[log_domain] = np.log10(data.y_raw[log_domain] + 1)
    y_glb = _minmax_scale(y_log, axis=None)

    if options.trace.use_sim:
        y_foot = pythia.y_hat
        p_foot = pythia.selection0
    else:
        y_foot = data.y_bin
        p_foot = data.p - 1

    for i in range(num_feats):
        filename = f"distribution_feature_{feature_stems[i]}.png"
        _draw_scatter(
            pilot.z,
            x_aux[:, i],
            data.feat_labels[i].replace("_", " "),
            output_directory / filename,
        )

    for i in range(num_algorithms):
        algo_label = data.algo_labels[i]
        algo_stem = algorithm_stems[i]

        filename = f"distribution_performance_global_normalized_{algo_stem}.png"
        _draw_scatter(
            pilot.z,
            y_glb[:, i],
            algo_label.replace("_", " "),
            output_directory / filename,
        )

        filename = f"distribution_performance_individual_normalized_{algo_stem}.png"
        _draw_scatter(
            pilot.z,
            y_ind[:, i],
            algo_label.replace("_", " "),
            output_directory / filename,
        )

        _draw_binary_performance(
            pilot.z,
            data.y_bin[:, i],
            algo_label.replace("_", " "),
            output_directory / f"binary_performance_{algo_stem}.png",
        )

        _draw_binary_performance(
            pilot.z,
            pythia.y_hat[:, i],
            algo_label.replace("_", " "),
            output_directory / f"binary_svm_{algo_stem}.png",
        )

        _draw_good_bad_footprint(
            pilot.z,
            trace.good[i],
            y_foot[:, i],
            algo_label.replace("_", " ") + " Footprint",
            output_directory / f"footprint_{algo_stem}.png",
        )

    _draw_scatter(
        pilot.z,
        data.num_good_algos / num_algorithms,
        "Percentage of good algorithms",
        output_directory / "distribution_number_good_algos.png",
    )

    _draw_portfolio_selections(
        pilot.z,
        data.p - 1,
        np.array(data.algo_labels),
        "Best algorithm",
        output_directory / "distribution_portfolio.png",
    )

    _draw_portfolio_selections(
        pilot.z,
        pythia.selection0,
        np.array(data.algo_labels),
        "Predicted best algorithm",
        output_directory / "distribution_svm_portfolio.png",
    )

    _draw_binary_performance(
        pilot.z,
        data.beta,
        "Beta score",
        output_directory / "distribution_beta_score.png",
    )

    if data.s is not None:
        _draw_sources(
            pilot.z,
            np.array(data.s),
            output_directory / "distribution_sources.png",
        )

    _draw_portfolio_footprint(
        pilot.z,
        trace.best,
        p_foot,
        np.array(data.algo_labels),
        output_directory / "footprint_portfolio.png",
    )


def _write_array_to_csv(
    data: NDArray[Any],
    column_names: pd.Series[str],
    row_names: pd.Series[str],
    filename: Path,
) -> None:
    _write_dataframe_to_csv(
        pd.DataFrame(data, index=row_names, columns=column_names),
        filename,
        index=True,
        index_label="Row",
    )


def _make_bind_labels(
    data: NDArray[Any],
) -> pd.Series[str]:
    return pd.Series([f"bnd_pnt_{i+1}" for i in range(data.shape[0])])


def _write_colour_array_to_csv(
    data: NDArray[np.double],
    column_names: pd.Series[str],
    row_names: pd.Series[str],
    filename: Path,
) -> None:
    """Write integer colours while keeping missing values empty."""
    frame = pd.DataFrame(data, index=row_names, columns=column_names).astype("Int64")
    _write_dataframe_to_csv(frame, filename, index=True, index_label="Row")


def _minmax_scale(
    data: NDArray[Any],
    axis: int | None,
) -> NDArray[np.double]:
    """Scale finite values to zero through one without empty-slice warnings."""
    values = np.asarray(data, dtype=np.double)
    scaled = np.full(values.shape, np.nan, dtype=np.double)

    if axis is None:
        finite = np.isfinite(values)
        if not np.any(finite):
            return scaled
        minimum = float(np.min(values[finite]))
        value_range = float(np.max(values[finite]) - minimum)
        scaled[finite] = 0.0
        if value_range > 0:
            scaled[finite] = (values[finite] - minimum) / value_range
        return scaled

    if axis != 0:
        raise ValueError("axis must be 0 or None.")

    for column in range(values.shape[1]):
        finite = np.isfinite(values[:, column])
        if not np.any(finite):
            continue
        minimum = float(np.min(values[finite, column]))
        value_range = float(np.max(values[finite, column]) - minimum)
        scaled[finite, column] = 0.0
        if value_range > 0:
            scaled[finite, column] = (values[finite, column] - minimum) / value_range
    return scaled


def _colour_scale(
    data: NDArray[Any],
) -> NDArray[np.double]:
    return np.round(255.0 * _minmax_scale(data, axis=0))


def _colour_scale_g(
    data: NDArray[Any],
) -> NDArray[np.double]:
    return np.round(255.0 * _minmax_scale(data, axis=None))


def _save_figure(fig: Figure, output: Path) -> None:
    """Save one figure and add the target path to write errors."""
    try:
        fig.savefig(output)
    except Exception as exc:
        raise SerializationError(f"Could not write image file '{output}'.") from exc


def _add_legend_if_present(ax: Axes) -> None:
    """Add a legend only when the plot has labeled artists."""
    handles, _ = ax.get_legend_handles_labels()
    if handles:
        ax.legend()


def _draw_sources(
    z: NDArray[Any],
    s: NDArray[np.str_],
    output: Path,
) -> None:
    source_labels = np.unique(s)
    num_sources = len(source_labels)

    cmap = plt.colormaps["viridis"]
    fig, ax2 = plt.subplots()
    try:
        ax: Axes = ax2
        ax.set_xlim((-5, 5))
        ax.set_ylim((-5, 5))
        fig.suptitle("Sources")

        denominator = max(num_sources - 1, 1)
        for i in reversed(range(num_sources)):
            ax.scatter(
                z[s == source_labels[i], 0],
                z[s == source_labels[i], 1],
                s=8,
                color=cmap(i / denominator),
                label=source_labels[i],
            )

        ax.set_xlabel(r"$z_{1}$")
        ax.set_ylabel(r"$z_{2}$")
        _add_legend_if_present(ax)
        _save_figure(fig, output)
    finally:
        plt.close(fig)


def _draw_scatter(
    z: NDArray[Any],
    x: NDArray[Any],
    title_label: str,
    output: Path,
) -> None:
    values = np.asarray(x, dtype=np.double)
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        lower_bound, upper_bound = 0.0, 1.0
    else:
        lower_bound = float(np.min(finite_values))
        upper_bound = float(np.max(finite_values))
        if lower_bound == upper_bound:
            delta = max(abs(lower_bound) * 0.01, 0.5)
            lower_bound -= delta
            upper_bound += delta

    cmap = plt.colormaps["viridis"].copy()
    cmap.set_bad("#bdbdbd")
    fig, ax2 = plt.subplots()
    try:
        ax: Axes = ax2
        fig.suptitle(title_label, size=14)
        norm = Normalize(lower_bound, upper_bound)

        finite = np.isfinite(values)
        if np.any(finite):
            ax.scatter(
                z[finite, 0],
                z[finite, 1],
                s=8,
                c=values[finite],
                norm=norm,
                cmap=cmap,
            )
        if np.any(~finite):
            ax.scatter(
                z[~finite, 0],
                z[~finite, 1],
                s=8,
                color="#bdbdbd",
            )
        ax.set_xlim((-5, 5))
        ax.set_ylim((-5, 5))
        ax.set_xlabel(r"$z_{1}$")
        ax.set_ylabel(r"$z_{2}$")
        fig.colorbar(
            plt.cm.ScalarMappable(
                norm=norm,
                cmap=cmap,
            ),
            ax=ax,
        )
        _save_figure(fig, output)
    finally:
        plt.close(fig)


def _draw_portfolio_selections(
    z: NDArray[Any],
    p: NDArray[Any],
    algorithm_labels: NDArray[np.str_],
    title_label: str,
    output: Path,
) -> None:
    """Plot a portfolio encoded with zero-based indices and ``-1`` for none."""
    num_algorithms = len(algorithm_labels)
    cmap = plt.colormaps["viridis"]
    fig, ax2 = plt.subplots()
    try:
        ax: Axes = ax2
        ax.set_xlim((-5, 5))
        ax.set_ylim((-5, 5))
        fig.suptitle(title_label)

        denominator = max(num_algorithms, 1)
        for selection in range(-1, num_algorithms):
            selected = p == selection
            if not np.any(selected):
                continue

            ax.scatter(
                z[selected, 0],
                z[selected, 1],
                s=8,
                color=cmap((selection + 1) / denominator),
                label=(
                    "None"
                    if selection == -1
                    else algorithm_labels[selection].replace("_", " ")
                ),
            )

        ax.set_xlabel(r"$z_{1}$")
        ax.set_ylabel(r"$z_{2}$")
        _add_legend_if_present(ax)
        _save_figure(fig, output)
    finally:
        plt.close(fig)


def _draw_portfolio_footprint(
    z: NDArray[Any],
    best: list[Footprint],
    p: NDArray[Any],
    algorithm_labels: NDArray[np.str_],
    output: Path,
) -> None:
    """Plot best footprints for a zero-based portfolio with ``-1`` for none."""
    num_algorithms = len(algorithm_labels)

    cmap = plt.colormaps["viridis"]
    fig, ax2 = plt.subplots()
    try:
        ax: Axes = ax2
        ax.set_xlim((-5, 5))
        ax.set_ylim((-5, 5))
        fig.suptitle("Portfolio footprints")

        denominator = max(num_algorithms, 1)
        for selection in range(-1, num_algorithms):
            selected = p == selection
            if not np.any(selected):
                continue

            colour = cmap((selection + 1) / denominator)
            ax.scatter(
                z[selected, 0],
                z[selected, 1],
                s=8,
                color=colour,
                label=(
                    "None"
                    if selection == -1
                    else algorithm_labels[selection].replace("_", " ")
                ),
            )

            if selection >= 0:
                _draw_footprint(ax, best[selection], colour, 0.3)

        ax.set_xlabel(r"$z_{1}$")
        ax.set_ylabel(r"$z_{2}$")
        _add_legend_if_present(ax)
        _save_figure(fig, output)
    finally:
        plt.close(fig)


def _draw_good_bad_footprint(
    z: NDArray[Any],
    good: Footprint,
    y_bin: NDArray[Any],
    title_label: str,
    output: Path,
) -> None:
    orange = (1.0, 0.6471, 0.0, 1.0)
    blue = (0.0, 0.0, 1.0, 1.0)

    fig, ax2 = plt.subplots()
    try:
        ax: Axes = ax2
        fig.suptitle(title_label)
        ax.set_xlim((-5, 5))
        ax.set_ylim((-5, 5))
        not_y_bin = y_bin == 0
        good_y_bin = y_bin == 1

        if np.any(not_y_bin):
            ax.scatter(
                z[not_y_bin, 0],
                z[not_y_bin, 1],
                s=8,
                c=[orange],
                label="BAD",
            )

        if np.any(good_y_bin):
            ax.scatter(
                z[good_y_bin, 0],
                z[good_y_bin, 1],
                s=8,
                c=[blue],
                label="GOOD",
            )
            _draw_footprint(ax, good, blue, 0.3)

        ax.set_xlabel(r"$z_{1}$")
        ax.set_ylabel(r"$z_{2}$")
        _add_legend_if_present(ax)
        _save_figure(fig, output)
    finally:
        plt.close(fig)


def _draw_footprint(
    ax: Axes,
    footprint: Footprint,
    colour: tuple[float, float, float, float],
    alpha: float,
) -> None:
    if footprint.polygon is None or footprint.polygon.is_empty:
        return

    parts = (
        footprint.polygon.geoms
        if isinstance(footprint.polygon, MultiPolygon)
        else [footprint.polygon]
    )
    for part in parts:
        oriented = orient(part, sign=1.0)
        vertices: list[tuple[float, float]] = []
        codes: list[int] = []
        for ring in [oriented.exterior, *oriented.interiors]:
            ring_coordinates = list(ring.coords)
            if not ring_coordinates:
                continue
            vertices.extend((float(x), float(y)) for x, y in ring_coordinates)
            codes.extend(
                [
                    int(MatplotlibPath.MOVETO),
                    *([int(MatplotlibPath.LINETO)] * (len(ring_coordinates) - 2)),
                    int(MatplotlibPath.CLOSEPOLY),
                ],
            )

        if vertices:
            path = MatplotlibPath(np.asarray(vertices), np.asarray(codes))
            ax.add_patch(
                PathPatch(
                    path,
                    facecolor=colour,
                    edgecolor=colour,
                    alpha=alpha,
                ),
            )


def _draw_binary_performance(
    z: NDArray[Any],
    y_bin: NDArray[Any],
    title_label: str,
    output: Path,
) -> None:
    orange = (1.0, 0.6471, 0.0, 1.0)
    blue = (0.0, 0.0, 1.0, 1.0)

    fig, ax2 = plt.subplots()
    try:
        ax: Axes = ax2
        fig.suptitle(title_label)
        ax.set_xlim((-5, 5))
        ax.set_ylim((-5, 5))
        not_y_bin = y_bin == 0
        good_y_bin = y_bin == 1

        if np.any(not_y_bin):
            ax.scatter(z[not_y_bin, 0], z[not_y_bin, 1], s=8, c=[orange], label="BAD")

        if np.any(good_y_bin):
            ax.scatter(z[good_y_bin, 0], z[good_y_bin, 1], s=8, c=[blue], label="GOOD")

        ax.set_xlabel(r"$z_{1}$")
        ax.set_ylabel(r"$z_{2}$")
        _add_legend_if_present(ax)
        _save_figure(fig, output)
    finally:
        plt.close(fig)


def save_instance_space_output_mat(
    output_directory: Path,
    data: Data,
) -> None:
    """Offline dashboard only use the algo labels from the data."""
    output = output_directory / "model.mat"
    try:
        savemat(
            output,
            {"data": {"algolabels": np.array(data.algo_labels)}},
        )
    except Exception as exc:
        raise SerializationError(f"Could not write MAT file '{output}'.") from exc
