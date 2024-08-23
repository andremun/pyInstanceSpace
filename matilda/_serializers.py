from pathlib import Path
from typing import Any, TypeVar

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap, Normalize
from numpy.typing import NDArray

from matilda.data.model import (
    CloisterOut,
    Data,
    Footprint,
    PilotOut,
    PrelimOut,
    PythiaOut,
    SiftedOut,
    StageState,
    TraceOut,
)

from .data.options import InstanceSpaceOptions


def save_instance_space_to_csv(
    output_directory: Path,
    data: Data,
    sifted_state: StageState[SiftedOut],
    trace_state: StageState[TraceOut],
    pilot_state: StageState[PilotOut],
    cloister_state: StageState[CloisterOut],
    pythia_state: StageState[PythiaOut],
) -> None:
    if not output_directory.is_dir():
        raise ValueError("output_directory isn't a directory.")

    num_algorithms = data.y.shape[1]

    for i in range(num_algorithms):
        best = trace_state.out.best[i]
        if best is not None and best.polygon is not None:
            best_vertices = np.array(best.polygon.boundary.coords.xy).T[:-1]
            best = trace_state.out.best[i]
            algorithm_labels = trace_state.data.algo_labels[i]
            _write_array_to_csv(
                best_vertices,
                pd.Series(["z_1", "z_2"]),
                _make_bind_labels(best_vertices),
                output_directory / f"footprint_{algorithm_labels}_best.csv",
            )

        good = trace_state.out.good[i]
        if good is not None and good.polygon is not None:
            good_vertices = np.array(good.polygon.boundary.coords.xy).T[:-1]
            algorithm_labels = trace_state.data.algo_labels[i]
            _write_array_to_csv(
                good_vertices,
                pd.Series(["z_1", "z_2"]),
                _make_bind_labels(good_vertices),
                output_directory / f"footprint_{algorithm_labels}_good.csv",
            )

    _write_array_to_csv(
        pilot_state.out.z,
        pd.Series(["z_1", "z_2"]),
        pilot_state.data.inst_labels,
        output_directory / "coordinates.csv",
    )

    if cloister_state is not None:
        _write_array_to_csv(
            cloister_state.out.z_edge,
            pd.Series(["z_1", "z_2"]),
            _make_bind_labels(cloister_state.out.z_edge),
            output_directory / "bounds.csv",
        )
        _write_array_to_csv(
            cloister_state.out.z_ecorr,
            pd.Series(["z_1", "z_2"]),
            _make_bind_labels(cloister_state.out.z_ecorr),
            output_directory / "bounds_prunned.csv",
        )

    _write_array_to_csv(
        data.x_raw[:, sifted_state.out.idx],
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
        pythia_state.out.y_hat,
        pd.Series(data.algo_labels),
        data.inst_labels,
        output_directory / "algorithm_svm.csv",
    )
    _write_array_to_csv(
        pythia_state.out.selection0,
        pd.Series(["Best_Algorithm"]),
        data.inst_labels,
        output_directory / "portfolio_svm.csv",
    )
    _write_cell_to_csv(
        trace_state.out.summary[1:, [2, 4, 5, 7, 9, 10]],
        trace_state.out.summary[0, [2, 4, 5, 7, 9, 10]],
        trace_state.out.summary[1:, 0],
        output_directory / "footprint_performance.csv",
    )
    if pilot_state.out.summary is not None:
        _write_cell_to_csv(
            pilot_state.out.summary[1:, 1:],
            pilot_state.out.summary[0, 1:],
            pilot_state.out.summary[1:, 0],
            output_directory / "projection_matrix.csv",
        )
    _write_cell_to_csv(
        pythia_state.out.summary[1:, 1:],
        pythia_state.out.summary[0, 1:],
        pythia_state.out.summary[1:, 0],
        output_directory / "svm_table.csv",
    )


def save_instance_space_for_web(
    output_directory: Path,
    prelim_state: StageState[PrelimOut],
    sifted_state: StageState[SiftedOut],
) -> None:
    if not output_directory.is_dir():
        raise ValueError("output_directory isn't a directory.")


    colours = (
        np.array(matplotlib.colormaps["viridis"].resampled(256).colors)[:, :3] * 255
    ).astype(np.int_)
    pd.DataFrame(colours, columns=["R", "G", "B"]).to_csv(
        output_directory / "color_table.csv",
        index_label=False,
    )

    _write_array_to_csv(
        _colour_scale(prelim_state.data.x_raw[:, sifted_state.out.idx]),
        pd.Series(prelim_state.data.feat_labels),
        prelim_state.data.inst_labels,
        output_directory / "feature_raw_color.csv",
    )
    _write_array_to_csv(
        _colour_scale(prelim_state.data.y_raw),
        pd.Series(prelim_state.data.algo_labels),
        prelim_state.data.inst_labels,
        output_directory / "algorithm_raw_single_color.csv",
    )
    _write_array_to_csv(
        _colour_scale(prelim_state.data.x),
        pd.Series(prelim_state.data.feat_labels),
        prelim_state.data.inst_labels,
        output_directory / "feature_process_color.csv",
    )
    _write_array_to_csv(
        _colour_scale(prelim_state.data.y),
        pd.Series(prelim_state.data.algo_labels),
        prelim_state.data.inst_labels,
        output_directory / "algorithm_process_single_color.csv",
    )
    _write_array_to_csv(
        _colour_scale_g(prelim_state.data.y_raw),
        pd.Series(prelim_state.data.algo_labels),
        prelim_state.data.inst_labels,
        output_directory / "algorithm_raw_color.csv",
    )
    _write_array_to_csv(
        _colour_scale_g(prelim_state.data.y),
        pd.Series(prelim_state.data.algo_labels),
        prelim_state.data.inst_labels,
        output_directory / "algorithm_process_color.csv",
    )
    _write_array_to_csv(
        _colour_scale_g(prelim_state.data.num_good_algos),
        pd.Series(["NumGoodAlgos"]),
        prelim_state.data.inst_labels,
        output_directory / "good_algos_color.csv",
    )


def save_instance_space_graphs(
    output_directory: Path,
    data: Data,
    options: InstanceSpaceOptions,
    pythia_state: StageState[PythiaOut],
    pilot_state: StageState[PilotOut],
    trace_state: StageState[TraceOut],
) -> None:
    if not output_directory.is_dir():
        raise ValueError("output_directory isn't a directory.")

    num_feats = data.x.shape[1]
    num_algorithms = data.y.shape[1]

    x_range = np.max(data.x, axis=0) - np.min(data.x, axis=0)
    x_aux = (data.x - np.min(data.x, axis=0)) / x_range

    y_raw_range = np.max(data.y_raw, axis=0) - np.min(data.y_raw, axis=0)
    y_ind = data.y_raw - np.min(data.y_raw, axis=0) / y_raw_range

    y_glb = np.log10(data.y_raw + 1)
    y_glb_range = np.max(y_glb, axis=0) - np.min(y_glb, axis=0)
    y_glb = (y_glb - np.min(y_glb)) / y_glb_range

    if options.trace.use_sim:
        y_foot = pythia_state.out.y_hat
        p_foot = pythia_state.out.selection0
    else:
        y_foot = data.y_bin
        p_foot = data.p

    for i in range(num_feats):
        filename = f"distribution_feature_{data.feat_labels[i]}.png"
        _draw_scatter(
            pilot_state.out.z,
            x_aux[:, i],
            data.feat_labels[i].replace("_", " "),
            output_directory / filename,
        )

    for i in range(num_algorithms):
        algo_label = data.algo_labels[i]

        filename = f"distribution_performance_global_normalized_{algo_label}.png"
        _draw_scatter(
            pilot_state.out.z,
            y_glb[:, i],
            algo_label.replace("_", " "),
            output_directory / filename,
        )

        filename = f"distribution_performance_individual_normalized_{algo_label}.png"
        _draw_scatter(
            pilot_state.out.z,
            y_ind[:, i],
            algo_label.replace("_", " "),
            output_directory / filename,
        )

        _draw_binary_performance(
            pilot_state.out.z,
            data.y_bin[:, i],
            algo_label.replace("_", " "),
            output_directory / f"binary_performance_{algo_label}.png",
        )

        # TODO: MATLAB has a try catch for this one, when pythia is done maybe make
        # optional? in model?
        _draw_binary_performance(
            pilot_state.out.z,
            pythia_state.out.y_hat,
            algo_label.replace("_", " "),
            output_directory / f"binary_svm_{algo_label}.png",
        )

        # TODO: Same as above
        _draw_good_bad_footprint(
            pilot_state.out.z,
            trace_state.out.good[i],
            y_foot,
            algo_label.replace("_", " "),
            output_directory / f"footprint_{algo_label}.png",
        )

    _draw_scatter(
        pilot_state.out.z,
        data.num_good_algos / num_algorithms,
        "Percentage of good algorithms",
        output_directory / "distribution_number_good_algos.png",
    )

    _draw_portfolio_selections(
        pilot_state.out.z,
        data.p,
        np.array(data.algo_labels),
        "Best algorithm",
        output_directory / "distribution_portfolio.png",
    )

    _draw_portfolio_selections(
        pilot_state.out.z,
        pythia_state.out.selection0,
        np.array(data.algo_labels),
        "Predicted best algorithm",
        output_directory / "distribution_svm_portfolio.png",
    )

    _draw_portfolio_footprint(
        pilot_state.out.z,
        trace_state.out.best,
        p_foot,
        np.array(data.algo_labels),
        output_directory / "footprint_portfolio.png",
    )

    _draw_binary_performance(
        pilot_state.out.z,
        data.beta,
        "Beta score",
        output_directory / "distribution_beta_score.png",
    )

    if data.s is not None:
        _draw_sources(
            pilot_state.out.z,
            np.array(data.s),
            output_directory / "distribution_sources.png",
        )


def _write_array_to_csv(
    data: NDArray[Any],  # TODO: Try to unify these
    column_names: pd.Series,  # TODO: Try to unify these
    row_names: pd.Series,  # type: ignore[type-arg]
    filename: Path,
) -> None:
    pd.DataFrame(data, index=row_names, columns=column_names).to_csv(
        filename,
        index_label="Row",
    )


def _write_cell_to_csv(
    data: pd.Series,  # TODO: Try to unify these
    column_names: pd.Series,  # TODO: Try to unify these
    row_names: pd.Series,  # type: ignore[type-arg]
    filename: Path,
) -> None:
    pd.DataFrame(data, index=row_names, columns=column_names).to_csv(
        filename,
        index_label="Row",
    )


def _make_bind_labels(
    data: NDArray[Any],
) -> pd.Series:
    return pd.Series([f"bnd_pnt_{i+1}" for i in range(data.shape[0])])


T = TypeVar("T", bound=np.generic)


def _colour_scale(
    data: NDArray[np.number],
) -> NDArray[np.int_]:
    data_range = np.max(data) - np.min(data)
    return np.floor(
        255 * ((data - np.min(data)) / data_range),
    ).astype(np.int_)


def _colour_scale_g(
    data: NDArray[np.number],
) -> NDArray[np.int_]:
    data_range = np.max(data) - np.min(data)
    return np.round(
        255 * ((data - np.min(data)) / data_range),
    ).astype(np.int_)


def _draw_sources(
    z: NDArray[Any],
    s: NDArray[np.str_],
    output: Path,
) -> None:
    upper_bound = np.ceil(np.max(z))
    lower_bound = np.floor(np.min(z))
    source_labels = np.unique(s)
    num_sources = len(source_labels)

    cmap = plt.colormaps["viridis"]
    fig, ax2 = plt.subplots()
    ax: Axes = ax2  # TODO: Remove this before PR, just for programming
    fig.suptitle("Sources")

    norm = Normalize(lower_bound, upper_bound)

    for i in reversed(range(num_sources)):
        ax.scatter(
            z[s == source_labels[i], 0],
            z[s == source_labels[i], 1],
            s=8,
            c=source_labels[i],
            norm=norm,
            cmap=cmap,
        )

    ax.set_xlabel("z_{1}")
    ax.set_ylabel("z_{2}")
    ax.legend()

    fig.savefig(output)


def _draw_scatter(
    z: NDArray[Any],
    x: NDArray[Any],
    title_label: str,
    output: Path,
) -> None:
    upper_bound = np.ceil(np.max(z))
    lower_bound = np.floor(np.min(z))

    cmap = plt.colormaps["viridis"]
    fig, ax2 = plt.subplots()
    ax: Axes = ax2  # TODO: Remove this before PR, just for programming
    fig.suptitle(title_label, size=14)

    norm = Normalize(lower_bound, upper_bound)

    ax.scatter(z[:, 0], z[:, 1], s=8, c=x, norm=norm, cmap=cmap)
    ax.set_xlabel("z_{1}")
    ax.set_ylabel("z_{2}")
    fig.colorbar(
        plt.cm.ScalarMappable(
            norm=norm,
            cmap=cmap,
        ),
    )

    fig.savefig(output)


def _draw_portfolio_selections(
    z: NDArray[Any],
    p: NDArray[Any],
    algorithm_labels: NDArray[np.str_],
    title_label: str,
    output: Path,
) -> None:
    upper_bound = np.ceil(np.max(z))
    lower_bound = np.floor(np.min(z))
    num_algorithms = len(algorithm_labels)
    # labels: list[str] = []
    # h = np.zeros((1, num_algorithms + 1))

    bsxfun_result = np.array(
        [[x == j for j in range(num_algorithms + 1)] for i, x in enumerate(p)],
    )
    is_worthy = np.sum(bsxfun_result) != 0

    cmap = plt.colormaps["viridis"]
    fig, ax2 = plt.subplots()
    ax: Axes = ax2  # TODO: Remove this before PR, just for programming
    fig.suptitle(title_label)

    norm = Normalize(lower_bound, upper_bound)

    for i in range(num_algorithms):
        if not is_worthy[i]:
            continue

        ax.scatter(
            z[p == i, 0],
            z[p == i, 1],
            s=8,
            c=i,
            norm=norm,
            cmap=cmap,
            label="None" if i == 0 else algorithm_labels[i - 1].replace("_", " "),
        )

    ax.set_xlabel("z_{1}")
    ax.set_ylabel("z_{2}")
    ax.legend()

    fig.savefig(output)


def _draw_portfolio_footprint(
    z: NDArray[Any],
    best: list[Footprint],
    p: NDArray[Any],
    algorithm_labels: NDArray[np.str_],
    output: Path,
) -> None:
    upper_bound = np.ceil(np.max(z))
    lower_bound = np.floor(np.min(z))
    num_algorithms = len(algorithm_labels)

    bsxfun_result = np.array(
        [[x == j for j in range(num_algorithms + 1)] for i, x in enumerate(p)],
    )
    is_worthy = np.sum(bsxfun_result) != 0

    cmap = plt.colormaps["viridis"]
    fig, ax2 = plt.subplots()
    ax: Axes = ax2  # TODO: Remove this before PR, just for programming
    fig.suptitle("Portfolio footprints")

    norm = Normalize(lower_bound, upper_bound)

    for i in range(num_algorithms):
        if not is_worthy[i]:
            continue

        ax.scatter(
            z[p == i, 0],
            z[p == i, 1],
            s=8,
            c=i,
            norm=norm,
            cmap=cmap,
            label="None" if i == 0 else algorithm_labels[i - 1].replace("_", " "),
        )

        _draw_footprint(ax, best[i], cmap(norm(i)), 0.3)

    ax.set_xlabel("z_{1}")
    ax.set_ylabel("z_{2}")
    ax.legend()

    fig.savefig(output)


def _draw_good_bad_footprint(
    z: NDArray[Any],
    good: Footprint,
    y_bin: NDArray[Any],
    title_label: str,
    output: Path,
) -> None:
    orange = (1.0, 0.6471, 0.0, 1.0)
    blue = (0.0, 0.0, 1.0, 1.0)

    labels = ["GOOD", "BAD"]

    fig, ax2 = plt.subplots()
    ax: Axes = ax2  # TODO: Remove this before PR, just for programming
    fig.suptitle(title_label)

    if np.any(not y_bin):
        ax.scatter(
            z[not y_bin, 0],
            z[not y_bin, 1],
            s=8,
            c=orange,
        )

    if np.any(y_bin):
        ax.scatter(
            z[y_bin, 0],
            z[y_bin, 1],
            s=8,
            c=blue,
        )
        _draw_footprint(ax, good, blue, 0.3)

    ax.set_xlabel("z_{1}")
    ax.set_ylabel("z_{2}")
    ax.legend()

    fig.savefig(output)


def _draw_footprint(
    ax: Axes,
    footprint: Footprint,
    colour: tuple[float, float, float, float],
    alpha: float,
) -> None:
    # TODO: Blockered on TRACE
    pass


def _draw_binary_performance(
    z: NDArray[Any],
    y_bin: NDArray[Any],
    title_label: str,
    output: Path,
) -> None:
    orange = (1.0, 0.6471, 0.0, 1.0)
    blue = (0.0, 0.0, 1.0, 1.0)

    labels = ["GOOD", "BAD"]

    fig, ax2 = plt.subplots()
    ax: Axes = ax2  # TODO: Remove this before PR, just for programming
    fig.suptitle(title_label)

    if np.any(not y_bin):
        ax.scatter(
            z[not y_bin, 0],
            z[not y_bin, 1],
            s=8,
            c=orange,
        )

    if np.any(y_bin):
        ax.scatter(
            z[y_bin, 0],
            z[y_bin, 1],
            s=8,
            c=blue,
        )

    ax.set_xlabel("z_{1}")
    ax.set_ylabel("z_{2}")
    ax.legend()

    fig.savefig(output)
