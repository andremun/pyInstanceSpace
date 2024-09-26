from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib as mpl
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from shapely import MultiPolygon, Polygon

from matilda.data.model import (
    CloisterOut,
    Data,
    FeatSel,
    PilotOut,
    PrelimOut,
    PythiaOut,
    SiftedOut,
    TraceOut,
)


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
        raise ValueError("output_directory isn't a directory.")

    num_algorithms = data.y.shape[1]

    for i in range(num_algorithms):
        best = trace_out.best[i]
        boundaries: NDArray[Any]
        if best is not None and best.polygon is not None:
            boundaries = np.empty((1, 2))
            if isinstance(best.polygon, Polygon):
                # Extract the boundary coordinates of a single Polygon
                x, y = best.polygon.exterior.xy
                boundary_coords = np.array([x, y]).T
                boundaries = np.concatenate((boundaries, boundary_coords))

            elif isinstance(best.polygon, MultiPolygon):
                # Extract the boundary coordinates of each Polygon in MultiPolygon
                for poly in best.polygon.geoms:
                    x, y = poly.exterior.xy
                    boundary_coords = np.array([x, y]).T
                    boundaries = np.concatenate((boundaries, boundary_coords))

            algorithm_labels = data.algo_labels[i]
            _write_array_to_csv(
                boundaries,
                pd.Series(["z_1", "z_2"]),
                _make_bind_labels(boundaries),
                output_directory / f"footprint_{algorithm_labels}_best.csv",
            )

        good = trace_out.good[i]
        if good is not None and good.polygon is not None:
            boundaries = np.empty((1, 2))
            if isinstance(good.polygon, Polygon):
                # Extract the boundary coordinates of a single Polygon
                x, y = good.polygon.exterior.xy
                boundary_coords = np.array([x, y]).T
                boundaries = np.concatenate((boundaries, boundary_coords))

            elif isinstance(good.polygon, MultiPolygon):
                # Extract the boundary coordinates of each Polygon in MultiPolygon
                for poly in good.polygon.geoms:
                    x, y = poly.exterior.xy
                    boundary_coords = np.array([x, y]).T
                    boundaries = np.concatenate((boundaries, boundary_coords))

            algorithm_labels = data.algo_labels[i]
            _write_array_to_csv(
                boundaries,
                pd.Series(["z_1", "z_2"]),
                _make_bind_labels(boundaries),
                output_directory / f"footprint_{algorithm_labels}_good.csv",
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
        data.x_raw[:, sifted_out.idx],
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
    _write_cell_to_csv(
        trace_out.summary[1:, [2, 4, 5, 7, 9, 10]],
        trace_out.summary[0, [2, 4, 5, 7, 9, 10]],
        trace_out.summary[1:, 0],
        output_directory / "footprint_performance.csv",
    )
    if pilot_out.summary is not None:
        _write_cell_to_csv(
            pilot_out.summary[1:, 1:],
            pilot_out.summary[0, 1:],
            pilot_out.summary[1:, 0],
            output_directory / "projection_matrix.csv",
        )
    _write_cell_to_csv(
        pythia_out.summary[1:, 1:],
        pythia_out.summary[0, 1:],
        pythia_out.summary[1:, 0],
        output_directory / "svm_table.csv",
    )


def save_instance_space_for_web(
    output_directory: Path,
    data: Data,
    prelim_state: PrelimOut,
    sifted_state: SiftedOut,
    feat_sel: FeatSel,
) -> None:
    if not output_directory.is_dir():
        raise ValueError("output_directory isn't a directory.")

    colours = (
        np.array(
            mpl.colormaps["viridis"].resampled(256).__dict__["colors"],
        )[:, :3]
        * 255
    ).astype(np.int_)

    pd.DataFrame(colours, columns=["R", "G", "B"]).to_csv(
        output_directory / "color_table.csv",
        index_label=False,
    )

    _write_array_to_csv(
        _colour_scale(data.x_raw[:, feat_sel.idx]),
        pd.Series(data.feat_labels),
        data.inst_labels,
        output_directory / "feature_raw_color.csv",
    )
    _write_array_to_csv(
        _colour_scale(data.y_raw),
        pd.Series(data.algo_labels),
        data.inst_labels,
        output_directory / "algorithm_raw_single_color.csv",
    )
    _write_array_to_csv(
        _colour_scale(data.x),
        pd.Series(data.feat_labels),
        data.inst_labels,
        output_directory / "feature_process_color.csv",
    )
    _write_array_to_csv(
        _colour_scale(data.y),
        pd.Series(data.algo_labels),
        data.inst_labels,
        output_directory / "algorithm_process_single_color.csv",
    )
    _write_array_to_csv(
        _colour_scale_g(data.y_raw),
        pd.Series(data.algo_labels),
        data.inst_labels,
        output_directory / "algorithm_raw_color.csv",
    )
    _write_array_to_csv(
        _colour_scale_g(data.y),
        pd.Series(data.algo_labels),
        data.inst_labels,
        output_directory / "algorithm_process_color.csv",
    )
    _write_array_to_csv(
        _colour_scale_g(data.num_good_algos),
        pd.Series(["NumGoodAlgos"]),
        data.inst_labels,
        output_directory / "good_algos_color.csv",
    )


def _write_array_to_csv(
    data: NDArray[Any],  # TODO: Try to unify these
    column_names: pd.Series[str],  # TODO: Try to unify these
    row_names: pd.Series[str],
    filename: Path,
) -> None:
    pd.DataFrame(data, index=row_names, columns=column_names).to_csv(
        filename,
        index_label="Row",
    )


def _write_cell_to_csv(
    data: pd.Series[Any],  # TODO: Try to unify these
    column_names: pd.Series[str],  # TODO: Try to unify these
    row_names: pd.Series[str],
    filename: Path,
) -> None:
    pd.DataFrame(data, index=row_names, columns=column_names).to_csv(
        filename,
        index_label="Row",
    )


def _make_bind_labels(
    data: NDArray[Any],
) -> pd.Series[str]:
    return pd.Series([f"bnd_pnt_{i+1}" for i in range(data.shape[0])])


def _colour_scale(
    data: NDArray[np._NumberType],
) -> NDArray[np.int_]:
    data_range = np.max(data, axis=0) - np.min(data, axis=0)
    out: NDArray[np.int_] = np.floor(
        255.0 * ((data - np.min(data, axis=0)) / data_range),
    ).astype(np.int_)

    return out


def _colour_scale_g(
    data: NDArray[np._NumberType],
) -> NDArray[np.int_]:
    data_range = np.max(data) - np.min(data)
    out: NDArray[np.int_] = np.round(
        255.0 * ((data - np.min(data)) / data_range),
    ).astype(np.int_)

    return out