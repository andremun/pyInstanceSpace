from pathlib import Path
from typing import Any, TypeVar

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from matilda.data.model import (
    CloisterOut,
    Data,
    PilotOut,
    PrelimOut,
    PythiaOut,
    StageState,
    TraceOut,
)


def save_instance_space_to_csv(
    output_directory: Path,
    data: Data,
    prelim_state: StageState[PrelimOut],
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
            best = trace_state.out.best[i]
            algorithm_labels = trace_state.data.algo_labels[i]
            _write_array_to_csv(
                best.polygon.vertices,
                pd.Series(["z_1", "z_2"]),
                _make_bind_labels(best.polygon.vertices),
                output_directory / f"footprint_{algorithm_labels}_good.csv",
            )

        good = trace_state.out.good[i]
        if good is not None and good.polygon is not None:
            algorithm_labels = trace_state.data.algo_labels[i]
            _write_array_to_csv(
                good.polygon.vertices,
                pd.Series(["z_1", "z_2"]),
                _make_bind_labels(good.polygon.vertices),
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
        data.x_raw[:, prelim_state.out.idx],
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
        pd.Series(data.feat_labels),
        data.inst_labels,
        output_directory / "algorithm_raw.csv",
    )
    _write_array_to_csv(
        data.y,
        pd.Series(data.feat_labels),
        data.inst_labels,
        output_directory / "algorithm_process.csv",
    )
    _write_array_to_csv(
        data.y_bin,
        pd.Series(data.feat_labels),
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
        trace_state.out.summary[2:, [3, 5, 6, 8, 10, 11]],
        trace_state.out.summary[1, [3, 5, 6, 8, 10, 11]],
        trace_state.out.summary[2:, 1],
        output_directory / "footprint_performance.csv",
    )
    if pilot_state.out.summary is not None:
        _write_cell_to_csv(
            pilot_state.out.summary[2:, 2:],
            pilot_state.out.summary[1, 2:],
            pilot_state.out.summary[2:, 1],
            output_directory / "footprint_performance.csv",
        )
    _write_cell_to_csv(
        pythia_state.out.summary[2:, 2:],
        pythia_state.out.summary[1, 2:],
        pythia_state.out.summary[2:, 1],
        output_directory / "svm_table.csv",
    )

def save_instance_space_for_web(
    output_directory: Path,
    prelim_state: StageState[PrelimOut],
) -> None:

    if not output_directory.is_dir():
        raise ValueError("output_directory isn't a directory.")


    _write_array_to_csv(
        _colour_scale(prelim_state.data.x_raw[:, prelim_state.out.idx]),
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



def _write_array_to_csv(
    data: NDArray[Any], # TODO: Try to unify these
    column_names: pd.Series, # TODO: Try to unify these
    row_names: pd.Series, # type: ignore[type-arg]
    filename: Path,
) -> None:
    pd.DataFrame(data, index=row_names, columns=column_names).to_csv(filename)

def _write_cell_to_csv(
    data: pd.Series, # TODO: Try to unify these
    column_names: pd.Series, # TODO: Try to unify these
    row_names: pd.Series, # type: ignore[type-arg]
    filename: Path,
) -> None:
    pd.DataFrame(data, index=row_names, columns=column_names).to_csv(filename)

def _make_bind_labels(
    data: NDArray[Any],
) -> pd.Series:
    return pd.Series([f"bnd_pnt_{i}" for i in range(data.shape[0])])

T = TypeVar("T", bound=np.generic)
def _colour_scale(
    data: NDArray[T],
) -> NDArray[T]:
    data_range: NDArray[T] = np.max(data, axis=0) - np.min(data, axis=0)
    out: NDArray[T] = np.round(255 * (data - np.min(data, axis=0)) / data_range)
    return out

def _colour_scale_g(
    data: NDArray[T],
) -> NDArray[T]:
    data_range: NDArray[T] = np.max(data, axis=0) - np.min(data, axis=0)
    out: NDArray[T] = np.round(255 * (data - np.min(data, axis=0)) / data_range)
    return out
