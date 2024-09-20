from pathlib import Path

import numpy as np
import pandas as pd

from matilda.data.options import SelvarsOptions
from matilda.stages.prelim_stage import PrelimStage

script_dir = Path(__file__).parent

csv_path_x_input = script_dir / "test_data/prelim/input/filter/input_X.csv"
csv_path_y_input = script_dir / "test_data/prelim/input/filter/input_Y.csv"
csv_path_p_input = script_dir / "test_data/prelim/input/filter/input_P.csv"
csv_path_x_raw_input = script_dir / "test_data/prelim/input/filter/input_Xraw.csv"
csv_path_y_raw_input = script_dir / "test_data/prelim/input/filter/input_Yraw.csv"
csv_path_beta_input = script_dir / "test_data/prelim/input/filter/input_beta.csv"
csv_path_num_good_algos_input = (
    script_dir / "test_data/prelim/input/filter/input_numGoodAlgos.csv"
)
csv_path_y_best_input = script_dir / "test_data/prelim/input/filter/input_Ybest.csv"
csv_path_y_bin_input = script_dir / "test_data/prelim/input/filter/input_Ybin.csv"
csv_path_y_raw_input = script_dir / "test_data/prelim/input/filter/input_Yraw.csv"
csv_path_beta_input = script_dir / "test_data/prelim/input/filter/input_beta.csv"
csv_path_num_good_algos_input = (
    script_dir / "test_data/prelim/input/filter/input_numGoodAlgos.csv"
)
csv_path_y_best_input = script_dir / "test_data/prelim/input/filter/input_Ybest.csv"
csv_path_y_bin_input = script_dir / "test_data/prelim/input/filter/input_Ybin.csv"

abs_perf = (True,)
beta_threshold = (0.5500,)
epsilon = (0.2000,)
max_perf = (False,)
bound = (True,)
norm = (True,)

x_input = pd.read_csv(csv_path_x_input, header=None).to_numpy()
y_input = pd.read_csv(csv_path_y_input, header=None).to_numpy()
p_input = pd.read_csv(csv_path_p_input, header=None).to_numpy()
x_raw_input = pd.read_csv(csv_path_x_raw_input, header=None).to_numpy()
y_raw_input = pd.read_csv(csv_path_y_raw_input, header=None).to_numpy()
beta_input = pd.read_csv(csv_path_beta_input, header=None).to_numpy()
num_good_algos_input = pd.read_csv(
    csv_path_num_good_algos_input,
    header=None,
).to_numpy()
y_best_input = pd.read_csv(csv_path_y_best_input, header=None).to_numpy()
y_bin_input = pd.read_csv(csv_path_y_bin_input, header=None).to_numpy()

abs_perf = (True,)
beta_threshold = (0.5500,)
epsilon = (0.2000,)
max_perf = (False,)
bound = (True,)
norm = True

subset_index_output_fractional = (
    pd.read_csv(
        script_dir / "test_data/prelim/output/filter/output_subsetIndex_fractional.csv",
        header=None,
    )
    .to_numpy()
    .flatten()
)


def test_filter_post_prelim_fractional() -> None:
    selvars = SelvarsOptions(
        small_scale_flag=True,
        small_scale=0.5,
        file_idx_flag=False,
        file_idx="",
        feats=None,
        algos=None,
        selvars_type="Ftr&Good",
        density_flag=False,
        min_distance=0.1,
    )

    prelim = PrelimStage(
        x_input,
        y_input,
        max_perf,
        abs_perf,
        epsilon,
        beta_threshold,
        bound,
        norm,
        selvars.small_scale_flag,
        selvars.small_scale,
        selvars.file_idx_flag,
        selvars.file_idx,
        selvars.feats,
        selvars.algos,
        selvars.selvars_type,
        selvars.density_flag,
        selvars.min_distance,
    )

    (subset_index, x, y, x_raw, y_raw, y_bin, beta, num_good_algos, y_best, p) = (
        prelim._filter_post_prelim(
            x_input,
            y_input,
            y_bin_input,
            y_best_input,
            x_raw_input,
            y_raw_input,
            p_input,
            num_good_algos_input,
            beta_input,
            selvars.small_scale_flag,
            selvars.small_scale,
            selvars.file_idx_flag,
            selvars.file_idx,
            selvars.feats,
            selvars.algos,
            selvars.selvars_type,
            selvars.min_distance,
            selvars.density_flag,
        )
    )
    subset_index = subset_index.astype(int)

    print("Actual subset_index", subset_index)
    print("Expected subset_index_output_fractional", subset_index_output_fractional)

    assert np.array_equal(subset_index, subset_index_output_fractional)
