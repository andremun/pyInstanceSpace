"""Test module for Prelim class to verify its functionality.

The file contains multiple unit tests to ensure that the `Prelim` class corretly
performs its tasks. The basic mechanism of the test is to compare its output against
output from MATLAB and check if the outputs are the same or reasonable similar.

Tests include:
- Value of feature matrix after removing extreme outliers.
-- Verifying the values for IQR, median, upper and lower bounds.
- Normalisation of the feature matrix and performance matrix.
-- Verifying the values for lambda, min, mu, and sigma.
- Verifying the values of the data.model after running the Prelim class.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from instancespace.data.options import PrelimOptions, SelvarsOptions
from instancespace.stages.prelim import PrelimInput, PrelimStage

script_dir = Path(__file__).parent

csv_path_x_input = script_dir / "test_data/prelim/input/model-data-x-input.csv"
csv_path_y_input = script_dir / "test_data/prelim/input/model-data-y.csv"

csv_path_beta = script_dir / "test_data/prelim/output/model-data-beta.csv"
csv_path_num_good_algos = (
    script_dir / "test_data/prelim/output/model-data-numGoodAlgos.csv"
)
csv_path_p = script_dir / "test_data/prelim/output/model-data-p.csv"
csv_path_ybest = script_dir / "test_data/prelim/output/model-data-ybest.csv"
csv_path_ybin = script_dir / "test_data/prelim/output/model-data-ybin.csv"
csv_path_x_output = script_dir / "test_data/prelim/output/model-data-x.csv"
csv_path_y_output = script_dir / "test_data/prelim/output/model-data-y.csv"
csv_path_x_output_after_bound = (
    script_dir / "test_data/prelim/output/model-data-x-after-bound.csv"
)
csv_path_prelim_output_hi_bound = (
    script_dir / "test_data/prelim/output/model-prelim-hibound.csv"
)
csv_path_prelim_output_iq_range = (
    script_dir / "test_data/prelim/output/model-prelim-iqrange.csv"
)
csv_path_prelim_output_med_val = (
    script_dir / "test_data/prelim/output/model-prelim-medval.csv"
)
csv_path_prelim_output_lo_bound = (
    script_dir / "test_data/prelim/output/model-prelim-lobound.csv"
)
csv_path_prelim_output_lambda_x = (
    script_dir / "test_data/prelim/output/model-prelim-lambdaX.csv"
)
csv_path_prelim_output_min_x = (
    script_dir / "test_data/prelim/output/model-prelim-minX.csv"
)
csv_path_prelim_output_lambda_y = (
    script_dir / "test_data/prelim/output/model-prelim-lambdaY.csv"
)
csv_path_prelim_output_min_y = (
    script_dir / "test_data/prelim/output/model-prelim-minY.csv"
)
csv_path_prelim_output_mu_x = (
    script_dir / "test_data/prelim/output/model-prelim-muX.csv"
)
csv_path_prelim_output_mu_y = (
    script_dir / "test_data/prelim/output/model-prelim-muY.csv"
)
csv_path_prelim_output_sigma_x = (
    script_dir / "test_data/prelim/output/model-prelim-sigmaX.csv"
)
csv_path_prelim_output_sigma_y = (
    script_dir / "test_data/prelim/output/model-prelim-sigmaY.csv"
)

csv_path_prelim_input_x_raw = (
    script_dir / "test_data/prelim/fractional/before/Xraw_split.txt"
)

csv_path_prelim_input_y_raw = (
    script_dir / "test_data/prelim/fractional/before/Yraw_split.txt"
)

csv_path_prelim_input_p = script_dir / "test_data/prelim/fractional/before/P_split.txt"

csv_path_prelim_inst_labels = (
    script_dir / "test_data/prelim/fractional/before/instlabels_split.txt"
)
# input data
x_input = pd.read_csv(csv_path_x_input, header=None).to_numpy()
y_input = pd.read_csv(csv_path_y_input, header=None).to_numpy()
x_raw = np.genfromtxt(csv_path_prelim_input_x_raw, delimiter=",")
y_raw = np.genfromtxt(csv_path_prelim_input_y_raw, delimiter=",")
s: pd.Series | None = None  # type: ignore[type-arg]
inst_labels = np.genfromtxt(csv_path_prelim_inst_labels, delimiter=",")

prelim_opts = PrelimOptions(
    abs_perf=True,
    beta_threshold=0.5500,
    epsilon=0.2000,
    max_perf=False,
    bound=True,
    norm=True,
)

selvars_opts = SelvarsOptions.default()


def test_bound() -> None:
    """Test the removal of outliers from the feature matrix."""
    prelim_hi_bound = np.genfromtxt(csv_path_prelim_output_hi_bound, delimiter=",")
    prelim_lo_bound = np.genfromtxt(csv_path_prelim_output_lo_bound, delimiter=",")
    prelim_med_val = np.genfromtxt(csv_path_prelim_output_med_val, delimiter=",")
    prelim_iq_range = np.genfromtxt(csv_path_prelim_output_iq_range, delimiter=",")
    prelim_x_after_bound = np.genfromtxt(csv_path_x_output_after_bound, delimiter=",")

    prelim = PrelimStage(
        x_input,
        y_input,
        x_raw,
        y_raw,
        s,
        pd.Series(inst_labels),
        prelim_opts,
        selvars_opts,
    )
    prelim_bound = prelim._bound()  # noqa: SLF001
    x = prelim_bound.x
    hi_bound = prelim_bound.hi_bound
    lo_bound = prelim_bound.lo_bound
    med_val = prelim_bound.med_val
    iq_range = prelim_bound.iq_range

    assert np.allclose(x, prelim_x_after_bound)
    assert np.allclose(hi_bound, prelim_hi_bound)
    assert np.allclose(lo_bound, prelim_lo_bound)
    assert np.allclose(med_val, prelim_med_val)
    assert np.allclose(iq_range, prelim_iq_range)


def test_normalise() -> None:
    """Test the normalisation of the feature matrix and performance matrix."""
    prelim_lambda_x = np.genfromtxt(csv_path_prelim_output_lambda_x, delimiter=",")
    prelim_min_x = np.genfromtxt(csv_path_prelim_output_min_x, delimiter=",")
    prelim_mu_x = np.genfromtxt(csv_path_prelim_output_mu_x, delimiter=",")
    prelim_sigma_x = np.genfromtxt(csv_path_prelim_output_sigma_x, delimiter=",")
    prelim_lambda_y = np.genfromtxt(csv_path_prelim_output_lambda_y, delimiter=",")
    prelim_min_y = np.genfromtxt(csv_path_prelim_output_min_y, delimiter=",").item()
    prelim_mu_y = np.genfromtxt(csv_path_prelim_output_mu_y, delimiter=",")
    prelim_sigma_y = np.genfromtxt(csv_path_prelim_output_sigma_y, delimiter=",")

    (
        x,
        y,
        y_bin,
        y_best,
        p,
        num_good_algos,
        beta,
        med_val,
        iq_range,
        hi_bound,
        lo_bound,
        min_x,
        lambda_x,
        mu_x,
        sigma_x,
        min_y,
        lambda_y,
        sigma_y,
        mu_y,
    ) = PrelimStage.prelim(
        x_input,
        y_input,
        x_raw,
        y_raw,
        s,
        pd.Series(inst_labels),
        prelim_opts,
        selvars_opts,
    )

    assert np.allclose(lambda_x, prelim_lambda_x)
    assert np.allclose(min_x, prelim_min_x)
    assert np.allclose(mu_x, prelim_mu_x)
    assert np.allclose(sigma_x, prelim_sigma_x)
    assert np.allclose(lambda_y, prelim_lambda_y)
    assert np.allclose(min_y, prelim_min_y)
    assert np.allclose(mu_y, prelim_mu_y)
    assert np.allclose(sigma_y, prelim_sigma_y)


def test_prelim() -> None:
    """Test the Prelim run method for the values of the data.model."""
    beta_output = pd.read_csv(csv_path_beta, sep=",", header=None).iloc[:, 0].values
    p_output = pd.read_csv(csv_path_p, sep=",", header=None).iloc[:, 0].values
    ybest_output = pd.read_csv(csv_path_ybest, sep=",", header=None).iloc[:, 0].values
    ybin_output = pd.read_csv(csv_path_ybin, sep=",", header=None)
    num_good_algos_output = pd.read_csv(csv_path_num_good_algos, header=None, sep=",")
    x_output = pd.read_csv(csv_path_x_output, header=None).to_numpy()
    y_output = pd.read_csv(csv_path_y_output, header=None).to_numpy()

    (
        x,
        y,
        y_bin,
        y_best,
        p,
        num_good_algos,
        beta,
        med_val,
        iq_range,
        hi_bound,
        lo_bound,
        min_x,
        lambda_x,
        mu_x,
        sigma_x,
        min_y,
        lambda_y,
        sigma_y,
        mu_y,
    ) = PrelimStage.prelim(
        x_input,
        y_input,
        x_raw,
        y_raw,
        s,
        pd.Series(inst_labels),
        prelim_opts,
        selvars_opts,
    )

    assert np.allclose(x, x_output)
    assert np.allclose(y, y_output)
    assert np.allclose(y_bin, ybin_output)
    assert np.allclose(
        np.array(y_best).flatten(),
        np.array(ybest_output, dtype=np.float64),
    )
    assert np.allclose(p, np.array(p_output, dtype=np.float64))
    assert np.allclose(num_good_algos, num_good_algos_output.values.flatten())
    assert np.allclose(beta, np.array(beta_output, dtype=bool))


csv_input_prelim_x_run = script_dir / "test_data/prelim/run/input/input_X.csv"
csv_input_prelim_y_run = script_dir / "test_data/prelim/run/input/input_Y.csv"
csv_input_prelim_x_raw_run = script_dir / "test_data/prelim/run/input/input_Xraw.csv"
csv_input_prelim_y_raw_run = script_dir / "test_data/prelim/run/input/input_Yraw.csv"
csv_input_inst_labels_run = (
    script_dir / "test_data/prelim/run/input/input_instlabels.csv"
)

csv_output_prelim_beta_run = script_dir / "test_data/prelim/run/output/output_beta.csv"
csv_output_prelim_num_good_algos_run = (
    script_dir / "test_data/prelim/run/output/output_numGoodAlgos.csv"
)
csv_output_prelim_p_run = script_dir / "test_data/prelim/run/output/output_P.csv"
csv_output_prelim_ybest_run = (
    script_dir / "test_data/prelim/run/output/output_Ybest.csv"
)
csv_output_prelim_ybin_run = script_dir / "test_data/prelim/run/output/output_Ybin.csv"
csv_output_prelim_x_run = script_dir / "test_data/prelim/run/output/output_X.csv"
csv_output_prelim_y_run = script_dir / "test_data/prelim/run/output/output_Y.csv"


def test_prelim_run() -> None:
    """Test the Prelim run method for the values of the data.model."""
    x_input_run = pd.read_csv(csv_input_prelim_x_run, header=None).to_numpy()
    y_input_run = pd.read_csv(csv_input_prelim_y_run, header=None).to_numpy()
    x_raw_run = np.genfromtxt(csv_input_prelim_x_raw_run, delimiter=",")
    y_raw_run = np.genfromtxt(csv_input_prelim_y_raw_run, delimiter=",")
    inst_labels_input_run = np.genfromtxt(csv_input_inst_labels_run, delimiter=",")

    p_output_run = (
        pd.read_csv(csv_output_prelim_p_run, sep=",", header=None).iloc[:, 0].values
    )
    ybest_output_run = (
        pd.read_csv(csv_output_prelim_ybest_run, sep=",", header=None).iloc[:, 0].values
    )
    ybin_output_run = pd.read_csv(csv_output_prelim_ybin_run, sep=",", header=None)

    x_output_run = pd.read_csv(csv_output_prelim_x_run, header=None).to_numpy()
    y_output_run = pd.read_csv(csv_output_prelim_y_run, header=None).to_numpy()

    s: pd.Series | None = None  # type: ignore[type-arg]

    prelim_opts = PrelimOptions(
        abs_perf=True,
        beta_threshold=0.5500,
        epsilon=0.2000,
        max_perf=False,
        bound=True,
        norm=True,
    )

    selvars_opts = SelvarsOptions(
        small_scale_flag=False,
        small_scale=0.50,
        file_idx_flag=False,
        file_idx="",
        feats=None,
        algos=None,
        selvars_type="Ftr&Good",
        min_distance=0.1,
        density_flag=False,
    )

    inputs = PrelimInput(
        x=x_input_run,
        y=y_input_run,
        x_raw=x_raw_run,
        y_raw=y_raw_run,
        s=s,
        inst_labels=pd.Series(inst_labels_input_run),
        prelim_options=prelim_opts,
        selvars_options=selvars_opts,
    )

    (
        med_val,
        iq_range,
        hi_bound,
        lo_bound,
        min_x,
        lambda_x,
        mu_x,
        sigma_x,
        min_y,
        lambda_y,
        sigma_y,
        mu_y,
        x,
        y,
        x_raw,
        y_raw,
        y_bin,
        y_best,
        p,
        num_good_algos,
        beta,
        inst_labels,
        data_dense,
        s,
    ) = PrelimStage._run(  # noqa: SLF001
        inputs,
    )

    assert np.allclose(x.shape, x_output_run.shape)
    assert np.allclose(y, y_output_run)
    assert np.allclose(y_bin, ybin_output_run)
    assert np.allclose(
        np.array(y_best).flatten(),
        np.array(ybest_output_run, dtype=np.float64),
    )
    assert np.allclose(p, np.array(p_output_run, dtype=np.float64))
