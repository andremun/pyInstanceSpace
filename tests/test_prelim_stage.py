from pathlib import Path

import numpy as np
import pandas as pd

from matilda.data.options import SelvarsOptions
from matilda.stages.prelim_stage import PrelimStage
from matilda.data.options import PrelimOptions, SelvarsOptions

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

csv_path_prelim_input_filter = (
    script_dir / "test_data/prelim/input/filter/model_data_input.mat"
)

# input data
x_input = pd.read_csv(csv_path_x_input, header=None).to_numpy()
y_input = pd.read_csv(csv_path_y_input, header=None).to_numpy()

prelim_opts = PrelimOptions(
    abs_perf=True,
    beta_threshold=0.5500,
    epsilon=0.2000,
    max_perf=False,
    bound=True,
    norm=True
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


