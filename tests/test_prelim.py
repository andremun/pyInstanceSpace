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

from matilda.data.option import PrelimOptions
from matilda.stages.prelim import Prelim

script_dir = Path(__file__).parent

csv_path_x_input = script_dir / "test_data/prelim/input/model-data-x-input.csv"
csv_path_y_input = script_dir / "test_data/prelim/input/model-data-y.csv"

csv_path_beta = script_dir / "test_data/prelim/output/model-data-beta.csv"
csv_path_num_good_algos = script_dir / "test_data/prelim/output/model-data-numGoodAlgos.csv"
csv_path_p = script_dir / "test_data/prelim/output/model-data-p.csv"
csv_path_ybest = script_dir / "test_data/prelim/output/model-data-ybest.csv"
csv_path_ybin = script_dir / "test_data/prelim/output/model-data-ybin.csv"
csv_path_x_output = script_dir / "test_data/prelim/output/model-data-x.csv"
csv_path_y_output = script_dir / "test_data/prelim/output/model-data-y.csv"
csv_path_x_output_after_bound = script_dir / "test_data/prelim/output/model-data-x-after-bound.csv"

csv_path_prelim_output_hi_bound = script_dir / "test_data/prelim/output/model-prelim-hibound.csv"
csv_path_prelim_output_iq_range = script_dir / "test_data/prelim/output/model-prelim-iqrange.csv"
csv_path_prelim_output_med_val = script_dir / "test_data/prelim/output/model-prelim-medval.csv"
csv_path_prelim_output_lo_bound = script_dir / "test_data/prelim/output/model-prelim-lobound.csv"
csv_path_prelim_output_lambda_x = script_dir / "test_data/prelim/output/model-prelim-lambdaX.csv"
csv_path_prelim_output_min_x = script_dir / "test_data/prelim/output/model-prelim-minX.csv"
csv_path_prelim_output_lambda_y = script_dir / "test_data/prelim/output/model-prelim-lambdaY.csv"
csv_path_prelim_output_min_y = script_dir / "test_data/prelim/output/model-prelim-minY.csv"
csv_path_prelim_output_mu_x = script_dir / "test_data/prelim/output/model-prelim-muX.csv"
csv_path_prelim_output_mu_y = script_dir / "test_data/prelim/output/model-prelim-muY.csv"
csv_path_prelim_output_sigma_x = script_dir / "test_data/prelim/output/model-prelim-sigmaX.csv"
csv_path_prelim_output_sigma_y = script_dir / "test_data/prelim/output/model-prelim-sigmaY.csv"

# input data
x_input = pd.read_csv(csv_path_x_input, header=None).to_numpy()
y_input = pd.read_csv(csv_path_y_input, header=None).to_numpy()

opts = PrelimOptions(
    abs_perf=1,
    beta_threshold=0.5500,
    epsilon=0.2000,
    max_perf=0,
    bound=1,
    norm=1,
)

def test_bound() -> None:
    """Test the removal of outliers from the feature matrix."""
    prelim_hi_bound = np.genfromtxt(csv_path_prelim_output_hi_bound, delimiter=",")
    prelim_lo_bound = np.genfromtxt(csv_path_prelim_output_lo_bound, delimiter=",")
    prelim_med_val = np.genfromtxt(csv_path_prelim_output_med_val, delimiter=",")
    prelim_iq_range = np.genfromtxt(csv_path_prelim_output_iq_range, delimiter=",")
    prelim_x_after_bound = np.genfromtxt(csv_path_x_output_after_bound, delimiter=",")

    prelim = Prelim(x_input, y_input, opts)
    x, med_val, iq_range, hi_bound, lo_bound = prelim.bound(x_input)

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

    _, prelim_out = Prelim.run(x_input, y_input, opts)

    assert np.allclose(prelim_out.lambda_x, prelim_lambda_x)
    assert np.allclose(prelim_out.min_x, prelim_min_x)
    assert np.allclose(prelim_out.mu_x, prelim_mu_x)
    assert np.allclose(prelim_out.sigma_x, prelim_sigma_x)
    assert np.allclose(prelim_out.lambda_y, prelim_lambda_y)
    assert np.allclose(prelim_out.min_y, prelim_min_y)
    assert np.allclose(prelim_out.mu_y, prelim_mu_y)
    assert np.allclose(prelim_out.sigma_y, prelim_sigma_y)

def test_prelim() -> None:
    """Test the Prelim run method for the values of the data.model."""
    beta_output = pd.read_csv(csv_path_beta, sep=",", header=None).iloc[:, 0].values
    p_output = pd.read_csv(csv_path_p, sep=",", header=None).iloc[:, 0].values
    ybest_output = pd.read_csv(csv_path_ybest, sep=",", header=None).iloc[:, 0].values
    ybin_output = pd.read_csv(csv_path_ybin, sep=",", header=None)
    num_good_algos_output = pd.read_csv(csv_path_num_good_algos, header=None, sep=",")
    x_output = pd.read_csv(csv_path_x_output, header=None).to_numpy()
    y_output = pd.read_csv(csv_path_y_output, header=None).to_numpy()

    data, _ = Prelim.run(x_input, y_input, opts)

    assert np.allclose(data.x, x_output)
    assert np.allclose(data.y, y_output)
    assert np.allclose(data.y_bin, ybin_output)
    assert np.allclose(np.array(data.y_best).flatten(), ybest_output)
    assert np.allclose(data.p, p_output)
    assert np.allclose(data.num_good_algos, num_good_algos_output.values.flatten())
    assert np.allclose(data.beta, beta_output)
