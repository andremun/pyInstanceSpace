from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from numpy.typing import NDArray

from matilda.data.model import Data, PrelimOut
from matilda.data.option import PrelimOptions
from matilda.stages.prelim import Prelim

script_dir = Path(__file__).parent

csv_path_x_input = script_dir / "prelim/input/model-data-x-input.csv"
csv_path_y_input = script_dir / "prelim/input/model-data-y.csv"

csv_path_beta = script_dir / "prelim/output/model-data-beta.csv"
csv_path_numGoodAlgos = script_dir / "prelim/output/model-data-numGoodAlgos.csv"
csv_path_p = script_dir / "prelim/output/model-data-p.csv"
csv_path_ybest = script_dir / "prelim/output/model-data-ybest.csv"
csv_path_ybin = script_dir / "prelim/output/model-data-ybin.csv"
csv_path_x_output = script_dir / "prelim/output/model-data-x.csv"
csv_path_y_output = script_dir / "prelim/output/model-data-y.csv"
csv_path_x_output_after_bound = script_dir / "prelim/output/model-data-x-after-bound.csv"

csv_path_prelim_output_hi_bound = script_dir / "prelim/output/model-prelim-hibound.csv"
csv_path_prelim_output_iq_range = script_dir / "prelim/output/model-prelim-iqrange.csv"
csv_path_prelim_output_med_val = script_dir / "prelim/output/model-prelim-medval.csv"
csv_path_prelim_output_lo_bound = script_dir / "prelim/output/model-prelim-lobound.csv"
csv_path_prelim_output_lambda_x = script_dir / "prelim/output/model-prelim-lambdaX.csv"
csv_path_prelim_output_min_x = script_dir / "prelim/output/model-prelim-minX.csv"
csv_path_prelim_output_lambda_y = script_dir / "prelim/output/model-prelim-lambdaY.csv"
csv_path_prelim_output_min_y = script_dir / "prelim/output/model-prelim-minY.csv"
csv_path_prelim_output_mu_x = script_dir / "prelim/output/model-prelim-muX.csv"
csv_path_prelim_output_mu_y = script_dir / "prelim/output/model-prelim-muY.csv"
csv_path_prelim_output_sigma_x = script_dir / "prelim/output/model-prelim-sigmaX.csv"
csv_path_prelim_output_sigma_y = script_dir / "prelim/output/model-prelim-sigmaY.csv"

# input data
x_input = pd.read_csv(csv_path_x_input, header=None).to_numpy()
y_input = pd.read_csv(csv_path_y_input, header=None).to_numpy()

print("y_input shapre:", y_input.shape)
print("y_input first row:", y_input[0, :])
print("y_input second row:", y_input[1, :])

# expected outputs
beta_output = pd.read_csv(csv_path_beta, sep=",").iloc[:, 0].values
p_output = pd.read_csv(csv_path_p, sep=",").iloc[:, 0].values
ybest_output = pd.read_csv(csv_path_ybest, sep=",").iloc[:, 0].values
ybin_output = pd.read_csv(csv_path_ybin, sep=",")
numGoodAlgos_output = pd.read_csv(csv_path_numGoodAlgos, sep=",")
x_output = pd.read_csv(csv_path_x_output).to_numpy()
y_output = pd.read_csv(csv_path_y_output).to_numpy()
# x_output_after_bound = pd.read_csv(csv_path_x_output_after_bound).to_numpy()

# prelim_hi_bound = pd.read_csv(csv_path_prelim_output_hi_bound, header=None).to_numpy()
# prelim_iq_range = pd.read_csv(csv_path_prelim_output_iq_range, header=None).to_numpy().flatten().tolist()
# prelim_med_val = pd.read_csv(csv_path_prelim_output_med_val, header=None).to_numpy().flatten().tolist()
# prelim_lo_bound = pd.read_csv(csv_path_prelim_output_lo_bound, header=None).to_numpy().flatten().tolist()
# prelim_lambda_x = pd.read_csv(csv_path_prelim_output_lambda_x, header=None).to_numpy().flatten().tolist()
# prelim_min_x = pd.read_csv(csv_path_prelim_output_min_x, header=None).to_numpy().flatten().tolist()
# prelim_lambda_y = pd.read_csv(csv_path_prelim_output_lambda_y, header=None).to_numpy().flatten().tolist()
# prelim_min_y = pd.read_csv(csv_path_prelim_output_min_y, header=None).to_numpy().flatten().tolist()
# prelim_mu_x = pd.read_csv(csv_path_prelim_output_mu_x, header=None).to_numpy().flatten().tolist()
# prelim_mu_y = pd.read_csv(csv_path_prelim_output_mu_y, header=None).to_numpy().flatten().tolist()
# prelim_sigma_x = pd.read_csv(csv_path_prelim_output_sigma_x, header=None).to_numpy().flatten().tolist()
# prelim_sigma_y = pd.read_csv(csv_path_prelim_output_sigma_y, header=None).to_numpy().flatten().tolist()

# prelim_min_x = np.genfromtxt(csv_path_prelim_output_min_x, delimiter=",")
# print("----------")
# print(prelim_min_x)
# print(type(prelim_min_x))
# prelim_min_y = np.genfromtxt(csv_path_prelim_output_min_y, delimiter=",")
# print("----------")
# print(prelim_min_y)
# print(type(prelim_min_y.item()))
# print(prelim_min_y.shape)

# print(x_input.shape)
# print(y_input.shape)
# print(type(prelim_hi_bound))
# print(prelim_hi_bound.shape)

opts = PrelimOptions(
    abs_perf=1,
    beta_threshold=0.5500,
    epsilon=0.2000,
    max_perf=0,
    bound=1,
    norm=1,
)

# @pytest.fixture()
# def run_prelim() -> tuple[Data, PrelimOut, NDArray[np.double]]:
#     return prelim(x_input, y_input, opts)

def test_bound() -> None:
    """Add docstring."""
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
    """Add docstring."""
    prelim_lambda_x = np.genfromtxt(csv_path_prelim_output_lambda_x, delimiter=",")
    prelim_min_x = np.genfromtxt(csv_path_prelim_output_min_x, delimiter=",")
    prelim_mu_x = np.genfromtxt(csv_path_prelim_output_mu_x, delimiter=",")
    prelim_sigma_x = np.genfromtxt(csv_path_prelim_output_sigma_x, delimiter=",")
    prelim_lambda_y = np.genfromtxt(csv_path_prelim_output_lambda_y, delimiter=",")
    prelim_min_y = np.genfromtxt(csv_path_prelim_output_min_y, delimiter=",").item()
    prelim_mu_y = np.genfromtxt(csv_path_prelim_output_mu_y, delimiter=",")
    prelim_sigma_y = np.genfromtxt(csv_path_prelim_output_sigma_y, delimiter=",")

    _, prelim_out = Prelim.run(x_input, y_input, opts)

    print("lambda_y:", prelim_out.lambda_y)
    print("min_y:", prelim_out.min_y)
    print("mu_y:", prelim_out.mu_y)
    print("sigma_y:", prelim_out.sigma_y)

    assert np.allclose(prelim_out.lambda_x, prelim_lambda_x)
    assert np.allclose(prelim_out.min_x, prelim_min_x)
    assert np.allclose(prelim_out.mu_x, prelim_mu_x)
    assert np.allclose(prelim_out.sigma_x, prelim_sigma_x)
    assert np.allclose(prelim_out.lambda_y, prelim_lambda_y)
    assert np.allclose(prelim_out.min_y, prelim_min_y)
    assert np.allclose(prelim_out.mu_y, prelim_mu_y)
    assert np.allclose(prelim_out.sigma_y, prelim_sigma_y)


# def test_prelim() -> None:
#     # Call the prelim function
#     # from matilda.prelim import bound, prelim
#     data, prelim_out, x_after_bound_prelim = Prelim(x_input, y_input, opts)

#     # Assert other attributes of the data object
#     assert data.inst_labels == ""
#     assert data.feat_labels == ""
#     assert data.algo_labels == ""
#     # fails
#     # print('data.x', data.x, data.x.shape)
#     # print('x_output', x_output, x_output.shape)
#     # assert np.allclose(data.x, x_output)

#     # fails
#     # print('data.y', data.y)
#     # print('y_output', y_output)
#     # assert np.allclose(data.y, y_output)

#     # fails
#     # print('data.x_raw', data.x_raw)
#     # print('x_output', x_output)
#     # assert np.allclose(data.x_raw, x_output)

#     # fails
#     # print('data.y_raw', data.y_raw)
#     # print('y_output', y_output)
#     # assert np.allclose(data.y_raw, y_output)

#     # fails
#     # print('data.y_bin', data.y_bin)
#     # print('ybin_output', ybin_output)
#     # assert not np.allclose(data.y_bin, ybin_output)

#     # passes 
#     assert np.allclose(np.array(data.y_best).flatten(), ybest_output) # passes but need to flatten output 
#     assert np.allclose(data.p, p_output) # passes
#     assert np.allclose(data.num_good_algos, numGoodAlgos_output.values.flatten()) # passes but need to flatten expected
#     assert np.allclose(data.beta, beta_output) # passes
#     assert data.s is None


#     # Assert other attributes of the prelim_out object

#     # bounding outliers output comparision
#     # assert np.all(np.isclose(prelim_out.med_val, prelim_med_val, atol=0.1)) # passes
#     # incorrect since I am hardcoding the iq_range
#     print('prelim_out.iq_range', prelim_out.iq_range)
#     print('prelim_iq_range', prelim_iq_range)
#     assert np.all(np.isclose(prelim_out.iq_range, prelim_iq_range, atol=0.1)) # fails with stats.iqr and np.percentile

#     # print('prelim_out.hi_bound', prelim_out.hi_bound)
#     # print('prelim_hi_bound', prelim_hi_bound)
#     # assert np.all(np.isclose(prelim_out.hi_bound, prelim_hi_bound, atol=0.01)) # passes
#     # assert np.all(np.isclose(prelim_out.lo_bound, prelim_lo_bound, atol=0.01)) # passes

#     # check if the x_after_bound_prelim is the same as the x_output_after_bound
#     assert np.all(np.isclose(x_after_bound_prelim, x_output_after_bound, atol=0.001)) # passes

#     # normalisation output comparision
#     assert np.all(np.isclose(prelim_out.min_x, prelim_min_x, atol=0.01)) # passes
#     assert np.all(np.isclose(prelim_out.min_y, prelim_min_y, atol=0.01)) # passes

#     # assert np.all(np.isclose(prelim_out.lambda_x, prelim_lambda_x, atol=0.01)) #fails

#     assert np.all(np.isclose(prelim_out.mu_x, prelim_mu_x, atol=0.1)) # passes
#     assert np.all(np.isclose(prelim_out.sigma_x, prelim_sigma_x, atol=0.1)) # passes
#     assert np.all(np.isclose(prelim_out.lambda_y, prelim_lambda_y, atol=0.1)) # passes
#     assert np.all(np.isclose(prelim_out.sigma_y, prelim_sigma_y, atol=0.1)) # passes
#     assert np.all(np.isclose(prelim_out.mu_y, prelim_mu_y, atol=0.1)) # passes
