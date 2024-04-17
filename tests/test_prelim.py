from pathlib import Path

import numpy as np
import pandas as pd

from matilda.data.option import PrelimOptions

script_dir = Path(__file__).parent

csv_path_x_input = script_dir / "prelim/input/model-data-x.csv"
csv_path_y_input = script_dir / "prelim/input/model-data-y.csv"

csv_path_beta = script_dir / "prelim/output/model-data-beta.csv"
csv_path_numGoodAlgos = script_dir / "prelim/output/model-data-numGoodAlgos.csv"
csv_path_p = script_dir / "prelim/output/model-data-p.csv"
csv_path_ybest = script_dir / "prelim/output/model-data-ybest.csv"
csv_path_ybin = script_dir / "prelim/output/model-data-ybin.csv"
csv_path_x_output = script_dir / "prelim/output/model-data-x.csv"
csv_path_y_output = script_dir / "prelim/output/model-data-y.csv"

# input data
x_input = np.genfromtxt(csv_path_x_input, delimiter=",")
y_input = np.genfromtxt(csv_path_y_input, delimiter=",")

# expected outputs
beta_output = pd.read_csv(csv_path_beta, sep=",").iloc[:, 0].values
p_output = pd.read_csv(csv_path_p, sep=",").iloc[:, 0].values
ybest_output = pd.read_csv(csv_path_ybest, sep=",").iloc[:, 0].values
ybin_output = pd.read_csv(csv_path_ybin, sep=",")
numGoodAlgos_output = pd.read_csv(csv_path_numGoodAlgos, sep=",")
x_output = np.genfromtxt(csv_path_x_output, delimiter=",")
y_output = np.genfromtxt(csv_path_y_output, delimiter=",")


opts = PrelimOptions(
    abs_perf=1,
    beta_threshold=0.5500,
    epsilon=0.2000,
    max_perf=0,
    bound=1,
    norm=1,
)

# passes
# def test_prelim_beta() -> None:
#     # Call the prelim function
#     from matilda.prelim import prelim
#     beta = prelim(x_input, y_input, opts)

#     # print(beta)
#     # print(beta_output)

#     assert np.allclose(beta, beta_output)


# fails
# def test_prelim_p() -> None:
#     # Call the prelim function
#     from matilda.prelim import prelim
#     p = prelim(x_input, y_input, opts)

#     # print(p)
#     # print(p_output)

#     assert np.allclose(p, p_output)


# passes
# def test_prelim_ybest() -> None:
#     # Call the prelim function
#     from matilda.prelim import prelim
#     ybest = prelim(x_input, y_input, opts)

#     print(ybest)
#     print(ybest_output)

#     assert np.allclose(ybest, ybest_output)


# passes
# def test_prelim_numGoodAlgos() -> None:
#     # Call the prelim function
#     from matilda.prelim import prelim
#     numGoodAlgos = prelim(x_input, y_input, opts)

#     print(numGoodAlgos)
#     print(numGoodAlgos_output)

#     assert np.allclose(numGoodAlgos, numGoodAlgos_output)


# def test_prelim_ybin() -> None:
#     # Call the prelim function
#     from matilda.prelim import prelim
#     ybin = prelim(x_input, y_input, opts)

#     print(ybin)
#     print(ybin_output)

#     assert np.allclose(ybin, ybin_output)

