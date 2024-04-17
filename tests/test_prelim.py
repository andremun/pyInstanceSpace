from pathlib import Path

import numpy as np
import pandas as pd

from matilda.data.option import PrelimOptions

script_dir = Path(__file__).parent

csv_path_x = script_dir / "prelim/input/model-data-x.csv"
csv_path_y = script_dir / "prelim/input/model-data-y.csv"
csv_path_beta = script_dir / "prelim/output/model-data-beta.csv"

x_input = np.genfromtxt(csv_path_x, delimiter=",")
y_input = np.genfromtxt(csv_path_y, delimiter=",")
beta_output = pd.read_csv(csv_path_beta, sep=",").iloc[:, 0].values

abs_perf = 1
beta_threshold = 0.5500
epsilon = 0.2000
max_perf = 0
bound = 1
norm = 1

opts = PrelimOptions(
    abs_perf=abs_perf,
    beta_threshold=beta_threshold,
    epsilon=epsilon,
    max_perf=max_perf,
    bound=bound,
    norm=norm,
)

def test_prelim() -> None:
    # Call the prelim function
    from matilda.prelim import prelim
    beta = prelim(x_input, y_input, opts)

    # print(beta)
    # print(beta_output)

    assert np.allclose(beta, beta_output)
