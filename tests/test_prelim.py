import csv
import numpy as np
import pandas as pd
import os

script_dir = os.path.dirname(__file__)

csv_path_x = os.path.join(script_dir, "prelim/input/model-data-x.csv")
csv_path_y = os.path.join(script_dir, "prelim/input/model-data-y.csv")
csv_path_beta = os.path.join(script_dir, "prelim/output/model-data-beta.csv")

x_input = np.genfromtxt(csv_path_x, delimiter=",")
y_input = np.genfromtxt(csv_path_y, delimiter=",")
beta_output = pd.read_csv(csv_path_beta, sep=",").iloc[:, 0].values

abs_perf = 1
beta_threshold = 0.5500
epsilon = 0.2000
max_perf = 0
bound = 1
norm = 1

def test_prelim() -> None:
    """
    The test case for demonstration.

    Returns
    -------
        None

    """
    # Call the prelim function
    from matilda.prelim import prelim
    beta = prelim(x_input, y_input, abs_perf, beta_threshold, epsilon, max_perf, bound, norm)

    # print(beta)
    # print(beta_output)
    

    assert np.allclose(beta, beta_output), "The output from the function is not as expected."
    assert True, "Something is wrong with the Github Workflow - please contact"
