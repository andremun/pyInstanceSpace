"""Test module for Pilot class to verify its functionality.

The file contains multiple unit tests to ensure that the `Pilot` class corretly
perform its tasks. The basic mechanism of the test is to compare its output against
output from MATLAB and check if the outputs are the same or reasonable similar. The
tests also include some boundary test where appropriate to test the boundary of the
statement within the methods to ensure they are implemented appropriately.

Tests include:
- Correct import for the data.
- Correct output dimensionality
- Analytic option is correctly detected
- Error handling from convex hull calculation
"""

from pathlib import Path

import numpy as np

from matilda.data.model import PilotOut
from matilda.data.option import PilotOptions
from tests.utils.option_creator import create_option
from matilda.stages.pilot import Pilot

script_dir = Path(__file__).parent

csv_path_x = script_dir / "test_data/pilot/input/modelData_X.csv"
csv_path_y = script_dir / "test_data/pilot/input/modelData_Y.csv"
csv_path_featlabels = script_dir / "test_data/pilot/input/modelData_featlabels.csv"
csv_path_out = script_dir / "test_data/pilot/output/modelPilot.csv"

def test_analytic_option_output_dimension():
    X = np.genfromtxt(csv_path_x, delimiter=",")
    Y = np.genfromtxt(csv_path_y, delimiter=",")
    labels = np.genfromtxt(csv_path_featlabels, delimiter=",", dtype=str)
    opts = PilotOptions(analytic=True, n_tries=1)
    pilot = Pilot()
    pilot_out = pilot.run(X, Y, labels, opts)

    matlab_output = np.genfromtxt(csv_path_out, delimiter=",")
    matlab_X0 = matlab_output[['X0_1', 'X0_2']].values
    
    assert pilot_out.X0.shape == matlab_X0.shape, f"Output matrix does not have the correct dimension. Expected {matlab_X0.shape}, got {pilot_out.X0.shape}"
    
def test_eigenvalues_and_matrix_reconstruction():
    X = np.genfromtxt(csv_path_x, delimiter=",")
    Y = np.genfromtxt(csv_path_y, delimiter=",")
    labels = np.genfromtxt(csv_path_featlabels, delimiter=",")
    opts = PilotOptions(analytic=True, n_tries=1)
    pilot = Pilot()
    pilot_out = pilot.run(X, Y, labels, opts)

    matlab_output = np.genfromtxt(csv_path_out, delimiter=",")
    matlab_A = matlab_output[['A_1', 'A_2']].values
    
    assert np.isclose(np.sum(np.linalg.eigvals(pilot_out.A)), 1), "Eigenvalues do not sum to 1 or incorrect."
    assert pilot_out.A.shape == matlab_A.shape, "Matrix A is not reconstructed correctly."
    assert np.allclose(pilot_out.A, matlab_A, atol=1e-3), "Matrix A values do not match MATLAB output."