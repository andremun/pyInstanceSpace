"""Test module for Cloister class to verify its functionality.

The file contains multiple unit tests to ensure that the `Cloister` class corretly
perform its tasks. The basic mechanism of the test is to compare its output against
output from MATLAB and check if the outputs are the same or reasonable similar. The
tests also include some boundary test where appropriate to test the boundary of the
statement within the methods to ensure they are implemented appropriately.

Tests include:
- Correlation calculations and boundary test.
- Generating binary matrix from decimal
- Running analysis from start to end
- Error handling from convex hull calculation
- Boundary generation and the boundary test.
"""

from pathlib import Path

import numpy as np

from matilda.data.option import CloisterOptions
from matilda.stages.cloister import Cloister

script_dir = Path(__file__).parent

csv_path_x = script_dir / "test_data/cloister/input/input_x.csv"
csv_path_a = script_dir / "test_data/cloister/input/input_a.csv"

input_x = np.genfromtxt(csv_path_x, delimiter=",")
input_a = np.genfromtxt(csv_path_a, delimiter=",")

default_option = CloisterOptions(p_val=0.05, c_thres=0.7)


def test_correlation_calculation() -> None:
    """Test correlation calculation against MATLAB's correlation output.

    Compare the calculated rho value between MATLAB's and Python's computation
    using same pval value.
    """
    csv_path_rho = script_dir / "test_data/cloister/output/rho.csv"
    rho_matlab = np.genfromtxt(csv_path_rho, delimiter=",")

    cloister = Cloister(input_x, input_a, default_option)
    rho = cloister.compute_correlation()

    assert np.allclose(rho_matlab, rho)


def test_correlation_calculation_boundary() -> None:
    """Test with pval value being zero.

    This will test both on point and off point boundary.
    """
    csv_path_rho = script_dir / "test_data/cloister/output/rho_zero_pval.csv"
    rho_matlab = np.genfromtxt(csv_path_rho, delimiter=",")

    option = CloisterOptions(p_val=0, c_thres=0.7)
    cloister = Cloister(input_x, input_a, option)
    rho = cloister.compute_correlation()

    assert np.allclose(rho_matlab, rho)


def test_decimal_to_binary() -> None:
    """Test generating binary matrix from decimal against MATLAB's de2bi ouput.

    Compare the generated binary matrix between MATLAB's de2bi function and custom
    function implemented in Python. Python's matrix should be 1 less than MATLAB's
    output since MATLAB use 1 base indexing while python use 0 base indexing.
    """
    csv_path_index = script_dir / "test_data/cloister/output/index.csv"
    index_matlab = np.genfromtxt(csv_path_index, delimiter=",")

    cloister = Cloister(input_x, input_a, default_option)
    index = cloister.decimal_to_binary_matrix()

    assert np.all(index_matlab == index + 1)


def test_decimal_to_binary_with_empty_x() -> None:
    """Test generating binary matrix with empty input."""
    empty_x = np.empty((0, 0))
    cloister = Cloister(empty_x, input_a, default_option)
    index = cloister.decimal_to_binary_matrix()

    assert index.shape == (1, 1)
    assert index[0, 0] == 0


def test_run() -> None:
    """Test run methods correctly run analysis from start to end.

    The test also test for convex hull calculation with valid input. The z_edge and
    z_ecorr output from MATLAB's convhull produce circular ouput, containing duplicated
    point for start value and have different ordering compared to Scipy's ConvexHull
    ouput. Thus, for the purpose of testing, MATLAB's ouput has been reordered.
    """
    csv_path_z_edge = script_dir / "test_data/cloister/output/z_edge.csv"
    csv_path_z_ecorr = script_dir / "test_data/cloister/output/z_ecorr.csv"
    z_edge_matlab = np.genfromtxt(csv_path_z_edge, delimiter=",")
    z_ecorr_matlab = np.genfromtxt(csv_path_z_ecorr, delimiter=",")

    z_edge, z_ecorr = Cloister.run(input_x, input_a, default_option)

    assert np.allclose(z_edge_matlab, z_edge)
    assert np.allclose(z_ecorr_matlab, z_ecorr)


def test_convex_hull_qhull_error() -> None:
    """Test convex hull function properly handles qhull error."""
    points_collinear = np.array([[0, 0], [1, 1], [2, 2]])
    cloister = Cloister(input_x, input_a, default_option)
    output = cloister.compute_convex_hull(points_collinear)
    assert output.size == 0


def test_convex_hull_value_error() -> None:
    """Test convex hull function properly handles value error."""
    points_one_dimension = np.array([[1], [2], [3]])
    cloister = Cloister(input_x, input_a, default_option)
    output = cloister.compute_convex_hull(points_one_dimension)
    assert output.size == 0


def test_boundary_generation() -> None:
    """Test boundary generation against MATLAB's output.

    Compare the z_edge and z_ecorr vaules obtained from the function with MATLAB's.
    output to verify Python implementation produce out within acceptable range.
    """
    csv_path_x_edge = script_dir / "test_data/cloister/output/x_edge.csv"
    csv_path_remove = script_dir / "test_data/cloister/output/remove.csv"
    x_edge_matlab = np.genfromtxt(csv_path_x_edge, delimiter=",")
    remove_matlab = np.genfromtxt(csv_path_remove, delimiter=",")

    cloister = Cloister(input_x, input_a, default_option)
    rho = cloister.compute_correlation()
    x_edge, remove = cloister.generate_boundaries(rho)

    assert np.allclose(x_edge_matlab, x_edge)
    assert np.all(remove_matlab == remove)


def test_boundary_generation_cthres_boundary() -> None:
    """Test cthres boundary."""
    csv_path_remove = script_dir / "test_data/cloister/output/remove.csv"
    remove_matlab = np.genfromtxt(csv_path_remove, delimiter=",")

    csv_path_rho = script_dir / "test_data/cloister/input/rho_boundary.csv"
    rho = np.genfromtxt(csv_path_rho, delimiter=",")

    cloister = Cloister(input_x, input_a, default_option)
    _, remove = cloister.generate_boundaries(rho)

    assert np.all(remove_matlab == remove)
