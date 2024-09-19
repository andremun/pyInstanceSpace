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
import pytest

from matilda.data.options import CloisterOptions
from matilda.stages.cloister import CloisterStage

script_dir = Path(__file__).parent


class CloisterMatlabInputs:
    """Class to store MATLAB input data for cloister tests."""

    def __init__(self) -> None:
        """Initialize the input data for the cloister tests."""
        csv_path_x = script_dir / "test_data/cloister/input/input_x.csv"
        csv_path_a = script_dir / "test_data/cloister/input/input_a.csv"

        self.input_x = np.genfromtxt(csv_path_x, delimiter=",")
        self.input_a = np.genfromtxt(csv_path_a, delimiter=",")


class CloisterMatlabOutput:
    """Class to store MATLAB output data for cloister tests."""

    def __init__(self) -> None:
        """Initialize the output data for the cloister tests."""
        csv_path_rho = script_dir / "test_data/cloister/output/rho.csv"
        csv_path_rho_zero = script_dir / "test_data/cloister/output/rho_zero_pval.csv"
        csv_path_z_edge = script_dir / "test_data/cloister/output/z_edge.csv"
        csv_path_z_ecorr = script_dir / "test_data/cloister/output/z_ecorr.csv"
        csv_path_x_edge = script_dir / "test_data/cloister/output/x_edge.csv"
        csv_path_remove = script_dir / "test_data/cloister/output/remove.csv"
        csv_path_index = script_dir / "test_data/cloister/output/index.csv"

        self.rho = np.genfromtxt(csv_path_rho, delimiter=",")
        self.rho_zero = np.genfromtxt(csv_path_rho_zero, delimiter=",")
        self.z_edge = np.genfromtxt(csv_path_z_edge, delimiter=",")
        self.z_ecorr = np.genfromtxt(csv_path_z_ecorr, delimiter=",")
        self.x_edge = np.genfromtxt(csv_path_x_edge, delimiter=",")
        self.remove = np.genfromtxt(csv_path_remove, delimiter=",")
        self.index = np.genfromtxt(csv_path_index, delimiter=",")


class TestCloister:
    """Test module for Cloister class to verify its functionality."""

    @pytest.fixture()
    def input_data(self) -> CloisterMatlabInputs:
        """Fixture to initialize MATLAB input data for cloister tests."""
        return CloisterMatlabInputs()

    @pytest.fixture()
    def output_data(self) -> CloisterMatlabOutput:
        """Fixture to initialize MATLAB output data for cloister tests."""
        return CloisterMatlabOutput()

    def test_correlation_calculation(
        self,
        input_data: CloisterMatlabInputs,
        output_data: CloisterMatlabOutput,
    ) -> None:
        """Test correlation calculation against MATLAB's correlation output.

        Compare the calculated rho value between MATLAB's and Python's computation
        using same pval value.
        """
        input_x = input_data.input_x
        options = CloisterOptions.default()

        rho_python = CloisterStage._compute_correlation(
            input_x, options,
        )
        rho_matlab = output_data.rho

        assert np.allclose(rho_matlab, rho_python)

    def test_correlation_calculation_boundary(
        self,
        input_data: CloisterMatlabInputs,
        output_data: CloisterMatlabOutput,
    ) -> None:
        """Test with pval value being zero.

        This will test both on point and off point boundary.
        """
        input_x = input_data.input_x
        option = CloisterOptions(p_val=0, c_thres=0.7)

        rho_python = CloisterStage._compute_correlation(input_x, option)  # noqa: SLF001
        rho_matlab = output_data.rho_zero

        assert np.allclose(rho_matlab, rho_python)

    def test_decimal_to_binary(
        self,
        input_data: CloisterMatlabInputs,
        output_data: CloisterMatlabOutput,
    ) -> None:
        """Test generating binary matrix from decimal against MATLAB's de2bi ouput.

        Compare the generated binary matrix between MATLAB's de2bi function and custom
        function implemented in Python. Python's matrix should be 1 less than MATLAB's
        output since MATLAB use 1 base indexing while python use 0 base indexing.
        """
        input_x = input_data.input_x
        nfeats = input_x.shape[1]

        index_python = CloisterStage._decimal_to_binary_matrix(nfeats)  # noqa: SLF001
        index_matlab = output_data.index

        assert np.all(index_matlab == index_python + 1)

    def test_decimal_to_binary_with_empty_x(self) -> None:
        """Test generating binary matrix with empty input."""
        empty_x = np.empty((0, 0))
        nfeats = empty_x.shape[1]

        index = CloisterStage._decimal_to_binary_matrix(nfeats)  # noqa: SLF001

        assert index.shape == (1, 1)
        assert index[0, 0] == 0

    def test_run(
        self,
        input_data: CloisterMatlabInputs,
        output_data: CloisterMatlabOutput,
    ) -> None:
        """Test run methods correctly run analysis from start to end.

        The test also test for convex hull calculation with valid input. The z_edge and
        z_ecorr output from MATLAB's convhull produce circular ouput, containing
        duplicated point for start value and have different ordering compared to Scipy's
        ConvexHull ouput. Thus, for the purpose of testing, MATLAB's ouput has been
        reordered.
        """
        input_x = input_data.input_x
        input_a = input_data.input_a
        options = CloisterOptions.default()

        cloister = CloisterStage(input_x, input_a)

        z_edge_python, z_ecorr_python = cloister._run(options)  # noqa: SLF001
        z_edge_matlab = output_data.z_edge
        z_ecorr_matlab = output_data.z_ecorr

        assert np.allclose(z_edge_matlab, z_edge_python)
        assert np.allclose(z_ecorr_matlab, z_ecorr_python)

    def test_convex_hull_qhull_error(self) -> None:
        """Test convex hull function properly handles qhull error."""
        points_collinear = np.array([[0, 0], [1, 1], [2, 2]])
        output = CloisterStage._compute_convex_hull(points_collinear)  # noqa: SLF001
        assert output.size == 0

    def test_convex_hull_value_error(self) -> None:
        """Test convex hull function properly handles value error."""
        points_one_dimension = np.array([[1], [2], [3]])
        output = CloisterStage._compute_convex_hull(
            points_one_dimension,
        )
        assert output.size == 0

    def test_boundary_generation(
        self,
        input_data: CloisterMatlabInputs,
        output_data: CloisterMatlabOutput,
    ) -> None:
        """Test boundary generation against MATLAB's output.

        Compare the z_edge and z_ecorr vaules obtained from the function with MATLAB's.
        output to verify Python implementation produce out within acceptable range.
        """
        input_x = input_data.input_x
        options = CloisterOptions.default()

        rho = CloisterStage._compute_correlation(input_x, options)  # noqa: SLF001

        x_edge_python, remove_python = (
            CloisterStage._generate_boundaries(  # noqa: SLF001
                input_x,
                rho,
                options,
            )
        )
        x_edge_matlab = output_data.x_edge
        remove_matlab = output_data.remove

        assert np.allclose(x_edge_matlab, x_edge_python)
        assert np.all(remove_matlab == remove_python)

    def test_boundary_generation_cthres_boundary(
        self,
        input_data: CloisterMatlabInputs,
        output_data: CloisterMatlabOutput,
    ) -> None:
        """Test cthres boundary."""
        csv_path_rho = script_dir / "test_data/cloister/input/rho_boundary.csv"
        rho_boundary = np.genfromtxt(csv_path_rho, delimiter=",")

        input_x = input_data.input_x
        options = CloisterOptions.default()

        _, remove = CloisterStage._generate_boundaries(
            input_x,
            rho_boundary,
            options,
        )
        remove_matlab = output_data.remove

        assert np.all(remove_matlab == remove)
