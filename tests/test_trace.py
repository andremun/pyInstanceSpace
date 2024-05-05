"""Test module for the Trace class to verify its functionality.

This file contains unit tests for the Trace class.
These tests are intended to validate the behavior of the Trace class using dummy data.
However, without the full implementation of the Trace class,
this testing module is incomplete.

Tests include:
- Verification of the `run` method with dummy data.
"""


import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from matilda.data.option import TraceOptions
from matilda.stages.trace import Trace

script_dir = Path(__file__).parent


csv_z = script_dir/"trace/input/Z.csv"
csv_y_bin = script_dir/"trace/input/Ybin.csv"
csv_p = script_dir/"trace/input/P.csv"
csv_beta = script_dir/"trace/input/Beta.csv"
csv_algo_labels = script_dir/"trace/input/Algolabels.csv"
csv_pi = script_dir/"trace/input/Opts_PI.csv"
csv_usesim = script_dir/"trace/input/Opts_usesim.csv"
csv_summary_with_mock_data = script_dir/"trace/output/summary_with_mock_data.csv"

class TestTraceRun(unittest.TestCase):
    """A test suite for testing the Trace class functionality in the Matilda project.

    This class uses unit tests to ensure the correct behavior of the methods
    including build, contra, and summary operations.
    It relies on mocking to isolate the tests from full implementation of trace class.

    Attributes:
    ----------
        z (np.ndarray): The z data loaded from a CSV file, used in trace operations.
        y_bin (np.ndarray): The binary y data loaded from a CSV file.
        p (np.ndarray): The p data loaded from a CSV file.
        beta (np.ndarray): The beta coefficients loaded from a CSV file.
        algo_labels (list): A list of algorithm labels extracted from a CSV file.
        pi (np.ndarray): The pi values loaded from a CSV file.
        usesim (np.ndarray): The usesim loaded from a CSV file and used in options.
        opts (TraceOptions): The trace options initialized with pi and usesim.

    Methods:
    -------
        setUp(self):
            Set up initial conditions for the Trace class tests.

        test_trace_function(self, mock_summary, mock_contra, mock_build):
            Tests the integrated functionality of the Trace class by simulating a
            complete run and verifying the output against expected results.
    """

    def setUp(self) -> None:
        """Read the initial data provided by matlab."""
        self.z = np.genfromtxt(csv_z,delimiter=",")
        self.y_bin = np.genfromtxt(csv_y_bin, delimiter = ",")
        self.p = np.genfromtxt(csv_p,delimiter=",")
        self.beta = np.genfromtxt(csv_beta,delimiter=",")
        self.algo_labels = pd.read_csv(csv_algo_labels, delimiter=",", header=None,
                                       dtype=str).iloc[0].tolist()
        self.pi = np.genfromtxt(csv_pi, delimiter=",", dtype=float)
        self.usesim = np.genfromtxt(csv_usesim)
        self.opts = TraceOptions(self.pi,self.usesim)

    # TODO: test for tracebuild,tracecontra,tracesummary
    @patch("matilda.stages.trace.Trace.build")
    @patch("matilda.stages.trace.Trace.contra")
    @patch("matilda.stages.trace.Trace.summary")
    def test_trace_function(self, mock_summary:list[float], mock_contra: MagicMock,
                            mock_build: MagicMock) -> None:
        """
        Tests the integrated functionality of the Trace class by calling the run method.

        This test ensures that the `Trace.run` method orchestrates its build, contra,
        and summary phases correctly using mocked components. The method tests the
        integration of these components by checking that the return values from mocks
        are handled correctly and the final format matches the original format of the
        matlab.

        Args:
        ----------
        mock_summary (list[float]): A mock of the Trace.summary method used to simulate
                                the summary phase without executing the real method.
        mock_contra (MagicMock): A mock of the Trace.contra method used to simulate
                                the contra phase without executing the real method.
        mock_build (MagicMock): A mock of the Trace.build method used to simulate
                                the build phase without executing the real method.

        Returns:
        -------
        None: This method does not return a value but asserts conditions.
        """
        # Set return values for the mocked functions
        mock_polygon = MagicMock()

        mock_build.return_value = mock_polygon
        mock_contra.return_value = [mock_polygon,mock_polygon]
        mock_summary.return_value = [0.01,0.02,0.03,0.04,0.05]

        # Call the trace function
        result = Trace.run(self.z, self.y_bin, self.p, self.beta,
                           self.algo_labels, self.opts)

        # Index_col is the list of algo_labels which set to unnamed for consistency
        # with matlab format

        # Summary.csv is the output of the summary function generated with sample option
        # and metadata

        summary_from_csv = pd.read_csv(csv_summary_with_mock_data,
                                       index_col="Unnamed: 0")

        assert result.summary.equals(summary_from_csv), "The DataFrames do not match."

        #TODO: Add assertions

