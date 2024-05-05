"""Test module for the Trace class to verify its functionality.

This file contains unit tests for the Trace class.
These tests are intended to validate the behavior of the Trace class using dummy data. 
However, without the full implementation of the Trace class, this testing module is incomplete.

Tests include:
- Verification of the `run` method with dummy data.
"""


from pathlib import Path

import unittest
from unittest.mock import patch, MagicMock
import numpy as np
from matilda.stages.trace import Trace
from matilda.data.option import TraceOptions
import pandas as pd

script_dir = Path(__file__).parent


csv_z = script_dir/'trace/input/Z.csv'
csv_yBin = script_dir/'trace/input/Ybin.csv'
csv_p = script_dir/'trace/input/P.csv'
csv_beta = script_dir/'trace/input/Beta.csv'
csv_algo_labels = script_dir/'trace/input/Algolabels.csv'
csv_pi = script_dir/'trace/input/Opts_PI.csv'
csv_usesim = script_dir/'trace/input/Opts_usesim.csv'
csv_summary_with_mock_data = script_dir/'trace/output/summary_with_mock_data.csv'
class TestTraceRun(unittest.TestCase):

    def setUp(self):
        self.z = np.genfromtxt(csv_z,delimiter=',')  
        self.y_bin = np.genfromtxt(csv_yBin, delimiter = ',')
        self.p = np.genfromtxt(csv_p,delimiter=',')
        self.beta = np.genfromtxt(csv_beta,delimiter=',')
        self.algo_labels = pd.read_csv(csv_algo_labels, delimiter=',', header=None, dtype=str).iloc[0].tolist()
        self.pi = np.genfromtxt(csv_pi, delimiter=',', dtype=float)
        self.usesim = np.genfromtxt(csv_usesim)
        self.opts = TraceOptions(self.pi,self.usesim)
    
    # TODO: test for tracebuild,tracecontra,tracesummary
    @patch('matilda.stages.trace.Trace.build')
    @patch('matilda.stages.trace.Trace.contra')
    @patch('matilda.stages.trace.Trace.summary')
    def test_trace_function(self, mock_summary, mock_contra, mock_build):
        # Set return values for the mocked functions
        mock_polygon = MagicMock()
        
        mock_build.return_value = mock_polygon
        mock_contra.return_value = [mock_polygon,mock_polygon]
        mock_summary.return_value = [0.01,0.02,0.03,0.04,0.05]
        
        # Call the trace function
        result = Trace.run(self.z, self.y_bin, self.p, self.beta, self.algo_labels, self.opts)
        
        # Index_col is the list of algo_labels which set to unnamed for consistency with matlab format
        # Summary.csv is the output of the summary function generated with sample options and metadata 

        summary_from_csv = pd.read_csv(csv_summary_with_mock_data, index_col='Unnamed: 0')

        assert result.summary.equals(summary_from_csv), "The DataFrames do not match."
    
        #TODO: Add assertions
        
