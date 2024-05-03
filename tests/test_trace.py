import unittest
from unittest.mock import patch, MagicMock
import numpy as np
from matilda.data.model import Footprint, PolyShape, TraceOut
from matilda.stages.trace import Trace
from matilda.data.option import TraceOptions

csv_z = 'trace/input/Z.csv'
csv_yBin = 'trace/input/Ybin.csv'
csv_p = 'trace/input/P.csv'
csv_beta = 'trace/input/Beta.csv'
csv_algo_labels = 'trace/input/Algolabels.csv'
csv_pi = 'trace/input/Opts_PI.csv'
csv_usesim = 'trace/input/Opts_usesim.csv'

class TestTraceFunction(unittest.TestCase):

    def setUp(self):
        self.z = np.genfromtxt(csv_z,delimiter=',')  
        self.y_bin = np.genfromtxt(csv_yBin, delimiter = ',')
        self.p = np.genfromtxt(csv_p,delimiter=',')
        self.beta = np.genfromtxt(csv_beta,delimiter=',')
        self.algo_labels = np.genfromtxt(csv_algo_labels,delimiter=',')
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
        # footprint = Footprint(mock_polygon,0.01,0.02,0.03,0.04,0.05)
        mock_build.return_value = mock_polygon
        mock_contra.return_value = [mock_polygon,mock_polygon]
        mock_summary.return_value = [0.01,0.02,0.03,0.04]
        
        # Call the trace function
        result = Trace(self.z, self.y_bin, self.p, self.beta, self.algo_labels, self.opts)
        
if __name__ == '__main__':
    unittest.main()
