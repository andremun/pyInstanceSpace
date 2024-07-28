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
from scipy.io import loadmat

from matilda.data.option import PilotOptions
from matilda.stages.pilot import Pilot

script_dir = Path(__file__).parent

class SampleData:
    def __init__(self):
        fp_sampledata = script_dir / "test_data/pilot/input/test_analytic.mat"
        data = loadmat(fp_sampledata)
        self.X_sample = data["X_test"]
        self.Y_sample = data["Y_test"]
        feat_labels_sample = data["featlabels"][0]
        self.feat_labels_sample = [str(label[0]) for label in feat_labels_sample]
        analytic = data["optsPilot"][0,0]["analytic"][0,0]
        n_tries = int(data["optsPilot"][0,0]["ntries"][0,0])
        self.opts_sample = PilotOptions(analytic, n_tries)

class SampleDataNum:
    def __init__(self):
        fp_sampledata = script_dir / "test_data/pilot/input/test_numerical.mat"
        data = loadmat(fp_sampledata)
        self.X_sample = data["X_test"]
        self.Y_sample = data["Y_test"]
        feat_labels = data["featlabels"][0]
        self.feat_labels_sample = [str(label[0]) for label in feat_labels]
        analytic = data["optsPilot"][0,0]["analytic"][0,0]
        n_tries = int(data["optsPilot"][0,0]["ntries"][0,0])
        self.opts_sample = PilotOptions(analytic, n_tries)


class MatlabResults:
    def __init__(self):
        fp_outdata = script_dir / "test_data/pilot/output/matlab_results_ana.mat"
        self.data = loadmat(fp_outdata)

class MatlabResultsNum:
    def __init__(self):
        fp_outdata = script_dir / "test_data/pilot/output/matlab_results_num.mat"
        self.data = loadmat(fp_outdata)

# def test_error_function():
#     sd = SampleDataNum()

#     mtr = MatlabResultsNum()

#     X_sample = sd.X_sample
#     Y_sample = sd.Y_sample
#     n = X_sample.shape[1]
#     m = X_sample.shape[1] + Y_sample.shape[1]  # Total number of features including appended Y

#     # alpha_sample = mtr.data['X0']
#     alpha_sample = mtr.data['alpha'][:, 0]
#     x_bar_sample = np.hstack([X_sample, Y_sample])
#     n = X_sample.shape[1]
#     m = x_bar_sample.shape[1]

#     pilot = Pilot()
#     error = pilot.error_function(alpha_sample, x_bar_sample, n, m)


#     matlab_error = mtr.data['eoptim'][0, 0]
#     assert (error == matlab_error)

# def test_run_analytic():
#     sd = SampleData()
#     mtr = MatlabResults()

#     X_sample = sd.X_sample
#     Y_sample = sd.Y_sample
#     feat_labels_sample = sd.feat_labels_sample
#     opts_sample = sd.opts_sample
#     opts = PilotOptions(opts_sample.analytic, opts_sample.n_tries)
#     pilot = Pilot()
#     result = pilot.run(X_sample, Y_sample, feat_labels_sample, opts)[0]

#     np.testing.assert_almost_equal(result.a, mtr.data['A'], decimal=6)
#     np.testing.assert_almost_equal(result.b, mtr.data['B'], decimal=6)
#     np.testing.assert_almost_equal(result.c, mtr.data['C'], decimal=6)
#     np.testing.assert_almost_equal(result.z, mtr.data['Z'], decimal=6)
#     np.testing.assert_almost_equal(result.error, mtr.data['error'], decimal=6)
#     np.testing.assert_almost_equal(result.r2, mtr.data['R2'], decimal=6)

def test_run_numerical():
    sd = SampleDataNum()
    mtr = MatlabResultsNum()

    X_sample = sd.X_sample
    Y_sample = sd.Y_sample
    feat_labels_sample = sd.feat_labels_sample
    opts_sample = sd.opts_sample
    opts = PilotOptions(opts_sample.analytic, opts_sample.n_tries)
    pilot = Pilot()
    result = pilot.run(X_sample, Y_sample, feat_labels_sample, opts)[0]


    np.testing.assert_almost_equal(result.eoptim, mtr.data['eoptim'], decimal=6)
    np.testing.assert_almost_equal(result.perf, mtr.data['perf'], decimal=1)
    np.testing.assert_almost_equal(result.a, mtr.data['A'], decimal=6)
    np.testing.assert_almost_equal(result.z, mtr.data['Z'], decimal=6)
    np.testing.assert_almost_equal(result.b, mtr.data['B'], decimal=6)
    np.testing.assert_almost_equal(result.c, mtr.data['C'], decimal=6)
    np.testing.assert_almost_equal(result.r2, mtr.data['R2'], decimal=6)
    np.testing.assert_almost_equal(result.error, mtr.data['error'], decimal=6)

