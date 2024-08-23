"""Test module for Pilot and Cloister to verify that they can integrate together

The file contains multiple unit tests to ensure that the `Pilot` and `Cloister` classes
corretly perform their tasks when they are integrated. The basic mechanism of the test 
is to compare its output against output from MATLAB and check if the outputs are the same 
or reasonably similar. 

"""

from pathlib import Path

import numpy as np
from scipy.io import loadmat

from matilda.data.options import PilotOptions
from matilda.stages.pilot import Pilot
from matilda.data.options import CloisterOptions
from matilda.stages.cloister import Cloister
from tests.utils.option_creator import create_option
from scipy.io import loadmat

script_dir = Path(__file__).parent
default_option = create_option(cloister=CloisterOptions(p_val=0.05, c_thres=0.7))

def test_run_analytic_Zedge():
    """Test Closister stage for analytic option from Pilot stage."""

    fp_smapledata = script_dir / "test_data/pilot-cloister/input/test_analytic.mat"
    data = loadmat(fp_smapledata)
    x_sample = data["X"]
    y_sample = data["Y"]
    feat_labels_sample = data["featlabels"][0]
    feat_labels_sample_str = [str(label[0]) for label in feat_labels_sample]
    opts = PilotOptions(True, 5)
    pilot = Pilot()
    result = pilot.run(x_sample, y_sample, feat_labels_sample_str, opts)[1]

    a = result.a

    cloister = Cloister(x_sample, a, default_option.cloister)
    _, result_cloister = cloister.run(x_sample, a, default_option.cloister)

    fp_outdata = script_dir / "test_data/pilot-cloister/output/matlab_results_ana.mat"
    data_out = loadmat(fp_outdata)

    np.testing.assert_almost_equal(data_out['Zedge'], result_cloister.z_edge, decimal=6)

def test_run_analytic_Zecorr():
    """Test Closister stage for analytic option from Pilot stage."""

    fp_smapledata = script_dir / "test_data/pilot-cloister/input/test_analytic.mat"
    data = loadmat(fp_smapledata)
    x_sample = data["X"]
    y_sample = data["Y"]
    feat_labels_sample = data["featlabels"][0]
    feat_labels_sample_str = [str(label[0]) for label in feat_labels_sample]
    opts = PilotOptions(True, 5)
    pilot = Pilot()
    result = pilot.run(x_sample, y_sample, feat_labels_sample_str, opts)[1]

    a = result.a

    cloister = Cloister(x_sample, a, default_option.cloister)
    _, result_cloister = cloister.run(x_sample, a, default_option.cloister)

    fp_outdata = script_dir / "test_data/pilot-cloister/output/matlab_results_ana.mat"
    data_out = loadmat(fp_outdata)

    np.testing.assert_almost_equal(data_out['Zecorr'], result_cloister.z_ecorr, decimal=6)

def test_run_numerical_Zedge():
    """Test Closister stage for analytic option from Pilot stage."""

    fp_smapledata = script_dir / "test_data/pilot-cloister/input/test_numerical.mat"
    data = loadmat(fp_smapledata)
    x_sample = data["X_test"]
    y_sample = data["Y_test"]
    feat_labels_sample = data["featlabels"][0]
    feat_labels_sample_str = [str(label[0]) for label in feat_labels_sample]
    opts = PilotOptions(False, 5)
    pilot = Pilot()
    result = pilot.run(x_sample, y_sample, feat_labels_sample_str, opts)[1]

    a = result.a

    cloister = Cloister(x_sample, a, default_option.cloister)
    _, result_cloister = cloister.run(x_sample, a, default_option.cloister)

    fp_outdata = script_dir / "test_data/pilot-cloister/output/matlab_results_num.mat"
    data_out = loadmat(fp_outdata)

    np.testing.assert_almost_equal(data_out['Zedge'], result_cloister.z_edge, decimal=6)

def test_run_numerical_Zecorr():
    """Test Closister stage for analytic option from Pilot stage."""

    fp_smapledata = script_dir / "test_data/pilot-cloister/input/test_numerical.mat"
    data = loadmat(fp_smapledata)
    x_sample = data["X_test"]
    y_sample = data["Y_test"]
    feat_labels_sample = data["featlabels"][0]
    feat_labels_sample_str = [str(label[0]) for label in feat_labels_sample]
    opts = PilotOptions(False, 5)
    pilot = Pilot()
    result = pilot.run(x_sample, y_sample, feat_labels_sample_str, opts)[1]

    a = result.a

    cloister = Cloister(x_sample, a, default_option.cloister)
    _, result_cloister = cloister.run(x_sample, a, default_option.cloister)

    fp_outdata = script_dir / "test_data/pilot-cloister/output/matlab_results_num.mat"
    data_out = loadmat(fp_outdata)

    np.testing.assert_almost_equal(data_out['Zecorr'], result_cloister.z_ecorr, decimal=6)






