"""Test module for Pilot and Cloister to verify that they can integrate together.

The file contains multiple unit tests to ensure that the `Pilot` and `Cloister`
classes corretly perform their tasks when they are integrated. The basic
mechanism of the test is to compare its output against output from MATLAB and
check if the outputs are the same or reasonably similar.

"""

from pathlib import Path

import numpy as np
from scipy.io import loadmat

from matilda.data.options import CloisterOptions, PilotOptions
from matilda.stages.cloister import Cloister
from matilda.stages.pilot import Pilot
from tests.utils.option_creator import create_option

script_dir = Path(__file__).parent
default_option = create_option(cloister=CloisterOptions(p_val=0.05, c_thres=0.7))


def test_run_analytic_pilot_cloister() -> None:
    """Test Closister stage for analytic option from Pilot stage."""
    fp_smapledata = script_dir / "test_data/pilot-cloister/input/test_analytic.mat"
    data = loadmat(fp_smapledata)
    x_sample = data["X"]
    y_sample = data["Y"]
    feat_labels_sample = data["featlabels"][0]
    feat_labels_sample_str = [str(label[0]) for label in feat_labels_sample]
    opts = PilotOptions(True, 5)
    pilot = Pilot()
    _, result_pilot = pilot.run(x_sample, y_sample, feat_labels_sample_str, opts)

    a = result_pilot.a

    cloister = Cloister(x_sample, a, default_option.cloister)
    _, result_cloister = cloister.run(x_sample, a, default_option.cloister)

    fp_outdata = script_dir / "test_data/pilot-cloister/output/matlab_results_ana.mat"
    data_out = loadmat(fp_outdata)

    np.testing.assert_almost_equal(
        np.sort(data_out["Zedge"][1:], axis=0),
        np.sort(result_cloister.z_edge, axis=0),
        decimal=1,
    )

    np.testing.assert_almost_equal(
        np.sort(data_out["Zecorr"][1:], axis=0),
        np.sort(result_cloister.z_ecorr, axis=0),
        decimal=1,
    )


def test_run_numerical_pilot_cloister_zedge() -> None:
    """Test Closister stage for analytic option from Pilot stage."""
    fp_smapledata = script_dir / "test_data/pilot-cloister/input/test_numerical.mat"
    data = loadmat(fp_smapledata)
    x_sample = data["X_test"]
    y_sample = data["Y_test"]
    feat_labels_sample = data["featlabels"][0]
    feat_labels_sample_str = [str(label[0]) for label in feat_labels_sample]
    opts = PilotOptions(False, 5)
    pilot = Pilot()
    _, result_pilot = pilot.run(x_sample, y_sample, feat_labels_sample_str, opts)

    a = result_pilot.a

    cloister = Cloister(x_sample, a, default_option.cloister)
    _, result_cloister = cloister.run(x_sample, a, default_option.cloister)

    fp_outdata = script_dir / "test_data/pilot-cloister/output/matlab_results_num.mat"
    data_out = loadmat(fp_outdata)

    dt_out_zedge = np.sort(data_out["Zedge"][1:], axis=0)
    czedge = np.sort(result_cloister.z_edge, axis=0)
    dt_out_zecorr = np.sort(data_out["Zecorr"][1:], axis=0)
    czecorr = np.sort(result_cloister.z_ecorr, axis=0)

    np.testing.assert_allclose(dt_out_zedge, czedge, atol=3)
    np.testing.assert_allclose(dt_out_zecorr, czecorr, atol=3)
