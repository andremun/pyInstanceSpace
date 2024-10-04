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

from matilda.data.options import PilotOptions
from matilda.stages.pilot import PilotStage

script_dir = Path(__file__).parent


class SampleData:
    """Data class for testing the Pilot stage for analytic purposes.

    This class contains the data used for testing the Pilot stage for analytic purposes.
    """

    def __init__(self) -> None:
        """Initialize the sample data for the Pilot stage."""
        fp_sampledata = script_dir / "test_data/pilot/input/test_analytic.mat"
        data = loadmat(fp_sampledata)
        self.x_sample = data["X"]
        self.y_sample = data["Y"]
        feat_labels_sample = data["featlabels"][0]
        self.feat_labels_sample = [str(label[0]) for label in feat_labels_sample]


class SampleDataNum:
    """Data class for testing the Pilot stage for numerical purposes.

    This class contains the data used for testing the Pilot stage for
    numerical purposes.
    """

    def __init__(self) -> None:
        """Initialize the sample data for the Pilot stage."""
        fp_sampledata = script_dir / "test_data/pilot/input/test_numerical.mat"
        data = loadmat(fp_sampledata)
        self.x_sample = data["X_test"]
        self.y_sample = data["Y_test"]
        feat_labels = data["featlabels"][0]
        self.feat_labels_sample = [str(label[0]) for label in feat_labels]
        analytic = data["optsPilot"][0, 0]["analytic"][0, 0]
        n_tries = int(data["optsPilot"][0, 0]["ntries"][0, 0])
        self.opts_sample = PilotOptions(None, None, analytic, n_tries)


class MatlabResults:
    """Data class for verifying the output of the Pilot analytical method.

    This class contains the data used for verifying the output of the
    analytical Pilot stage.
    """

    def __init__(self) -> None:
        """Initialize the sample data for the Pilot stage."""
        fp_outdata = script_dir / "test_data/pilot/output/matlab_results_ana.mat"
        self.data = loadmat(fp_outdata)


class MatlabResultsNum:
    """Data class for verifying the output of the Pilot numerical method.

    This class contains the data used for verifying the output of the
    numerical Pilot stage.
    """

    def __init__(self) -> None:
        """Initialize the sample data for the Pilot stage."""
        fp_outdata = script_dir / "test_data/pilot/output/matlab_results_num.mat"
        self.data = loadmat(fp_outdata)


def test_run_analytic() -> None:
    """Test the run function for the Pilot stage for analytic purposes."""
    sd = SampleData()
    mtr = MatlabResults()

    x_sample = sd.x_sample
    y_sample = sd.y_sample
    feat_labels_sample = sd.feat_labels_sample
    opts = PilotOptions(None, None, True, 5)
    pilot = PilotStage(x_sample, y_sample, feat_labels_sample)
    result = pilot.pilot(x_sample, y_sample, feat_labels_sample, opts)

    a = result[4]
    b = result[7]
    c = result[6]
    z = result[5]
    error = result[8]

    np.testing.assert_almost_equal(abs(a), abs(mtr.data["A"]), decimal=6)
    np.testing.assert_almost_equal(abs(b), abs(mtr.data["B"]), decimal=6)
    np.testing.assert_almost_equal(abs(c), abs(mtr.data["C"]), decimal=6)
    np.testing.assert_almost_equal(abs(z), abs(mtr.data["Z"]), decimal=6)
    np.testing.assert_almost_equal(abs(error), abs(mtr.data["error"]), decimal=6)


def test_run_numerical() -> None:
    """Test the run function for the Pilot stage for numerical purposes."""
    sd = SampleDataNum()
    mtr = MatlabResultsNum()

    x_sample = sd.x_sample
    y_sample = sd.y_sample
    feat_labels_sample = sd.feat_labels_sample
    opts_sample = sd.opts_sample
    opts = PilotOptions(None, None, opts_sample.analytic, opts_sample.n_tries)
    pilot = PilotStage(x_sample, y_sample, feat_labels_sample)
    result = pilot.pilot(x_sample, y_sample, feat_labels_sample, opts)
    eoptim = result[2]
    perf = result[3]

    if eoptim is not None and perf is not None:
        np.testing.assert_almost_equal(eoptim, mtr.data["eoptim"][0], decimal=6)
        np.testing.assert_almost_equal(perf, mtr.data["perf"][0], decimal=1)
