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
import pytest
from scipy.io import loadmat
from scipy.spatial.distance import pdist

from instancespace.data.options import GeneralOptions, PilotOptions
from instancespace.stages.pilot import PilotStage

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
    result = pilot.pilot(
        x_sample,
        y_sample,
        feat_labels_sample,
        opts,
        GeneralOptions.default(),
    )

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
    result = pilot.pilot(
        x_sample,
        y_sample,
        feat_labels_sample,
        opts,
        GeneralOptions.default(),
    )
    eoptim = result[2]
    perf = result[3]

    if eoptim is not None and perf is not None:
        np.testing.assert_almost_equal(eoptim, mtr.data["eoptim"][0], decimal=6)
        np.testing.assert_almost_equal(perf, mtr.data["perf"][0], decimal=1)


def test_pilot_seed_reproducibility() -> None:
    """Same seed gives identical output; a different seed gives different output.

    Regression test for Q9 (general.seed threading): the numerical solve branch
    picks its BFGS starting points via `general_options.seed`, so this is the
    one place PILOT's output actually depends on the seed.
    """
    rng = np.random.default_rng(42)
    x = rng.random((30, 4))
    y = rng.random((30, 2))
    feat_labels = ["f0", "f1", "f2", "f3"]
    opts = PilotOptions(None, None, False, 3)

    result_a = PilotStage.pilot(
        x,
        y,
        feat_labels,
        opts,
        GeneralOptions(verbose=False, seed=0),
        _do_output=False,
    )
    result_b = PilotStage.pilot(
        x,
        y,
        feat_labels,
        opts,
        GeneralOptions(verbose=False, seed=0),
        _do_output=False,
    )
    result_c = PilotStage.pilot(
        x,
        y,
        feat_labels,
        opts,
        GeneralOptions(verbose=False, seed=1),
        _do_output=False,
    )

    np.testing.assert_array_equal(result_a.z, result_b.z)
    assert not np.array_equal(result_a.z, result_c.z)


def test_adjust_rotation_preserves_pairwise_distances() -> None:
    """R1: `adjust_rotation()` only rotates Z - pairwise distances are unchanged.

    Ported from PyISpace's `pilot.adjust_rotation()`
    (gitlab.com/ita-ml/pyispace). Rotation is a rigid transform, so the
    pairwise-distance matrix of Z must be identical before and after.
    """
    rng = np.random.default_rng(7)
    z = rng.random((25, 2))
    bad_instances = np.zeros(25, dtype=bool)
    bad_instances[[2, 5, 9, 14]] = True

    z_rot, rot = PilotStage.adjust_rotation(z, bad_instances)

    np.testing.assert_allclose(pdist(z), pdist(z_rot), atol=1e-10)
    # A rotation matrix is orthonormal: R @ R.T == I.
    np.testing.assert_allclose(rot @ rot.T, np.eye(2), atol=1e-10)


def test_adjust_rotation_places_bad_centroid_at_theta() -> None:
    """The bad-instance centroid lands at the requested angle after rotation."""
    rng = np.random.default_rng(11)
    z = rng.random((20, 2)) - 0.5
    bad_instances = np.zeros(20, dtype=bool)
    bad_instances[[1, 4, 7]] = True

    z_rot, _ = PilotStage.adjust_rotation(z, bad_instances, theta=135.0)

    centroid = np.mean(z_rot[bad_instances], axis=0)
    angle = np.degrees(np.arctan2(centroid[1], centroid[0]))
    np.testing.assert_allclose(angle, 135.0, atol=1e-6)


def test_pilot_adjust_rotation_matches_unrotated_up_to_rotation() -> None:
    """`adjust_rotation=True` rotates Z but leaves pairwise distances unchanged.

    Roadmap R1 checkpoint: confirms Z's pairwise distances are unchanged
    before/after the rotation adjustment.
    """
    rng = np.random.default_rng(3)
    x = rng.random((30, 4))
    y = rng.random((30, 2))
    feat_labels = ["f0", "f1", "f2", "f3"]
    y_bin = np.zeros((30, 2), dtype=bool)
    y_bin[::3, 0] = True

    opts_plain = PilotOptions(None, None, False, 3, adjust_rotation=False)
    opts_rotated = PilotOptions(None, None, False, 3, adjust_rotation=True)
    general_options = GeneralOptions(verbose=False, seed=0)

    plain = PilotStage.pilot(
        x,
        y,
        feat_labels,
        opts_plain,
        general_options,
        y_bin=y_bin,
        _do_output=False,
    )
    rotated = PilotStage.pilot(
        x,
        y,
        feat_labels,
        opts_rotated,
        general_options,
        y_bin=y_bin,
        _do_output=False,
    )

    np.testing.assert_allclose(pdist(plain.z), pdist(rotated.z), atol=1e-8)
    assert not np.allclose(plain.z, rotated.z)


def test_pilot_adjust_rotation_reproducible_across_runs() -> None:
    """Two independent runs on the same data rotate to the same orientation."""
    rng = np.random.default_rng(3)
    x = rng.random((30, 4))
    y = rng.random((30, 2))
    feat_labels = ["f0", "f1", "f2", "f3"]
    y_bin = np.zeros((30, 2), dtype=bool)
    y_bin[::3, 0] = True
    opts = PilotOptions(None, None, False, 3, adjust_rotation=True)
    general_options = GeneralOptions(verbose=False, seed=0)

    run_a = PilotStage.pilot(
        x,
        y,
        feat_labels,
        opts,
        general_options,
        y_bin=y_bin,
        _do_output=False,
    )
    run_b = PilotStage.pilot(
        x,
        y,
        feat_labels,
        opts,
        general_options,
        y_bin=y_bin,
        _do_output=False,
    )

    np.testing.assert_array_equal(run_a.z, run_b.z)


def test_pilot_adjust_rotation_requires_y_bin() -> None:
    """`adjust_rotation=True` without `y_bin` fails loudly, not silently."""
    rng = np.random.default_rng(3)
    x = rng.random((20, 3))
    y = rng.random((20, 2))
    feat_labels = ["f0", "f1", "f2"]
    opts = PilotOptions(None, None, False, 2, adjust_rotation=True)

    with pytest.raises(ValueError, match="y_bin"):
        PilotStage.pilot(
            x,
            y,
            feat_labels,
            opts,
            GeneralOptions(verbose=False, seed=0),
            _do_output=False,
        )


def test_pilot_adjust_rotation_no_bad_instances_leaves_z_unchanged() -> None:
    """No poorly-solved instances: rotation is skipped, Z is returned as-is."""
    rng = np.random.default_rng(3)
    x = rng.random((20, 3))
    y = rng.random((20, 2))
    feat_labels = ["f0", "f1", "f2"]
    y_bin_all_good = np.ones((20, 2), dtype=bool)
    opts_plain = PilotOptions(None, None, False, 2, adjust_rotation=False)
    opts_rotated = PilotOptions(None, None, False, 2, adjust_rotation=True)
    general_options = GeneralOptions(verbose=False, seed=0)

    plain = PilotStage.pilot(
        x,
        y,
        feat_labels,
        opts_plain,
        general_options,
        y_bin=y_bin_all_good,
        _do_output=False,
    )
    rotated = PilotStage.pilot(
        x,
        y,
        feat_labels,
        opts_rotated,
        general_options,
        y_bin=y_bin_all_good,
        _do_output=False,
    )

    np.testing.assert_array_equal(plain.z, rotated.z)
