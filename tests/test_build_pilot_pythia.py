"""Integration contracts for a PILOT projection followed by PYTHIA training.

Historical CSV comparisons formerly in this module have no generator commit, options,
or MATLAB release metadata. They remain classified as ``legacy-unknown`` rather than
being used as numerical oracles. Current-MATLAB parity belongs to the verified fixture
bundle; these tests cover the live cross-stage Python contract.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from numpy.typing import NDArray
from scipy.io import loadmat
from sklearn.svm import SVC

from instancespace.data.options import (
    GeneralOptions,
    ParallelOptions,
    PilotOptions,
    PythiaOptions,
)
from instancespace.stages.pilot import PilotStage
from instancespace.stages.pythia import PythiaOutput, PythiaStage

parallel_opts = ParallelOptions(
    flag=False,
    n_cores=1,
)

POLYNOMIAL_DEGREE = 3
INTEGRATION_BOX_CONSTRAINT = 5.0
INTEGRATION_KERNEL_SCALE = 2.0

script_dir = Path(__file__).parent

csv_path_z_input = script_dir / "test_data/pythia/input/Z.csv"
csv_path_y_input = script_dir / "test_data/pythia/input/y.csv"
csv_path_algo_input = script_dir / "test_data/pythia/input/algolabels.csv"
csv_path_y_best_input = script_dir / "test_data/pythia/input/ybest.csv"
csv_path_y_bin_input = script_dir / "test_data/pythia/input/ybin.csv"

z = np.genfromtxt(csv_path_z_input, delimiter=",")
y = np.genfromtxt(csv_path_y_input, delimiter=",")
algo = pd.read_csv(csv_path_algo_input, header=None).squeeze().tolist()
y_best = np.genfromtxt(csv_path_y_best_input, delimiter=",")
y_bin = np.genfromtxt(csv_path_y_bin_input, delimiter=",")


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
        analytic = bool(data["optsPilot"][0, 0]["analytic"][0, 0])
        n_tries = int(data["optsPilot"][0, 0]["ntries"][0, 0])
        self.opts_sample = PilotOptions(None, None, analytic, n_tries)


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


def _train_pythia_from_projection(
    projection: NDArray[np.double],
    *,
    polynomial: bool,
) -> PythiaOutput:
    """Train PYTHIA from one PILOT projection with fixed MATLAB-unit parameters."""
    params = np.tile(
        [INTEGRATION_BOX_CONSTRAINT, INTEGRATION_KERNEL_SCALE],
        (len(algo), 1),
    )
    return PythiaStage.pythia(
        projection,
        y,
        y_bin,
        y_best,
        algo,
        PythiaOptions(
            cv_folds=5,
            is_poly_krnl=polynomial,
            use_weights=False,
            params=params,
            classifier="svm",
            tuning="none",
        ),
        parallel_opts,
        GeneralOptions.default(),
    )


def _assert_integrated_output(
    output: PythiaOutput,
    projection: NDArray[np.double],
    *,
    polynomial: bool,
) -> None:
    """Assert stable shapes and MATLAB-unit SVM parameter boundaries."""
    expected_shape = y_bin.shape
    assert projection.shape == (expected_shape[0], 2)
    assert output.y_hat.shape == expected_shape
    assert output.y_sub.shape == expected_shape
    assert output.pr0_hat.shape == expected_shape
    assert output.pr0_sub.shape == expected_shape
    assert output.selection0.shape == (expected_shape[0],)
    assert output.pythia_summary.shape[0] == len(algo) + 2
    assert np.all(np.isfinite(output.mu))
    assert np.all(np.isfinite(output.sigma))

    expected_kernel = "poly" if polynomial else "rbf"
    for estimator, kernel_scale in zip(output.svm, output.k_scale, strict=True):
        assert isinstance(estimator, SVC)
        assert estimator.kernel == expected_kernel
        assert estimator.degree == POLYNOMIAL_DEGREE
        assert estimator.gamma == pytest.approx(1.0 / kernel_scale**2)


@pytest.mark.parametrize("polynomial", [False, True])
def test_numerical_pilot_to_pythia_contract(*, polynomial: bool) -> None:
    """Numerical PILOT output is a valid input to Gaussian and polynomial PYTHIA."""
    sample_data = SampleDataNum()
    pilot_output = PilotStage(
        sample_data.x_sample,
        sample_data.y_sample,
        sample_data.feat_labels_sample,
    ).pilot(
        sample_data.x_sample,
        sample_data.y_sample,
        sample_data.feat_labels_sample,
        sample_data.opts_sample,
        GeneralOptions.default(),
    )

    output = _train_pythia_from_projection(pilot_output.z, polynomial=polynomial)

    _assert_integrated_output(output, pilot_output.z, polynomial=polynomial)


@pytest.mark.parametrize("polynomial", [False, True])
def test_analytic_pilot_to_pythia_contract(*, polynomial: bool) -> None:
    """Analytic PILOT output is a valid input to Gaussian and polynomial PYTHIA."""
    sample_data = SampleData()
    pilot_output = PilotStage(
        sample_data.x_sample,
        sample_data.y_sample,
        sample_data.feat_labels_sample,
    ).pilot(
        sample_data.x_sample,
        sample_data.y_sample,
        sample_data.feat_labels_sample,
        PilotOptions.default(analytic=True, n_tries=5),
        GeneralOptions.default(),
    )

    output = _train_pythia_from_projection(pilot_output.z, polynomial=polynomial)

    _assert_integrated_output(output, pilot_output.z, polynomial=polynomial)
