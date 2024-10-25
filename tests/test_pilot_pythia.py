"""
Test the integration of the pilot stage with the Pythia stage.

The file contains the integration test for the Pilot Stage followed by the Pythia
Stage to verify the functionality of the stages when integrated together. The test
will check the output of the Pythia stage with the expected output from the MATLAB
"""

from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.io import loadmat
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC

from matilda.data.options import ParallelOptions, PilotOptions, PythiaOptions
from matilda.stages.pilot import PilotStage
from matilda.stages.pythia import PythiaStage

parallel_opts = ParallelOptions(
    flag=True,
    n_cores=8,
)

script_dir = Path(__file__).parent

output_dir = script_dir / "test_data/pythia/output"

csv_path_z_input = script_dir / "test_data/pythia/input/Z.csv"
csv_path_y_input = script_dir / "test_data/pythia/input/y.csv"
csv_path_algo_input = script_dir / "test_data/pythia/input/algolabels.csv"
csv_path_y_best_input = script_dir / "test_data/pythia/input/ybest.csv"
csv_path_y_bin_input = script_dir / "test_data/pythia/input/ybin.csv"

csv_path_znorm_input = script_dir / "test_data/pythia/output/znorm.csv"
csv_path_mu_input = script_dir / "test_data/pythia/output/mu.csv"
csv_path_sig_input = script_dir / "test_data/pythia/output/sigma.csv"

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
        analytic = data["optsPilot"][0, 0]["analytic"][0, 0]
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


class MatlabResultsNum:
    """Data class for verifying the output of the Pilot numerical method.

    This class contains the data used for verifying the output of the
    numerical Pilot stage.
    """

    def __init__(self) -> None:
        """Initialize the sample data for the Pilot stage."""
        fp_outdata = script_dir / "test_data/pilot/output/matlab_results_num.mat"
        self.data = loadmat(fp_outdata)


class MatlabResults:
    """Data class for verifying the output of the Pilot analytical method.

    This class contains the data used for verifying the output of the
    analytical Pilot stage.
    """

    def __init__(self) -> None:
        """Initialize the sample data for the Pilot stage."""
        fp_outdata = script_dir / "test_data/pilot/output/matlab_results_ana.mat"
        self.data = loadmat(fp_outdata)


def compare_performance(
    python_output: tuple[
        list[float],
        list[float],
        NDArray[np.double],
        StratifiedKFold,
        list[SVC],
        NDArray[np.double],
        NDArray[np.bool_],
        NDArray[np.bool_],
        NDArray[np.double],
        NDArray[np.double],
        list[float],
        list[float],
        list[float],
        list[float],
        list[float],
        NDArray[np.int_],
        NDArray[np.int_],
        pd.DataFrame,
    ],
    matlab_accuracy: NDArray[np.double],
    matlab_precision: NDArray[np.double],
    matlab_recall: NDArray[np.double],
    algo_num: int,
    tol: float,
) -> None:
    """Test that whether the performance of model is as expected."""
    total = 0
    correct = 0
    threshold = 0.90

    # tolerance
    tol = 2.5

    # compare the performance of the model with the expected values
    # if the performance is greater than the expected value, it is considered correct
    # if the performance is within the tolerance, it is considered correct
    for i in range(algo_num):
        total += 3

        if (
            python_output[12][i] * 100 >= matlab_accuracy[i]
            or abs(python_output[12][i] * 100 - matlab_accuracy[i]) <= tol
        ):
            correct += 1

        if (
            python_output[13][i] * 100 >= matlab_precision[i]
            or abs(python_output[13][i] * 100 - matlab_precision[i]) <= tol
        ):
            correct += 1

        if (
            python_output[14][i] * 100 >= matlab_recall[i]
            or abs(python_output[14][i] * 100 - matlab_recall[i]) <= tol
        ):
            correct += 1

    assert correct / total >= threshold


def test_pilot_num_pythia_bayes_gaussian() -> None:
    """Test the integration of the Pilot and Pythia stages."""
    sample_data = SampleDataNum()

    x_sample = sample_data.x_sample
    y_sample = sample_data.y_sample
    feat_labels_sample = sample_data.feat_labels_sample
    opts_sample = sample_data.opts_sample
    pilot_opts = PilotOptions(None, None, opts_sample.analytic, opts_sample.n_tries)
    pilot = PilotStage(x_sample, y_sample, feat_labels_sample)
    pilot_result = pilot.pilot(x_sample, y_sample, feat_labels_sample, pilot_opts)

    pythia_options = PythiaOptions(
        cv_folds=5,
        is_poly_krnl=False,
        use_weights=False,
        use_grid_search=True,
        params=None,
    )
    pythia = PythiaStage(pilot_result[5], y, y_bin, y_best, algo)
    pythia_result = pythia.pythia(
        pilot_result[5],
        y,
        y_bin,
        y_best,
        algo,
        pythia_options,
        parallel_opts,
    )

    # read the actual output
    matlab_output = pd.read_csv(output_dir / "BO_gaussian/gaussian.csv")

    # get the accuracy, precision, recall
    matlab_accuracy = matlab_output["CV_model_accuracy"].values.astype(np.double)
    matlab_precision = matlab_output["CV_model_precision"].values.astype(np.double)
    matlab_recall = matlab_output["CV_model_recall"].values.astype(np.double)

    compare_performance(
        pythia_result,
        matlab_accuracy,
        matlab_precision,
        matlab_recall,
        len(algo),
        2.5,
    )


def test_pilot_num_pythia_bayes_poly() -> None:
    """Test the integration of the Pilot and Pythia stages."""
    sample_data = SampleDataNum()

    x_sample = sample_data.x_sample
    y_sample = sample_data.y_sample
    feat_labels_sample = sample_data.feat_labels_sample
    opts_sample = sample_data.opts_sample
    pilot_opts = PilotOptions(None, None, opts_sample.analytic, opts_sample.n_tries)
    pilot = PilotStage(x_sample, y_sample, feat_labels_sample)
    pilot_result = pilot.pilot(x_sample, y_sample, feat_labels_sample, pilot_opts)

    opts = PythiaOptions(
        cv_folds=5,
        is_poly_krnl=True,
        use_weights=False,
        use_grid_search=False,
        params=None,
    )
    pythia = PythiaStage(pilot_result[5], y, y_bin, y_best, algo)
    pythia_result = pythia.pythia(
        pilot_result[5],
        y,
        y_bin,
        y_best,
        algo,
        opts,
        parallel_opts,
    )

    # read the actual output
    matlab_output = pd.read_csv(output_dir / "BO_poly/poly.csv")

    # get the accuracy, precision, recall
    matlab_accuracy = matlab_output["CV_model_accuracy"].values.astype(np.double)
    matlab_precision = matlab_output["CV_model_precision"].values.astype(np.double)
    matlab_recall = matlab_output["CV_model_recall"].values.astype(np.double)

    compare_performance(
        pythia_result,
        matlab_accuracy,
        matlab_precision,
        matlab_recall,
        len(algo),
        2.5,
    )


def test_pilot_num_pythia_grid_gaussian() -> None:
    """Test the integration of the Pilot and Pythia stages."""
    sample_data = SampleDataNum()

    x_sample = sample_data.x_sample
    y_sample = sample_data.y_sample
    feat_labels_sample = sample_data.feat_labels_sample
    opts_sample = sample_data.opts_sample
    pilot_opts = PilotOptions(None, None, opts_sample.analytic, opts_sample.n_tries)
    pilot = PilotStage(x_sample, y_sample, feat_labels_sample)
    pilot_result = pilot.pilot(x_sample, y_sample, feat_labels_sample, pilot_opts)

    opts = PythiaOptions(
        cv_folds=5,
        is_poly_krnl=False,
        use_weights=False,
        use_grid_search=True,
        params=None,
    )
    pythia = PythiaStage(pilot_result[5], y, y_bin, y_best, algo)
    pythia_result = pythia.pythia(
        pilot_result[5],
        y,
        y_bin,
        y_best,
        algo,
        opts,
        parallel_opts,
    )

    # read the actual output
    matlab_output = pd.read_csv(output_dir / "GS_gaussian/gridsearch_gaussian.csv")

    # get the accuracy, precision, recall
    matlab_accuracy = matlab_output["CV_model_accuracy"].values.astype(np.double)
    matlab_precision = matlab_output["CV_model_precision"].values.astype(np.double)
    matlab_recall = matlab_output["CV_model_recall"].values.astype(np.double)

    compare_performance(
        pythia_result,
        matlab_accuracy,
        matlab_precision,
        matlab_recall,
        len(algo),
        2.5,
    )


def test_pilot_num_pythia_grid_poly() -> None:
    """Test the integration of the Pilot and Pythia stages."""
    sample_data = SampleDataNum()

    x_sample = sample_data.x_sample
    y_sample = sample_data.y_sample
    feat_labels_sample = sample_data.feat_labels_sample
    opts_sample = sample_data.opts_sample
    pilot_opts = PilotOptions(None, None, opts_sample.analytic, opts_sample.n_tries)
    pilot = PilotStage(x_sample, y_sample, feat_labels_sample)
    pilot_result = pilot.pilot(x_sample, y_sample, feat_labels_sample, pilot_opts)

    opts = PythiaOptions(
        cv_folds=5,
        is_poly_krnl=True,
        use_weights=False,
        use_grid_search=True,
        params=None,
    )
    pythia = PythiaStage(pilot_result[5], y, y_bin, y_best, algo)
    pythia_result = pythia.pythia(
        pilot_result[5],
        y,
        y_bin,
        y_best,
        algo,
        opts,
        parallel_opts,
    )

    # read the actual output
    matlab_output = pd.read_csv(output_dir / "GS_poly/gridsearch_poly.csv")

    # get the accuracy, precision, recall
    matlab_accuracy = matlab_output["CV_model_accuracy"].values.astype(np.double)
    matlab_precision = matlab_output["CV_model_precision"].values.astype(np.double)
    matlab_recall = matlab_output["CV_model_recall"].values.astype(np.double)

    compare_performance(
        pythia_result,
        matlab_accuracy,
        matlab_precision,
        matlab_recall,
        len(algo),
        2.5,
    )


def test_pilot_analytic_pythia_grid_gaussian() -> None:
    sample_data = SampleData()

    x_sample = sample_data.x_sample
    y_sample = sample_data.y_sample
    feat_labels_sample = sample_data.feat_labels_sample
    opts_sample = PilotOptions(None, None, True, 5)
    pilot_opts = PilotOptions(None, None, opts_sample.analytic, opts_sample.n_tries)
    pilot = PilotStage(x_sample, y_sample, feat_labels_sample)
    pilot_result = pilot.pilot(x_sample, y_sample, feat_labels_sample, pilot_opts)

    opts = PythiaOptions(
        cv_folds=5,
        is_poly_krnl=False,
        use_weights=False,
        use_grid_search=True,
        params=None,
    )

    pythia = PythiaStage(pilot_result[5], y, y_bin, y_best, algo)
    pythia_result = pythia.pythia(
        pilot_result[5],
        y,
        y_bin,
        y_best,
        algo,
        opts,
        parallel_opts,
    )

    output_dir = script_dir / "pilot_pythia"
    # read the actual output
    matlab_output = pd.read_csv(output_dir / "analytic_gaussian_grid.csv")

    # get the accuracy, precision, recall
    matlab_accuracy = matlab_output["CV_model_accuracy"].values.astype(np.double)
    matlab_precision = matlab_output["CV_model_precision"].values.astype(np.double)
    matlab_recall = matlab_output["CV_model_recall"].values.astype(np.double)

    compare_performance(
        pythia_result,
        matlab_accuracy,
        matlab_precision,
        matlab_recall,
        len(algo),
        2.5,
    )


def test_pilot_analytic_pythia_grid_poly() -> None:
    sample_data = SampleData()

    x_sample = sample_data.x_sample
    y_sample = sample_data.y_sample
    feat_labels_sample = sample_data.feat_labels_sample
    opts_sample = PilotOptions(None, None, True, 5)
    pilot_opts = PilotOptions(None, None, opts_sample.analytic, opts_sample.n_tries)
    pilot = PilotStage(x_sample, y_sample, feat_labels_sample)
    pilot_result = pilot.pilot(x_sample, y_sample, feat_labels_sample, pilot_opts)

    opts = PythiaOptions(
        cv_folds=5,
        is_poly_krnl=True,
        use_weights=False,
        use_grid_search=True,
        params=None,
    )

    pythia = PythiaStage(pilot_result[5], y, y_bin, y_best, algo)
    pythia_result = pythia.pythia(
        pilot_result[5],
        y,
        y_bin,
        y_best,
        algo,
        opts,
        parallel_opts,
    )

    output_dir = script_dir / "pilot_pythia"
    # read the actual output
    matlab_output = pd.read_csv(output_dir / "analytic_poly_grid.csv")

    # get the accuracy, precision, recall
    matlab_accuracy = matlab_output["CV_model_accuracy"].values.astype(np.double)
    matlab_precision = matlab_output["CV_model_precision"].values.astype(np.double)
    matlab_recall = matlab_output["CV_model_recall"].values.astype(np.double)

    compare_performance(
        pythia_result,
        matlab_accuracy,
        matlab_precision,
        matlab_recall,
        len(algo),
        2.5,
    )


def test_pilot_analytic_pythia_BO_gaussian() -> None:
    sample_data = SampleData()

    x_sample = sample_data.x_sample
    y_sample = sample_data.y_sample
    feat_labels_sample = sample_data.feat_labels_sample
    opts_sample = PilotOptions(None, None, True, 5)
    pilot_opts = PilotOptions(None, None, opts_sample.analytic, opts_sample.n_tries)
    pilot = PilotStage(x_sample, y_sample, feat_labels_sample)
    pilot_result = pilot.pilot(x_sample, y_sample, feat_labels_sample, pilot_opts)

    opts = PythiaOptions(
        cv_folds=5,
        is_poly_krnl=False,
        use_weights=False,
        use_grid_search=False,
        params=None,
    )

    pythia = PythiaStage(pilot_result[5], y, y_bin, y_best, algo)
    pythia_result = pythia.pythia(
        pilot_result[5],
        y,
        y_bin,
        y_best,
        algo,
        opts,
        parallel_opts,
    )

    output_dir = script_dir / "pilot_pythia"
    # read the actual output
    matlab_output = pd.read_csv(output_dir / "analytic_gaussian_BO.csv")

    # get the accuracy, precision, recall
    matlab_accuracy = matlab_output["CV_model_accuracy"].values.astype(np.double)
    matlab_precision = matlab_output["CV_model_precision"].values.astype(np.double)
    matlab_recall = matlab_output["CV_model_recall"].values.astype(np.double)

    compare_performance(
        pythia_result,
        matlab_accuracy,
        matlab_precision,
        matlab_recall,
        len(algo),
        2.5,
    )


def test_pilot_analytic_pythia_BO_poly() -> None:
    sample_data = SampleData()

    x_sample = sample_data.x_sample
    y_sample = sample_data.y_sample
    feat_labels_sample = sample_data.feat_labels_sample
    opts_sample = PilotOptions(None, None, True, 5)
    pilot_opts = PilotOptions(None, None, opts_sample.analytic, opts_sample.n_tries)
    pilot = PilotStage(x_sample, y_sample, feat_labels_sample)
    pilot_result = pilot.pilot(x_sample, y_sample, feat_labels_sample, pilot_opts)

    opts = PythiaOptions(
        cv_folds=5,
        is_poly_krnl=True,
        use_weights=False,
        use_grid_search=False,
        params=None,
    )

    pythia = PythiaStage(pilot_result[5], y, y_bin, y_best, algo)
    pythia_result = pythia.pythia(
        pilot_result[5],
        y,
        y_bin,
        y_best,
        algo,
        opts,
        parallel_opts,
    )

    output_dir = script_dir / "pilot_pythia"
    # read the actual output
    matlab_output = pd.read_csv(output_dir / "analytic_poly_BO.csv")

    # get the accuracy, precision, recall
    matlab_accuracy = matlab_output["CV_model_accuracy"].values.astype(np.double)
    matlab_precision = matlab_output["CV_model_precision"].values.astype(np.double)
    matlab_recall = matlab_output["CV_model_recall"].values.astype(np.double)

    compare_performance(
        pythia_result,
        matlab_accuracy,
        matlab_precision,
        matlab_recall,
        len(algo),
        2.5,
    )
