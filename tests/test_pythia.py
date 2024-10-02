"""Test module for Pythia class to verify its functionality.

The file contains the tests for the Pythia class to verify its functionality.
The tests are compare the performance matrics including accurancy, precision and
recall of the Pythia class with the expected output
from the MATLAB implementation with diffcult kernel and optimisation.

Tests includes:
    - test_compute_znorm: Test that the output of the compute_znorm.
    - test_compare_output: Test that the output of the compute_znorm is as expected.
    - test_generate_params_true: Test that the output of the compute_znorm is as
        expected.
    - test_bayes_opt: Test that the output of the function is as expected when BO is
        required.
    - test_bayes_opt_poly: Test that the output of the function is as expected when BO
        and polykernal is required.
    - test_grid_gaussian: Test that the performance of model is asexpected when grid
        search & gaussian.
    - test_grid_poly: Test that the performance of model is asexpected when grid search
        & poly.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from skopt.space import Real

from matilda.data.options import PythiaOptions
from matilda.stages.pythia import PythiaStage

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
default_opts = PythiaOptions.default()
opt = PythiaOptions(
    cv_folds=5,
    is_poly_krnl=False,
    use_weights=False,
    use_grid_search=True,
    params=None,
)


def test_compute_znorm() -> None:
    """Test that the output of the compute_znorm."""
    znorm = np.genfromtxt(csv_path_znorm_input, delimiter=",")

    pythia = PythiaStage(z, y, y_bin, y_best, algo)
    _, _, znorm_test = pythia._compute_znorm(z)  # noqa: SLF001
    assert np.allclose(znorm, znorm_test)


def test_compare_output() -> None:
    """Test that the output of the compute_znorm is as expected."""
    pythia = PythiaStage(z, y, y_bin, y_best, algo)
    pythia_out = pythia.pythia(z, y, y_bin, y_best, algo, opt)
    mu = np.genfromtxt(csv_path_mu_input, delimiter=",")

    assert np.allclose(mu, pythia_out[0])
    assert pythia_out[3].get_n_splits() == opt.cv_folds


def test_generate_params_true() -> None:
    """Test that the output of the compute_znorm is as expected."""
    min_value = 2**-10
    max_value = 2**4
    rng = np.random.default_rng(seed=0)

    params = PythiaStage._generate_params(opt.use_grid_search, rng) # noqa: SLF001
    assert all(min_value <= param <= max_value for param in params["C"])
    assert all(min_value <= param <= max_value for param in params["gamma"])


def test_bayes_opt() -> None:
    """Test that the output of the function is as expected when BO is required."""
    opts = PythiaOptions(
        cv_folds=5,
        is_poly_krnl=False,
        use_weights=False,
        use_grid_search=False,
        params=None,
    )
    pythia = PythiaStage(z, y, y_bin, y_best, algo)
    pythia_out = pythia.pythia(z, y, y_bin, y_best, algo, opts)

    # read the actual output
    matlab_output = pd.read_csv(output_dir / "BO_gaussian/gaussian.csv")

    # get the accuracy, precision, recall
    matlab_accuracy = matlab_output["CV_model_accuracy"].values.astype(np.double)
    matlab_precision = matlab_output["CV_model_precision"].values.astype(np.double)
    matlab_recall = matlab_output["CV_model_recall"].values.astype(np.double)

    print(pythia_out[12])
    print("====================================")
    print(matlab_accuracy)
    print("====================================")
    print(pythia_out[13])
    print("====================================")
    print(matlab_precision)
    print("====================================")
    print(pythia_out[14])
    print("====================================")
    print(matlab_recall)

    compare_performance(
        pythia_out,
        matlab_accuracy,
        matlab_precision,
        matlab_recall,
        len(algo),
        2.5,
    )


def test_bayes_opt_poly() -> None:
    """Test that the output of the function is as expected when BO and polykernal is required."""  # noqa: E501
    opts = PythiaOptions(
        cv_folds=5,
        is_poly_krnl=True,
        use_weights=False,
        use_grid_search=False,
        params=None,
    )
    pythia = PythiaStage(z, y, y_bin, y_best, algo)
    pythia_out = pythia.pythia(z, y, y_bin, y_best, algo, opts)

    # read the actual output
    matlab_output = pd.read_csv(output_dir / "BO_poly/poly.csv")

    # get the accuracy, precision, recall
    matlab_accuracy = matlab_output["CV_model_accuracy"].values.astype(np.double)
    matlab_precision = matlab_output["CV_model_precision"].values.astype(np.double)
    matlab_recall = matlab_output["CV_model_recall"].values.astype(np.double)

    compare_performance(
        pythia_out,
        matlab_accuracy,
        matlab_precision,
        matlab_recall,
        len(algo),
        2.5,
    )


def test_grid_gaussian() -> None:
    """Test that the performance of model is asexpected when grid search & gaussian."""
    opts = PythiaOptions(
        cv_folds=5,
        is_poly_krnl=False,
        use_weights=False,
        use_grid_search=True,
        params=None,
    )
    pythia = PythiaStage(z, y, y_bin, y_best, algo)
    pythia_out = pythia.pythia(z, y, y_bin, y_best, algo, opts)
    # read the actual output
    matlab_accuracy = pd.read_csv(
        output_dir / "gridsearch_gaussian/accuracy.csv",
        header=None,
    ).values
    matlab_precision = pd.read_csv(
        output_dir / "gridsearch_gaussian/precision.csv",
        header=None,
    ).values
    matlab_recall = pd.read_csv(
        output_dir / "gridsearch_gaussian/recall.csv",
        header=None,
    ).values
    compare_performance(
        pythia_out,
        matlab_accuracy,
        matlab_precision,
        matlab_recall,
        len(algo),
        2.5,
    )


def test_grid_poly() -> None:
    """Test that the performance of model is asexpected when grid search & poly ."""
    opts = PythiaOptions(
        cv_folds=5,
        is_poly_krnl=True,
        use_weights=False,
        use_grid_search=True,
        params=None,
    )
    pythia = PythiaStage(z, y, y_bin, y_best, algo)
    pythia_out = pythia.pythia(z, y, y_bin, y_best, algo, opts)

    # read the actual output
    matlab_accuracy = pd.read_csv(
        output_dir / "gridsearch_polynomial/accuracy.csv",
        header=None,
    ).values
    matlab_precision = pd.read_csv(
        output_dir / "gridsearch_polynomial/precision.csv",
        header=None,
    ).values
    matlab_recall = pd.read_csv(
        output_dir / "gridsearch_polynomial/recall.csv",
        header=None,
    ).values

    compare_performance(
        pythia_out,
        matlab_accuracy,
        matlab_precision,
        matlab_recall,
        len(algo),
        2.5,
    )


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
    threshold = 0.9

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
