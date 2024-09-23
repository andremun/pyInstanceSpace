"""Test input parameters, particularly metrics, are accurately parsed and stored."""

from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from matilda.data.options import PythiaOptions
from matilda.stages.pythia import Pythia, PythiaOut

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
pythia = Pythia(z, y, y_bin, y_best, algo, default_opts)


def test_compute_znorm() -> None:
    """Test that the output of the compute_znorm."""
    znorm = np.genfromtxt(csv_path_znorm_input, delimiter=",")
    # Test the compute_znorm function from the Pythia class
    znorm_test = pythia.compute_znorm(z)
    # Check if the results from compute_znorm match the expected values in znorm
    assert np.allclose(znorm, znorm_test)


def test_compare_output() -> None:
    """Test that the output of the compute_znorm is as expected."""
    pythia_out = Pythia.run(z, y, y_bin, y_best, algo, opt)[1]
    print(pythia_out.sigma)
    mu = np.genfromtxt(csv_path_mu_input, delimiter=",")
    assert np.allclose(mu, pythia_out.mu)

    assert pythia_out.cp.get_n_splits() == opt.cv_folds


def test_generate_params_true() -> None:
    """Test that the output of the generate_params function is as expected."""
    min_value = 2**-10
    max_value = 2**4

    params = pythia.generate_params(True)
    assert all(min_value <= param <= max_value for param in params["C"])
    assert all(min_value <= param <= max_value for param in params["gamma"])

def test_generate_params_false() -> None:
    """Test that the output of the generate_params function is as expected."""
    min_value = 2**-10
    max_value = 2**4
    params = pythia.generate_params(False)
    # Check the bounds of the 'gamma' parameter
    assert params["C"].low == min_value
    assert params["C"].high == max_value
    assert params["C"].prior == "log-uniform"

    assert params["gamma"].low == min_value
    assert params["gamma"].high == max_value
    assert params["gamma"].prior == "log-uniform"


def test_bayes_opt() -> None:
    """Test that the output of the function is as expected when BO is required."""
    opts = PythiaOptions(
        cv_folds=5,
        is_poly_krnl=False,
        use_weights=False,
        use_grid_search=False,
        params=None,
    )
    pythia_output = Pythia.run(
        z,
        y,
        y_bin,
        y_best,
        algo,
        opts,
    )[1]

    # read the actual output
    matlab_output = pd.read_csv(output_dir / "BO_gaussian/gaussian.csv")

    # get the accuracy, precision, recall
    matlab_accuracy = matlab_output["CV_model_accuracy"].values.astype(np.double)
    matlab_precision = matlab_output["CV_model_precision"].values.astype(np.double)
    matlab_recall = matlab_output["CV_model_recall"].values.astype(np.double)

    compare_performance(
        pythia_output,
        matlab_accuracy,
        matlab_precision,
        matlab_recall,
        len(algo),
        2.5,
    )


def test_bayes_opt_poly() -> None:
    """Test that the output of the function is as expected when BO is required."""
    opts = PythiaOptions(
        cv_folds=5,
        is_poly_krnl=True,
        use_weights=False,
        use_grid_search=False,
        params=None,
    )
    pythia_output = Pythia.run(
        z,
        y,
        y_bin,
        y_best,
        algo,
        opts,
    )[1]

    # read the actual output
    matlab_output = pd.read_csv(output_dir / "BO_poly/poly.csv")

    # get the accuracy, precision, recall
    matlab_accuracy = matlab_output["CV_model_accuracy"].values.astype(np.double)
    matlab_precision = matlab_output["CV_model_precision"].values.astype(np.double)
    matlab_recall = matlab_output["CV_model_recall"].values.astype(np.double)

    compare_performance(
        pythia_output,
        matlab_accuracy,
        matlab_precision,
        matlab_recall,
        len(algo),
        2.5,
    )


def test_grid_gaussian() -> None:
    """Test that the performance of model is asexpected when grid search & gaussian ."""
    opts = PythiaOptions(
        cv_folds=5,
        is_poly_krnl=False,
        use_weights=False,
        use_grid_search=True,
        params=None,
    )
    pythia_output = Pythia.run(
        z,
        y,
        y_bin,
        y_best,
        algo,
        opts,
    )[1]

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
        pythia_output,
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
    pythia_output = Pythia.run(
        z,
        y,
        y_bin,
        y_best,
        algo,
        opts,
    )[1]

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
        pythia_output,
        matlab_accuracy,
        matlab_precision,
        matlab_recall,
        len(algo),
        2.5,
    )


def compare_performance(
    python_output: PythiaOut,
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
            python_output.accuracy[i] * 100 >= matlab_accuracy[i]
            or abs(python_output.accuracy[i] * 100 - matlab_accuracy[i]) <= tol
        ):
            correct += 1

        if (
            python_output.precision[i] * 100 >= matlab_precision[i]
            or abs(python_output.precision[i] * 100 - matlab_precision[i]) <= tol
        ):
            correct += 1

        if (
            python_output.recall[i] * 100 >= matlab_recall[i]
            or abs(python_output.recall[i] * 100 - matlab_recall[i]) <= tol
        ):
            correct += 1

    assert correct / total >= threshold
