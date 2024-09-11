"""Test input parameters, particularly metrics, are accurately parsed and stored."""

from pathlib import Path

import numpy as np
import pandas as pd

from matilda.data.options import PythiaOptions
from matilda.stages.pythia import Pythia

script_dir = Path(__file__).parent
output_dir = script_dir / "test_data/pythia/output"

csv_path_z_input = script_dir / "test_data/pythia/input/Z.csv"
csv_path_y_input = script_dir / "test_data/pythia/input/y.csv"
csv_path_algo_input = script_dir / "test_data/pythia/input/algolabels.csv"
csv_path_y_best_input = script_dir / "test_data/pythia/input/ybest.csv"
csv_path_y_bin_input = script_dir / "test_data/pythia/input/ybin.csv"


def test_bayes_opt() -> None:
    """Test that the output of the function is as expected when BO is required."""
    pythia_option = PythiaOptions(
        cv_folds=5,
        is_poly_krnl=False,
        use_weights=False,
        use_grid_search=False,
    )
    z_input = pd.read_csv(csv_path_z_input, header=None).values
    y_input = pd.read_csv(csv_path_y_input, header=None).values
    algo_input = pd.read_csv(csv_path_algo_input, header=None).squeeze().tolist()
    y_best_input = pd.read_csv(csv_path_y_best_input, header=None).values
    y_bin_input = pd.read_csv(csv_path_y_bin_input, header=None).values
    z_input = np.array(z_input)
    y_input = np.array(y_input)
    y_best_input = np.array(y_best_input)
    y_bin_input = np.array(y_bin_input)
    [_, pythiaOut] = Pythia.run(  # noqa: N806
        z_input,
        y_input,
        y_bin_input,
        y_best_input,
        algo_input,
        pythia_option,
    )

    # read the actual output
    matlab_output = pd.read_csv(output_dir / "BO/gaussian.csv")

    # get the accuracy, precision, recall
    matlab_accuracy = matlab_output["CV_model_accuracy"].values
    matlab_precision = matlab_output["CV_model_precision"].values
    matlab_recall = matlab_output["CV_model_recall"].values

    tol = 2.5

    # compare the output and check the tolerance, the tolerance should within 2.5%
    # if 90% passed, the test is considered passed
    total = 0
    correct = 0
    threshold = 0.9

    for i in range(len(algo_input)):
        total += 3
        if np.allclose(matlab_accuracy[i], pythiaOut.accuracy[i] * 100, atol=tol):
            correct += 1

        if np.allclose(matlab_precision[i], pythiaOut.precision[i] * 100, atol=tol):
            correct += 1

        if np.allclose(matlab_recall[i], pythiaOut.recall[i] * 100, atol=tol):
            correct += 1

    assert correct / total >= threshold


def test_grid_gaussian() -> None:
    """Test that the output of the function is as expected when grid search & gaussian ."""
    pythia_option = PythiaOptions(
        cv_folds=5,
        is_poly_krnl=False,
        use_weights=False,
        use_grid_search=True,
    )
    z_input = pd.read_csv(csv_path_z_input, header=None).values
    y_input = pd.read_csv(csv_path_y_input, header=None).values
    algo_input = pd.read_csv(csv_path_algo_input, header=None).squeeze().tolist()
    y_best_input = pd.read_csv(csv_path_y_best_input, header=None).values
    y_bin_input = pd.read_csv(csv_path_y_bin_input, header=None).values
    z_input = np.array(z_input)
    y_input = np.array(y_input)
    y_best_input = np.array(y_best_input)
    y_bin_input = np.array(y_bin_input)
    [_, pythiaOut] = Pythia.run(  # noqa: N806
        z_input,
        y_input,
        y_bin_input,
        y_best_input,
        algo_input,
        pythia_option,
    )

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

    tol = 2.5

    # compare the output and check the tolerance, the tolerance should within 2.5%
    # if 90% passed, the test is considered passed
    total = 0
    correct = 0
    threshold = 0.9

    for i in range(len(algo_input)):
        total += 3
        # check if the accuracy is higher than the matlab output
        if pythiaOut.accuracy[i] * 100 >= matlab_accuracy[i] or abs(pythiaOut.accuracy[i] * 100 - matlab_accuracy[i]) <= tol:
            correct += 1

        # check precision
        if pythiaOut.precision[i] * 100 >= matlab_precision[i] or abs(pythiaOut.precision[i] * 100 - matlab_precision[i]) <= tol:
            correct += 1

        # check recall
        if pythiaOut.recall[i] * 100 >= matlab_recall[i] or abs(pythiaOut.recall[i] * 100 - matlab_recall[i]) <= tol:
            correct += 1

    assert correct / total >= threshold

if __name__ == "__main__":
    test_bayes_opt()
    test_grid_gaussian()
