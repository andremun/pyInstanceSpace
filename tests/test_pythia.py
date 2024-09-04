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
    """Test that the output of the `bayes_opt` function is as expected."""
    pythia_option = PythiaOptions(
        cv_folds=5,
        is_poly_krnl=False,
        use_weights=False,
        use_lib_svm=False,
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
        None,
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
    for i in range(len(algo_input)):
        if np.allclose(matlab_accuracy[i], pythiaOut.accuracy[i] * 100, atol=tol):
            pass
        else:
            print(f"Accuracy mismatch at index {i}")
            print(
                f"matlab accuracy is: {matlab_accuracy[i]}, python accuracy is: {pythiaOut.accuracy[i] * 100}"
            )

        if np.allclose(matlab_precision[i], pythiaOut.precision[i] * 100, atol=tol):
            pass
        else:
            print(f"Precision mismatch at index {i}")
            print(
                f"matlab precision is: {matlab_precision[i]}, python precision is: {pythiaOut.precision[i] * 100}"
            )

        if np.allclose(matlab_recall[i], pythiaOut.recall[i] * 100, atol=tol):
            pass
        else:
            print(f"Recall mismatch at index {i}")
            print(
                f"matlab recall is: {matlab_recall[i]}, python recall is: {pythiaOut.recall[i] * 100}"
            )

    for i in range(len(algo_input)):
        assert np.isclose(
            matlab_accuracy[i],
            pythiaOut.accuracy[i] * 100,
            atol=tol,
        ), f"Accuracy mismatch at index {i}"

        assert np.isclose(
            matlab_precision[i],
            pythiaOut.precision[i] * 100,
            atol=tol,
        ), f"Precision mismatch at index {i}"

        assert np.isclose(
            matlab_recall[i],
            pythiaOut.recall[i] * 100,
            atol=tol,
        ), f"Recall mismatch at index {i}"


if __name__ == "__main__":
    test_bayes_opt()
