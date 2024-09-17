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
    pythia_out = Pythia.run(z, y, y_bin, y_best, algo, opt)[1]
    print(pythia_out.sigma)
    print(pythia_out.mu)

    print(pythia_out.cp)
    print(pythia_out.svm)
    # sigma = np.genfromtxt(csv_path_sig_input, delimiter=",")
    mu = np.genfromtxt(csv_path_mu_input, delimiter=",")
    # assert np.allclose(sigma, pythia_out.sigma)
    assert np.allclose(mu, pythia_out.mu)

    assert pythia_out.cp.get_n_splits() == opt.cv_folds


def test_generate_params() -> None:
    """Test that the output of the generate_params function is as expected."""
    min_value = 2**-10
    max_value = 2**4

    params = pythia.generate_params(True)
    assert all(min_value <= param <= max_value for param in params["C"])
    assert all(min_value <= param <= max_value for param in params["gamma"])

    params = pythia.generate_params(False)
    # Check the bounds of the 'gamma' parameter
    assert params["C"].low == min_value
    assert params["C"].high == max_value
    assert params["C"].prior == "log-uniform"

    assert params["gamma"].low == min_value
    assert params["gamma"].high == max_value
    assert params["gamma"].prior == "log-uniform"




# def test_performance():


if __name__ == "__main__":
    # test_compute_znorm()
    test_compare_output()
    # test_generate_params()
# def test_bayes_opt() -> None:
#     """Test that the output of the function is as expected when BO is required."""
#     pythia_option = PythiaOptions(
#         cv_folds=5,
#         is_poly_krnl=False,
#         use_weights=False,
#         use_grid_search=False,
#         params=None,
#     )
#     z_input = pd.read_csv(csv_path_z_input, header=None).values
#     y_input = pd.read_csv(csv_path_y_input, header=None).values
#     algo_input = pd.read_csv(csv_path_algo_input, header=None).squeeze().tolist()
#     y_best_input = pd.read_csv(csv_path_y_best_input, header=None).values
#     y_bin_input = pd.read_csv(csv_path_y_bin_input, header=None).values
#     z_input = np.array(z_input)
#     y_input = np.array(y_input)
#     y_best_input = np.array(y_best_input)
#     y_bin_input = np.array(y_bin_input)
#     [_, pythiaOut] = Pythia.run(
#         z_input,
#         y_input,
#         y_bin_input,
#         y_best_input,
#         algo_input,
#         pythia_option,
#     )

#     # read the actual output
#     matlab_output = pd.read_csv(output_dir / "BO_gaussian/gaussian.csv")

#     # get the accuracy, precision, recall
#     matlab_accuracy = matlab_output["CV_model_accuracy"].values
#     matlab_precision = matlab_output["CV_model_precision"].values
#     matlab_recall = matlab_output["CV_model_recall"].values

#     tol = 2.5

#     # compare the output and check the tolerance, the tolerance should within 2.5%
#     # if 90% passed, the test is considered passed
#     total = 0
#     correct = 0
#     threshold = 0.9

#     for i in range(len(algo_input)):
#         total += 3
#         # check if the accuracy is higher than the matlab output
#         if pythiaOut.accuracy[i] * 100 >= matlab_accuracy[i]:
#             correct += 1
#         # if lower, check if the difference is within the tolerance
#         elif abs(pythiaOut.accuracy[i] * 100 - matlab_accuracy[i]) <= tol:
#             correct += 1

#         # check precision
#         if pythiaOut.precision[i] * 100 >= matlab_precision[i]:
#             correct += 1
#         elif abs(pythiaOut.precision[i] * 100 - matlab_precision[i]) <= tol:
#             correct += 1

#         # check recall
#         if pythiaOut.recall[i] * 100 >= matlab_recall[i]:
#             correct += 1
#         elif abs(pythiaOut.recall[i] * 100 - matlab_recall[i]) <= tol:
#             correct += 1

#     assert correct / total >= threshold


# def test_bayes_opt_poly() -> None:
#     """Test that the output of the function is as expected when BO is required."""
#     pythia_option = PythiaOptions(
#         cv_folds=5,
#         is_poly_krnl=True,
#         use_weights=False,
#         use_grid_search=False,
#         params=None,
#     )
#     z_input = pd.read_csv(csv_path_z_input, header=None).values
#     y_input = pd.read_csv(csv_path_y_input, header=None).values
#     algo_input = pd.read_csv(csv_path_algo_input, header=None).squeeze().tolist()
#     y_best_input = pd.read_csv(csv_path_y_best_input, header=None).values
#     y_bin_input = pd.read_csv(csv_path_y_bin_input, header=None).values
#     z_input = np.array(z_input)
#     y_input = np.array(y_input)
#     y_best_input = np.array(y_best_input)
#     y_bin_input = np.array(y_bin_input)
#     [_, pythiaOut] = Pythia.run(
#         z_input,
#         y_input,
#         y_bin_input,
#         y_best_input,
#         algo_input,
#         pythia_option,
#     )

#     # read the actual output
#     matlab_output = pd.read_csv(output_dir / "BO_poly/poly.csv")

#     # get the accuracy, precision, recall
#     matlab_accuracy = matlab_output["CV_model_accuracy"].values
#     matlab_precision = matlab_output["CV_model_precision"].values
#     matlab_recall = matlab_output["CV_model_recall"].values

#     tol = 2.5

#     # compare the output and check the tolerance, the tolerance should within 2.5%
#     # if 90% passed, the test is considered passed
#     total = 0
#     correct = 0
#     threshold = 0.9

#     for i in range(len(algo_input)):
#         total += 3
#         # check if the accuracy is higher than the matlab output
#         if pythiaOut.accuracy[i] * 100 >= matlab_accuracy[i]:
#             correct += 1
#         # if lower, check if the difference is within the tolerance
#         elif abs(pythiaOut.accuracy[i] * 100 - matlab_accuracy[i]) <= tol:
#             correct += 1

#         # check precision
#         if pythiaOut.precision[i] * 100 >= matlab_precision[i]:
#             correct += 1
#         elif abs(pythiaOut.precision[i] * 100 - matlab_precision[i]) <= tol:
#             correct += 1

#         # check recall
#         if pythiaOut.recall[i] * 100 >= matlab_recall[i]:
#             correct += 1
#         elif abs(pythiaOut.recall[i] * 100 - matlab_recall[i]) <= tol:
#             correct += 1

#     assert correct / total >= threshold


# def test_grid_gaussian() -> None:
#     """Test that the output of the function is as expected when grid search & gaussian ."""
#     pythia_option = PythiaOptions(
#         cv_folds=5,
#         is_poly_krnl=False,
#         use_weights=False,
#         use_grid_search=True,
#         params=None,
#     )
#     z_input = pd.read_csv(csv_path_z_input, header=None).values
#     y_input = pd.read_csv(csv_path_y_input, header=None).values
#     algo_input = pd.read_csv(csv_path_algo_input, header=None).squeeze().tolist()
#     y_best_input = pd.read_csv(csv_path_y_best_input, header=None).values
#     y_bin_input = pd.read_csv(csv_path_y_bin_input, header=None).values
#     z_input = np.array(z_input)
#     y_input = np.array(y_input)
#     y_best_input = np.array(y_best_input)
#     y_bin_input = np.array(y_bin_input)
#     [_, pythiaOut] = Pythia.run(
#         z_input,
#         y_input,
#         y_bin_input,
#         y_best_input,
#         algo_input,
#         pythia_option,
#     )

#     # read the actual output
#     matlab_accuracy = pd.read_csv(
#         output_dir / "gridsearch_gaussian/accuracy.csv",
#         header=None,
#     ).values
#     matlab_precision = pd.read_csv(
#         output_dir / "gridsearch_gaussian/precision.csv",
#         header=None,
#     ).values
#     matlab_recall = pd.read_csv(
#         output_dir / "gridsearch_gaussian/recall.csv",
#         header=None,
#     ).values

#     tol = 2.5

#     # compare the output and check the tolerance, the tolerance should within 2.5%
#     # if 90% passed, the test is considered passed
#     total = 0
#     correct = 0
#     threshold = 0.9

#     for i in range(len(algo_input)):
#         total += 3
#         # check if the accuracy is higher than the matlab output
#         if pythiaOut.accuracy[i] * 100 >= matlab_accuracy[i]:
#             correct += 1
#         # if lower, check if the difference is within the tolerance
#         elif abs(pythiaOut.accuracy[i] * 100 - matlab_accuracy[i]) <= tol:
#             correct += 1

#         # check precision
#         if pythiaOut.precision[i] * 100 >= matlab_precision[i]:
#             correct += 1
#         elif abs(pythiaOut.precision[i] * 100 - matlab_precision[i]) <= tol:
#             correct += 1

#         # check recall
#         if pythiaOut.recall[i] * 100 >= matlab_recall[i]:
#             correct += 1
#         elif abs(pythiaOut.recall[i] * 100 - matlab_recall[i]) <= tol:
#             correct += 1

#     assert correct / total >= threshold


# def test_grid_poly() -> None:
#     """Test that the output of the function is as expected when grid search & gaussian ."""
#     pythia_option = PythiaOptions(
#         cv_folds=5,
#         is_poly_krnl=True,
#         use_weights=False,
#         use_grid_search=True,
#         params=None,
#     )
#     z_input = pd.read_csv(csv_path_z_input, header=None).values
#     y_input = pd.read_csv(csv_path_y_input, header=None).values
#     algo_input = pd.read_csv(csv_path_algo_input, header=None).squeeze().tolist()
#     y_best_input = pd.read_csv(csv_path_y_best_input, header=None).values
#     y_bin_input = pd.read_csv(csv_path_y_bin_input, header=None).values
#     z_input = np.array(z_input)
#     y_input = np.array(y_input)
#     y_best_input = np.array(y_best_input)
#     y_bin_input = np.array(y_bin_input)
#     [_, pythiaOut] = Pythia.run(
#         z_input,
#         y_input,
#         y_bin_input,
#         y_best_input,
#         algo_input,
#         pythia_option,
#     )

#     # read the actual output
#     matlab_accuracy = pd.read_csv(
#         output_dir / "gridsearch_polynomial/accuracy.csv",
#         header=None,
#     ).values
#     matlab_precision = pd.read_csv(
#         output_dir / "gridsearch_polynomial/precision.csv",
#         header=None,
#     ).values
#     matlab_recall = pd.read_csv(
#         output_dir / "gridsearch_polynomial/recall.csv",
#         header=None,
#     ).values

#     tol = 2.5

#     # compare the output and check the tolerance, the tolerance should within 2.5%
#     # if 90% passed, the test is considered passed
#     total = 0
#     correct = 0
#     threshold = 0.9

#     for i in range(len(algo_input)):
#         total += 3
#         # check if the accuracy is higher than the matlab output
#         if pythiaOut.accuracy[i] * 100 >= matlab_accuracy[i]:
#             correct += 1
#         # if lower, check if the difference is within the tolerance
#         elif abs(pythiaOut.accuracy[i] * 100 - matlab_accuracy[i]) <= tol:
#             correct += 1

#         # check precision
#         if pythiaOut.precision[i] * 100 >= matlab_precision[i]:
#             correct += 1
#         elif abs(pythiaOut.precision[i] * 100 - matlab_precision[i]) <= tol:
#             correct += 1

#         # check recall
#         if pythiaOut.recall[i] * 100 >= matlab_recall[i]:
#             correct += 1
#         elif abs(pythiaOut.recall[i] * 100 - matlab_recall[i]) <= tol:
#             correct += 1

#     assert correct / total >= threshold

# def test_compute_znorm():
#     pd.read_csv(csv_path_znorm_input, header=None).values
#     np.allclose()
