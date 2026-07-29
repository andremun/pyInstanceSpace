"""Test module for Pythia class to verify its functionality.

The file contains the tests for the Pythia class to verify its functionality.
The tests are compare the performance matrics including accurancy, precision and
recall of the Pythia class with the expected output
from the MATLAB implementation with diffcult kernel and optimisation.

Tests includes:
    - test_compute_znorm: Test that the output of the compute_znorm.
    - test_generate_params: Test that the generated param space is expected for BO
    - test_bayes_opt_gaussian: Test that the output of the function is as expected
        when BO is required.
    - test_bayes_opt_poly: Test that the output of the function is as expected
        when BO and polykernal is required.
    - test_compare_output: Test that the output of the compute_znorm is as expected.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from numpy.typing import NDArray
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC

from instancespace.data.options import GeneralOptions, ParallelOptions, PythiaOptions
from instancespace.stages.pythia import PythiaStage
from instancespace.utils.get_classifier_fcn import get_classifier_fcn

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
    params=None,
)

parallel_opts = ParallelOptions(
    flag=True,
    n_cores=2,
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
    pythia_out = pythia.pythia(
        z,
        y,
        y_bin,
        y_best,
        algo,
        opt,
        ParallelOptions.default(),
        GeneralOptions.default(),
    )
    mu = np.genfromtxt(csv_path_mu_input, delimiter=",")

    assert np.allclose(mu, pythia_out[0])
    assert pythia_out[3].get_n_splits() == opt.cv_folds


def test_pythia_does_not_mutate_y_raw() -> None:
    """Regression test for #229: PYTHIA must not mutate the caller's y array.

    _generate_summary previously mutated its `y` argument in place
    (`y[~sel0] = np.nan`) without copying it first, unlike the y_full/y_svms
    variables derived from it. Since `y` is the same array object as the
    caller's y_raw, this silently corrupted the caller's data.
    """
    y_input = y.copy()
    y_before = y_input.copy()
    PythiaStage.pythia(
        z,
        y_input,
        y_bin,
        y_best,
        algo,
        opt,
        ParallelOptions.default(),
        GeneralOptions.default(),
    )
    assert np.array_equal(y_input, y_before)


def test_pythia_seed_reproducibility() -> None:
    """Same seed gives identical output; a different seed gives different output.

    Regression test for Q9 (general.seed threading): a tiny synthetic dataset
    keeps this test fast while still exercising the SVM/cross-validation code
    paths that consume `general_options.seed`.
    """
    rng = np.random.default_rng(0)
    ninst = 20
    nalgos = 2
    coin_flip_threshold = 0.5
    z_small = rng.random((ninst, 2))
    y_small = rng.random((ninst, nalgos))
    y_bin_small = rng.random((ninst, nalgos)) > coin_flip_threshold
    y_best_small = rng.random(ninst)
    algo_small = ["a0", "a1"]
    small_opts = PythiaOptions(
        cv_folds=2,
        is_poly_krnl=False,
        use_weights=False,
        params=None,
    )

    def run(seed: int) -> NDArray[np.double]:
        out = PythiaStage.pythia(
            z_small,
            y_small,
            y_bin_small,
            y_best_small,
            algo_small,
            small_opts,
            ParallelOptions.default(),
            GeneralOptions(verbose=False, seed=seed),
        )
        return out.pr0_hat

    pr0_hat_a = run(0)
    pr0_hat_b = run(0)
    pr0_hat_c = run(1)

    np.testing.assert_array_equal(pr0_hat_a, pr0_hat_b)
    assert not np.array_equal(pr0_hat_a, pr0_hat_c)


def test_generate_params() -> None:
    """Test that the range of generated param space is expected."""
    min_value = 2**-10
    max_value = 2**4
    rng = np.random.default_rng(seed=0)

    params = PythiaStage._generate_params(rng)  # noqa: SLF001
    assert all(min_value <= param <= max_value for param in params["C"])
    assert all(min_value <= param <= max_value for param in params["gamma"])


def test_bayes_opt_gaussian() -> None:
    """Test that the output of the function is as expected when BO is required."""
    opts = PythiaOptions(
        cv_folds=5,
        is_poly_krnl=False,
        use_weights=False,
        params=None,
        tuning="bayes",  # exercise the pre-F10 Bayes-search path, not sobol
    )
    pythia = PythiaStage(z, y, y_bin, y_best, algo)
    pythia_out = pythia.pythia(
        z,
        y,
        y_bin,
        y_best,
        algo,
        opts,
        parallel_opts,
        GeneralOptions.default(),
    )

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
        params=None,
        tuning="bayes",  # exercise the pre-F10 Bayes-search path, not sobol
    )
    pythia = PythiaStage(z, y, y_bin, y_best, algo)
    pythia_out = pythia.pythia(
        z,
        y,
        y_bin,
        y_best,
        algo,
        opts,
        parallel_opts,
        GeneralOptions.default(),
    )

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


def _small_pythia_dataset() -> tuple[
    NDArray[np.double],
    NDArray[np.double],
    NDArray[np.bool_],
    NDArray[np.double],
    list[str],
]:
    """Build a tiny synthetic dataset for fast, classifier-agnostic PYTHIA tests."""
    rng = np.random.default_rng(0)
    ninst = 20
    nalgos = 2
    coin_flip_threshold = 0.5
    z_small = rng.random((ninst, 2))
    y_small = rng.random((ninst, nalgos))
    y_bin_small = rng.random((ninst, nalgos)) > coin_flip_threshold
    y_best_small = rng.random(ninst)
    algo_small = ["a0", "a1"]
    return z_small, y_small, y_bin_small, y_best_small, algo_small


@pytest.mark.parametrize("classifier", ["knn", "tree", "nb", "linear", "ensemble"])
@pytest.mark.parametrize("tuning", ["sobol", "bayes"])
def test_pythia_trains_each_registered_classifier(classifier: str, tuning: str) -> None:
    """PYTHIA can train and tune every registered non-SVM classifier (#65).

    Runs end-to-end on a tiny synthetic dataset (not the MATLAB reference,
    which only covers 'svm') and checks the output has the right shape, the
    right estimator type, and a hyperparameter reported inside that
    classifier's own registered search range (`get_classifier_fcn`) - not
    numeric parity with MATLAB, since these five classifiers have no
    MATLAB-verified reference to compare against.
    """
    z_small, y_small, y_bin_small, y_best_small, algo_small = _small_pythia_dataset()
    opts = PythiaOptions(
        cv_folds=2,
        is_poly_krnl=False,
        use_weights=False,
        params=None,
        classifier=classifier,
        tuning=tuning,
        n_tuning_iter=4,
    )

    out = PythiaStage.pythia(
        z_small,
        y_small,
        y_bin_small,
        y_best_small,
        algo_small,
        opts,
        ParallelOptions.default(),
        GeneralOptions(verbose=False, seed=0),
    )

    assert len(out.svm) == len(algo_small)
    for clf in out.svm:
        assert type(clf).__name__ != "SVC"
    assert out.y_hat.shape == y_bin_small.shape
    assert out.pr0_hat.shape == y_bin_small.shape

    spec = get_classifier_fcn(classifier)
    # Every registered classifier now has a real, tuned first hyperparameter.
    assert all(not np.isnan(c) for c in out.box_consnt)
    if spec.param2 is None:
        # tree/nb/linear only have one tunable hyperparameter - k_scale stays NaN.
        assert all(np.isnan(g) for g in out.k_scale)
    else:
        assert all(not np.isnan(g) for g in out.k_scale)
        assert spec.param2.label in out.pythia_summary.columns


def test_pythia_default_classifier_is_svm() -> None:
    """PythiaOptions.default() still trains SVMs, matching pre-F1 behaviour."""
    assert PythiaOptions.default().classifier == "svm"

    z_small, y_small, y_bin_small, y_best_small, algo_small = _small_pythia_dataset()
    small_opts = PythiaOptions(
        cv_folds=2,
        is_poly_krnl=False,
        use_weights=False,
        params=None,
    )
    out = PythiaStage.pythia(
        z_small,
        y_small,
        y_bin_small,
        y_best_small,
        algo_small,
        small_opts,
        ParallelOptions.default(),
        GeneralOptions(verbose=False, seed=0),
    )
    assert all(isinstance(clf, SVC) for clf in out.svm)


def test_pythia_knn_ignores_use_weights_with_a_warning() -> None:
    """A classifier without sample_weight support degrades gracefully, not a crash."""
    z_small, y_small, y_bin_small, y_best_small, algo_small = _small_pythia_dataset()
    opts = PythiaOptions(
        cv_folds=2,
        is_poly_krnl=False,
        use_weights=True,
        params=None,
        classifier="knn",
    )

    # Should not raise, despite use_weights=True and KNN not supporting it.
    out = PythiaStage.pythia(
        z_small,
        y_small,
        y_bin_small,
        y_best_small,
        algo_small,
        opts,
        ParallelOptions.default(),
        GeneralOptions(verbose=False, seed=0),
    )
    assert len(out.svm) == len(algo_small)


def test_pythia_default_tuning_is_sobol() -> None:
    """F10: `PythiaOptions.default()` uses Sobol tuning, matching MATLAB's default."""
    assert PythiaOptions.default().tuning == "sobol"


def test_pythia_sobol_produces_valid_svm_hyperparameters() -> None:
    """F10: the Sobol search picks a C/gamma pair inside the searched range.

    Mirrors MATLAB's `sobolToParams` range for SVM: C, gamma in [2^-10, 2^4].
    """
    min_value = 2**-10
    max_value = 2**4
    z_small, y_small, y_bin_small, y_best_small, algo_small = _small_pythia_dataset()
    opts = PythiaOptions(
        cv_folds=2,
        is_poly_krnl=False,
        use_weights=False,
        params=None,
        tuning="sobol",
        n_tuning_iter=5,
    )

    out = PythiaStage.pythia(
        z_small,
        y_small,
        y_bin_small,
        y_best_small,
        algo_small,
        opts,
        ParallelOptions.default(),
        GeneralOptions(verbose=False, seed=0),
    )

    assert all(min_value <= c <= max_value for c in out.box_consnt)
    assert all(min_value <= g <= max_value for g in out.k_scale)
    assert all(isinstance(clf, SVC) for clf in out.svm)


def test_pythia_sobol_seed_reproducibility() -> None:
    """F10: same seed gives identical Sobol-tuned output; a different seed differs."""
    z_small, y_small, y_bin_small, y_best_small, algo_small = _small_pythia_dataset()
    opts = PythiaOptions(
        cv_folds=2,
        is_poly_krnl=False,
        use_weights=False,
        params=None,
        tuning="sobol",
        n_tuning_iter=5,
    )

    def run(seed: int) -> NDArray[np.double]:
        out = PythiaStage.pythia(
            z_small,
            y_small,
            y_bin_small,
            y_best_small,
            algo_small,
            opts,
            ParallelOptions.default(),
            GeneralOptions(verbose=False, seed=seed),
        )
        return out.pr0_hat

    pr0_hat_a = run(0)
    pr0_hat_b = run(0)
    pr0_hat_c = run(1)

    np.testing.assert_array_equal(pr0_hat_a, pr0_hat_b)
    assert not np.array_equal(pr0_hat_a, pr0_hat_c)


def test_pythia_tuning_none_requires_params() -> None:
    """F10: `tuning='none'` without pre-calculated params fails loudly."""
    z_small, y_small, y_bin_small, y_best_small, algo_small = _small_pythia_dataset()
    opts = PythiaOptions(
        cv_folds=2,
        is_poly_krnl=False,
        use_weights=False,
        params=None,
        tuning="none",
    )

    with pytest.raises(ValueError, match="tuning='none'"):
        PythiaStage.pythia(
            z_small,
            y_small,
            y_bin_small,
            y_best_small,
            algo_small,
            opts,
            ParallelOptions.default(),
            GeneralOptions(verbose=False, seed=0),
        )


def test_pythia_tuning_invalid_value_raises() -> None:
    """F10: an unrecognised `tuning` value fails loudly, not silently."""
    z_small, y_small, y_bin_small, y_best_small, algo_small = _small_pythia_dataset()
    opts = PythiaOptions(
        cv_folds=2,
        is_poly_krnl=False,
        use_weights=False,
        params=None,
        tuning="not-a-real-strategy",
    )

    with pytest.raises(ValueError, match="not recognised"):
        PythiaStage.pythia(
            z_small,
            y_small,
            y_bin_small,
            y_best_small,
            algo_small,
            opts,
            ParallelOptions.default(),
            GeneralOptions(verbose=False, seed=0),
        )


@pytest.mark.parametrize(
    "classifier",
    ["svm", "knn", "tree", "nb", "linear", "ensemble"],
)
def test_pythia_precalc_params_does_not_crash(classifier: str) -> None:
    """Regression test for #292: real pre-calculated params must not crash.

    Every registered classifier previously crashed when given real
    (non-`None`) `PythiaOptions.params` - `_fit_classifier` fed scalar
    values straight into `BayesSearchCV`, which requires real search-space
    `Dimension`s, not bare numbers. `_fit_precalculated` now bypasses
    search entirely for this case, matching MATLAB's own `precalcparams`
    branch (a direct `crossValPredict`/`trainFinalClassifier` call).
    """
    z_small, y_small, y_bin_small, y_best_small, algo_small = _small_pythia_dataset()
    params = np.array([[5.0, 2.0], [3.0, 1.0]])
    opts = PythiaOptions(
        cv_folds=2,
        is_poly_krnl=False,
        use_weights=False,
        params=params,
        classifier=classifier,
        tuning="none",
    )

    out = PythiaStage.pythia(
        z_small,
        y_small,
        y_bin_small,
        y_best_small,
        algo_small,
        opts,
        ParallelOptions.default(),
        GeneralOptions(verbose=False, seed=0),
    )

    spec = get_classifier_fcn(classifier)
    np.testing.assert_allclose(out.box_consnt, params[:, 0])
    if spec.param2 is not None:
        np.testing.assert_allclose(out.k_scale, params[:, 1])
    else:
        assert all(np.isnan(g) for g in out.k_scale)


def test_pythia_precalc_params_knn_distance_round_trips_through_category_index() -> (
    None
):
    """#292: KNN's categorical `Distance` param survives precalc's numeric round-trip.

    MATLAB stores `Distance` as a 1-based index into
    `{'euclidean','cityblock','cosine','correlation'}` even for
    pre-calculated params (`fitOneClassifier`'s
    `distOpts{max(1,min(4,round(p2)))}`) - verify Python's `from_precalc`/
    `reported` round-trip matches that exactly, not just "doesn't crash".
    """
    z_small, y_small, y_bin_small, y_best_small, algo_small = _small_pythia_dataset()
    # Distance index 2 -> 'cityblock' (1-based order: euclidean, cityblock, cosine,
    # correlation).
    params = np.array([[5.0, 2.0], [3.0, 2.0]])
    opts = PythiaOptions(
        cv_folds=2,
        is_poly_krnl=False,
        use_weights=False,
        params=params,
        classifier="knn",
        tuning="none",
    )

    out = PythiaStage.pythia(
        z_small,
        y_small,
        y_bin_small,
        y_best_small,
        algo_small,
        opts,
        ParallelOptions.default(),
        GeneralOptions(verbose=False, seed=0),
    )

    np.testing.assert_allclose(out.k_scale, [2.0, 2.0])
    for clf in out.svm:
        assert clf.metric == "cityblock"
