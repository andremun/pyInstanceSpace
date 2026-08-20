# SPDX-License-Identifier: LicenseRef-PolyForm-Noncommercial-1.0.0
# Copyright (c) 2024-2026 Mario Andrés Muñoz
"""Test module for the PythiaOptions.classifier registry (F1)."""

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from instancespace.utils.get_classifier_fcn import KernelNB, get_classifier_fcn


@pytest.mark.parametrize(
    ("name", "expected_class"),
    [
        ("svm", SVC),
        ("knn", KNeighborsClassifier),
        ("tree", DecisionTreeClassifier),
        ("nb", KernelNB),
        ("linear", LogisticRegression),
        ("ensemble", RandomForestClassifier),
    ],
)
def test_registered_classifier_builds_expected_type(
    name: str,
    expected_class: type,
) -> None:
    """Each registered name builds an unfitted instance of the right sklearn class."""
    spec = get_classifier_fcn(name)
    estimator = spec.build(0, False)
    assert isinstance(estimator, expected_class)


def test_unknown_classifier_raises() -> None:
    """An unregistered classifier name is rejected with a clear error."""
    with pytest.raises(ValueError, match="Unknown PythiaOptions.classifier"):
        get_classifier_fcn("not_a_real_classifier")


@pytest.mark.parametrize(
    ("name", "has_param2"),
    [
        ("svm", True),
        ("knn", True),
        ("tree", False),
        ("nb", False),
        ("linear", False),
        ("ensemble", True),
    ],
)
def test_every_registered_classifier_is_tunable(name: str, has_param2: bool) -> None:
    """Every registered classifier (#65) has at least one real tunable hyperparameter.

    Matches MATLAB's `classifierBayesVars`/`sobolToParams` (`core/PYTHIA.m`), which
    tunes all six classifiers, not just `'svm'` - the old `tunable` field this test
    used to check no longer exists (removed once every classifier became tunable).
    """
    spec = get_classifier_fcn(name)
    assert spec.param1 is not None
    assert (spec.param2 is not None) == has_param2


def test_knn_does_not_support_sample_weight() -> None:
    """KNeighborsClassifier.fit() has no sample_weight parameter."""
    assert not get_classifier_fcn("knn").supports_sample_weight


def test_knn_preserves_matlab_num_neighbors_search_domain() -> None:
    """Per-fit normalization must not shrink MATLAB's public search domain."""
    expected_max_neighbors = 25
    parameter = get_classifier_fcn("knn").param1

    assert (parameter.low, parameter.high) == (1, expected_max_neighbors)
    assert parameter.dimension().bounds == (1, expected_max_neighbors)
    assert parameter.sample(1.0) == expected_max_neighbors


def test_knn_fit_caps_neighbors_to_the_available_training_rows() -> None:
    """Match R2026a ``fitcknn`` by normalizing on each individual fit."""
    requested_neighbors = 25
    n_training_rows = 6
    estimator = get_classifier_fcn("knn").build(0, False)
    assert isinstance(estimator, KNeighborsClassifier)
    estimator.set_params(n_neighbors=requested_neighbors)
    x_train = np.arange(2 * n_training_rows, dtype=np.double).reshape(-1, 2)
    y_train = np.arange(n_training_rows) % 2 == 0

    estimator.fit(x_train, y_train)

    assert estimator.n_neighbors == n_training_rows
    assert estimator.get_params(deep=False)["n_neighbors"] == requested_neighbors
    assert estimator.predict(x_train[:2]).shape == (2,)


def test_knn_repeated_fit_recomputes_cap_from_requested_neighbors() -> None:
    """A small first fit must not make its effective cap sticky."""
    requested_neighbors = 25
    estimator = get_classifier_fcn("knn").build(0, False)
    assert isinstance(estimator, KNeighborsClassifier)
    estimator.set_params(n_neighbors=requested_neighbors)
    x_small = np.arange(12, dtype=np.double).reshape(-1, 2)
    y_small = np.arange(x_small.shape[0]) % 2 == 0
    x_large = np.arange(40, dtype=np.double).reshape(-1, 2)
    y_large = np.arange(x_large.shape[0]) % 2 == 0

    estimator.fit(x_small, y_small)
    assert estimator.n_neighbors == x_small.shape[0]
    estimator.fit(x_large, y_large)

    assert estimator.n_neighbors == x_large.shape[0]
    assert estimator.get_params(deep=False)["n_neighbors"] == requested_neighbors
    assert estimator.predict(x_large[:2]).shape == (2,)


def test_knn_clone_after_fit_retains_requested_neighbors() -> None:
    """Scikit cloning must carry the nominal candidate, not the prior fit cap."""
    requested_neighbors = 25
    estimator = get_classifier_fcn("knn").build(0, False)
    assert isinstance(estimator, KNeighborsClassifier)
    estimator.set_params(n_neighbors=requested_neighbors)
    x_small = np.arange(12, dtype=np.double).reshape(-1, 2)
    y_small = np.arange(x_small.shape[0]) % 2 == 0
    estimator.fit(x_small, y_small)

    cloned = clone(estimator)

    assert isinstance(cloned, KNeighborsClassifier)
    assert cloned.n_neighbors == requested_neighbors
    x_larger = np.arange(40, dtype=np.double).reshape(-1, 2)
    y_larger = np.arange(x_larger.shape[0]) % 2 == 0
    cloned.fit(x_larger, y_larger)
    assert cloned.n_neighbors == x_larger.shape[0]
    assert cloned.get_params(deep=False)["n_neighbors"] == requested_neighbors


def test_knn_fit_accepts_array_like_inputs_before_applying_cap() -> None:
    """Let sklearn validate/normalize public array-like inputs before capping."""
    requested_neighbors = 25
    x_list = [[0.0, 0.0], [1.0, 1.0], [2.0, 0.0], [3.0, 1.0]]
    y_list = [False, True, False, True]
    estimator = get_classifier_fcn("knn").build(0, False)
    assert isinstance(estimator, KNeighborsClassifier)
    estimator.set_params(n_neighbors=requested_neighbors)

    estimator.fit(x_list, y_list)

    assert estimator.n_neighbors == len(x_list)
    assert estimator.get_params(deep=False)["n_neighbors"] == requested_neighbors
    assert estimator.predict([[0.5, 0.5]]).shape == (1,)


def test_svm_kernel_choice_follows_is_poly_krnl() -> None:
    """The SVM builder follows the kernel flag and MATLAB's cubic default."""
    matlab_default_polynomial_order = 3
    spec = get_classifier_fcn("svm")
    rbf_svm = spec.build(0, False)
    poly_svm = spec.build(0, True)
    assert isinstance(rbf_svm, SVC)
    assert isinstance(poly_svm, SVC)
    assert rbf_svm.kernel == "rbf"
    assert poly_svm.kernel == "poly"
    assert poly_svm.degree == matlab_default_polynomial_order


def test_svm_build_matches_seed() -> None:
    """The seed is threaded through to random_state for classifiers that support it."""
    seed = 42
    spec = get_classifier_fcn("svm")
    estimator = spec.build(seed, False)
    assert estimator.random_state == seed


@pytest.mark.parametrize("kernel_scale", [2**-10, 1.0, 2**4])
def test_svm_kernel_scale_precalc_round_trip(kernel_scale: float) -> None:
    """SVM params cross the API in MATLAB units and SVC in gamma units."""
    parameter = get_classifier_fcn("svm").param2
    assert parameter is not None

    gamma = parameter.from_precalc(kernel_scale)

    assert isinstance(gamma, float)
    assert gamma == pytest.approx(1.0 / kernel_scale**2)
    assert parameter.reported(gamma) == pytest.approx(kernel_scale)


def test_svm_kernel_scale_sobol_and_bayes_use_estimator_units() -> None:
    """Sobol samples and the Bayes dimension share the KernelScale conversion."""
    parameter = get_classifier_fcn("svm").param2
    assert parameter is not None
    expected_midpoint = np.sqrt(parameter.low * parameter.high)

    for point, expected_scale in (
        (0.0, parameter.low),
        (0.5, expected_midpoint),
        (1.0, parameter.high),
    ):
        gamma = parameter.sample(point)
        assert isinstance(gamma, float)
        assert parameter.reported(gamma) == pytest.approx(expected_scale)

    dimension = parameter.dimension()
    assert dimension.bounds == pytest.approx((2**-8, 2**20))


def test_linear_lambda_precalc_sobol_and_bayes_round_trip() -> None:
    """Linear Lambda remains the public unit while sklearn receives inverse C."""
    parameter = get_classifier_fcn("linear").param1
    regularization = 1e-4

    inverse_regularization = parameter.from_precalc(regularization)

    assert isinstance(inverse_regularization, float)
    assert inverse_regularization == pytest.approx(1e4)
    assert parameter.reported(inverse_regularization) == pytest.approx(regularization)
    sampled = parameter.sample(0.5)
    assert isinstance(sampled, float)
    assert parameter.reported(sampled) == pytest.approx(
        np.sqrt(parameter.low * parameter.high),
    )
    assert parameter.dimension().bounds == pytest.approx((1e-3, 1e6))


def test_precalculated_discrete_parameters_use_matlab_normalization() -> None:
    """Integer params round away from zero and apply MATLAB's lower clamps."""
    rounded_two_point_five = 3
    minimum_ensemble_cycles = 10
    knn = get_classifier_fcn("knn")
    tree = get_classifier_fcn("tree")
    ensemble = get_classifier_fcn("ensemble")
    assert knn.param2 is not None
    assert ensemble.param2 is not None

    assert knn.param1.from_precalc(2.5) == rounded_two_point_five
    assert knn.param1.from_precalc(-2.5) == 1
    assert knn.param2.from_precalc(0.4) == "euclidean"
    assert knn.param2.from_precalc(2.5) == "cosine"
    assert knn.param2.from_precalc(4.6) == "correlation"
    assert tree.param1.from_precalc(2.5) == rounded_two_point_five
    assert tree.param1.from_precalc(-2.5) == 1
    assert ensemble.param1.from_precalc(9.4) == minimum_ensemble_cycles
    assert ensemble.param2.from_precalc(0.4) == 1


@pytest.mark.parametrize(
    ("point", "expected"),
    [
        (0.0, "euclidean"),
        (0.25, "euclidean"),
        (0.2500001, "cityblock"),
        (1.0, "correlation"),
    ],
)
def test_knn_categorical_sobol_sampling_matches_matlab_boundaries(
    point: float,
    expected: str,
) -> None:
    """Categorical Sobol bins use MATLAB's ceil-based one-based indexing."""
    parameter = get_classifier_fcn("knn").param2
    assert parameter is not None

    assert parameter.sample(point) == expected
