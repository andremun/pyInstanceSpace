# SPDX-License-Identifier: LicenseRef-PolyForm-Noncommercial-1.0.0
# Copyright (c) 2024-2026 Mario Andrés Muñoz
"""Test module for the PythiaOptions.classifier registry (F1)."""

import pytest
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


def test_svm_kernel_choice_follows_is_poly_krnl() -> None:
    """The SVM builder respects the poly-kernel flag, matching prior behaviour."""
    spec = get_classifier_fcn("svm")
    rbf_svm = spec.build(0, False)
    poly_svm = spec.build(0, True)
    assert isinstance(rbf_svm, SVC)
    assert isinstance(poly_svm, SVC)
    assert rbf_svm.kernel == "rbf"
    assert poly_svm.kernel == "poly"


def test_svm_build_matches_seed() -> None:
    """The seed is threaded through to random_state for classifiers that support it."""
    seed = 42
    spec = get_classifier_fcn("svm")
    estimator = spec.build(seed, False)
    assert estimator.random_state == seed
