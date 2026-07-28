# SPDX-License-Identifier: LicenseRef-PolyForm-Noncommercial-1.0.0
# Copyright (c) 2024-2026 Mario Andrés Muñoz
"""Registry mapping ``PythiaOptions.classifier`` names to scikit-learn estimators.

Mirrors MATLAB's ``ISAgetClassifierFcn.m`` dispatch table structurally, not
its exact tuning ranges: only ``'svm'`` (Python's original, only classifier)
is wired into Pythia's existing two-numeric-hyperparameter search (`C`/
`gamma` via `RandomizedSearchCV`/`BayesSearchCV`). The other five entries are
fit with scikit-learn's own default hyperparameters rather than a
hand-invented tuning range this repo has no MATLAB reference to verify
against - extending real tuning support to each of them is follow-on work,
not claimed here just because the classifier is registered and runnable.
"""

from collections.abc import Callable
from typing import NamedTuple

from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


class ClassifierSpec(NamedTuple):
    """A registry entry describing how to build one classifier type.

    Attributes
    ----------
    build : Callable[[int | None, bool], ClassifierMixin]
        Returns an unfitted estimator, given a random seed (``None`` for
        non-deterministic) and whether a polynomial kernel was requested
        (meaningful only for ``'svm'``; ignored otherwise).
    tunable : bool
        Whether this classifier is wired into Pythia's existing `C`/`gamma`
        hyperparameter search. Only true for ``'svm'`` in this pass.
    supports_sample_weight : bool
        Whether the estimator's `fit()` accepts a `sample_weight` argument -
        needed to honour `PythiaOptions.use_weights` (cost-sensitive
        classification). Not every scikit-learn classifier does.
    """

    build: Callable[[int | None, bool], ClassifierMixin]
    tunable: bool
    supports_sample_weight: bool


def _build_svm(seed: int | None, is_poly_krnl: bool) -> ClassifierMixin:
    return SVC(
        kernel="poly" if is_poly_krnl else "rbf",
        random_state=seed,
        probability=True,
        degree=2,
        coef0=1,
    )


def _build_knn(seed: int | None, is_poly_krnl: bool) -> ClassifierMixin:
    # KNeighborsClassifier has no random_state - its predictions are already
    # deterministic given fixed training data.
    return KNeighborsClassifier()


def _build_tree(seed: int | None, is_poly_krnl: bool) -> ClassifierMixin:
    return DecisionTreeClassifier(random_state=seed)


def _build_nb(seed: int | None, is_poly_krnl: bool) -> ClassifierMixin:
    # GaussianNB has no random_state - fitting is a closed-form calculation.
    return GaussianNB()


def _build_linear(seed: int | None, is_poly_krnl: bool) -> ClassifierMixin:
    return LogisticRegression(random_state=seed)


def _build_ensemble(seed: int | None, is_poly_krnl: bool) -> ClassifierMixin:
    # MATLAB's ensembleMethod sub-option picks among several equivalents
    # (RandomForest/AdaBoost/GradientBoosting); a full sub-registry for that
    # choice is follow-on work. RandomForestClassifier is the most commonly
    # used default among the three.
    return RandomForestClassifier(random_state=seed)


_REGISTRY: dict[str, ClassifierSpec] = {
    "svm": ClassifierSpec(build=_build_svm, tunable=True, supports_sample_weight=True),
    "knn": ClassifierSpec(
        build=_build_knn,
        tunable=False,
        supports_sample_weight=False,
    ),
    "tree": ClassifierSpec(
        build=_build_tree,
        tunable=False,
        supports_sample_weight=True,
    ),
    "nb": ClassifierSpec(build=_build_nb, tunable=False, supports_sample_weight=True),
    "linear": ClassifierSpec(
        build=_build_linear,
        tunable=False,
        supports_sample_weight=True,
    ),
    "ensemble": ClassifierSpec(
        build=_build_ensemble,
        tunable=False,
        supports_sample_weight=True,
    ),
}


def get_classifier_fcn(name: str) -> ClassifierSpec:
    """Look up a classifier registry entry by ``PythiaOptions.classifier`` name.

    Raises
    ------
    ValueError
        If `name` isn't a registered classifier.
    """
    try:
        return _REGISTRY[name]
    except KeyError:
        raise ValueError(
            f"Unknown PythiaOptions.classifier {name!r}. Registered classifiers: "
            f"{sorted(_REGISTRY)}",
        ) from None
