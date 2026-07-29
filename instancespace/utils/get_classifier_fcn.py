# SPDX-License-Identifier: LicenseRef-PolyForm-Noncommercial-1.0.0
# Copyright (c) 2024-2026 Mario Andrés Muñoz
"""Registry mapping ``PythiaOptions.classifier`` names to scikit-learn estimators.

Mirrors MATLAB's ``ISAgetClassifierFcn.m`` dispatch table, including its
per-classifier tunable-hyperparameter ranges (``classifierBayesVars``/
``sobolToParams`` in ``core/PYTHIA.m``) - every registered classifier is
tunable, not just ``'svm'``. ``'nb'`` needs a custom estimator (`KernelNB`)
since MATLAB tunes it via kernel-density estimation (``fitcnb(...,
'DistributionNames', 'kernel', 'Width', ...)``), which scikit-learn's
``GaussianNB`` has no equivalent for - it only fits a per-feature Gaussian,
with no bandwidth to tune.

Ranges are ported for parity of *search space*, not claimed to reproduce
MATLAB's exact tuned values bit-for-bit: no MATLAB reference output exists
for these five classifiers to verify against (only `'svm'` does).
"""

from collections.abc import Callable
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KernelDensity, KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from skopt.space import Categorical, Dimension, Integer, Real


class KernelNB(ClassifierMixin, BaseEstimator):  # type: ignore[misc]
    """Kernel-density-estimated Naive Bayes, matching MATLAB's `fitcnb`.

    MATLAB tunes `'nb'` via ``fitcnb(..., 'DistributionNames', 'kernel',
    'Width', p1)`` - a per-feature Gaussian-kernel density estimate per
    class, not the single fitted Gaussian `GaussianNB` assumes. There is no
    scikit-learn estimator for this, so this fits one univariate
    `sklearn.neighbors.KernelDensity` per (class, feature) and classifies by
    the class with the highest total log-density plus log-prior - the
    standard kernel-NB decision rule.
    """

    def __init__(self, bandwidth: float = 1.0) -> None:
        """Create an unfitted KernelNB with the given kernel bandwidth."""
        self.bandwidth = bandwidth

    def fit(
        self,
        x: NDArray[np.double],
        y: NDArray[np.bool_],
        sample_weight: NDArray[np.double] | None = None,
    ) -> "KernelNB":
        """Fit one `KernelDensity` per (class, feature), plus class priors."""
        self.classes_ = np.unique(y)
        total_weight = (
            x.shape[0] if sample_weight is None else float(sample_weight.sum())
        )

        log_priors = {}
        kdes: dict[bool, list[KernelDensity]] = {}
        for label in self.classes_:
            mask = y == label
            class_weight = (
                mask.sum() if sample_weight is None else sample_weight[mask].sum()
            )
            log_priors[label] = float(np.log(class_weight / total_weight))
            x_class = x[mask]
            weight_class = None if sample_weight is None else sample_weight[mask]
            kdes[label] = [
                KernelDensity(bandwidth=self.bandwidth).fit(
                    x_class[:, [feature]],
                    sample_weight=weight_class,
                )
                for feature in range(x.shape[1])
            ]

        self._log_priors = log_priors
        self._kdes = kdes
        return self

    def predict_proba(self, x: NDArray[np.double]) -> NDArray[np.double]:
        """Posterior class probabilities, columns ordered by `self.classes_`."""
        log_joint = np.zeros((x.shape[0], len(self.classes_)))
        for i, label in enumerate(self.classes_):
            log_density = sum(
                kde.score_samples(x[:, [feature]])
                for feature, kde in enumerate(self._kdes[label])
            )
            log_joint[:, i] = self._log_priors[label] + log_density

        log_joint -= log_joint.max(axis=1, keepdims=True)
        joint = np.exp(log_joint)
        proba: NDArray[np.double] = joint / joint.sum(axis=1, keepdims=True)
        return proba

    def predict(self, x: NDArray[np.double]) -> NDArray[np.bool_]:
        """Predict the class with the highest posterior probability."""
        proba = self.predict_proba(x)
        predictions: NDArray[np.bool_] = self.classes_[np.argmax(proba, axis=1)]
        return predictions


class ParamSpec(NamedTuple):
    """One tunable hyperparameter's search range.

    Mirrors one column of MATLAB's `classifierBayesVars`/`sobolToParams`
    (`core/PYTHIA.m`) - the range a classifier's hyperparameter is searched
    over, for both the `'sobol'` and `'bayes'` tuning strategies.

    Attributes
    ----------
    sklearn_name : str
        The estimator's own constructor keyword for this hyperparameter.
    label : str
        Human-readable label, matching `ISAgetClassifierFcn.m`'s
        p1label/p2label (used for the PYTHIA summary table's columns).
    low : float
        Lower bound of the search range, in `sklearn_name`'s own units.
    high : float
        Upper bound of the search range, in `sklearn_name`'s own units.
    log_scale : bool
        Whether to sample/search log-uniformly rather than linearly.
    is_int : bool
        Whether a sampled value must be rounded to an integer.
    categories : tuple[str, ...] | None
        Set only for categorical parameters (KNN's distance metric);
        `low`/`high` are unused when this is set.
    report : Callable[[float], float] | None
        Converts a raw sklearn value back to the unit MATLAB reports it in,
        for parameters where the two differ (e.g. logistic regression's
        `C` back to `Lambda = 1/C`). Identity if omitted.
    """

    sklearn_name: str
    label: str
    low: float
    high: float
    log_scale: bool = False
    is_int: bool = False
    categories: tuple[str, ...] | None = None
    report: Callable[[float], float] | None = None

    def sample(self, x: float) -> float | int | str:
        """Map a uniform sample `x` in [0,1] to this parameter's real value."""
        if self.categories is not None:
            index = min(len(self.categories) - 1, int(x * len(self.categories)))
            return self.categories[index]
        if self.log_scale:
            log_low, log_high = np.log(self.low), np.log(self.high)
            value = float(np.exp(log_low + x * (log_high - log_low)))
        else:
            value = self.low + x * (self.high - self.low)
        return int(max(self.low, round(value))) if self.is_int else float(value)

    def dimension(self) -> Dimension:
        """Return this parameter's search space as a `skopt` dimension (`'bayes'`)."""
        if self.categories is not None:
            return Categorical(list(self.categories), name=self.sklearn_name)
        if self.is_int:
            return Integer(int(self.low), int(self.high), name=self.sklearn_name)
        prior = "log-uniform" if self.log_scale else "uniform"
        return Real(self.low, self.high, prior=prior, name=self.sklearn_name)

    def reported(self, value: float | int | str) -> float:
        """Return the value to store in `PythiaOutput`, in MATLAB's own units."""
        if isinstance(value, str):
            if self.categories is None:
                return float("nan")
            return float(self.categories.index(value) + 1)
        return self.report(value) if self.report is not None else float(value)


class ClassifierSpec(NamedTuple):
    """A registry entry describing how to build and tune one classifier type.

    Attributes
    ----------
    build : Callable[[int | None, bool], ClassifierMixin]
        Returns an unfitted estimator, given a random seed (``None`` for
        non-deterministic) and whether a polynomial kernel was requested
        (meaningful only for ``'svm'``; ignored otherwise).
    supports_sample_weight : bool
        Whether the estimator's `fit()` accepts a `sample_weight` argument -
        needed to honour `PythiaOptions.use_weights` (cost-sensitive
        classification). Not every scikit-learn classifier does.
    param1 : ParamSpec
        The first (and, for some classifiers, only) tunable hyperparameter.
    param2 : ParamSpec | None
        The second tunable hyperparameter, or `None` if this classifier only
        has one (`ISAgetClassifierFcn.m`'s `p2label = 'N/A'` case).
    """

    build: Callable[[int | None, bool], ClassifierMixin]
    supports_sample_weight: bool
    param1: ParamSpec
    param2: ParamSpec | None


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
    # deterministic given fixed training data. algorithm='brute' is required
    # once the distance metric is tunable: 'cosine'/'correlation' aren't
    # supported by the default ball_tree/kd_tree algorithms.
    return KNeighborsClassifier(algorithm="brute")


def _build_tree(seed: int | None, is_poly_krnl: bool) -> ClassifierMixin:
    return DecisionTreeClassifier(random_state=seed)


def _build_nb(seed: int | None, is_poly_krnl: bool) -> ClassifierMixin:
    # KernelNB has no random_state - kernel density fitting is deterministic.
    return KernelNB()


def _build_linear(seed: int | None, is_poly_krnl: bool) -> ClassifierMixin:
    return LogisticRegression(random_state=seed)


def _build_ensemble(seed: int | None, is_poly_krnl: bool) -> ClassifierMixin:
    # MATLAB's ensembleMethod sub-option picks among several equivalents
    # (RandomForest/AdaBoost/GradientBoosting); a full sub-registry for that
    # choice is follow-on work. RandomForestClassifier is the most commonly
    # used default among the three.
    return RandomForestClassifier(random_state=seed)


_REGISTRY: dict[str, ClassifierSpec] = {
    "svm": ClassifierSpec(
        build=_build_svm,
        supports_sample_weight=True,
        param1=ParamSpec("C", "BoxConstraint", 2**-10, 2**4, log_scale=True),
        param2=ParamSpec("gamma", "KernelScale", 2**-10, 2**4, log_scale=True),
    ),
    "knn": ClassifierSpec(
        build=_build_knn,
        supports_sample_weight=False,
        param1=ParamSpec("n_neighbors", "NumNeighbors", 1, 25, is_int=True),
        param2=ParamSpec(
            "metric",
            "Distance",
            0,
            0,
            categories=("euclidean", "cityblock", "cosine", "correlation"),
        ),
    ),
    "tree": ClassifierSpec(
        build=_build_tree,
        supports_sample_weight=True,
        param1=ParamSpec("min_samples_leaf", "MinLeafSize", 1, 100, is_int=True),
        param2=None,
    ),
    "nb": ClassifierSpec(
        build=_build_nb,
        supports_sample_weight=True,
        param1=ParamSpec("bandwidth", "Bandwidth", 1e-3, 10, log_scale=True),
        param2=None,
    ),
    "linear": ClassifierSpec(
        build=_build_linear,
        supports_sample_weight=True,
        # Lambda (MATLAB's regularization strength, log-uniform [1e-6,1e3])
        # is inversely related to sklearn's C (inverse regularization
        # strength); sampling C log-uniformly over the inverted range
        # [1/1e3, 1/1e-6] is the equivalent search, reporting back via
        # Lambda = 1/C to match MATLAB's own label/units.
        param1=ParamSpec(
            "C",
            "Lambda",
            1e-3,
            1e6,
            log_scale=True,
            report=lambda c: 1.0 / c,
        ),
        param2=None,
    ),
    "ensemble": ClassifierSpec(
        build=_build_ensemble,
        supports_sample_weight=True,
        param1=ParamSpec("n_estimators", "NumLearningCycles", 10, 200, is_int=True),
        param2=ParamSpec("min_samples_leaf", "MinLeafSize", 1, 20, is_int=True),
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
