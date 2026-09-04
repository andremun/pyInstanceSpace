# SPDX-License-Identifier: LicenseRef-PolyForm-Noncommercial-1.0.0
# Copyright (c) 2024-2026 Mario Andrés Muñoz
"""PYTHIA: Automated Algorithm Selection.

By training Support Vector Machines (SVMs) to predict the best-performing
algorithm for a given problem instance, using the coordinates of that
instance in a two-dimensional instance space. PYTHIA uses the trained models
generate overall summary of each algorithm performance and
recommend the best algorithm for a new problem instance.

Key steps for PYTHIA:
1. Normalize the instance space.
2. Train SVM models for each algorithm.
3. Evaluate the performance of the SVM models.
4. Generate a summary of the results.

This module is structured around the `PythiaStage` class

Dependencies:
- numpy
- pandas
- scipy
- sklearn
- skopt

Classes:
--------
- PythiaStage: The main class for the Pythia stage.

Functions:
----------
- pythia: The main function for the Pythia stage.
- _fit_classifier: Train the configured classifier (see PythiaOptions.classifier).
- _display_overall_perf: Output overall performance metrics.
- _compute_znorm: Compute normalized instance space.
- _check_precalcparams: Check pre-calculated hyper-parameters.
- _determine_selections: Determine the selections based on the precision metrics.
- _generate_summary: Generate a summary of the results.
"""

import warnings
from dataclasses import dataclass, replace
from time import perf_counter
from typing import Any, NamedTuple

import numpy as np
import pandas as pd
from loguru import logger
from numpy.typing import NDArray
from scipy import stats
from sklearn.base import ClassifierMixin
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from skopt import BayesSearchCV

from instancespace.data.model import PythiaOut
from instancespace.data.options import GeneralOptions, ParallelOptions, PythiaOptions
from instancespace.stages.stage import PredictiveStage, Stage
from instancespace.utils.get_classifier_fcn import ClassifierSpec, get_classifier_fcn

LARGE_NUM_INSTANCE: int = 1000
PYTHIA_ARRAY_DIMENSIONS = 2

# MATLAB PYTHIA explicitly selects bayesopt's
# ``expected-improvement-plus`` acquisition and leaves ``NumSeedPoints`` at
# its R2026a default of 4.  skopt has no equivalent to MATLAB's additional
# anti-overexploitation "plus" loop; plain EI is its closest base acquisition.
# Keep the same four-seed/evaluation-budget split without claiming identical
# optimizer trajectories.  Historical #304 CSVs are ``legacy-unknown`` and
# are not evidence for changing the production budget or defaults.
_BAYES_OPTIMIZER_KWARGS: dict[str, object] = {
    "n_initial_points": 4,
    "acq_func": "EI",
}


def _algorithm_seed(base_seed: int | None, algorithm_index: int) -> int | None:
    """Translate Python's zero-based index to MATLAB's ``seed + i`` stream."""
    return None if base_seed is None else base_seed + algorithm_index + 1


def _safe_ratio(numerator: int | float, denominator: int | float) -> float:
    """Return NaN when a rate has no observations."""
    return float(numerator / denominator) if denominator != 0 else float("nan")


def _nanmean(
    values: NDArray[Any],
    axis: int | None = None,
) -> NDArray[np.double]:
    """Calculate a NaN-aware mean without an empty-slice warning."""
    array = np.asarray(values, dtype=np.double)
    present = np.logical_not(np.isnan(array))
    totals = np.sum(np.where(present, array, 0.0), axis=axis, dtype=np.double)
    counts = np.sum(present, axis=axis, dtype=np.int_)
    result = np.full(np.asarray(totals).shape, np.nan, dtype=np.double)
    return np.divide(totals, counts, out=result, where=counts > 0)


def _matlab_nanstd(
    values: NDArray[Any],
    axis: int | None = None,
) -> NDArray[np.double]:
    """Calculate MATLAB's NaN-aware sample standard deviation without warnings."""
    array = np.asarray(values, dtype=np.double)
    present = np.logical_not(np.isnan(array))
    means = _nanmean(array, axis=axis)
    expanded_means = means if axis is None else np.expand_dims(means, axis=axis)
    squared_error = np.where(present, (array - expanded_means) ** 2, 0.0)
    totals = np.sum(squared_error, axis=axis, dtype=np.double)
    counts = np.sum(present, axis=axis, dtype=np.int_)
    denominator = np.maximum(counts - 1, 1)
    variance = np.full(np.asarray(totals).shape, np.nan, dtype=np.double)
    np.divide(totals, denominator, out=variance, where=counts > 0)
    return np.sqrt(variance)


@dataclass(frozen=True)
class _ClassifierResult:
    """Trained-classifier result, generic over PythiaOptions.classifier's choice."""

    classifier: ClassifierMixin
    Ysub: NDArray[np.bool_]
    Psub: NDArray[np.double]
    Yhat: NDArray[np.bool_]
    Phat: NDArray[np.double]
    c: float
    g: float


class _ConstantClassifier:
    """Sklearn-classifier-shaped sentinel for a degenerate (single-class) label.

    Mirrors MATLAB's `struct('constant', true, 'value', yi(1))` sentinel,
    used when an algorithm is good (or bad) for every training instance, so
    `StratifiedKFold` cannot stratify a single-class label and
    cross-validation/tuning is skipped entirely. Implements just enough of
    scikit-learn's classifier interface (`classes_`, `predict`,
    `predict_proba`) that every consumer of `PythiaOutput.svm` - including
    `InstanceSpace._explore_pythia` - can call it exactly like a real fitted
    classifier, without special-casing it.
    """

    def __init__(self, value: bool) -> None:
        self.value = value
        self.classes_ = np.array([False, True])

    def predict(self, x: NDArray[np.double]) -> NDArray[np.bool_]:
        """Predict the constant label for every row of `x`."""
        return np.full(x.shape[0], self.value, dtype=bool)

    def predict_proba(self, x: NDArray[np.double]) -> NDArray[np.double]:
        """Predict [P(class 0), P(class 1)] as [0, 1] or [1, 0] for every row."""
        proba = np.zeros((x.shape[0], 2), dtype=np.double)
        proba[:, int(self.value)] = 1.0
        return proba


class PythiaInput(NamedTuple):
    """Inputs for the Pythia stage.

    Attributes
    ----------
    z : NDArray[np.double]
        The feature matrix.
    y : NDArray[np.double]
        The performance metrics.
    y_bin : NDArray[np.bool_]
        The binary labels.
    y_best : NDArray[np.double]
        The best performance metrics.
    algo_labels : list[str]
        The algorithm labels.
    pythia_options : PythiaOptions
        The options for the Pythia stage.
    parallel_options: ParallelOptions
        The parallel options, specifiy whether run in parallel and number of cores.
    general_options : GeneralOptions
        General options (e.g. the RNG seed), not specific to any one stage.
    """

    z: NDArray[np.double]
    y_raw: NDArray[np.double]
    y_bin: NDArray[np.bool_]
    y_best: NDArray[np.double]
    algo_labels: list[str]
    pythia_options: PythiaOptions
    parallel_options: ParallelOptions
    general_options: GeneralOptions


class PythiaOutput(NamedTuple):
    """Outputs from the Pythia stage.

    Attributes
    ----------
    mu : list[float]
        The mean values of the normalized features.
    sigma : list[float]
        The standard deviations of the normalized features.
    w : NDArray[np.double]
        The weight matrix used for cost-sensitive classification.
    cp : list[StratifiedKFold | None]
        The per-algorithm Stratified K-Fold cross-validators. A degenerate or
        skipped algorithm has ``None``, matching MATLAB's empty cell entry.
    svm : list[ClassifierMixin | None]
        The trained classifiers, one per algorithm - `SVC` instances unless
        `PythiaOptions.classifier` selected a different registered type. The
        field is still named `svm` for backward compatibility; it holds
        whatever `PythiaOptions.classifier` chose to train, `'svm'` by
        default. A skipped algorithm has ``None``, matching MATLAB's empty
        classifier cell.
    cvcmat : NDArray[np.double]
        Confusion matrix for each algorithm
    y_sub : NDArray[np.bool_]
        The binary predicted labels for each algorithm.
    y_hat : NDArray[np.bool_]
        The final predicted labels for each algorithm.
    pr0_sub : NDArray[np.double]
        The predicted cross-validated probabilities of class 0 ("bad"),
        matching MATLAB's `Pr0sub` convention.
    pr0_hat : NDArray[np.double]
        The predicted probabilities of class 0 ("bad") on the full data,
        matching MATLAB's `Pr0hat` convention.
    box_consnt : list[float]
        Regularization parameters `C`.
    k_scale : list[float]
        MATLAB-facing KernelScale values. SVC stores the equivalent
        estimator coefficient as `gamma = 1 / KernelScale**2`.
    accuracy : list[float]
        Accuracy scores of each SVM model.
    precision : list[float]
        Precision scores for each SVM model.
    recall : list[float]
        Recall scores for each algorithm
    selection0 : NDArray[np.int_]
        The selected algorithm indices for each instance.
    selection1 : NDArray[np.int_]
        The backup selected algorithm indices for each instance.
    summary : pd.DataFrame
        A summary table for performance statistics of all algorithms.
    """

    mu: list[float]
    sigma: list[float]
    w: NDArray[np.double]
    cp: list[StratifiedKFold | None]
    svm: list[ClassifierMixin | None]
    cvcmat: NDArray[np.double]
    y_sub: NDArray[np.bool_]
    y_hat: NDArray[np.bool_]
    pr0_sub: NDArray[np.double]
    pr0_hat: NDArray[np.double]
    box_consnt: list[float]
    k_scale: list[float]
    accuracy: list[float]
    precision: list[float]
    recall: list[float]
    selection0: NDArray[np.int_]
    selection1: NDArray[np.int_]
    pythia_summary: pd.DataFrame


class PythiaPredictInput(NamedTuple):
    """Inputs for applying fitted PYTHIA classifiers to projected instances."""

    z: NDArray[np.double]
    n_new_algos: int = 0


class PythiaPredictOutput(NamedTuple):
    """Predictions and precision-weighted selections for unseen instances."""

    y_hat: NDArray[np.bool_]
    pr0_hat: NDArray[np.double]
    selection0: NDArray[np.int_]


class PythiaEvaluateInput(NamedTuple):
    """Inputs for evaluating fitted-classifier predictions against test truth."""

    y_true: NDArray[np.bool_]
    y_pred: NDArray[np.bool_]


class PythiaEvaluateOutput(NamedTuple):
    """MATLAB-compatible per-algorithm confusion counts and rates."""

    accuracy: NDArray[np.double]
    precision: NDArray[np.double]
    recall: NDArray[np.double]
    cvcmat: NDArray[np.double]


class PythiaStage(
    Stage[PythiaInput, PythiaOutput],
    PredictiveStage[PythiaPredictInput, PythiaOut, PythiaPredictOutput],
):
    """Pythia stage for automated algorithm selection.

    The `PythiaStage` class is the main class for the Pythia stage. It
    contains the main function `pythia` that runs the Pythia stage.

    Methods
    -------
    _inputs() -> type[PythiaInput]
        Return the input type for the Pythia stage.

    _outputs() -> type[PythiaOutput]
        Return the output type for the Pythia stage.

    _run(inputs: PythiaInput) -> PythiaOutput
        Run the Pythia stage.

    pythia(z: NDArray[np.double], y: NDArray[np.double], y_bin: NDArray[np.bool_],
              y_best: NDArray[np.double], algo_labels: list[str], opts: PythiaOptions,
                parallel_options: ParallelOptions) -> PythiaOutput
        Main method that perform automated algorithm selection.

    _fit_classifier(z: NDArray[np.double], y_bin: NDArray[np.bool_],
                w: NDArray[np.double], skf: StratifiedKFold, classifier_name: str,
                is_poly_kernel: bool, param_space: dict[str, list[float]] | None,
                use_weights: bool, parallel_options: ParallelOptions,
                general_options: GeneralOptions,
                n_tuning_iter: int) -> _ClassifierResult
        Train the classifier selected by PythiaOptions.classifier.

    _display_overall_perf(precision: list[float], accuracy: list[float]) -> None
        Output overall performance metrics.

    _compute_znorm(z: NDArray[np.double]) -> tuple[list[float], list[float],
                NDArray[np.double]]
        Compute normalized feature matrix.

    _check_precalcparams(params: NDArray[np.double] | None, nalgos: int) ->
                NDArray[np.double] | None
        Check pre-calculated hyper-parameters.

    _determine_selections(nalgos: int, precision: list[float], y_hat: NDArray[np.bool_],
                            y_bin: NDArray[np.bool_]) -> tuple[NDArray[np.int_],
                            NDArray[np.int_]]
        Determine the selections based on the precision metrics.

    _generate_summary(nalgos: int, algo_labels: list[str], y: NDArray[np.double],
                        y_hat: NDArray[np.bool_], y_bin: NDArray[np.bool_],
                        y_best: NDArray[np.double],
                        selection0: NDArray[np.int_], selection1: NDArray[np.int_],
                        accuracy: list[float], precision: list[float],
                        recall: list[float],
                        box_consnt: list[float],
                        k_scale: list[float]) -> pd.DataFrames
        Generate a summary of the results.
    """

    def __init__(
        self,
        z: NDArray[np.double],
        y_raw: NDArray[np.double],
        y_bin: NDArray[np.bool_],
        y_best: NDArray[np.double],
        algo_labels: list[str],
    ) -> None:
        """Define the input for the Pythia stage.

        Parameters
        ----------
        z : NDArray[np.double]
            The feature matrix.
        y_raw : NDArray[np.double]
            The performance metrics.
        y_bin : NDArray[np.bool_]
            The binary labels.
        y_best : NDArray[np.double]
            The best performance metrics.
        algo_labels : list[str]
            The algorithm labels.
        """
        super().__init__()
        self.z = z
        self.y = y_raw
        self.y_bin = y_bin
        self.y_best = y_best
        self.algo_labels = algo_labels

    @staticmethod
    def _inputs() -> type[PythiaInput]:
        return PythiaInput

    @staticmethod
    def _outputs() -> type[PythiaOutput]:
        return PythiaOutput

    @staticmethod
    def predict(
        inputs: PythiaPredictInput,
        fitted: PythiaOut,
    ) -> PythiaPredictOutput:
        """Apply persisted PYTHIA state without fitting or mutating classifiers.

        This is MATLAB's ``PYTHIAevalMode`` prediction core: normalize new
        coordinates with the parameters stored by the training run, apply each
        fitted classifier, pad test-only algorithms with empty predictions, and
        make the same precision-weighted recommendation as the build path.
        """
        if inputs.z.ndim != PYTHIA_ARRAY_DIMENSIONS:
            msg = "PYTHIA predict requires Z to be a two-dimensional matrix."
            raise ValueError(msg)
        if inputs.n_new_algos < 0:
            msg = "PYTHIA predict requires n_new_algos to be non-negative."
            raise ValueError(msg)

        mu = np.asarray(fitted.mu, dtype=np.double)
        sigma = np.asarray(fitted.sigma, dtype=np.double)
        if mu.shape != (inputs.z.shape[1],) or sigma.shape != mu.shape:
            msg = "PYTHIA predict normalization parameters must match Z columns."
            raise ValueError(msg)

        classifiers = PythiaStage._fitted_classifier_slots(fitted)
        precision = np.asarray(fitted.precision, dtype=np.double)
        n_trained = len(classifiers)
        if precision.shape != (n_trained,):
            msg = "PYTHIA predict precision must have one value per classifier."
            raise ValueError(msg)

        z_norm = (inputs.z - mu) / sigma
        n_instances = z_norm.shape[0]
        n_algorithms = n_trained + inputs.n_new_algos
        y_hat = np.zeros((n_instances, n_algorithms), dtype=np.bool_)
        pr0_hat = np.zeros((n_instances, n_algorithms), dtype=np.double)

        for index, classifier in enumerate(classifiers):
            if classifier is None:
                continue
            y_hat[:, index] = np.asarray(
                classifier.predict(z_norm),
                dtype=np.bool_,
            )
            probabilities = np.asarray(
                classifier.predict_proba(z_norm),
                dtype=np.double,
            )
            pr0_hat[:, index] = PythiaStage._bad_class_proba(
                classifier,
                probabilities,
            )

        if n_algorithms == 0:
            return PythiaPredictOutput(
                y_hat,
                pr0_hat,
                np.full(n_instances, -1, dtype=np.int_),
            )

        selection_precision = np.zeros(n_algorithms, dtype=np.double)
        selection_precision[:n_trained] = precision
        best, selection0 = PythiaStage._weighted_selection(
            n_algorithms,
            selection_precision,
            y_hat,
        )
        selection0[best <= 0] = -1
        return PythiaPredictOutput(y_hat, pr0_hat, selection0)

    @staticmethod
    def evaluate(
        inputs: PythiaEvaluateInput,
        fitted: PythiaOut,
    ) -> PythiaEvaluateOutput:
        """Evaluate predictions with MATLAB's explicit confusion-count formulas.

        MATLAB stores each confusion matrix as ``cm(:)'`` in column-major order,
        hence ``[TN, FN, FP, TP]``. MATLAB v0.9.1's
        ``core/PYTHIA.m::PYTHIAevalMode`` loops over every trained classifier and
        calls ``confusionmat`` against its reconciled truth column. Therefore,
        every non-empty trained-classifier slot is scored, including a training
        algorithm absent from the test metadata (whose truth column is all
        false). Empty trained slots retain a zero confusion row and therefore
        zero accuracy with undefined precision and recall. Test-only algorithms
        also retain zero confusion rows, but their rates stay ``NaN`` because no
        trained-model slot exists for them.
        """
        if (
            inputs.y_true.ndim != PYTHIA_ARRAY_DIMENSIONS
            or inputs.y_pred.ndim != PYTHIA_ARRAY_DIMENSIONS
        ):
            msg = "PYTHIA evaluate requires two-dimensional truth and predictions."
            raise ValueError(msg)
        if inputs.y_true.shape != inputs.y_pred.shape:
            msg = "PYTHIA evaluate truth and predictions must have matching shapes."
            raise ValueError(msg)

        n_instances, n_algorithms = inputs.y_pred.shape
        classifiers = PythiaStage._fitted_classifier_slots(fitted)
        n_trained_algorithms = len(classifiers)
        if n_trained_algorithms > n_algorithms:
            msg = "PYTHIA evaluate has more trained classifiers than algorithms."
            raise ValueError(msg)

        y_true = np.asarray(inputs.y_true, dtype=np.bool_)
        y_pred = np.asarray(inputs.y_pred, dtype=np.bool_)
        accuracy = np.full(n_algorithms, np.nan, dtype=np.double)
        precision = np.full(n_algorithms, np.nan, dtype=np.double)
        recall = np.full(n_algorithms, np.nan, dtype=np.double)
        cvcmat = np.zeros((n_algorithms, 4), dtype=np.double)

        for index, classifier in enumerate(classifiers):
            if classifier is None:
                continue
            truth = y_true[:, index]
            prediction = y_pred[:, index]
            true_positive = int(np.sum(truth & prediction))
            true_negative = int(np.sum(~truth & ~prediction))
            false_positive = int(np.sum(~truth & prediction))
            false_negative = int(np.sum(truth & ~prediction))
            cvcmat[index] = [
                true_negative,
                false_negative,
                false_positive,
                true_positive,
            ]

        for index in range(n_trained_algorithms):
            true_negative, false_negative, false_positive, true_positive = cvcmat[index]
            accuracy[index] = _safe_ratio(
                true_positive + true_negative,
                n_instances,
            )
            precision[index] = _safe_ratio(
                true_positive,
                true_positive + false_positive,
            )
            recall[index] = _safe_ratio(
                true_positive,
                true_positive + false_negative,
            )

        return PythiaEvaluateOutput(accuracy, precision, recall, cvcmat)

    @staticmethod
    def _fitted_classifier_slots(
        fitted: PythiaOut,
    ) -> list[ClassifierMixin | None]:
        """Return fitted slots, recognizing pre-fix skip placeholders.

        Current skip-mode models store ``None`` just like MATLAB's empty cell
        entries. Older Python checkpoints stored an always-bad constant
        classifier in every slot so inference could call a classifier-shaped
        object unconditionally. That representation is safely identifiable as
        skip mode only when every slot is that sentinel and all three recorded
        model rates are ``NaN``; a genuinely trained always-bad algorithm has
        finite accuracy and must keep returning ``P(bad)=1``.
        """
        classifiers: list[ClassifierMixin | None] = list(fitted.svm)
        if not classifiers or not all(
            isinstance(classifier, _ConstantClassifier) and not classifier.value
            for classifier in classifiers
        ):
            return classifiers

        expected_shape = (len(classifiers),)
        for field in ("accuracy", "precision", "recall"):
            values = np.asarray(getattr(fitted, field), dtype=np.double)
            if values.shape != expected_shape or not np.all(np.isnan(values)):
                return classifiers
        return [None] * len(classifiers)

    @staticmethod
    def _run(inputs: PythiaInput) -> PythiaOutput:
        return PythiaStage.pythia(
            inputs.z,
            inputs.y_raw,
            inputs.y_bin,
            inputs.y_best,
            inputs.algo_labels,
            inputs.pythia_options,
            inputs.parallel_options,
            general_options=inputs.general_options,
        )

    @staticmethod
    def pythia(
        z: NDArray[np.double],
        y: NDArray[np.double],
        y_bin: NDArray[np.bool_],
        y_best: NDArray[np.double],
        algo_labels: list[str],
        opts: PythiaOptions,
        parallel_options: ParallelOptions,
        general_options: GeneralOptions,
    ) -> PythiaOutput:
        """Run the Pythia stage.

        Parameters
        ----------
        z : NDArray[np.double]
            The feature matrix.
        y : NDArray[np.double]
            The performance metrics.
        y_bin : NDArray[np.bool_]
            The binary labels.
        y_best : NDArray[np.double]
            The best performance metrics.
        algo_labels : list[str]
            The algorithm labels.
        opts : PythiaOptions
            The options for the Pythia stage.
        parallel_options : ParallelOptions
            The parallel options, specifiy whether run in parallel and number of cores.
        general_options : GeneralOptions
            General options (e.g. the RNG seed), not specific to any one stage.

        Returns
        -------
        PythiaOutput
            The output of the Pythia stage.
        """
        logger.info(
            "[PYTHIA] ================================================================"
            "=========",
        )
        logger.info("[PYTHIA] -> Summoning PYTHIA to train the prediction models.")
        logger.info(
            "[PYTHIA] ================================================================"
            "=========",
        )
        logger.info("[PYTHIA]   -> Initializing PYTHIA.")

        # Initialize variables
        ninst, nalgos = y_bin.shape

        y_sub = np.zeros(y_bin.shape, dtype=bool)
        y_hat = np.zeros(y_bin.shape, dtype=bool)
        pr0sub = np.zeros(y_bin.shape, dtype=np.double)
        pr0hat = np.zeros(y_bin.shape, dtype=np.double)

        # Resolving the classifier spec validates opts.classifier even in
        # skip mode - matching MATLAB, which calls ISAgetClassifierFcn
        # before checking opts.skip.
        classifier_spec = get_classifier_fcn(opts.classifier)

        # Section 1: Normalize the feature matrix. mu/sigma still describe
        # the real z even under opts.skip=True - MATLAB computes zscore
        # before checking opts.skip and copies mu/sigma onto its otherwise-
        # empty output, so a later eval-mode caller can still normalise new
        # data correctly.
        mu, sigma, z = PythiaStage._compute_znorm(z)

        if opts.skip:
            logger.info(
                "[PYTHIA]  -> opts.skip is True; bypassing classifier "
                "training entirely. TRACE will need to build footprints "
                "from raw labels only (trace.use_sim=False).",
            )
            return PythiaStage._empty_output(
                ninst,
                nalgos,
                algo_labels,
                y,
                y_bin,
                y_best,
                mu,
                sigma,
            )

        return PythiaStage._train(
            z,
            y,
            y_bin,
            y_best,
            algo_labels,
            opts,
            parallel_options,
            general_options,
            classifier_spec,
            mu,
            sigma,
            ninst,
            nalgos,
            y_sub,
            y_hat,
            pr0sub,
            pr0hat,
        )

    @staticmethod
    def _train(
        z: NDArray[np.double],
        y: NDArray[np.double],
        y_bin: NDArray[np.bool_],
        y_best: NDArray[np.double],
        algo_labels: list[str],
        opts: PythiaOptions,
        parallel_options: ParallelOptions,
        general_options: GeneralOptions,
        classifier_spec: ClassifierSpec,
        mu: list[float],
        sigma: list[float],
        ninst: int,
        nalgos: int,
        y_sub: NDArray[np.bool_],
        y_hat: NDArray[np.bool_],
        pr0sub: NDArray[np.double],
        pr0hat: NDArray[np.double],
    ) -> PythiaOutput:
        """Train a classifier per algorithm and assemble the `PythiaOutput`.

        Split out of `pythia()` so `opts.skip=True`'s early return doesn't
        push `pythia()` itself over the branch-count lint threshold; carries
        the whole training-mode body (Sections 2-4) unchanged.
        """
        PythiaStage._validate_cv_folds(
            y_bin,
            algo_labels,
            opts.cv_folds,
        )
        precalcparams = PythiaStage._check_precalcparams(
            opts.params,
            nalgos,
            classifier_spec,
            opts.classifier,
        )
        cp: list[StratifiedKFold | None] = [None] * nalgos
        svm: list[ClassifierMixin | None] = []
        cvcmat = np.zeros((nalgos, 4), dtype=int)
        box_consnt = []
        k_scale = []
        accuracy_record = []
        precision_record = []
        recall_record = []

        w = np.ones((z.shape[0], nalgos), dtype=np.double)
        PythiaStage._validate_tuning(opts, precalcparams)
        logger.info(
            f"[PYTHIA]  -> PYTHIA is training a '{opts.classifier}' classifier.",
        )

        if opts.classifier == "svm":
            PythiaStage._log_kernel_choice(ninst, opts.is_poly_krnl)

        # Section 2: Configure hyperparameter optimization
        PythiaStage._log_tuning_strategy(opts, precalcparams)

        # Cost-sensitive classification
        if opts.use_weights:
            logger.info("[PYTHIA]  -> PYTHIA is using cost-sensitive classification.")
            performance_mean = float(_nanmean(y))
            w = np.abs(y - performance_mean)
            finite_nonzero = w[(w != 0) & np.isfinite(w)]
            if finite_nonzero.size == 0:
                # Degenerate case: y is constant or entirely NaN, so every
                # weight is 0 or NaN. Fall back to uniform weighting rather
                # than erroring on min([])/max([]).
                logger.warning(
                    "[PYTHIA] use_weights=True but performance data yields "
                    "all-zero/NaN weights (constant or all-NaN y). Falling "
                    "back to uniform weights.",
                )
                w = np.ones((ninst, nalgos), dtype=np.double)
            else:
                w[w == 0] = np.min(finite_nonzero)
                w[~np.isfinite(w)] = np.max(finite_nonzero)
        else:
            logger.info(
                "[PYTHIA]  -> PYTHIA is not using cost-sensitive classification.",
            )
            w = np.ones((ninst, nalgos), dtype=int)
        logger.info(
            "[PYTHIA] -------------------------------------------------------"
            "------------------",
        )

        logger.info(
            "[PYTHIA]   -> Using a "
            + str(opts.cv_folds)
            + "-fold stratified cross-validation experiment to evaluate the SVMs.",
        )
        logger.info(
            "[PYTHIA] -------------------------------------------------------"
            "------------------",
        )
        logger.info(
            "[PYTHIA]   -> Training has started. PYTHIA may take a while to"
            " complete...",
        )

        # Section 3: Train SVM model for each algorithm & Evaluate performance.
        overall_start_time = perf_counter()

        for i in range(nalgos):
            algo_start_time = perf_counter()
            yi = y_bin[:, i]
            algorithm_seed = _algorithm_seed(general_options.seed, i)
            algorithm_options = replace(general_options, seed=algorithm_seed)
            algorithm_cp: StratifiedKFold | None = None
            # `np.any`/`np.all` (not `~yi`) since some callers still pass
            # y_bin as 0.0/1.0 floats rather than true booleans; both read
            # identically for "all true"/"all false" either way.
            if bool(np.all(yi)) or not bool(np.any(yi)):
                # StratifiedKFold cannot stratify a single-class label and
                # raises immediately; skip CV/tuning for this algorithm
                # entirely rather than crash the whole run over it.
                res = PythiaStage._fit_degenerate(yi, algo_labels[i])
            else:
                algorithm_cp = StratifiedKFold(
                    n_splits=opts.cv_folds,
                    shuffle=True,
                    random_state=algorithm_seed,
                )
                cp[i] = algorithm_cp
                # Matching the integer ``seed + i`` boundary does not imply
                # that sklearn and MATLAB choose identical fold membership:
                # their stratifiers use different implementations. Verified
                # parity fixtures must export fold IDs when exact membership
                # matters.
                precalc_params: dict[str, float | int | str] | None = None
                param_space: dict[str, Any] | None = None
                if precalcparams is not None:
                    # Pre-calculated hyperparameters always win, regardless of
                    # tuning strategy, and bypass search entirely - matching
                    # MATLAB's precalcparams branch (a direct crossValPredict/
                    # trainFinalClassifier call, no bayesSearch/sobolSearch).
                    precalc_params = PythiaStage._precalc_param_space(
                        classifier_spec,
                        precalcparams[i],
                    )
                elif opts.tuning == "bayes":
                    param_space = PythiaStage._bayes_param_space(classifier_spec)
                # opts.tuning == "sobol": param_space stays None:
                # _fit_classifier dispatches to _sobol_search, which draws
                # its own candidates.
                res = PythiaStage._fit_classifier(
                    z=z,
                    y_bin=yi,
                    w=w[:, i].flatten(),
                    skf=algorithm_cp,
                    classifier_name=opts.classifier,
                    is_poly_kernel=opts.is_poly_krnl,
                    precalc_params=precalc_params,
                    param_space=param_space,
                    use_weights=opts.use_weights,
                    parallel_options=parallel_options,
                    general_options=algorithm_options,
                    n_tuning_iter=opts.n_tuning_iter,
                )

            # Record performance metrics
            y_sub[:, [i]] = res.Ysub.reshape(-1, 1)
            pr0sub[:, [i]] = res.Psub.reshape(-1, 1)
            y_hat[:, [i]] = res.Yhat.reshape(-1, 1)
            pr0hat[:, [i]] = res.Phat.reshape(-1, 1)
            box_consnt.append(res.c)
            k_scale.append(res.g)
            svm.append(res.classifier)

            # Reported metrics must reflect cross-validated performance
            # (Ysub), not training-set performance (Yhat fit on, and
            # evaluated on, all the data) - matching MATLAB, which derives
            # accuracy/precision/recall from the same confusion matrix as
            # Ysub. labels=[False, True] pins the matrix to 2x2 even for a
            # degenerate algorithm whose Ysub/y_bin only ever take one value.
            cm = confusion_matrix(y_bin[:, i], res.Ysub, labels=[False, True])
            tn, fp, fn, tp = cm.ravel()

            accuracy = float((tp + tn) / ninst)
            precision = _safe_ratio(tp, tp + fp)
            recall = _safe_ratio(tp, tp + fn)

            cvcmat[i, :] = [tn, fp, fn, tp]
            accuracy_record.append(accuracy)
            precision_record.append(precision)
            recall_record.append(recall)

            if i == nalgos - 1:
                logger.info(
                    "[PYTHIA]     -> PYTHIA has trained a model for"
                    f" '{algo_labels[i]}', there are no models left to train.",
                )
            else:
                logger.info(
                    f"[PYTHIA]     -> PYTHIA has trained a model for '{algo_labels[i]}'"
                    f",there are {nalgos - i - 1} models left to train.",
                )
            logger.info(
                f"[PYTHIA]       -> Elapsed time:"
                f" {perf_counter() - algo_start_time:.2f}s",
            )

        logger.info(
            f"[PYTHIA] Total elapsed time:  {perf_counter() - overall_start_time:.2f}s",
        )
        logger.info(
            "[PYTHIA] -------------------------------------------------------"
            "------------------",
        )
        logger.info("[PYTHIA]  -> PYTHIA has completed training the models.")
        PythiaStage._display_overall_perf(precision_record, accuracy_record)

        # Select the algorithm with the highest precision
        selection0, selection1 = PythiaStage._determine_selections(
            nalgos,
            precision_record,
            y_hat,
            y_bin,
        )

        logger.info(
            "[PYTHIA] -------------------------------------------------------"
            "------------------",
        )

        # Section4: Generate summary of the results
        summary = PythiaStage._generate_summary(
            nalgos=nalgos,
            algo_labels=algo_labels,
            y=y,
            y_hat=y_hat,
            y_bin=y_bin,
            y_best=y_best,
            selection0=selection0,
            selection1=selection1,
            accuracy=accuracy_record,
            precision=precision_record,
            recall=recall_record,
            box_consnt=box_consnt,
            k_scale=k_scale,
            param1_label=classifier_spec.param1.label,
            param2_label=(
                classifier_spec.param2.label
                if classifier_spec.param2 is not None
                else None
            ),
        )

        return PythiaOutput(
            mu,
            sigma,
            w,
            cp,
            svm,
            cvcmat,
            y_sub,
            y_hat,
            pr0sub,
            pr0hat,
            box_consnt,
            k_scale,
            accuracy_record,
            precision_record,
            recall_record,
            selection0,
            selection1,
            summary,
        )

    @staticmethod
    def _empty_output(
        ninst: int,
        nalgos: int,
        algo_labels: list[str],
        y: NDArray[np.double],
        y_bin: NDArray[np.bool_],
        y_best: NDArray[np.double],
        mu: list[float],
        sigma: list[float],
    ) -> PythiaOutput:
        """Build a "nothing trained" `PythiaOutput` for `opts.skip=True`.

        Matches MATLAB's `emptyPYTHIAout` - every classifier-derived field is
        a placeholder rather than a real training result: `svm` holds one
        ``None`` entry per algorithm, matching MATLAB's empty classifier cells,
        `y_sub`/`y_hat` are all-False, `precision`/`accuracy`/`recall` are
        NaN, `cp` contains one ``None`` per algorithm, and
        `selection0`/`selection1` are both -1 - Python's
        established "no selection at all" sentinel (`_determine_selections`),
        translating MATLAB's `zeros(ninst, 1)` (which relies on MATLAB's
        1-based indexing never colliding with a real selection - 0 would
        collide with "algorithm index 0 selected" here). `mu`/`sigma` are
        the caller's real zscore parameters, not placeholders - MATLAB
        overwrites `emptyPYTHIAout`'s own zeroed/unit-scaled mu/sigma with
        these immediately after, so eval-mode callers can still normalise
        new data correctly.
        """
        svm: list[ClassifierMixin | None] = [None] * nalgos
        y_sub = np.zeros((ninst, nalgos), dtype=bool)
        y_hat = np.zeros((ninst, nalgos), dtype=bool)
        pr0sub = np.zeros((ninst, nalgos), dtype=np.double)
        pr0hat = np.zeros((ninst, nalgos), dtype=np.double)
        cvcmat = np.zeros((nalgos, 4), dtype=int)
        accuracy_record: list[float] = [float("nan")] * nalgos
        precision_record: list[float] = [float("nan")] * nalgos
        recall_record: list[float] = [float("nan")] * nalgos
        box_consnt: list[float] = [float("nan")] * nalgos
        k_scale: list[float] = [float("nan")] * nalgos
        selection0 = np.full(ninst, -1, dtype=np.int_)
        selection1 = np.full(ninst, -1, dtype=np.int_)
        w = np.ones((ninst, nalgos), dtype=np.double)
        cp: list[StratifiedKFold | None] = [None] * nalgos

        summary = PythiaStage._generate_summary(
            nalgos=nalgos,
            algo_labels=algo_labels,
            y=y,
            y_hat=y_hat,
            y_bin=y_bin,
            y_best=y_best,
            selection0=selection0,
            selection1=selection1,
            accuracy=accuracy_record,
            precision=precision_record,
            recall=recall_record,
            box_consnt=box_consnt,
            k_scale=k_scale,
            param1_label=None,
            param2_label=None,
        )

        return PythiaOutput(
            mu,
            sigma,
            w,
            cp,
            svm,
            cvcmat,
            y_sub,
            y_hat,
            pr0sub,
            pr0hat,
            box_consnt,
            k_scale,
            accuracy_record,
            precision_record,
            recall_record,
            selection0,
            selection1,
            summary,
        )

    @staticmethod
    def _precalc_param_space(
        spec: ClassifierSpec,
        precalc_row: NDArray[np.double],
    ) -> dict[str, float | int | str]:
        """Build the classifier's `set_params()` kwargs for pre-calculated params."""
        space: dict[str, float | int | str] = {
            spec.param1.sklearn_name: spec.param1.from_precalc(float(precalc_row[0])),
        }
        if spec.param2 is not None:
            space[spec.param2.sklearn_name] = spec.param2.from_precalc(
                float(precalc_row[1]),
            )
        return space

    @staticmethod
    def _bayes_param_space(spec: ClassifierSpec) -> dict[str, Any]:
        """Build the `tuning='bayes'` search space for `BayesSearchCV`.

        Every classifier - including `'svm'` - gets a proper continuous/
        integer/categorical `skopt` dimension per parameter, matching
        MATLAB's `classifierBayesVars` (`core/PYTHIA.m`): every case there,
        `'svm'` included, is an `optimizableVariable` over a continuous
        (log-transformed where relevant) range, the same range as
        `spec.param1`/`spec.param2`. There is no MATLAB precedent for a
        discrete pre-sampled candidate list - `'svm'` used to get one here
        (`_generate_params`, a pre-F10 leftover), which both discretised its
        continuous range and, since `BayesSearchCV` treats a list as a
        `Categorical` dimension per-parameter independently, silently
        dropped the paired 2D coverage that sampling was meant to give.
        """
        space: dict[str, Any] = {spec.param1.sklearn_name: spec.param1.dimension()}
        if spec.param2 is not None:
            space[spec.param2.sklearn_name] = spec.param2.dimension()
        return space

    @staticmethod
    def _bad_class_proba(
        estimator: ClassifierMixin,
        proba: NDArray[np.double],
    ) -> NDArray[np.double]:
        """Extract P(class 0 = "bad") from a fitted classifier's predict_proba output.

        Matches MATLAB's `Pr0sub`/`Pr0hat` convention (P(class 0), not P(class
        1)) and `InstanceSpace._explore_pythia`'s already-correct lookup:
        don't assume column 0 is "bad" - look up `estimator.classes_`
        explicitly, since `predict_proba`'s column order follows whatever
        order the estimator recorded its classes in. `estimator` must already
        be fitted (its `classes_` populated) before calling this.
        """
        # np.logical_not (not `~`) since some callers still pass y_bin as
        # 0.0/1.0 floats rather than true booleans, and bitwise invert isn't
        # defined for float dtypes.
        classes = np.asarray(estimator.classes_)
        bad_idx = int(np.where(np.logical_not(classes))[0][0])
        return proba[:, bad_idx]

    @staticmethod
    def _warn_unsupported_weights(
        classifier_name: str,
        use_weights: bool,
        supports_sample_weight: bool,
    ) -> None:
        if use_weights and not supports_sample_weight:
            logger.warning(
                f"[PYTHIA] '{classifier_name}' does not support sample "
                "weights - training without cost-sensitive classification "
                "for this algorithm.",
            )

    @staticmethod
    def _cv_fit_params(
        use_weights: bool,
        supports_sample_weight: bool,
        w: NDArray[np.double],
    ) -> dict[str, NDArray[np.double]] | None:
        """`cross_val_predict`'s `params`, weighting each fold's fit.

        #298 audit finding, issue 6: MATLAB threads `Wtrain` into every CV
        fold's fit (`evalFoldClassifier` -> `fitOneClassifier`, `core/
        PYTHIA.m`) for both Sobol- and Bayes-candidate ranking and the final
        full-data fit; Python previously weighted only the final fit,
        leaving candidate ranking's cross-validated predictions unweighted
        even when `use_weights=True`. Matches MATLAB precisely: only the fit
        is weighted, not the resulting misclassification-rate/error metric
        itself (verified directly against `sobolSearch`'s `errs = mean(...)`,
        which stays a plain unweighted rate even in MATLAB).
        """
        if use_weights and supports_sample_weight:
            return {"sample_weight": w}
        return None

    @staticmethod
    def _fit_degenerate(
        y_bin_i: NDArray[np.bool_],
        algo_label: str,
    ) -> _ClassifierResult:
        """Build a constant-prediction result for an all-good/all-bad algorithm.

        `StratifiedKFold` cannot stratify a single-class label and raises
        immediately; MATLAB has the same problem with `cvpartition(...,
        'Stratify', true)` and handles it by skipping cross-validation/tuning
        entirely for that algorithm and using a constant prediction instead
        (`core/PYTHIA.m`). Ysub/Yhat both equal the constant label; Psub/Phat
        are the deterministic P(class 0 = "bad") that implies (1.0 when
        always-bad, 0.0 when always-good).
        """
        value = bool(y_bin_i[0])
        label_word = "good" if value else "bad"
        logger.warning(
            f"[PYTHIA] Algorithm '{algo_label}' is {label_word} for every "
            "instance; skipping cross-validation/tuning and using a "
            "constant prediction instead.",
        )
        n = y_bin_i.shape[0]
        y_const = np.full(n, value, dtype=bool)
        p_bad = np.full(n, 0.0 if value else 1.0, dtype=np.double)
        return _ClassifierResult(
            classifier=_ConstantClassifier(value),
            Ysub=y_const,
            Psub=p_bad,
            Yhat=y_const,
            Phat=p_bad,
            c=float("nan"),
            g=float("nan"),
        )

    @staticmethod
    def _fit_precalculated(
        estimator: ClassifierMixin,
        spec: ClassifierSpec,
        precalc_params: dict[str, float | int | str],
        use_weights: bool,
        z: NDArray[np.double],
        y_bin: NDArray[np.bool_],
        w: NDArray[np.double],
        skf: StratifiedKFold,
    ) -> _ClassifierResult:
        """Fit a classifier at pre-calculated hyperparameters - no search at all.

        Matches MATLAB's `precalcparams` branch (`core/PYTHIA.m`): a direct
        `crossValPredict`/`trainFinalClassifier` call, bypassing
        `bayesSearch`/`sobolSearch` entirely rather than running a
        degenerate single-point search over them (#292 - the crash this
        replaces came from feeding those scalars into `BayesSearchCV`, which
        requires real search-space `Dimension`s).
        """
        estimator.set_params(**precalc_params)
        if use_weights and spec.supports_sample_weight:
            estimator.fit(z, y_bin, sample_weight=w)
        else:
            estimator.fit(z, y_bin)

        fit_params = PythiaStage._cv_fit_params(
            use_weights,
            spec.supports_sample_weight,
            w,
        )
        y_sub = cross_val_predict(
            estimator,
            z,
            y_bin,
            cv=skf,
            method="predict",
            params=fit_params,
        )
        p_sub = PythiaStage._bad_class_proba(
            estimator,
            cross_val_predict(
                estimator,
                z,
                y_bin,
                cv=skf,
                method="predict_proba",
                params=fit_params,
            ),
        )
        y_hat = estimator.predict(z)
        p_hat = PythiaStage._bad_class_proba(estimator, estimator.predict_proba(z))

        c = spec.param1.reported(precalc_params[spec.param1.sklearn_name])
        g = (
            spec.param2.reported(precalc_params[spec.param2.sklearn_name])
            if spec.param2 is not None
            else float("nan")
        )

        return _ClassifierResult(
            classifier=estimator,
            Yhat=y_hat,
            Ysub=y_sub,
            Psub=p_sub,
            Phat=p_hat,
            c=c,
            g=g,
        )

    @staticmethod
    def _fit_classifier(
        z: NDArray[np.double],
        y_bin: NDArray[np.bool_],
        w: NDArray[np.double],
        skf: StratifiedKFold,
        classifier_name: str,
        is_poly_kernel: bool,
        precalc_params: dict[str, float | int | str] | None,
        param_space: dict[str, Any] | None,
        use_weights: bool,
        parallel_options: ParallelOptions,
        general_options: GeneralOptions,
        n_tuning_iter: int,
    ) -> _ClassifierResult:
        """Train one classifier (per `PythiaOptions.classifier`) for one algorithm.

        Every registered classifier is tunable (its own per-classifier
        hyperparameter range, matching MATLAB's `classifierBayesVars`/
        `sobolToParams`). Dispatches on, in order: `precalc_params` (set and
        fit directly, no search, matching MATLAB - see `_fit_precalculated`);
        else `param_space is None` (`tuning='sobol'`, dispatching to
        `_sobol_search`); else `param_space` (`tuning='bayes'`, using
        `BayesSearchCV` below).

        Parameters
        ----------
        z : NDArray[np.double]
            The instance space.
        y_bin : NDArray[np.bool_]
            The binary labels.
        w : NDArray[np.double]
            The sample weights.
        skf : StratifiedKFold
            The stratified k-fold cross-validation object.
        classifier_name : str
            `PythiaOptions.classifier` - which registered classifier to train.
        is_poly_kernel : bool
            Whether to use a polynomial kernel. Only meaningful for `'svm'`.
        precalc_params : dict | None
            `PythiaOptions.params`' hyperparameters for this algorithm, already
            converted to this classifier's own units - bypasses tuning
            entirely when not `None` (see `_fit_precalculated`).
        param_space : dict | None
            The hyperparameters to search, when `PythiaOptions.tuning` isn't
            `'sobol'` (see above). Ignored when `precalc_params` is given.
        use_weights : bool
            Whether cost-sensitive classification was requested - honoured only
            for classifiers whose `fit()` accepts `sample_weight`.
        parallel_options : ParallelOptions
            The parallel options, specifiy whether run in parallel and number of cores.
        general_options : GeneralOptions
            General options (e.g. the RNG seed), not specific to any one stage.
        n_tuning_iter : int
            `PythiaOptions.n_tuning_iter` - the search's evaluation budget,
            for both `_sobol_search` and `BayesSearchCV` below (matching
            MATLAB's `opts.nTuningIter`, used identically for `'sobol'` and
            `'bayes'` - see `core/PYTHIA.m`'s `sobolSearch`/`bayesSearch`).

        Returns
        -------
        _ClassifierResult
        The trained-classifier result.
        """
        spec = get_classifier_fcn(classifier_name)
        estimator = spec.build(general_options.seed, is_poly_kernel)
        PythiaStage._warn_unsupported_weights(
            classifier_name,
            use_weights,
            spec.supports_sample_weight,
        )

        if precalc_params is not None:
            return PythiaStage._fit_precalculated(
                estimator,
                spec,
                precalc_params,
                use_weights,
                z,
                y_bin,
                w,
                skf,
            )

        if param_space is None:
            # tuning='sobol' with no pre-calculated params.
            return PythiaStage._sobol_search(
                estimator,
                spec,
                use_weights,
                z,
                y_bin,
                w,
                skf,
                n_tuning_iter,
                general_options.seed,
            )

        # tuning='bayes': Bayesian optimisation over param_space. Sobol
        # (F10's default) superseded the old RandomizedSearchCV ("grid
        # search") alternative that used to sit here for 'svm' - it covered
        # the same lightweight/random-ish search role, just done properly
        # (space-filling quasi-random, not sklearn's uniform random).
        optimization = BayesSearchCV(
            estimator=estimator,
            n_iter=n_tuning_iter,
            search_spaces=param_space,
            optimizer_kwargs=_BAYES_OPTIMIZER_KWARGS,
            cv=skf,
            verbose=0,
            random_state=general_options.seed,
            n_jobs=(parallel_options.n_cores if parallel_options.flag else 1),
            # A sampled candidate can be untrainable on a given fold (e.g. a
            # degenerate kernel bandwidth); disqualify it with the worst
            # possible score instead of crashing the whole search, mirroring
            # _evaluate_sobol_candidates' per-candidate catch below. Unlike
            # GridSearchCV, skopt's Bayesian optimiser feeds this score straight
            # into its Gaussian-process surrogate, which can't handle NaN - so
            # this must be a finite value, not `np.nan`.
            error_score=0.0,
        )
        fit_kwargs = {"sample_weight": w} if spec.supports_sample_weight else {}
        optimization.fit(z, y_bin, **fit_kwargs)
        best_estimator = optimization.best_estimator_
        c = spec.param1.reported(optimization.best_params_[spec.param1.sklearn_name])
        g = (
            spec.param2.reported(optimization.best_params_[spec.param2.sklearn_name])
            if spec.param2 is not None
            else float("nan")
        )

        # Perform cross-validated predictions using the best estimator
        bayes_fit_params = PythiaStage._cv_fit_params(
            use_weights,
            spec.supports_sample_weight,
            w,
        )
        y_sub = cross_val_predict(
            best_estimator,
            z,
            y_bin,
            cv=skf,
            method="predict",
            params=bayes_fit_params,
        )
        p_sub = PythiaStage._bad_class_proba(
            best_estimator,
            cross_val_predict(
                best_estimator,
                z,
                y_bin,
                cv=skf,
                method="predict_proba",
                params=bayes_fit_params,
            ),
        )
        # Predict the labels and probabilities for the entire dataset
        y_hat = best_estimator.predict(z)
        p_hat = PythiaStage._bad_class_proba(
            best_estimator,
            best_estimator.predict_proba(z),
        )

        return _ClassifierResult(
            classifier=best_estimator,
            Yhat=y_hat,
            Ysub=y_sub,
            Psub=p_sub,
            Phat=p_hat,
            c=c,
            g=g,
        )

    @staticmethod
    def _evaluate_sobol_candidates(
        estimator: ClassifierMixin,
        spec: ClassifierSpec,
        points: NDArray[np.double],
        z: NDArray[np.double],
        y_bin: NDArray[np.bool_],
        skf: StratifiedKFold,
        use_weights: bool,
        w: NDArray[np.double],
    ) -> tuple[list[dict[str, float | int | str]], NDArray[np.double]]:
        """Evaluate each Sobol candidate's CV misclassification error.

        A candidate that fails to train (e.g. a degenerate bandwidth) is
        disqualified with an infinite error rather than raising, mirroring
        MATLAB's `evalFoldClassifier` catch/NaN-disqualify behaviour. Each
        fold's fit is weighted when requested (#298 audit finding, issue 6 -
        see `_cv_fit_params`); the error rate itself stays an unweighted
        misclassification rate, matching MATLAB's own `sobolSearch`.
        """
        candidates: list[dict[str, float | int | str]] = []
        errs = np.zeros(len(points))
        fit_params = PythiaStage._cv_fit_params(
            use_weights,
            spec.supports_sample_weight,
            w,
        )
        for i, point in enumerate(points):
            candidate: dict[str, float | int | str] = {
                spec.param1.sklearn_name: spec.param1.sample(point[0]),
            }
            if spec.param2 is not None:
                candidate[spec.param2.sklearn_name] = spec.param2.sample(point[1])
            candidates.append(candidate)
            try:
                estimator.set_params(**candidate)
                y_pred = cross_val_predict(
                    estimator,
                    z,
                    y_bin,
                    cv=skf,
                    method="predict",
                    params=fit_params,
                )
                errs[i] = np.mean(y_pred != y_bin)
            except ValueError as error:
                logger.warning(
                    f"[PYTHIA] Sobol candidate {candidate} failed to train: {error}",
                )
                errs[i] = np.inf
        return candidates, errs

    @staticmethod
    def _sobol_search(
        estimator: ClassifierMixin,
        spec: ClassifierSpec,
        use_weights: bool,
        z: NDArray[np.double],
        y_bin: NDArray[np.bool_],
        w: NDArray[np.double],
        skf: StratifiedKFold,
        n_iter: int,
        seed: int | None,
    ) -> _ClassifierResult:
        """Lightweight quasi-random hyperparameter search (F10, `tuning='sobol'`).

        Evaluates `n_iter` scrambled-Sobol candidates (one or two
        hyperparameters, per `spec.param1`/`spec.param2`) via k-fold CV and
        keeps the one with the lowest misclassification error - a direct,
        much lighter-weight port of MATLAB's `sobolSearch`/`sobolToParams`
        (`core/PYTHIA.m`), used in place of `skopt.BayesSearchCV`'s heavier
        sequential-optimisation machinery. `w`/`use_weights` weight every CV
        fold's fit, both during candidate ranking (`_evaluate_sobol_
        candidates`) and the final full-data fit - matching MATLAB's own
        `Wtrain` threading through `evalFoldClassifier` (#298 audit finding,
        issue 6; see `_cv_fit_params`).
        """
        n_dims = 2 if spec.param2 is not None else 1
        sampler = stats.qmc.Sobol(d=n_dims, scramble=True, seed=seed)
        # MATLAB accepts an arbitrary tuning budget (20 by default).  SciPy's
        # warning merely notes that non-power-of-two prefixes lose one Sobol
        # balance property; it does not indicate an invalid search.  Filter
        # only that exact advisory so all other numerical warnings stay loud.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="The balance properties of Sobol.*",
                category=UserWarning,
                module=r"scipy\.stats\._qmc",
            )
            points = sampler.random(n_iter)

        candidates, errs = PythiaStage._evaluate_sobol_candidates(
            estimator,
            spec,
            points,
            z,
            y_bin,
            skf,
            use_weights,
            w,
        )
        if np.all(np.isinf(errs)):
            logger.warning(
                "[PYTHIA] All Sobol candidates failed to train; using the "
                "first candidate.",
            )
            best_index = 0
        else:
            best_index = int(np.argmin(errs))
        best_params = candidates[best_index]

        estimator.set_params(**best_params)
        if use_weights and spec.supports_sample_weight:
            estimator.fit(z, y_bin, sample_weight=w)
        else:
            estimator.fit(z, y_bin)

        sobol_fit_params = PythiaStage._cv_fit_params(
            use_weights,
            spec.supports_sample_weight,
            w,
        )
        y_sub = cross_val_predict(
            estimator,
            z,
            y_bin,
            cv=skf,
            method="predict",
            params=sobol_fit_params,
        )
        p_sub = PythiaStage._bad_class_proba(
            estimator,
            cross_val_predict(
                estimator,
                z,
                y_bin,
                cv=skf,
                method="predict_proba",
                params=sobol_fit_params,
            ),
        )
        y_hat = estimator.predict(z)
        p_hat = PythiaStage._bad_class_proba(estimator, estimator.predict_proba(z))

        c = spec.param1.reported(best_params[spec.param1.sklearn_name])
        g = (
            spec.param2.reported(best_params[spec.param2.sklearn_name])
            if spec.param2 is not None
            else float("nan")
        )

        return _ClassifierResult(
            classifier=estimator,
            Yhat=y_hat,
            Ysub=y_sub,
            Psub=p_sub,
            Phat=p_hat,
            c=c,
            g=g,
        )

    @staticmethod
    def _log_kernel_choice(ninst: int, is_poly_krnl: bool) -> None:
        """Log the SVM kernel choice and the large-dataset kernel suggestion.

        Only meaningful when `PythiaOptions.classifier == 'svm'`.
        """
        if ninst > LARGE_NUM_INSTANCE and not is_poly_krnl:
            logger.info(
                "[PYTHIA]   -> For datasets larger than 1K Instances, "
                "PYTHIA works better with a Polynomial kernel.",
            )
            logger.info(
                "[PYTHIA]   -> Consider changing the kernel if the results are"
                " unsatisfactory.",
            )
            logger.info(
                "[PYTHIA] ---------------------------------------------------"
                "----------------",
            )

        if is_poly_krnl:
            logger.info("[PYTHIA]  => PYTHIA is using polynomial kernel")
        else:
            logger.info("[PYTHIA]  => PYTHIA is using gaussian kernel")

        logger.info(
            "[PYTHIA] -------------------------------------------------------"
            "------------------",
        )

    @staticmethod
    def _display_overall_perf(precision: list[float], accuracy: list[float]) -> None:
        """Calculate overall performance.

        Parameters
        ----------
        precision : list[float]
            The precision metrics.
        accuracy : list[float]
            The accuracy metrics.

        Returns
        -------
        None
        """
        precision_mean = float(_nanmean(np.asarray(precision, dtype=np.double)))
        accuracy_mean = float(_nanmean(np.asarray(accuracy, dtype=np.double)))
        logger.info(
            "[PYTHIA]  -> The average cross validated precision is: "
            + str(np.round(100 * precision_mean, 1))
            + "%",
        )

        logger.info(
            "[PYTHIA]  -> The average cross validated accuracy is: "
            + str(np.round(100 * accuracy_mean, 1))
            + "%",
        )

    @staticmethod
    def _compute_znorm(
        z: NDArray[np.double],
    ) -> tuple[list[float], list[float], NDArray[np.double]]:
        """Compute mormalized z, standard deviations and mean.

        Parameters
        ----------
        z : NDArray[np.double]
            The feature coordinates.

        Returns
        -------
        tuple[list[float], list[float], NDArray[np.double]]
        The mean, standard deviation and normalized feature coordinates.
        """
        # mu/sigma must describe the *raw* z (matching MATLAB's zscore, which
        # returns them alongside its normalised output) so a later caller can
        # normalise new data as (z_new - mu) / sigma. Computing them from the
        # already-normalised z (the previous order here) gave mu ~= 0, sigma
        # ~= 1 regardless of the original feature scale.
        mu = np.mean(z, axis=0)
        sigma = np.std(z, ddof=1, axis=0)
        z = stats.zscore(z, ddof=1)
        return (mu, sigma, z)

    @staticmethod
    def _check_precalcparams(
        params: NDArray[Any] | None,
        nalgos: int,
        spec: ClassifierSpec,
        classifier_name: str,
    ) -> NDArray[np.double] | None:
        """Check pre-calculated hyper-parameters.

        Parameters
        ----------
        params : NDArray | None
            The pre-calculated hyper-parameters.
        nalgos : int
            The number of algorithms.
        spec : ClassifierSpec
            The registered classifier (`PythiaOptions.classifier`) - determines
            the expected column count: 1 for single-parameter classifiers
            (tree/nb/linear), 2 for two-parameter ones (svm/knn/ensemble).
        classifier_name : str
            `PythiaOptions.classifier`, for the log message only.

        Returns
        -------
        NDArray[np.double] | None
        The pre-calculated hyper-parameters or None.
        """
        if params is None:
            return None
        params_array = np.asarray(params)
        n_params = 1 + int(spec.param2 is not None)
        expected_shape = (nalgos, n_params)
        if params_array.shape != expected_shape:
            logger.warning(
                "[PYTHIA] -> PythiaOptions.params has shape "
                f"{params_array.shape}. Classifier '{classifier_name}' requires "
                f"shape {expected_shape}; ignoring the pre-calculated values "
                "and falling back to the configured tuning strategy, matching "
                "MATLAB v0.9.1 core/PYTHIA.m.",
            )
            return None
        if not np.issubdtype(params_array.dtype, np.number) or not np.isrealobj(
            params_array,
        ):
            raise ValueError("PythiaOptions.params must contain real numbers.")
        params_array = np.asarray(params_array, dtype=np.double)
        if not np.all(np.isfinite(params_array)):
            raise ValueError("PythiaOptions.params must contain finite values.")

        parameter_specs = [spec.param1]
        if spec.param2 is not None:
            parameter_specs.append(spec.param2)
        for row_index, row in enumerate(params_array):
            for column_index, parameter_spec in enumerate(parameter_specs):
                value = float(row[column_index])
                if (
                    parameter_spec.categories is None
                    and not parameter_spec.is_int
                    and value <= 0
                ):
                    raise ValueError(
                        "PythiaOptions.params"
                        f"[{row_index}, {column_index}] "
                        f"({parameter_spec.label} for classifier "
                        f"'{classifier_name}') must be strictly positive; "
                        f"got {value}.",
                    )
        logger.info(
            f"[PYTHIA] -> Using pre-calculated hyper-parameters for the "
            f"'{classifier_name}' classifier.",
        )
        return params_array

    @staticmethod
    def _validate_cv_folds(
        y_bin: NDArray[np.bool_],
        algo_labels: list[str],
        cv_folds: int,
    ) -> None:
        """Reject impossible CV layouts and warn about sparse test folds.

        MATLAB permits ``kFold`` to exceed the minority-class count.  Scikit-learn
        does too (with a warning): training remains valid when the minority class
        has at least two observations, although some test folds omit that class.
        A singleton minority class is different because the fold containing it
        leaves a single-class training set, which the active classifiers cannot
        train on.
        """
        n_instances = y_bin.shape[0]
        if cv_folds > n_instances:
            raise ValueError(
                f"PythiaOptions.cv_folds={cv_folds} exceeds the number of "
                f"instances {n_instances}.",
            )

        largest_test_fold = (n_instances + cv_folds - 1) // cv_folds
        if n_instances - largest_test_fold < 1:
            raise ValueError(
                f"PythiaOptions.cv_folds={cv_folds} leaves no usable "
                "cross-validation training split.",
            )

        for index, algo_label in enumerate(algo_labels):
            labels = np.asarray(y_bin[:, index], dtype=np.bool_)
            good_count = int(np.count_nonzero(labels))
            bad_count = int(labels.size - good_count)
            if good_count == 0 or bad_count == 0:
                continue
            smallest_class = min(good_count, bad_count)
            if smallest_class == 1:
                raise ValueError(
                    f"PythiaOptions.cv_folds={cv_folds} leaves a "
                    "single-class cross-validation training split for "
                    f"algorithm '{algo_label}' (smallest class count 1).",
                )
            if cv_folds > smallest_class:
                logger.warning(
                    f"PythiaOptions.cv_folds={cv_folds} exceeds class count "
                    f"{smallest_class} for algorithm '{algo_label}'; continuing "
                    "to match MATLAB, with some test folds missing that class.",
                )

    @staticmethod
    def _validate_tuning(
        opts: PythiaOptions,
        precalcparams: NDArray[np.double] | None,
    ) -> None:
        """Validate `PythiaOptions.tuning`/`n_tuning_iter` (F10).

        Fails loudly and early - before any training starts - rather than
        letting a bad value surface as a confusing error deep inside a
        search loop. Every registered classifier is tunable (F1's follow-on
        registry extension), so this always applies.
        """
        if opts.tuning not in ("sobol", "bayes", "none"):
            msg = (
                f"PythiaOptions.tuning={opts.tuning!r} is not recognised; must "
                "be one of 'sobol', 'bayes', 'none'."
            )
            raise ValueError(msg)
        if opts.tuning == "none" and precalcparams is None:
            msg = (
                "PythiaOptions.tuning='none' requires PythiaOptions.params to be "
                "a valid pre-calculated hyperparameters array. Either supply "
                "params or set tuning to 'sobol' or 'bayes'."
            )
            raise ValueError(msg)
        # n_tuning_iter is only ever consumed by the Sobol/Bayes search paths
        # (precalcparams bypasses tuning entirely) - matching MATLAB's own
        # nIter check (core/PYTHIA.m), which applies identically to both
        # tuning strategies rather than singling one out.
        if opts.tuning in ("sobol", "bayes") and opts.n_tuning_iter < 1:
            msg = (
                "PythiaOptions.n_tuning_iter must be a positive integer (it is "
                "the Sobol/Bayes search's evaluation budget); got "
                f"{opts.n_tuning_iter}."
            )
            raise ValueError(msg)

    @staticmethod
    def _log_tuning_strategy(
        opts: PythiaOptions,
        precalcparams: NDArray[np.double] | None,
    ) -> None:
        """Log which hyperparameter search strategy this run will use."""
        if precalcparams is not None:
            return  # _check_precalcparams already logged this above.
        if opts.tuning == "sobol":
            logger.info(
                f"[PYTHIA]  -> PYTHIA is using a Sobol quasi-random search "
                f"({opts.n_tuning_iter} candidates) for hyper-parameter"
                " optimization.",
            )
        else:
            logger.info(
                f"[PYTHIA]  -> PYTHIA is using Bayesian optimization "
                f"({opts.n_tuning_iter} evaluations) for hyper-parameter"
                " optimization.",
            )

    @staticmethod
    def _weighted_selection(
        nalgos: int,
        precision: list[float] | NDArray[np.double],
        y_hat: NDArray[np.bool_],
    ) -> tuple[NDArray[np.double], NDArray[np.int_]]:
        """Precision-weighted per-instance best score and its raw algorithm index.

        The numerical core shared by `_determine_selections` (training - also
        needs `y_bin` for the `selection1` default-fallback below) and
        `InstanceSpace._explore_pythia` (F8 - only ever needs `selection0`,
        which doesn't depend on `y_bin` at all). `best <= 0` marks "no
        algorithm was selected"; callers apply their own sentinel/default
        substitution on top of this shared core, since what replaces "no
        selection" differs between the two (training's `selection0`/
        `selection1` pair vs. `explore()`'s `selection0`-only path).

        Parameters
        ----------
        nalgos : int
            The number of algorithms.
        precision : list[float] | NDArray[np.double]
            The precision metrics (per algorithm).
        y_hat : NDArray[np.bool_]
            The predicted labels.
        """
        precision_array = np.asarray(precision, dtype=np.double).reshape(1, nalgos)
        precision_array = np.where(np.isnan(precision_array), 0.0, precision_array)
        weighted_yhat = y_hat * precision_array
        best = np.max(weighted_yhat, axis=1)
        raw_selection = np.argmax(weighted_yhat, axis=1).astype(np.int_)

        return best, raw_selection

    @staticmethod
    def _determine_selections(
        nalgos: int,
        precision: list[float],
        y_hat: NDArray[np.bool_],
        y_bin: NDArray[np.bool_],
    ) -> tuple[NDArray[np.int_], NDArray[np.int_]]:
        """Determine the selections based on the predicted labels and precision.

        Selects the best-performing algorithm for each instance using
        precision-weighted predictions (`_weighted_selection`). If no
        algorithm is selected (i.e., all scores are non-positive),
        `selection1` defaults to the algorithm with the best average
        performance; `selection0` uses -1 instead (see
        `_weighted_selection`'s docstring for why the two callers need
        different "no selection" conventions).

        Parameters
        ----------
        nalgos : int
            The number of algorithms.
        precision : list[float]
            The precision metrics.
        y_hat : NDArray[np.bool_]
            The predicted labels.
        y_bin : NDArray[np.bool_]
            The binary labels.
        """
        # Stores the index of the column with the highest mean value.
        # Index starts from 0
        default = np.argmax(np.mean(y_bin, axis=0))

        best, raw_selection = PythiaStage._weighted_selection(nalgos, precision, y_hat)

        # -1 (not 0) marks "no selection", matching
        # `InstanceSpace._explore_pythia`'s already-established convention -
        # 0 must stay free to mean "algorithm index 0 was genuinely
        # selected", which it cannot also do if reused as the sentinel.
        selection0 = np.copy(raw_selection)
        selection1 = np.copy(raw_selection)
        selection0[best <= 0] = -1
        selection1[best <= 0] = default
        return (selection0, selection1)

    @staticmethod
    def _generate_summary(
        nalgos: int,
        algo_labels: list[str],
        y: NDArray[np.double],
        y_hat: NDArray[np.bool_],
        y_bin: NDArray[np.bool_],
        y_best: NDArray[np.double],
        selection0: NDArray[np.int_],
        selection1: NDArray[np.int_],
        accuracy: list[float],
        precision: list[float],
        recall: list[float],
        box_consnt: list[float],
        k_scale: list[float],
        param1_label: str | None,
        param2_label: str | None,
    ) -> pd.DataFrame:
        """Generate a summary of the results.

        Parameters
        ----------
        nalgos : int
            The number of algorithms.
        algo_labels : list[str]
            The algorithm labels.
        y : NDArray[np.double]
            The performance metrics.
        y_hat : NDArray[np.bool_]
            The predicted labels.
        y_bin : NDArray[np.bool_]
            The binary labels.
        y_best : NDArray[np.double]
            The best performance metrics.
        selection0 : NDArray[np.integer]
            The selected algorithms.
        selection1 : NDArray[np.integer]
            Backup selected algorithm.
        accuracy : list[float]
            The accuracy metrics.
        precision : list[float]
            The precision metrics.
        recall : list[float]
            The recall metrics.
        box_consnt : list[float]
            Each algorithm's tuned first hyperparameter (`param1_label`'s units).
        k_scale : list[float]
            Each algorithm's tuned second hyperparameter (`param2_label`'s
            units), or NaN for classifiers with only one tunable parameter.
        param1_label : str | None
            Column header for `box_consnt`, matching `ISAgetClassifierFcn.m`'s
            `p1label` for `PythiaOptions.classifier`, or `None` to drop both
            hyperparameter columns entirely (`opts.skip=True`'s empty
            output - matching MATLAB's `buildSummary(..., [], [])` call from
            `emptyPYTHIAout`, whose `~isempty(p1label)` check gates
            `param2_label` too).
        param2_label : str | None
            Column header for `k_scale`, or `None` for classifiers with only
            one tunable parameter (`p2label = 'N/A'`), which drops the column
            entirely - matching MATLAB's `buildSummary`. Ignored when
            `param1_label` is `None`.
        """
        logger.info("[PYTHIA]   -> PYTHIA is preparing the summary table.")

        # Obtain the corresponding selection matrix for the two selections.
        # selection0/1 are 0-based algorithm indices (-1 in selection0 means
        # "no algorithm selected", which np.arange(nalgos) never matches, so
        # it naturally drops out of sel0 here).
        sel0 = selection0[:, np.newaxis] == np.arange(nalgos)
        sel1 = selection1[:, np.newaxis] == np.arange(nalgos)

        # Compute the average performance of the selected algorithms
        avgperf = np.round(_nanmean(y, axis=0), 3)
        stdperf = np.round(_matlab_nanstd(y, axis=0), 3)

        """This variable stores the full performance of the algorithms,
        but filtered based on selection1
        """
        y_full = y.copy()

        # This variable stores the performance of the selected algorithms
        y_svms = y.copy()

        # y is the caller's y_raw array; copy it before mutating so the
        # caller's data isn't silently changed by generating this summary.
        y = y.copy()
        y[~sel0] = np.nan
        y_full[~sel1] = np.nan
        y_svms[~y_hat] = np.nan

        # Compute the probability of "good"
        pgood = np.mean(np.any(np.logical_and(y_bin, sel1), axis=1))

        # Selector precision/recall, matching MATLAB's per-instance `any(...)`
        # definition (core/PYTHIA.m) rather than sklearn's flattened
        # (instance x algorithm)-pair precision_score/recall_score, which
        # answers a different question (agreement per pair, not "was the
        # selected algorithm good for this instance").
        not_y_bin = np.logical_not(y_bin)
        not_sel0 = np.logical_not(sel0)
        tg = np.sum(np.any(np.logical_and(y_bin, sel0), axis=1))  # selected, good
        fg = np.sum(np.any(np.logical_and(not_y_bin, sel0), axis=1))  # selected, bad
        fb = np.sum(np.any(np.logical_and(y_bin, not_sel0), axis=1))  # good, unselected
        precisionsel = _safe_ratio(tg, tg + fg)
        recallsel = _safe_ratio(tg, tg + fb)

        # Prepare the data for the summary table
        data = {
            "Algorithms": [*algo_labels, "Oracle", "Selector"],
            "Avg_Perf_all_instances": np.round(
                np.append(avgperf, [_nanmean(y_best), _nanmean(y_full)]),
                3,
            ),
            "Std_Perf_all_instances": np.round(
                np.append(
                    stdperf,
                    [_matlab_nanstd(y_best), _matlab_nanstd(y_full)],
                ),
                3,
            ),
            "Probability_of_good": np.round(
                np.append(_nanmean(y_bin, axis=0), [1, pgood]),
                3,
            ),
            "Avg_Perf_selected_instances": np.round(
                np.append(
                    _nanmean(y_svms, axis=0),
                    np.array([np.nan, _nanmean(y)]),
                ),
                3,
            ),
            "Std_Perf_selected_instances": np.round(
                np.append(
                    _matlab_nanstd(y_svms, axis=0),
                    np.array([np.nan, _matlab_nanstd(y)]),
                ),
                3,
            ),
            "CV_model_accuracy": np.round(
                100 * np.append(accuracy, [np.nan, np.nan]),
                1,
            ),
            "CV_model_precision": np.round(
                100 * np.append(precision, [np.nan, precisionsel]),
                1,
            ),
            "CV_model_recall": np.round(
                100 * np.append(recall, [np.nan, recallsel]),
                1,
            ),
        }
        if param1_label is not None:
            data[param1_label] = np.round(np.append(box_consnt, [np.nan, np.nan]), 3)
            if param2_label is not None:
                data[param2_label] = np.round(
                    np.append(k_scale, [np.nan, np.nan]),
                    3,
                )

        df = pd.DataFrame(data)
        logger.info(
            f"[PYTHIA]   -> PYTHIA has completed! Performance of the models:\n{df}",
        )
        return df
