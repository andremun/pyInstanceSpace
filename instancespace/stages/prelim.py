# SPDX-License-Identifier: LicenseRef-PolyForm-Noncommercial-1.0.0
# Copyright (c) 2024-2026 Mario Andrés Muñoz
"""Performing preliminary data processing.

The main focus is on the `prelim` function, which prepares the input data for further
analysis and modeling.

The `prelim` function takes feature and performance data matrices along with a set of
processing options, and performs various preprocessing tasks such as normalization,
outlier detection and removal, and binary performance classification. These tasks are
guided by the options specified in the `InstanceSpaceOptions` object.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
from loguru import logger
from numpy.typing import NDArray
from scipy import optimize, stats
from sklearn.model_selection import train_test_split

from instancespace.data.model import DataDense, PrelimOut
from instancespace.data.options import (
    GeneralOptions,
    PerformanceOptions,
    PrelimOptions,
    SelvarsOptions,
)
from instancespace.stages.stage import PredictiveStage, Stage
from instancespace.utils.filter import do_filter

# Fraction of instances with a best-algorithm performance of exactly zero above
# which a data-quality warning fires (matches MATLAB's PRELIM.m,
# ISA:PRELIM:manyZeroBest).
_MANY_ZERO_BEST_THRESHOLD = 0.05
_OOD_CLIPPED_FRACTION_THRESHOLD = 0.05


def _log_many_zero_best_warning(y_best: NDArray[np.double], log_prefix: str) -> None:
    """Warn if too many instances have a best-algorithm performance of zero.

    Matches MATLAB's ISA:PRELIM:manyZeroBest: once these zeros are
    substituted with `eps` below, the relative-performance matrix becomes
    uninformative (close to 1 everywhere) for those instances. Shared by
    `PrelimStage._warn_many_zero_best()` (training) and
    `compute_binary_performance()` (F9's explore()-time evaluation path).
    """
    frac_zero_best = np.mean(y_best == 0)
    if frac_zero_best > _MANY_ZERO_BEST_THRESHOLD:
        logger.warning(
            f"[{log_prefix}] {frac_zero_best:.1%} of instances have a best-algorithm "
            "performance of exactly zero; the relative-performance matrix will be "
            "close to 1 everywhere for these instances.",
        )


class BinaryPerformance(NamedTuple):
    """Ground-truth binary-performance fields derived from raw algorithm performance.

    Attributes
    ----------
    y : NDArray[np.double]
        Relative (or absolute) performance matrix - `y_raw` transformed by the
        same `max_perf`/`abs_perf` branching used to derive `y_bin`.
    y_bin : NDArray[np.bool_]
        Binary matrix indicating instances with good algorithm performance.
    y_best : NDArray[np.double]
        Best observed performance value for each instance across all algorithms
        (zero values substituted with `eps`, matching MATLAB).
    p : NDArray[np.int_]
        1-based index of the best-performing algorithm per instance, ties
        broken with MATLAB-compatible seeded random selection.
    num_good_algos : NDArray[np.double]
        Number of algorithms with good performance, per instance.
    beta : NDArray[np.bool_]
        Whether each instance clears `beta_threshold`'s fraction of good
        algorithms.
    """

    y: NDArray[np.double]
    y_bin: NDArray[np.bool_]
    y_best: NDArray[np.double]
    p: NDArray[np.int_]
    num_good_algos: NDArray[np.double]
    beta: NDArray[np.bool_]


def compute_binary_performance(
    y_raw: NDArray[np.double],
    perf_opts: PerformanceOptions,
    general_opts: GeneralOptions,
    log_prefix: str = "PRELIM",
) -> BinaryPerformance:
    """Compute the binary measure of algorithm performance from raw `Y`.

    Ports MATLAB PRELIM.m's binary-performance section: an algorithm is
    "good" for an instance if its performance is within `epsilon` of the best
    (relative) or better than `epsilon` outright (absolute), per
    `perf_opts.max_perf`/`abs_perf`. Ties for the best algorithm use MATLAB's
    seeded twister stream and ``randi`` selection rule.

    Shared by `PrelimStage._prelim()` (training, ground truth for the
    training set) and `InstanceSpace.explore()`'s evaluation path (F9, ground
    truth for a test set using the *trained* `PerformanceOptions`) - one
    implementation, not a second one written to match it by hand.
    """
    logger.info(
        f"[{log_prefix}] -------------------------------------------------"
        "------------------------",
    )
    logger.info(f"[{log_prefix}] -> Calculating the binary measure of performance")

    y = y_raw.copy()
    nalgos = y.shape[1]

    msg = "An algorithm is good if its performance is "
    if perf_opts.max_perf:
        y_aux = y.copy()
        y_aux[np.isnan(y_aux)] = -np.inf

        y_best = np.max(y_aux, axis=1)
        # Snapshot before the zero-value eps substitution below, so tied
        # instances whose best performance is exactly zero can still be
        # detected as ties (matches MATLAB's YbestTie).
        y_best_tie = y_best.copy()
        # add 1 to the index to match the MATLAB code
        p = np.argmax(y_aux, axis=1) + 1

        if perf_opts.abs_perf:
            y_bin = y_aux >= perf_opts.epsilon
            msg = msg + "higher than " + str(perf_opts.epsilon)
        else:
            _log_many_zero_best_warning(y_best, log_prefix)
            y_best[y_best == 0] = np.finfo(float).eps
            y[y == 0] = np.finfo(float).eps
            y = 1 - y / y_best[:, np.newaxis]
            y_bin = (1 - y_aux / y_best[:, np.newaxis]) <= perf_opts.epsilon
            msg = (
                msg + "within " + str(round(100 * perf_opts.epsilon)) + "% of the best."
            )
    else:
        logger.info(f"[{log_prefix}] -> Minimizing performance.")
        y_aux = y.copy()
        y_aux[np.isnan(y_aux)] = np.inf

        y_best = np.min(y_aux, axis=1)
        # Snapshot before the zero-value eps substitution below, so tied
        # instances whose best performance is exactly zero can still be
        # detected as ties (matches MATLAB's YbestTie).
        y_best_tie = y_best.copy()
        # add 1 to the index to match the MATLAB code
        p = np.argmin(y_aux, axis=1) + 1

        if perf_opts.abs_perf:
            y_bin = y_aux <= perf_opts.epsilon
            msg = msg + "less than " + str(perf_opts.epsilon)
        else:
            _log_many_zero_best_warning(y_best, log_prefix)
            y_best[y_best == 0] = np.finfo(float).eps
            y[y == 0] = np.finfo(float).eps
            y = y / y_best[:, np.newaxis] - 1
            y_bin = (y_aux / y_best[:, np.newaxis] - 1) <= perf_opts.epsilon
            msg = (
                msg + "within " + str(round(100 * perf_opts.epsilon)) + "% of the best."
            )

    logger.info(f"[{log_prefix}] {msg}")

    best_algos = np.equal(y_raw, y_best_tie[:, np.newaxis])
    multiple_best_algos = np.sum(best_algos, axis=1) > 1
    aidx = np.arange(1, nalgos + 1)
    rng = np.random.RandomState(general_opts.seed)
    for i in range(y_raw.shape[0]):
        if multiple_best_algos[i]:
            aux = aidx[best_algos[i]]
            # MATLAB randi(n) is floor(rand*n)+1. RandomState uses the same
            # legacy twister and 53-bit double conversion as MATLAB.
            choice = int(np.floor(rng.random_sample() * aux.size))
            p[i] = aux[choice]

    logger.info(
        f"[{log_prefix}] -> For {round(100 * np.mean(multiple_best_algos))}% of the "
        "instances there is more than one best algorithm.",
    )
    logger.info(f"[{log_prefix}] Random selection is used to break ties.")

    num_good_algos = np.sum(y_bin, axis=1)
    if general_opts.verbose:
        logger.debug(f"[{log_prefix}] beta_threshold: {perf_opts.beta_threshold}")
        logger.debug(f"[{log_prefix}] nalgos: {nalgos}")
        logger.debug(f"[{log_prefix}] num_good_algos: {num_good_algos}")

    beta = num_good_algos > (perf_opts.beta_threshold * nalgos)

    return BinaryPerformance(
        y=y,
        y_bin=y_bin,
        y_best=y_best,
        p=p,
        num_good_algos=num_good_algos,
        beta=beta,
    )


def apply_bound_clip(
    x: NDArray[np.double],
    hi_bound: NDArray[np.double],
    lo_bound: NDArray[np.double],
) -> NDArray[np.double]:
    """Clip each feature column of `x` to `[lo_bound, hi_bound]`.

    Equivalent to (and verified bit-for-bit identical, including NaN
    handling, against) the mask-multiply arithmetic this replaces -
    `np.clip` simply expresses the same operation without hand-writing it
    twice. Shared by `PrelimStage._bound()` (training, clipping to bounds
    just fit from this same data) and `InstanceSpace._explore_prelim()`
    (test data, clipping to the trained model's stored bounds) - one
    implementation, not a second one written to match it by hand.
    """
    return np.clip(x, lo_bound, hi_bound)


def apply_boxcox_zscore(
    x: NDArray[np.double],
    lambda_: float,
    mu: float,
    sigma: float,
) -> NDArray[np.double]:
    """Apply a Box-Cox transform at a known `lambda`, then z-score at known mu/sigma.

    Verified bit-for-bit identical to `scipy.stats.zscore(stats.boxcox(x,
    lambda_), ddof=1)` when `mu`/`sigma` are themselves `x`'s own
    box-cox'd mean/std (ddof=1) - i.e. this is a drop-in replacement for
    recomputing the z-score from scratch, not a new formula. Shared by
    `PrelimStage._normalise()` (training, applying the lambda/mu/sigma it
    just fit from this same column) and `InstanceSpace._explore_prelim()`
    (test data, applying the trained model's stored lambda/mu/sigma) - one
    implementation, not a second one written to match it by hand. `x` must
    already be positive (the min-shift-by-1 step happens before this call).
    """
    transformed = np.asarray(stats.boxcox(x, lambda_), dtype=np.double)
    return np.asarray((transformed - mu) / sigma, dtype=np.double)


class PrelimInput(NamedTuple):
    """Inputs for the Prelim stage.

    Attributes
    ----------
    x : NDArray[np.double]
        Feature matrix where each row represents an instance, and each column represents
        a feature.
    y : NDArray[np.double]
        Performance matrix of algorithms, with rows as instances and columns as
        algorithms.
    x_raw : NDArray[np.double]
        Unprocessed feature matrix, containing raw values of each instance-feature pair.
    y_raw : NDArray[np.double]
        Unprocessed performance matrix, containing raw performance values for
        each instance-algorithm pair.
    s : pd.Series | None
        Optional series for additional selection during processing, if available.
    inst_labels : pd.Series
        Labels for each instance in the dataset, used for identification.
    prelim_options : PrelimOptions
        Configuration options specific to the Prelim stage.
    selvars_options : SelvarsOptions
        Options for selecting variables within the Prelim stage, affecting criteria
        and file indices.
    general_options : GeneralOptions
        General options (e.g. the RNG seed), not specific to any one stage.
    """

    x: NDArray[np.double]
    y: NDArray[np.double]
    x_raw: NDArray[np.double]
    y_raw: NDArray[np.double]
    s: pd.Series | None  # type: ignore[type-arg]
    inst_labels: pd.Series  # type: ignore[type-arg]
    prelim_options: PrelimOptions
    selvars_options: SelvarsOptions
    general_options: GeneralOptions


class PrelimPredictInput(NamedTuple):
    """Inputs for applying fitted PRELIM feature transformations."""

    x: NDArray[np.double]
    auto_preproc: bool
    bound_enabled: bool
    norm_enabled: bool


# needs to be changes to output including prelim output, and data changed by stage
class PrelimOutput(NamedTuple):
    """Outputs for the Prelim stage.

    Attributes
    ----------
    med_val : NDArray[np.double]
        Median values of each feature across instances in the processed data.
    iq_range : NDArray[np.double]
        Interquartile range of each feature, representing the spread of data between
         the 25th and 75th percentiles.
    hi_bound : NDArray[np.double]
        Upper bound values for each feature based on specified statistical measures.
    lo_bound : NDArray[np.double]
        Lower bound values for each feature based on specified statistical measures.
    min_x : NDArray[np.double]
        Minimum values for each feature in the raw feature matrix.
    lambda_x : NDArray[np.double]
        Box-Cox transformation parameters for each feature, if applicable.
    mu_x : NDArray[np.double]
        Mean values of each feature across instances in the processed data.
    sigma_x : NDArray[np.double]
        Standard deviation of each feature across instances in the processed data.
    min_y : float
        Minimum value observed in the raw performance data.
    lambda_y : NDArray[np.double]
        Box-Cox transformation parameters for the performance matrix, if applicable.
    sigma_y : NDArray[np.double]
        Standard deviation of performance values across instances.
    mu_y : NDArray[np.double]
        Mean values of performance across instances for each algorithm.
    x : NDArray[np.double]
        Processed feature matrix, where each row represents an instance and each column
         represents a feature.
    y : NDArray[np.double]
        Processed performance matrix, containing performance values for each
         instance-algorithm pair.
    x_raw : NDArray[np.double]
        Original, unprocessed feature matrix containing raw values of each
         instance-feature pair.
    y_raw : NDArray[np.double]
        Original, unprocessed performance matrix containing raw values for each
          instance-algorithm pair.
    y_bin : NDArray[np.bool_]
        Binary matrix indicating instances with good algorithm performance
          (True if performance is good).
    y_best : NDArray[np.double]
        Best observed performance value for each instance across all algorithms.
    p : NDArray[np.int_]
        Index of the best-performing algorithm for each instance (1-based,
          matching MATLAB).
    num_good_algos : NDArray[np.double]
        Number of algorithms per feature that meet a certain performance threshold.
    beta : NDArray[np.bool_]
        Binary array indicating selected features based on certain criteria
          (True if selected).
    instlabels : pd.Series | None
        Labels for each instance in the dataset, if provided.
    data_dense : DataDense | None
        Dense data representation, if available, containing compressed or alternative
          feature representations.
    s : pd.Series | None
        Optional series used for additional selection or processing criteria,
          if available.
    """

    med_val: NDArray[np.double]
    iq_range: NDArray[np.double]
    hi_bound: NDArray[np.double]
    lo_bound: NDArray[np.double]
    min_x: NDArray[np.double]
    lambda_x: NDArray[np.double]
    mu_x: NDArray[np.double]
    sigma_x: NDArray[np.double]
    min_y: float
    lambda_y: NDArray[np.double]
    sigma_y: NDArray[np.double]
    mu_y: NDArray[np.double]
    x: NDArray[np.double]
    y: NDArray[np.double]
    x_raw: NDArray[np.double]
    y_raw: NDArray[np.double]
    y_bin: NDArray[np.bool_]
    y_best: NDArray[np.double]
    p: NDArray[np.int_]
    num_good_algos: NDArray[np.double]
    beta: NDArray[np.bool_]
    instlabels: pd.Series | None  # type: ignore[type-arg]
    data_dense: DataDense | None
    s: pd.Series | None  # type: ignore[type-arg]


@dataclass(frozen=True)
class _BoundOut:
    x: NDArray[np.double]
    med_val: NDArray[np.double]
    iq_range: NDArray[np.double]
    hi_bound: NDArray[np.double]
    lo_bound: NDArray[np.double]


@dataclass(frozen=True)
class _NormaliseOut:
    x: NDArray[np.double]
    min_x: NDArray[np.double]
    lambda_x: NDArray[np.double]
    mu_x: NDArray[np.double]
    sigma_x: NDArray[np.double]
    y: NDArray[np.double]
    min_y: float
    lambda_y: NDArray[np.double]
    sigma_y: NDArray[np.double]
    mu_y: NDArray[np.double]


class PrelimStage(
    Stage[PrelimInput, PrelimOutput],
    PredictiveStage[PrelimPredictInput, PrelimOut, NDArray[np.double]],
):
    """See file docstring."""

    # need to add variables for data changed by stage as null initially
    def __init__(
        self,
        x: NDArray[np.double],
        y: NDArray[np.double],
        x_raw: NDArray[np.double],
        y_raw: NDArray[np.double],
        s: pd.Series | None,  # type: ignore[type-arg]
        inst_labels: pd.Series,  # type: ignore[type-arg]
        prelim_opts: PrelimOptions,
        selvars_opts: SelvarsOptions,
        general_opts: GeneralOptions,
    ) -> None:
        """See file docstring."""
        self.x = x
        self.y = y
        self.prelim_opts = prelim_opts
        self.selvars_opts = selvars_opts
        self.general_opts = general_opts
        self.x_raw = x_raw
        self.y_raw = y_raw
        self.s = s
        self.inst_labels = inst_labels

    def _log(self, msg: str) -> None:
        """Log a top-level, always-shown stage message."""
        logger.info(f"[PRELIM] {msg}")

    def _log_detail(self, msg: str) -> None:
        """Log per-trial/per-iteration detail, only shown when general.verbose."""
        if self.general_opts.verbose:
            logger.debug(f"[PRELIM] {msg}")

    def _warn_many_zero_best(self, y_best: NDArray[np.double]) -> None:
        """Warn if too many instances have a best-algorithm performance of zero.

        Matches MATLAB's ISA:PRELIM:manyZeroBest. Delegates to the shared
        `_log_many_zero_best_warning()` also used by `compute_binary_performance()`.
        """
        _log_many_zero_best_warning(y_best, "PRELIM")

    @staticmethod
    def _inputs() -> type[PrelimInput]:
        return PrelimInput

    @staticmethod
    def _outputs() -> type[PrelimOutput]:
        return PrelimOutput

    @staticmethod
    def predict(
        inputs: PrelimPredictInput,
        fitted: PrelimOut,
    ) -> NDArray[np.double]:
        """Apply fitted bounds and normalisation without re-fitting PRELIM."""
        x = inputs.x
        bound = inputs.auto_preproc and inputs.bound_enabled
        norm = inputs.auto_preproc and inputs.norm_enabled

        if bound:
            clipped = np.any(
                (x < fitted.lo_bound) | (x > fitted.hi_bound),
                axis=1,
            )
            frac_clipped = np.mean(clipped)
            if frac_clipped > _OOD_CLIPPED_FRACTION_THRESHOLD:
                logger.warning(
                    f"explore(): {frac_clipped:.1%} of test instances have at least "
                    "one feature outside the training bounds and were clipped to "
                    "them. This suggests the test set may not be well represented "
                    "by the trained instance space; consider retraining with a "
                    "combined dataset.",
                )
            x = apply_bound_clip(x, fitted.hi_bound, fitted.lo_bound)

        if not norm:
            return x

        x_transformed = x.copy()
        for i in range(x.shape[1]):
            x_transformed[:, i] = x_transformed[:, i] - fitted.min_x[i] + 1
            # MATLAB InstanceSpace.evaluateTestSet clamps shifted explore
            # values before Box-Cox even when bound.flag is disabled. This
            # protects the fitted transform's positive domain without
            # pretending the raw observation was inside its training bounds.
            finite_below_one = np.isfinite(x_transformed[:, i]) & (
                x_transformed[:, i] < 1
            )
            x_transformed[finite_below_one, i] = 1

            idx_valid = ~np.isnan(x_transformed[:, i])
            if np.any(idx_valid):
                x_transformed[idx_valid, i] = apply_boxcox_zscore(
                    x_transformed[idx_valid, i],
                    fitted.lambda_x[i],
                    fitted.mu_x[i],
                    fitted.sigma_x[i],
                )

        return x_transformed

    # will run prelim, filter_post_prelim, return prelim output and data changed by
    # stage
    @staticmethod
    def _run(inputs: PrelimInput) -> PrelimOutput:
        """See file docstring."""
        (
            x,
            y,
            y_bin,
            y_best,
            p,
            num_good_algos,
            beta,
            med_val,
            iq_range,
            hi_bound,
            lo_bound,
            min_x,
            lambda_x,
            mu_x,
            sigma_x,
            min_y,
            lambda_y,
            sigma_y,
            mu_y,
        ) = PrelimStage.prelim(
            inputs.x,
            inputs.y,
            inputs.x_raw,
            inputs.y_raw,
            inputs.s,
            inputs.inst_labels,
            inputs.prelim_options,
            inputs.selvars_options,
            inputs.general_options,
        )

        prelim = PrelimStage(
            x,
            y,
            inputs.x_raw,
            inputs.y_raw,
            inputs.s,
            inputs.inst_labels,
            inputs.prelim_options,
            inputs.selvars_options,
            inputs.general_options,
        )

        (
            subset_index,
            x,
            y,
            x_raw,
            y_raw,
            y_bin,
            beta,
            num_good_algos,
            y_best,
            p,
            inst_labels,
            s,
            data_dense,
        ) = prelim._filter(  # noqa: SLF001
            inputs.inst_labels,
            x,
            y,
            y_bin,
            y_best,
            inputs.x_raw,
            inputs.y_raw,
            p,
            num_good_algos,
            beta,
            inputs.s,
            inputs.selvars_options,
        )

        return PrelimOutput(
            med_val,
            iq_range,
            hi_bound,
            lo_bound,
            min_x,
            lambda_x,
            mu_x,
            sigma_x,
            min_y,
            lambda_y,
            sigma_y,
            mu_y,
            x,
            y,
            x_raw,
            y_raw,
            y_bin,
            y_best,
            p,
            num_good_algos,
            beta,
            inst_labels,
            data_dense,
            s,
        )

    # prelim matlab file implementation, will return only prelim output
    @staticmethod
    def prelim(
        x: NDArray[np.double],
        y: NDArray[np.double],
        x_raw: NDArray[np.double],
        y_raw: NDArray[np.double],
        s: pd.Series | None,  # type: ignore[type-arg]
        inst_labels: pd.Series,  # type: ignore[type-arg]
        prelim_opts: PrelimOptions,
        selvars_opts: SelvarsOptions,
        general_opts: GeneralOptions,
    ) -> tuple[
        NDArray[np.double],  # PrelimDataChanged.x
        NDArray[np.double],  # PrelimDataChanged.y
        NDArray[np.bool_],  # PrelimDataChanged.y_bin
        NDArray[np.double],  # PrelimDataChanged.y_best
        NDArray[np.int_],  # PrelimDataChanged.p
        NDArray[np.double],  # PrelimDataChanged.num_good_algos
        NDArray[np.bool_],  # PrelimDataChanged.beta
        NDArray[np.double],  # PrelimOut.med_val
        NDArray[np.double],  # PrelimOut.iq_range
        NDArray[np.double],  # PrelimOut.hi_bound
        NDArray[np.double],  # PrelimOut.lo_bound
        NDArray[np.double],  # PrelimOut.min_x
        NDArray[np.double],  # PrelimOut.lambda_x
        NDArray[np.double],  # PrelimOut.mu_x
        NDArray[np.double],  # PrelimOut.sigma_x
        float,  # PrelimOut.min_y
        NDArray[np.double],  # PrelimOut.lambda_y
        NDArray[np.double],  # PrelimOut.sigma_y
        NDArray[np.double],  # PrelimOut.mu_y
    ]:
        """Perform preliminary processing on the input data 'x' and 'y'.

        Args
            x: The feature matrix (instances x features) to process.
            y: The performance matrix (instances x algorithms) to
                process.
            prelim_opts: An object of type PrelimOptions containing options for
                processing.

        Returns
        -------
            A tuple containing the processed data (as 'Data' object) and
            preliminary output information (as 'PrelimOut' object).
        """
        prelim_stage = PrelimStage(
            x,
            y,
            x_raw,
            y_raw,
            s,
            inst_labels,
            prelim_opts,
            selvars_opts,
            general_opts,
        )

        return prelim_stage._prelim(  # noqa: SLF001
            x,
            y,
            prelim_opts,
        )

    def _bound(self) -> _BoundOut:
        """Remove extreme outliers from the feature values.

        Returns
        -------
            x: The feature matrix with extreme outliers removed.
            med_val: The median value of the feature matrix.
            iq_range: The interquartile range of the feature matrix.
            hi_bound: The upper bound for the feature values.
            lo_bound: The lower bound for the feature values.
        """
        self._log("-> Removing extreme outliers from the feature values.")
        med_val = np.nanmedian(self.x, axis=0)

        iq_range = stats.iqr(
            self.x,
            axis=0,
            interpolation="midpoint",
            nan_policy="omit",
        )

        multiplier = self.prelim_opts.iqr_multiplier
        hi_bound = med_val + multiplier * iq_range
        lo_bound = med_val - multiplier * iq_range

        self.x = apply_bound_clip(self.x, hi_bound, lo_bound)

        return _BoundOut(
            x=self.x,
            med_val=med_val,
            iq_range=iq_range,
            hi_bound=hi_bound,
            lo_bound=lo_bound,
        )

    def _normalise(self) -> _NormaliseOut:
        """Normalize the data using Box-Cox and Z transformations.

        Returns
        -------
            x: The normalized feature matrix.
            min_x: The minimum value of the feature matrix.
            lambda_x: The lambda values for the Box-Cox transformation of the
                      feature matrix.
            mu_x: The mean of the feature matrix.
            sigma_x: The standard deviation of the feature matrix.
            y: The normalized performance matrix.
            min_y: The minimum value of the performance matrix.
            lambda_y: The lambda values for the Box-Cox transformation of the
                      performance matrix.
            sigma_y: The standard deviation of the performance matrix.
            mu_y: The mean of the performance matrix.
        """
        self._log("-> Auto-normalizing the data using Box-Cox and Z transformations.")

        def boxcox_fmin(
            data: NDArray[np.double],
            lmbda_init: float = 0,
        ) -> tuple[NDArray[np.double], float]:
            """Perform Box-Cox transformation on data using fmin to optimize lambda.

            Args
            ----
                data (ArrayLike): The input data array which must contain only
                                 positive values.
                lmbda_init (float): Initial guess for the lambda parameter.

            Returns
            -------
                tuple[np.ndarray, float]: A tuple containing the transformed data
                                        and the optimal
                lambda value.

            """

            def neg_log_likelihood(lmbda: NDArray[np.double]) -> float:
                """Calculate the negative log-likelihood for the Box-Cox transformation.

                Args
                ----
                    lmbda: The lambda value for the Box-Cox transformation.

                Returns
                -------
                    Any: The negative log-likelihood value.
                """
                result = stats.boxcox_llf(lmbda, data)
                if isinstance(result, list | np.ndarray):
                    return -float(result[0])
                return -float(result)

            # Find the lambda that minimizes the negative log-likelihood
            # We minimize the negative log-likelihood because fmin performs minimization
            optimal_lambda = optimize.fmin(neg_log_likelihood, lmbda_init, disp=False)

            # Use the optimal lambda to perform the Box-Cox transformation
            transformed_data = stats.boxcox(data, optimal_lambda)

            return transformed_data, optimal_lambda[0]

        nfeats = self.x.shape[1]
        nalgos = self.y.shape[1]

        # nanmin (not min): a column with even one NaN must not have every
        # other, otherwise-valid entry in that column turned into NaN by the
        # shift below (plain `min` propagates NaN as the column minimum,
        # which then poisons `x - min_x` for every row) - matches MATLAB's
        # own `min(X, [], 1, 'omitnan')` and the unconditional nanmin already
        # used elsewhere in `_prelim()` for this same statistic.
        min_x = np.nanmin(self.x, axis=0)
        self.x = self.x - min_x + 1
        lambda_x = np.zeros(nfeats)
        mu_x = np.zeros(nfeats)
        sigma_x = np.zeros(nfeats)

        for i in range(nfeats):
            aux = self.x[:, i]
            idx = np.isnan(aux)
            valid = aux[~idx]
            fit_transformed, lambda_x[i] = boxcox_fmin(valid)
            mu_x[i] = np.mean(fit_transformed)
            sigma_x[i] = np.std(fit_transformed, ddof=1)
            self.x[~idx, i] = apply_boxcox_zscore(
                valid,
                lambda_x[i],
                mu_x[i],
                sigma_x[i],
            )

        min_y = float(np.nanmin(self.y))

        self.y = (self.y - min_y) + np.finfo(float).eps

        lambda_y = np.zeros(nalgos)
        mu_y = np.zeros(nalgos)
        sigma_y = np.zeros(nalgos)

        for i in range(nalgos):
            aux = self.y[:, i]
            idx = np.isnan(aux)
            aux, lambda_y[i] = boxcox_fmin(aux[~idx])
            mu_y[i] = np.mean(aux)
            sigma_y[i] = np.std(aux, ddof=1)
            aux = stats.zscore(aux, ddof=1)
            self.y[~idx, i] = aux

        return _NormaliseOut(
            x=self.x,
            min_x=min_x,
            lambda_x=lambda_x,
            mu_x=mu_x,
            sigma_x=sigma_x,
            y=self.y,
            min_y=min_y,
            lambda_y=lambda_y,
            sigma_y=sigma_y,
            mu_y=mu_y,
        )

    # prelim matlab file implementation, will return only prelim output
    def _prelim(
        self,
        x: NDArray[np.double],
        y: NDArray[np.double],
        prelim_opts: PrelimOptions,
    ) -> tuple[
        NDArray[np.double],  # PrelimDataChanged.x
        NDArray[np.double],  # PrelimDataChanged.y
        NDArray[np.bool_],  # PrelimDataChanged.y_bin
        NDArray[np.double],  # PrelimDataChanged.y_best
        NDArray[np.int_],  # PrelimDataChanged.p
        NDArray[np.double],  # PrelimDataChanged.num_good_algos
        NDArray[np.bool_],  # PrelimDataChanged.beta
        NDArray[np.double],  # PrelimOut.med_val
        NDArray[np.double],  # PrelimOut.iq_range
        NDArray[np.double],  # PrelimOut.hi_bound
        NDArray[np.double],  # PrelimOut.lo_bound
        NDArray[np.double],  # PrelimOut.min_x
        NDArray[np.double],  # PrelimOut.lambda_x
        NDArray[np.double],  # PrelimOut.mu_x
        NDArray[np.double],  # PrelimOut.sigma_x
        float,  # PrelimOut.min_y
        NDArray[np.double],  # PrelimOut.lambda_y
        NDArray[np.double],  # PrelimOut.sigma_y
        NDArray[np.double],  # PrelimOut.mu_y
    ]:
        y_raw = y.copy()

        perf = compute_binary_performance(
            y_raw,
            PerformanceOptions(
                max_perf=prelim_opts.max_perf,
                abs_perf=prelim_opts.abs_perf,
                epsilon=prelim_opts.epsilon,
                beta_threshold=prelim_opts.beta_threshold,
            ),
            self.general_opts,
            log_prefix="PRELIM",
        )
        y, y_bin, y_best, p, num_good_algos, beta = (
            perf.y,
            perf.y_bin,
            perf.y_best,
            perf.p,
            perf.num_good_algos,
            perf.beta,
        )

        med_val = np.nanmedian(x, axis=0)
        iq_range = stats.iqr(x, axis=0, interpolation="midpoint", nan_policy="omit")
        multiplier = prelim_opts.iqr_multiplier
        hi_bound = med_val + multiplier * iq_range
        lo_bound = med_val - multiplier * iq_range

        if prelim_opts.preproc and prelim_opts.bound:
            bound_out = self._bound()
            x = bound_out.x
            med_val = bound_out.med_val
            iq_range = bound_out.iq_range
            hi_bound = bound_out.hi_bound
            lo_bound = bound_out.lo_bound

        nfeats = x.shape[1]
        nalgos = y.shape[1]
        min_x = np.nanmin(x, axis=0)
        lambda_x = np.zeros(nfeats)
        mu_x = np.zeros(nfeats)
        sigma_x = np.zeros(nfeats)
        min_y = float(np.nanmin(y))
        lambda_y = np.zeros(nalgos)
        mu_y = np.zeros(nalgos)
        sigma_y = np.zeros(nalgos)

        if prelim_opts.preproc and prelim_opts.norm:
            # ``compute_binary_performance`` may replace raw Y with relative
            # performance. Normalise that transformed matrix, not the raw Y
            # retained by the stage constructor.
            self.x = x
            self.y = y
            normalise_out = self._normalise()
            x = normalise_out.x
            min_x = normalise_out.min_x
            lambda_x = normalise_out.lambda_x
            mu_x = normalise_out.mu_x
            sigma_x = normalise_out.sigma_x
            y = normalise_out.y
            min_y = normalise_out.min_y
            lambda_y = normalise_out.lambda_y
            sigma_y = normalise_out.sigma_y
            mu_y = normalise_out.mu_y

        return (
            x,
            y,
            y_bin,
            y_best,
            p,
            num_good_algos,
            beta,
            med_val,
            iq_range,
            hi_bound,
            lo_bound,
            min_x,
            lambda_x,
            mu_x,
            sigma_x,
            min_y,
            lambda_y,
            sigma_y,
            mu_y,
        )

    def _filter(
        self,
        inst_labels: pd.Series,  # type: ignore[type-arg]
        x: NDArray[np.double],
        y: NDArray[np.double],
        y_bin: NDArray[np.bool_],
        y_best: NDArray[np.double],
        x_raw: NDArray[np.double],
        y_raw: NDArray[np.double],
        p: NDArray[np.int_],
        num_good_algos: NDArray[np.double],
        beta: NDArray[np.bool_],
        s: pd.Series | None,  # type: ignore[type-arg]
        selvars_opts: SelvarsOptions,
    ) -> tuple[  # type: ignore[type-arg]
        NDArray[np.bool_],  # subset_index
        NDArray[np.double],  # x
        NDArray[np.double],  # y
        NDArray[np.double],  # x_raw
        NDArray[np.double],  # y_raw
        NDArray[np.bool_],  # y_bin
        NDArray[np.bool_],  # beta
        NDArray[np.double],  # num_good_algos
        NDArray[np.double],  # y_best
        NDArray[np.int_],  # p
        pd.Series,  # inst_labels
        pd.Series | None,  # s
        DataDense | None,  # data_dense
    ]:
        data_dense = None
        # If we are only meant to take some observations
        self._log("-------------------------------------------------------------------")
        ninst = x.shape[0]
        fractional = selvars_opts.small_scale_flag and isinstance(
            selvars_opts.small_scale,
            float,
        )

        path = Path(selvars_opts.file_idx)
        self._log_detail(f"path: {path}")
        self._log_detail(f"path.is_file(file_idx): {path.is_file()}")
        if selvars_opts.file_idx_flag and not path.is_file():
            msg = f"Subset index file does not exist: {path}"
            raise FileNotFoundError(msg)
        fileindexed = selvars_opts.file_idx_flag

        bydensity = (
            selvars_opts.density_flag
            and isinstance(selvars_opts.min_distance, float)
            and isinstance(selvars_opts.selvars_type, str)
        )

        if fractional:
            self._log(
                f"-> Creating a small scale experiment for validation. \
                Percentage of subset: \
                {round(100 * selvars_opts.small_scale, 2)}%",
            )
            _, subset_idx = train_test_split(
                np.arange(ninst),
                test_size=selvars_opts.small_scale,
                random_state=self.general_opts.seed,
            )
            subset_index = np.zeros(ninst, dtype=bool)
            subset_index[subset_idx] = True

        elif fileindexed:
            self._log("-> Using a subset of instances.")
            subset_index = np.zeros(ninst, dtype=bool)
            loaded = np.genfromtxt(path, delimiter=",", dtype=float)
            indices = np.atleast_1d(loaded).ravel()
            self._log_detail(f"indices (1-based): {indices}")

            if indices.size == 0 or not np.all(np.isfinite(indices)):
                msg = f"Subset index file must contain finite 1-based indices: {path}"
                raise ValueError(msg)
            if not np.all(indices == np.floor(indices)):
                msg = f"Subset index file must contain integer indices: {path}"
                raise ValueError(msg)

            indices_1_based = indices.astype(np.int_)
            invalid = (indices_1_based < 1) | (indices_1_based > ninst)
            if np.any(invalid):
                bad = indices_1_based[invalid].tolist()
                msg = (
                    f"Subset indices must be in MATLAB's 1-based range "
                    f"[1, {ninst}]; got {bad}."
                )
                raise ValueError(msg)

            subset_index[indices_1_based - 1] = True

        elif bydensity:
            self._log(
                "-> Creating a small scale experiment for validation based on density.",
            )
            subset_index, _, _, _ = do_filter(
                x,
                y,
                y_bin,
                selvars_opts.selvars_type,
                selvars_opts.min_distance,
            )
            subset_index = ~subset_index
            self._log(
                f"-> Percentage of instances retained: \
                {round(100 * np.mean(subset_index), 2)}%",
            )
        else:
            self._log("-> Using the complete set of the instances.")
            subset_index = np.ones(ninst, dtype=bool)

        if fileindexed or fractional or bydensity:
            if bydensity:
                data_dense = DataDense(
                    x=x,
                    y=y,
                    x_raw=x_raw,
                    y_raw=y_raw,
                    y_bin=y_bin,
                    y_best=y_best,
                    p=p,
                    num_good_algos=num_good_algos,
                    beta=beta,
                    inst_labels=inst_labels,
                    s=s,
                )

            x = x[subset_index, :]
            y = y[subset_index, :]
            x_raw = x_raw[subset_index, :]
            y_raw = y_raw[subset_index, :]
            y_bin = y_bin[subset_index, :]
            beta = beta[subset_index]
            num_good_algos = num_good_algos[subset_index]
            y_best = y_best[subset_index]
            p = p[subset_index]
            inst_labels = inst_labels[subset_index]

            if s is not None:
                s = s[subset_index]

        return (
            subset_index,
            x,
            y,
            x_raw,
            y_raw,
            y_bin,
            beta,
            num_good_algos,
            y_best,
            p,
            inst_labels,
            s,
            data_dense,
        )
