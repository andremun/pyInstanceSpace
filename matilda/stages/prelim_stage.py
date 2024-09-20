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
from types import UnionType

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import optimize, stats
from sklearn.model_selection import train_test_split

from matilda.data.options import PrelimOptions, SelvarsOptions
from matilda.stages.stage import Stage
from matilda.utils.filter import filter


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


@dataclass(frozen=True)
class DataDense:
    x_data_dense: NDArray[np.double]
    y_data_dense: NDArray[np.double]
    x_raw_data_dense: NDArray[np.double]
    y_raw_data_dense: NDArray[np.double]
    y_bin_data_dense: NDArray[np.bool_]
    y_best_data_dense: NDArray[np.double]
    p_data_dense: NDArray[np.double]
    num_good_algos_data_dense: NDArray[np.double]
    beta_data_dense: NDArray[np.bool_]
    inst_labels_data_dense: pd.Series
    s_data_dense: pd.Series | None


class PrelimStage(Stage):
    """See file docstring."""

    # need to add variables for data changed by stage as null initially
    def __init__(
        self,
        x: NDArray[np.double],
        y: NDArray[np.double],
        x_raw: NDArray[np.double],
        y_raw: NDArray[np.double],
        s: pd.Series | None,
        inst_labels: pd.Series,
        prelim_opts: PrelimOptions,
        selvars_opts: SelvarsOptions,
    ) -> None:
        """See file docstring."""
        self.x = x
        self.y = y
        self.prelim_opts = prelim_opts
        self.selvars_opts = selvars_opts

    @staticmethod
    def _inputs() -> list[tuple[str, type | UnionType]]:
        """See file docstring."""
        return [
            ("x", NDArray[np.double]),
            ("y", NDArray[np.double]),
            ("x_raw", NDArray[np.double]),
            ("y_raw", NDArray[np.double]),
            ("s", pd.Series | None),
            ("inst_labels", pd.Series | None),
            ("prelim_opts", PrelimOptions),
            ("selvars_opts", SelvarsOptions),
        ]

    # needs to be changes to output including prelim output, and data changed by stage
    @staticmethod
    def _outputs() -> list[tuple[str, type | UnionType]]:
        """See file docstring."""
        return [
            ("med_val", NDArray[np.double]),
            ("iq_range", NDArray[np.double]),
            ("hi_bound", NDArray[np.double]),
            ("lo_bound", NDArray[np.double]),
            ("min_x", NDArray[np.double]),
            ("lambda_x", NDArray[np.double]),
            ("mu_x", NDArray[np.double]),
            ("sigma_x", NDArray[np.double]),
            ("min_y", float),
            ("lambda_y", NDArray[np.double]),
            ("sigma_y", NDArray[np.double]),
            ("mu_y", NDArray[np.double]),
            ("x", NDArray[np.double]),
            ("y", NDArray[np.double]),
            ("x_raw", NDArray[np.double]),
            ("y_raw", NDArray[np.double]),
            ("y_bin", NDArray[np.bool_]),
            ("y_best", NDArray[np.double]),
            ("p", NDArray[np.double]),
            ("num_good_algos", NDArray[np.double]),
            ("beta", NDArray[np.bool_]),
            ("instlabels", pd.Series | None),
            ("data_dense", DataDense | None),
            ("s", pd.Series | None),
        ]

    # will run prelim, filter_post_prelim, return prelim output and data changed by
    # stage
    def _run(
        self,
        x: NDArray[np.double],
        y: NDArray[np.double],
        x_raw: NDArray[np.double],
        y_raw: NDArray[np.double],
        s: pd.Series | None,
        inst_labels: pd.Series,
        prelim_opts: PrelimOptions,
        selvars_opts: SelvarsOptions,
    ) -> tuple[
        NDArray[np.double],  # med_val
        NDArray[np.double],  # iq_range
        NDArray[np.double],  # hi_bound
        NDArray[np.double],  # lo_bound
        NDArray[np.double],  # min_x
        NDArray[np.double],  # lambda_x
        NDArray[np.double],  # mu_x
        NDArray[np.double],  # sigma_x
        float,  # min_y
        NDArray[np.double],  # lambda_y
        NDArray[np.double],  # sigma_y
        NDArray[np.double],  # mu_y
        NDArray[np.double],  # x
        NDArray[np.double],  # y
        NDArray[np.double],  # x_raw
        NDArray[np.double],  # y_raw
        NDArray[np.bool_],  # y_bin
        NDArray[np.double],  # y_best
        NDArray[np.double],  # p
        NDArray[np.double],  # num_good_algos
        NDArray[np.bool_],  # beta
        pd.Series | None,  # inst_labels
        DataDense | None,  # data_dense
        pd.Series | None,  # s
    ]:
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
        ) = self._prelim(
            x,
            y,
            prelim_opts,
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
        ) = self._filter(
            inst_labels,
            x,
            y,
            y_bin,
            y_best,
            x_raw,
            y_raw,
            p,
            num_good_algos,
            beta,
            s,
            selvars_opts,
        )

        return (
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

    def _select_best_algorithms(
        self,
        y_raw: NDArray[np.double],
        y_best: NDArray[np.double],
        y_bin: NDArray[np.bool_],
        nalgos: int,
        beta_threshold: float,
        p: NDArray[np.double],
    ) -> tuple[NDArray[np.double], NDArray[np.double], NDArray[np.bool_]]:
        """Select the best algorithms based on the given criteria.

        Args
        ----
            y_raw: Raw algorithm predictions.
            y_best: Best algorithm predictions.
            y_bin: Binary labels.
            nalgos: Number of algorithms.
            betaThreshold: Beta threshold.
            p: Placeholder for selected algorithms.

        Returns
        -------
            num_good_algos: Number of good algorithms.
            beta: Beta values.
            p: Selected algorithms.
        """
        # testing for ties
        # If there is a tie, we pick an algorithm at random
        y_best = y_best[:, np.newaxis]

        best_algos = np.equal(y_raw, y_best)
        multiple_best_algos = np.sum(best_algos, axis=1) > 1
        aidx = np.arange(1, nalgos + 1)

        for i in range(self.y.shape[0]):
            if multiple_best_algos[i].any():
                aux = aidx[best_algos[i]]
                # changed to pick the first one for testing purposes
                # will need to change it back to random after testing complete
                p[i] = aux[0]

        print(
            "-> For",
            round(100 * np.mean(multiple_best_algos)),
            "% of the instances there is more than one best algorithm.",
        )
        print("Random selection is used to break ties.")

        num_good_algos = np.sum(y_bin, axis=1)
        print("beta_threshold:", beta_threshold)
        print("nalgos:", nalgos)
        print("num_good_algos:", num_good_algos)

        beta = num_good_algos > (beta_threshold * nalgos)

        return num_good_algos, p, beta

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
        print("-> Removing extreme outliers from the feature values.")
        med_val = np.median(self.x, axis=0)

        iq_range = stats.iqr(self.x, axis=0, interpolation="midpoint")

        hi_bound = med_val + 5 * iq_range
        lo_bound = med_val - 5 * iq_range

        hi_mask = self.x > hi_bound
        lo_mask = self.x < lo_bound

        self.x = self.x * ~(hi_mask | lo_mask)
        self.x += np.multiply(hi_mask, np.broadcast_to(hi_bound, self.x.shape))
        self.x += np.multiply(lo_mask, np.broadcast_to(lo_bound, self.x.shape))

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
        print("-> Auto-normalizing the data using Box-Cox and Z transformations.")

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
                return -float(stats.boxcox_llf(lmbda, data)[0])

            # Find the lambda that minimizes the negative log-likelihood
            # We minimize the negative log-likelihood because fmin performs minimization
            optimal_lambda = optimize.fmin(neg_log_likelihood, lmbda_init, disp=False)

            # Use the optimal lambda to perform the Box-Cox transformation
            transformed_data = stats.boxcox(data, optimal_lambda)

            return transformed_data, optimal_lambda[0]

        nfeats = self.x.shape[1]
        nalgos = self.y.shape[1]

        min_x = np.min(self.x, axis=0)
        self.x = self.x - min_x + 1
        lambda_x = np.zeros(nfeats)
        mu_x = np.zeros(nfeats)
        sigma_x = np.zeros(nfeats)

        for i in range(nfeats):
            aux = self.x[:, i]
            idx = np.isnan(aux)
            aux, lambda_x[i] = boxcox_fmin(aux[~idx])
            mu_x[i] = np.mean(aux)
            sigma_x[i] = np.std(aux, ddof=1)
            aux = stats.zscore(aux, ddof=1)
            self.x[~idx, i] = aux

        min_y = float(np.min(self.y))

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
    ) -> tuple[
        NDArray[np.double],  # PrelimDataChanged.x
        NDArray[np.double],  # PrelimDataChanged.y
        NDArray[np.bool_],  # PrelimDataChanged.y_bin
        NDArray[np.double],  # PrelimDataChanged.y_best
        NDArray[np.double],  # PrelimDataChanged.p
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
        )

        return prelim_stage._prelim(  # noqa: SLF001
            x,
            y,
            prelim_opts,
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
        NDArray[np.double],  # PrelimDataChanged.p
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
        nalgos = y.shape[1]

        print(
            "-------------------------------------------------------------------------",
        )
        print("-> Calculating the binary measure of performance")

        msg = "An algorithm is good if its performance is "
        if prelim_opts.max_perf:
            print("-> Maximizing performance.")
            y_aux = y.copy()
            y_aux[np.isnan(y_aux)] = -np.inf

            y_best = np.max(y_aux, axis=1)
            # add 1 to the index to match the MATLAB code
            p = np.argmax(y_aux, axis=1) + 1

            if prelim_opts.abs_perf:
                y_bin = y_aux >= prelim_opts.epsilon
                msg = msg + "higher than " + str(prelim_opts.epsilon)
            else:
                y_best[y_best == 0] = np.finfo(float).eps
                y[y == 0] = np.finfo(float).eps
                y = 1 - y / y_best[:, np.newaxis]
                y_bin = (1 - y_aux / y_best[:, np.newaxis]) <= prelim_opts.epsilon
                msg = (
                    msg
                    + "within "
                    + str(round(100 * prelim_opts.epsilon))
                    + "% of the best."
                )

        else:
            print("-> Minimizing performance.")
            y_aux = y.copy()
            y_aux[np.isnan(y_aux)] = np.inf

            y_best = np.min(y_aux, axis=1)
            # add 1 to the index to match the MATLAB code
            p = np.argmin(y_aux, axis=1) + 1

            if prelim_opts.abs_perf:
                y_bin = y_aux <= prelim_opts.epsilon
                msg = msg + "less than " + str(prelim_opts.epsilon)
            else:
                y_best[y_best == 0] = np.finfo(float).eps
                y[y == 0] = np.finfo(float).eps
                y = 1 - y_best[:, np.newaxis] / y
                y_bin = (1 - y_best[:, np.newaxis] / y_aux) <= prelim_opts.epsilon
                msg = (
                    msg
                    + "within "
                    + str(round(100 * prelim_opts.epsilon))
                    + "% of the worst."
                )

        print(msg)

        num_good_algos, p, beta = self._select_best_algorithms(
            y_raw,
            y_best,
            y_bin,
            nalgos,
            prelim_opts.beta_threshold,
            p,
        )

        if prelim_opts.bound:
            bound_out = self._bound()
            x = bound_out.x
            med_val = bound_out.med_val
            iq_range = bound_out.iq_range
            hi_bound = bound_out.hi_bound
            lo_bound = bound_out.lo_bound

        if prelim_opts.norm:
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
        p: NDArray[np.double],
        num_good_algos: NDArray[np.double],
        beta: NDArray[np.bool_],
        s: pd.Series | None,
        selvars_opts: SelvarsOptions,
    ) -> tuple[
        NDArray[np.bool_],  # subset_index
        NDArray[np.double],  # x
        NDArray[np.double],  # y
        NDArray[np.double],  # x_raw
        NDArray[np.double],  # y_raw
        NDArray[np.bool_],  # y_bin
        NDArray[np.bool_],  # beta
        NDArray[np.double],  # num_good_algos
        NDArray[np.double],  # y_best
        NDArray[np.double],  # p
        pd.Series,  # inst_labels
        pd.Series | None,  # s
        DataDense | None,  # data_dense
    ]:
        data_dense = None
        # If we are only meant to take some observations
        print("-------------------------------------------------------------------")
        ninst = x.shape[0]
        fractional = selvars_opts.small_scale_flag and isinstance(
            selvars_opts.small_scale,
            float,
        )

        path = Path(selvars_opts.file_idx)
        print("path:", path)
        print("path.is_file(file_idx):", path.is_file())
        fileindexed = (
            selvars_opts.file_idx_flag and Path(selvars_opts.file_idx).is_file()
        )

        bydensity = (
            selvars_opts.density_flag
            and isinstance(selvars_opts.min_distance, float)
            and isinstance(selvars_opts.selvars_type, str)
        )

        if fractional:
            print(
                f"-> Creating a small scale experiment for validation. \
                Percentage of subset: \
                {round(100 * selvars_opts.small_scale, 2)}%",
            )
            _, subset_idx = train_test_split(
                np.arange(ninst),
                test_size=selvars_opts.small_scale,
                random_state=0,
            )
            subset_index = np.zeros(ninst, dtype=bool)
            subset_index[subset_idx] = True

        elif fileindexed:
            print("-> Using a subset of instances.")
            subset_index = np.zeros(ninst, dtype=bool)
            aux = np.genfromtxt(selvars_opts.file_idx, delimiter=",", dtype=int)
            print("aux:", aux)
            aux = aux[aux < ninst]
            # for some reason, this makes the indices perform correctly.
            for i in range(len(aux)):
                aux[i] = aux[i] - 1
            subset_index[aux] = True

        elif bydensity:
            print(
                "-> Creating a small scale experiment for validation based on density.",
            )
            subset_index, _, _, _ = filter(
                x,
                y,
                y_bin,
                selvars_opts.selvars_type,
                selvars_opts.min_distance,
            )
            subset_index = ~subset_index
            print(
                f"-> Percentage of instances retained: \
                {round(100 * np.mean(subset_index), 2)}%",
            )
        else:
            print("-> Using the complete set of the instances.")
            subset_index = np.ones(ninst, dtype=bool)

        if fileindexed or fractional or bydensity:
            if bydensity:
                data_dense = DataDense(
                    x_data_dense=x,
                    y_data_dense=y,
                    x_raw_data_dense=x_raw,
                    y_raw_data_dense=y_raw,
                    y_bin_data_dense=y_bin,
                    y_best_data_dense=y_best,
                    p_data_dense=p,
                    num_good_algos_data_dense=num_good_algos,
                    beta_data_dense=beta,
                    inst_labels_data_dense=inst_labels,
                    s_data_dense=s,
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
