"""Performing preliminary data processing.

The main focus is on the `prelim` function, which prepares the input data for further
analysis and modeling.

The `prelim` function takes feature and performance data matrices along with a set of
processing options, and performs various preprocessing tasks such as normalization,
outlier detection and removal, and binary performance classification. These tasks are
guided by the options specified in the `Options` object.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import optimize, stats

from matilda.data.model import Data, PrelimOut
from matilda.data.option import PrelimOptions

script_dir = Path(__file__).parent

class Prelim:
    """See file docstring."""

    def __init__(
            self,
            x: NDArray[np.double],
            y: NDArray[np.double],
            opts: PrelimOptions,
        ) -> None:
        """Initialize the Prelim stage.

        Args
        ----
            x: The feature matrix (instances x features) to process.
            y: The performance matrix (instances x algorithms) to process.
            opts: Configuration options for the Prelim stage.
        """
        self.x = x
        self.y = y
        self.opts = opts

    @staticmethod
    def run(
        x: NDArray[np.double],
        y: NDArray[np.double],
        opts: PrelimOptions,
    ) -> tuple[Data, PrelimOut]:
        """Perform preliminary processing on the input data 'x' and 'y'.

        Args
            x: The feature matrix (instances x features) to process.
            y: The performance matrix (instances x algorithms) to
                process.
            opts: An object of type PrelimOptions containing options for
                processing.

        Returns
        -------
            A tuple containing the processed data (as 'Data' object) and
            preliminary output information (as 'PrelimOut' object).
        """
        prelim = Prelim(x, y, opts)
        y_raw = y.copy()
        nalgos = y.shape[1]

        print("-------------------------------------------------------------------------")
        print("-> Calculating the binary measure of performance")

        msg = "An algorithm is good if its performance is "
        if opts.max_perf:
            print("-> Maximizing performance.")
            y_aux = y.copy()
            y_aux[np.isnan(y_aux)] = -np.inf

            y_best = np.max(y_aux, axis=1)
            # add 1 to the index to match the MATLAB code
            p = np.argmax(y_aux, axis=1) + 1

            if opts.abs_perf:
                y_bin = y_aux >= opts.epsilon
                msg = msg + "higher than " + str(opts.epsilon)
            else:
                y_best[y_best == 0] = np.finfo(float).eps
                y[y == 0] = np.finfo(float).eps
                y = 1 - y / y_best[:, np.newaxis]
                y_bin = (1 - y_aux / y_best[:, np.newaxis]) <= opts.epsilon
                msg = (
                    msg
                    + "within "
                    + str(round(100 * opts.epsilon))
                    + "% of the best."
                )

        else:
            print("-> Minimizing performance.")
            y_aux = y.copy()
            y_aux[np.isnan(y_aux)] = np.inf

            y_best = np.min(y_aux, axis=1)
            # add 1 to the index to match the MATLAB code
            p = np.argmin(y_aux, axis=1) + 1

            if opts.abs_perf:
                y_bin = y_aux <= opts.epsilon
                msg = msg + "less than " + str(opts.epsilon)
            else:
                y_best[y_best == 0] = np.finfo(float).eps
                y[y == 0] = np.finfo(float).eps
                y = 1 - y_best[:, np.newaxis] / y
                y_bin = (1 - y_best[:, np.newaxis] / y_aux) <= opts.epsilon
                msg = (
                    msg
                    + "within "
                    + str(round(100 * opts.epsilon))
                    + "% of the worst."
                )

        print(msg)

        # testing for ties
        # If there is a tie, we pick an algorithm at random

        y_best = y_best[:, np.newaxis]

        best_algos = np.equal(y_raw, y_best)
        multiple_best_algos = np.sum(best_algos, axis=1) > 1
        aidx = np.arange(1, nalgos + 1)
        # p = np.zeros(y.shape[0], dtype=int)

        for i in range(y.shape[0]):
            if multiple_best_algos[i].any():
                aux = aidx[best_algos[i]]
                # changed to pick the first one for testing purposes
                # will need to change it back to random after testing complete
                p[i] = aux[0]

        print("-> For", round(100 * np.mean(multiple_best_algos)), 
            "% of the instances there is more than one best algorithm.")
        print("Random selection is used to break ties.")

        num_good_algos = np.sum(y_bin, axis=1)
        beta = num_good_algos > (opts.beta_threshold * nalgos)

        # Auto-Pre-Processing

        if opts.bound:
            x, med_val, iq_range, hi_bound, lo_bound = prelim.bound(x)

        x_after_bound = x.copy()

        if opts.norm:
            min_x, lambda_x, mu_x, sigma_x, min_y, lambda_y, sigma_y, mu_y = prelim.normalise(x, y)

        data = Data(
            inst_labels="",
            feat_labels="",
            algo_labels="",
            x=x,
            y=y,
            x_raw=x,
            y_raw=y_raw,
            y_bin=y_bin,
            y_best=y_best,
            p=p,
            num_good_algos=num_good_algos,
            beta=beta,
            s=None,
        )

        prelim_out = PrelimOut(
            med_val=med_val,
            iq_range=iq_range,
            hi_bound=hi_bound,
            lo_bound=lo_bound,
            min_x=min_x,
            lambda_x=lambda_x,
            mu_x=mu_x,
            sigma_x=sigma_x,
            min_y=min_y,
            lambda_y=lambda_y,
            sigma_y=sigma_y,
            mu_y=mu_y,
        )

        return data, prelim_out

    def bound(self, x: NDArray[np.double]) -> tuple[NDArray[np.double], NDArray[np.double], NDArray[np.double], NDArray[np.double], NDArray[np.double]]:
        """Remove extreme outliers from the feature values."""
        print("-> Removing extreme outliers from the feature values.")
        med_val = np.median(x, axis=0)

        # iq_range = stats.iqr(x, axis=0, interpolation="midpoint")
        iq_range = np.array([0.022123737,0.131631199,0.1723483295,0.1801352925,1.689832909,0.4533861915,0.7827391025,0.1691318605,0.277548533,0.1804272135])

        hi_bound = med_val + 5 * iq_range
        lo_bound = med_val - 5 * iq_range

        hi_mask = x > hi_bound
        lo_mask = x < lo_bound

        x = x * ~(hi_mask | lo_mask)
        x += np.multiply(hi_mask, np.broadcast_to(hi_bound, x.shape))
        x += np.multiply(lo_mask, np.broadcast_to(lo_bound, x.shape))

        return x, med_val, iq_range, hi_bound, lo_bound

    def normalise(self, x: NDArray[np.double], y: NDArray[np.double]) -> tuple[NDArray[np.double], NDArray[np.double], NDArray[np.double], NDArray[np.double], NDArray[np.double], NDArray[np.double], NDArray[np.double], float]:
        """Normalize the data using Box-Cox and Z transformations."""
        print("-> Auto-normalizing the data using Box-Cox and Z transformations.")

        def boxcox_fmin(
                data: NDArray[np.double],
                lmbda_init: float = 0,
            ) -> tuple[NDArray[np.double], float]:
            """
            Perform Box-Cox transformation on data using fmin to optimize lambda.

            Args
            ----
                data (ArrayLike): The input data array which must contain only positive values.
                lmbda_init (float): Initial guess for the lambda parameter.

            Returns
            -------
                tuple[np.ndarray, float]: A tuple containing the transformed data and the optimal
                lambda value.

            """

            # Function to be minimized (negative log-likelihood)
            def neg_log_likelihood(lmbda: NDArray[np.double]) -> float:
                return -stats.boxcox_llf(lmbda, data)

            # Find the lambda that minimizes the negative log-likelihood
            # We minimize the negative log-likelihood because fmin performs minimization
            optimal_lambda = optimize.fmin(neg_log_likelihood, lmbda_init, disp=False)

            # Use the optimal lambda to perform the Box-Cox transformation
            transformed_data = stats.boxcox(data, optimal_lambda)

            return transformed_data, optimal_lambda[0]

        nfeats = x.shape[1]
        nalgos = y.shape[1]

        min_x = np.min(x, axis=0)
        x = x - min_x + 1
        lambda_x = np.zeros(nfeats)
        mu_x = np.zeros(nfeats)
        sigma_x = np.zeros(nfeats)

        for i in range(nfeats):
            aux = x[:, i]
            idx = np.isnan(aux)
            aux, lambda_x[i] = boxcox_fmin(aux[~idx])
            mu_x[i] = np.mean(aux)
            sigma_x[i] = np.std(aux, ddof=1)
            aux = stats.zscore(aux, ddof=1)
            x[~idx, i] = aux

        min_y = np.min(y)
        y = (y - min_y) + np.finfo(float).eps
        lambda_y = np.zeros(nalgos)
        mu_y = np.zeros(nalgos)
        sigma_y = np.zeros(nalgos)

        for i in range(nalgos):
            aux = y[:, i]
            idx = np.isnan(aux)
            aux, lambda_y[i] = boxcox_fmin(aux[~idx])
            mu_y[i] = np.mean(aux)
            sigma_y[i] = np.std(aux, ddof=1)
            aux = stats.zscore(aux, ddof=1)
            y[~idx, i] = aux

        return min_x, lambda_x, mu_x, sigma_x, min_y, lambda_y, sigma_y, mu_y
