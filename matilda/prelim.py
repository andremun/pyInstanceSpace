"""
Performing preliminary data processing.

The main focus is on the `prelim` function, which prepares the input data for further
analysis and modeling.

The `prelim` function takes feature and performance data matrices along with a set of
processing options, and performs various preprocessing tasks such as normalization,
outlier detection and removal, and binary performance classification. These tasks are
guided by the options specified in the `Opts` object.
"""

import csv
from pathlib import Path
import numpy as np
from numpy.typing import NDArray
from scipy import stats

from matilda.data.model import Data, PrelimOut
from matilda.data.option import Opts
import json

def prelim(
    x: NDArray[np.double],
    y: NDArray[np.double],
    abs_perf, beta_threshold, epsilon, max_perf, bound, norm,  # noqa: ANN001
    # opts: Opts,
):
# ) -> tuple[Data, PrelimOut]:
    """
    Perform preliminary processing on the input data 'x' and 'y'.

    :param x: The feature matrix (instances x features) to process.
    :param y: The performance matrix (instances x algorithms) to process.
    :param opts: An object of type Opts containing options for processing.
    :return: A tuple containing the processed data (as 'Data' object) and preliminary
             output information (as 'PrelimOut' object).
    """
    # TODO: Rewrite PRELIM logic in python

    y_raw = y.copy()
    nalgos = y.shape[1]
    out = {}

    print("-------------------------------------------------------------------------")
    print("-> Calculating the binary measure of performance")

    msg = "An algorithm is good if its performance is "
    if max_perf:
        y_aux = y.copy()
        y_aux[np.isnan(y_aux)] = -np.inf

        y_best = np.max(y_aux, axis=1)
        p = np.argmax(y_aux, axis=1)

        if abs_perf:
            y_bin = y_aux >= epsilon
            msg = msg + "higher than " + epsilon
        else:
            y_best[y_best == 0] = np.finfo(float).eps
            y[y == 0] = np.finfo(float).eps
            y = 1 - y / y_best[:, np.newaxis]
            y_bin = (1 - y_aux / y_best[:, np.newaxis]) <= epsilon
            msg = (
                msg
                + "within "
                + str(round(100 * epsilon))
                + "% of the best."
            )

    else:
        y_aux = y.copy()
        y_aux[np.isnan(y_aux)] = np.inf
        y_best = np.min(y_aux, axis=1)

        if abs_perf:
            y_bin = y_aux <= epsilon
            # msg = msg + "less than " + epsilon
        else:
            y_best[y_best == 0] = np.finfo(float).eps
            y[y == 0] = np.finfo(float).eps
            y = 1 - y_best[:, np.newaxis] / y
            y_bin = (1 - y_best[:, np.newaxis] / y_aux) <= epsilon
            msg = (
                msg
                + "within "
                + str(round(100 * epsilon))
                + "% of the worst."
            )

    print(msg)


    # best_algos = np.equal(y_raw, y_best)
    best_algos = y_raw == y_best[:, np.newaxis]
    multiple_best_algos = np.sum(best_algos, axis=1) > 1
    aidx = np.arange(1, nalgos + 1)

    p = np.zeros(y.shape[0], dtype=int)

    for i in range(y.shape[0]):
        if multiple_best_algos[i]:
            aux = aidx[best_algos[i]]
            # changed to pick the first one for testing purposes
            # will need to change it back to random after testing complete
            p[i] = aux[0]

    print("-> For", round(100 * np.mean(multiple_best_algos)), 
          "% of the instances there is more than one best algorithm.")
    print("Random selection is used to break ties.")

    num_good_algos = np.sum(y_bin, axis=1)
    beta = num_good_algos > (beta_threshold * nalgos)

    if bound:
        print("-> Removing extreme outliers from the feature values.")
        med_val = np.nanmedian(x, axis=0)
        iq_range = np.subtract(*np.percentile(x, [75, 25], axis=0))
        hi_bound = med_val + 5 * iq_range
        lo_bound = med_val - 5 * iq_range
        himask = x > hi_bound
        lomask = x < lo_bound
        x = np.where(himask | lomask, x, hi_bound)
        x = np.where(lomask, x, lo_bound)

    # if norm:
    #     nfeats = x.shape[1]
    #     nalgos = y.shape[1]
    #     print("-> Auto-normalizing the data using Box-Cox and Z transformations.")
    #     min_x = np.min(x, axis=0)
    #     x = x - min_x + 1
    #     lambda_x = np.zeros(nfeats)
    #     mu_x = np.zeros(nfeats)
    #     sigma_x = np.zeros(nfeats)
    #     for i in range(nfeats):
    #         aux = x[:, i]
    #         idx = np.isnan(aux)
    #         aux[~idx], lambda_x[i] = stats.boxcox(aux[~idx])
    #         aux, mu_x[i], sigma_x[i] = stats.zscore(aux, ddof=1)
    #         # ddof set to 1 as python defaults to 0; while MATLAB defaults to 1
    #         x[~idx, i] = aux

    #     min_y = np.min(y)
    #     y = (y - min_y) + np.finfo(float).eps
    #     lambda_x = np.zeros(nalgos)
    #     mu_x = np.zeros(nalgos)
    #     sigma_x = np.zeros(nalgos)
    #     for i in range(nalgos):
    #         aux = y[:, i]
    #         idx = np.isnan(aux)
    #         aux[~idx], lambda_x[i] = stats.boxcox(aux[~idx])
    #         aux, mu_x[i], sigma_x[i] = stats.zscore(aux, ddof=1)
    #         y[~idx, i] = aux

    print("p", p)

def main() -> None:
    # Create sample feature and performance data matrices
    # read x, y from test/input/model-data-x and y

    print('hello world')

    x = np.genfromtxt("prelim/input/model-data-x.csv", delimiter=",")
    y = np.genfromtxt("prelim/input/model-data-y.csv", delimiter=",")

    # with open("prelim/input/options.json", "r") as f:
    #     data = json.load(f)
    # Create sample options
    abs_perf = 1
    beta_threshold = 0.5500
    epsilon = 0.2000
    max_perf = 0
    bound = 1
    norm = 1
    # read input from json file

    # Call the prelim function
    prelim(x, y, abs_perf, beta_threshold, epsilon, max_perf, bound, norm)

if __name__ == "__main__":
    main()

