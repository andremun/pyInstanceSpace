"""
Performing preliminary data processing.

The main focus is on the `prelim` function, which prepares the input data for further
analysis and modeling.

The `prelim` function takes feature and performance data matrices along with a set of
processing options, and performs various preprocessing tasks such as normalization,
outlier detection and removal, and binary performance classification. These tasks are
guided by the options specified in the `Opts` object.
"""

import numpy as np
from numpy.typing import NDArray
from scipy import stats

from matilda.data.model import Data, PrelimOut
from matilda.data.option import Opts


def prelim(
    x: NDArray[np.double],
    y: NDArray[np.double],
    opts: Opts,
) -> tuple[Data, PrelimOut]:
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
    out = PrelimOut()

    print("-------------------------------------------------------------------------")
    print("-> Calculating the binary measure of performance")

    msg = "An algorithm is good if its performance is "
    if opts.perf.max_perf:
        y_aux = y.copy()
        y_aux[np.isnan(y_aux)] = -np.inf

        y_best = np.max(y_aux, axis=1)
        p = np.argmax(y_aux, axis=1)

        if opts.perf.abs_perf:
            y_bin = y_aux >= opts.perf.epsilon
            msg = msg + "higher than " + opts.perf.epsilon
        else:
            y_best[y_best == 0] = np.finfo(float).eps
            y[y == 0] = np.finfo(float).eps
            y = 1 - y / y_best[:, np.newaxis]
            y_bin = (1 - y_aux / y_best[:, np.newaxis]) <= opts.perf.epsilon
            msg = (
                msg
                + "within "
                + str(round(100 * opts.perf.epsilon))
                + "% of the best."
            )

    else:
        y_aux = y.copy()
        y_aux[np.isnan(y_aux)] = np.inf
        y_best = np.min(y_aux, axis=1)

        if opts.perf.abs_perf:
            y_bin = y_aux <= opts.epsilon
            msg = msg + "less than " + opts.perf.epsilon
        else:
            y_best[y_best == 0] = np.finfo(float).eps
            y[y == 0] = np.finfo(float).eps
            y = 1 - y_best[:, np.newaxis] / y
            y_bin = (1 - y_best[:, np.newaxis] / y_aux) <= opts.perf.epsilon
            msg = (
                msg
                + "within "
                + str(round(100 * opts.perf.epsilon))
                + "% of the worst."
            )

    print(msg)


    best_algos = np.equal(y_raw, y_best)
    multiple_best_algos = np.sum(best_algos, axis=1) > 1
    aidx = np.arange(1, nalgos + 1)

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
    beta = num_good_algos > (opts.perf.beta_threshold * nalgos)

    if opts.bound:
        print("-> Removing extreme outliers from the feature values.")
        out.med_val = np.nanmedian(x, axis=0)
        out.iq_range = np.subtract(*np.percentile(x, [75, 25], axis=0))
        out.hi_bound = out.medval + 5 * out.iqrange
        out.lo_bound = out.medval - 5 * out.iqrange
        himask = x > out.hi_bound
        lomask = x < out.lo_bound
        x = np.where(himask | lomask, x, out.hi_bound)
        x = np.where(lomask, x, out.lo_bound)

    if opts.norm:
        nfeats = x.shape[1]
        nalgos = y.shape[1]
        print("-> Auto-normalizing the data using Box-Cox and Z transformations.")
        out.min_x = np.min(x, axis=0)
        x = x - out.min_x + 1
        out.lambda_x = np.zeros(nfeats)
        out.mu_x = np.zeros(nfeats)
        out.sigma_x = np.zeros(nfeats)
        for i in range(nfeats):
            aux = x[:, i]
            idx = np.isnan(aux)
            aux[~idx], out.lambda_x[i] = stats.boxcox(aux[~idx])
            aux, out.mu_x[i], out.sigma_x[i] = stats.zscore(aux, ddof=1)
            # ddof set to 1 as python defaults to 0; while MATLAB defaults to 1
            x[~idx, i] = aux

        out.min_x = np.min(y)
        y = (y - out.minY) + np.finfo(float).eps
        out.lambda_x = np.zeros(nalgos)
        out.mu_x = np.zeros(nalgos)
        out.sigma_x = np.zeros(nalgos)
        for i in range(nalgos):
            aux = y[:, i]
            idx = np.isnan(aux)
            aux[~idx], out.lambda_x[i] = stats.boxcox(aux[~idx])
            aux, out.mu_x[i], out.sigma_x[i] = stats.zscore(aux, ddof=1)
            y[~idx, i] = aux
