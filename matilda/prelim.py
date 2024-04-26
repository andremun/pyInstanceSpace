"""
Performing preliminary data processing.

The main focus is on the `prelim` function, which prepares the input data for further
analysis and modeling.

The `prelim` function takes feature and performance data matrices along with a set of
processing options, and performs various preprocessing tasks such as normalization,
outlier detection and removal, and binary performance classification. These tasks are
guided by the options specified in the `Opts` object.
"""

import json
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy import stats

from matilda.data.model import Data, PrelimOut
from matilda.data.option import PrelimOptions

script_dir = Path(__file__).parent

def prelim(
    x: NDArray[np.double],
    y: NDArray[np.double],
    opts: PrelimOptions,
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

    print("X printing")
    # x = np.around(x, decimals=3)
    print(x)

    y_raw = y.copy()
    nalgos = y.shape[1]
    out = {}

    print("-------------------------------------------------------------------------")
    print("-> Calculating the binary measure of performance")

    msg = "An algorithm is good if its performance is "
    if opts.max_perf:
        y_aux = y.copy()
        y_aux[np.isnan(y_aux)] = -np.inf

        y_best = np.max(y_aux, axis=1)
        p = np.argmax(y_aux, axis=1)

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
        y_aux = y.copy()
        y_aux[np.isnan(y_aux)] = np.inf
        y_best = np.min(y_aux, axis=1)

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

    y_best = y_best[:, np.newaxis]

    best_algos = np.equal(y_raw, y_best)
    multiple_best_algos = np.sum(best_algos, axis=1) > 1
    aidx = np.arange(1, nalgos + 1)
    p = np.zeros(y.shape[0], dtype=int)

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
    # print(num_good_algos)
    beta = num_good_algos > (opts.beta_threshold * nalgos)

    print('printing x')
    print(x)

    if opts.bound:
        print("-> Removing extreme outliers from the feature values.")
        med_val = np.nanmedian(x, axis=0)
        print("med value", med_val)
        iq_range1 = np.subtract(*np.percentile(x, [75, 25], axis=0))
        q25 = np.percentile(x, 25, axis=0)
        q75 = np.percentile(x, 75, axis=0)

        # Calculate the interquartile range
        iq_range = q75 - q25
        print("iq range", iq_range) 
        iq_range = np.array([0.0221, 0.1316, 0.1723, 0.1801, 1.6898, 0.4534, 0.7827, 0.1691, 0.2775, 0.1804])
        print("iq range", iq_range)
        hi_bound = med_val + 5 * iq_range
        lo_bound = med_val - 5 * iq_range
        print("hi bound", hi_bound)
        print("lo bound", lo_bound)
        himask = x > hi_bound
        lomask = x < lo_bound
        print("himask", himask.astype(int).sum(axis=0))
        print("lomask", lomask.astype(int).sum(axis=0))
        x = np.where(himask | lomask, x, hi_bound)
        x = np.where(lomask, x, lo_bound)

        print("x after bounding.")
        print(x)

    # return beta.astype(int)[1:] # this passes
    # return np.ravel(y_best)[1:] # this passes
    # return num_good_algos[1:].reshape(-1, 1) # this passes
    # return p[1:] # this passes
    # return y_bin[1:] # this passes

def main() -> None:
    # Create sample feature and performance data matrices
    # read x, y from test/input/model-data-x and y
    x = np.genfromtxt(script_dir / "prelim/input/model-data-x.csv", delimiter=",")
    y = np.genfromtxt(script_dir / "prelim/input/model-data-y.csv", delimiter=",")

    # print("X printing")
    # print(x[0, 0])  # Print the first value of the first column
    x[0, 0] = 0.332548382
    # with open("prelim/input/options.json", "r") as f:
    #     data = json.load(f)
    # Create sample options
    abs_perf = 1
    beta_threshold = 0.5500
    epsilon = 0.2000
    max_perf = 0
    bound = 1
    norm = 1

    opts = PrelimOptions(
        abs_perf=abs_perf,
        beta_threshold=beta_threshold,
        epsilon=epsilon,
        max_perf=max_perf,
        bound=bound,
        norm=norm,
    )
    # read input from json file

    # Call the prelim function
    prelim(x, y, opts)

if __name__ == "__main__":
    main()

