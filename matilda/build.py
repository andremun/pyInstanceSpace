"""
Perform instance space analysis on given dataset and configuration.

Construct an instance space from data and configuration files located in a specified
directory. The instance space is represented as a Model object, which encapsulates the
analytical results and metadata of the instance space analysis.

The main function in this module, `build_instance_space`, reads the necessary
data from the provided directory, performs instance space analysis, and then
constructs a Model object that represents this analysis. This Model object can
then be used for further analysis, visualization, or processing within the
larger framework of the Matilda data analysis suite.

Functions:
    build_instance_space(rootdir: str) -> Model:
        Construct and return a Model object after performing instance space analysis
        on the data and configurations found in the specified root directory.

Example usage:
    python your_module_name.py /path/to/data
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.model_selection import train_test_split

from matilda.data.model import Model
from matilda.filter import filter_by_us
from matilda.prelim import prelim

MAX_DUPLICATES_RATIO = 0.5  # Constant


def build_instance_space(rootdir: str) -> Model:
    """
    Construct and return a Model object after instance space analysis.

    :param rootdir: The root directory containing the data and configuration files
    :return: A Model object representing the built instance space.
    """
    # TODO: Rewrite buildIS logic in Python
    raise NotImplementedError


if __name__ == "__main__":
    rootdir = sys.argv[1]
    build_instance_space(rootdir)


def data_processing(idx: NDArray[np.bool_], model: Model) -> int:
    # fix upper line late
    """
    Process data for instance space analysis.

    :param idx: A boolean array indicating features to be retained or removed.
    """
    if np.any(idx):
        # Remove instances with too many missing values
        warnings.warn(
            "There are features with too many missing values. They are\
being removed to increase speed.",
        )
        model.data.x = model.data.x[:, ~idx]
        model.data.feat_labels = [
            label for label, keep in zip(model.data.feat_labels, ~idx) if keep
        ]

    # get the number of instances and unique instances
    ninst = model.data.x.shape[0]
    nuinst = len(np.unique(model.data.x, axis=0))

    # check if there are too many repeated instances
    if nuinst / ninst < MAX_DUPLICATES_RATIO:
        warnings.warn(
            "-> There are too many repeated instances. It is unlikely\
that this run will produce good results.",
        )

    # Storing the raw data for further processing, e.g., graphs
    model.data.x_raw = model.data.x.copy()
    model.data.y_raw = model.data.y.copy()

    # Removing the template data such that it can be used in the labels of graphs
    # and figures
    model.data.feat_labels = [
        label.replace("feature_", "") for label in model.data.feat_labels
    ]
    model.data.algo_labels = [
        label.replace("algo_", "") for label in model.data.algo_labels
    ]

    # Running PRELIM as to preprocess the data, including scaling and bounding
    model.opts.prelim = model.opts.perf
    model.opts.prelim.bound = model.opts.bound.flag
    model.opts.prelim.norm = model.opts.norm.flag
    [
        model.data.x,
        model.data.y,
        model.data.y_best,
        model.data.y_bin,
        model.data.p,
        model.data.num_good_algos,
        model.data.beta,
        model.prelim,
    ] = prelim(model.data.x, model.data.y, model.opts)

    idx = np.all(~model.data.y_bin, axis=0)
    if np.any(idx):
        warnings.warn(
            '-> There are algorithms with no "good" instances. They are being\
removed to increase speed.',
        )
        model.data.y_raw = model.data.y_raw[:, ~idx]
        model.data.y = model.data.y[:, ~idx]
        model.data.y_bin = model.data.y_bin[:, ~idx]
        model.data.algo_labels = model.data.algo_labels[~idx]
        nalgos = model.data.y.shape[1]
        if nalgos == 0:
            raise Exception(
                "'-> There are no ''good'' algorithms. Please verify\
                             the binary performance measure. STOPPING!'",
            )

    # If we are only meant to take some observations
    print("-------------------------------------------------------------------")
    ninst = model.data.x.shape[0]
    fractional = (
        hasattr(model.opts, "selvars")
        and hasattr(model.opts.selvars, "small_scale_flag")
        and model.opts.selvars.small_scale_flag
        and hasattr(model.opts.selvars, "small_scale")
        and isinstance(model.opts.selvars.small_scale, float)
    )
    fileindexed = (
        hasattr(model.opts, "selvars")
        and hasattr(model.opts.selvars, "file_idx_flag")
        and model.opts.selvars.file_idx_flag
        and hasattr(model.opts.selvars, "file_idx")
        and os.path.isfile(model.opts.selvars.file_idx)
    )
    bydensity = (
        hasattr(model.opts, "selvars")
        and hasattr(model.opts.selvars, "density_flag")
        and model.opts.selvars.density_flag
        and hasattr(model.opts.selvars, "min_distance")
        and isinstance(model.opts.selvars.min_distance, float)
        and hasattr(model.opts.selvars, "type")
        and isinstance(model.opts.selvars.type, str)
    )
    if fractional:
        print(f"-> Creating a small scale experiment for validation. \
              Percentage of subset: \
              {round(100 * model.opts.selvars.small_scale, 2)}%")
        _, subset_idx = train_test_split(
            np.arange(ninst),
            test_size=model.opts.selvars.small_scale,
            random_state=0,
        )
        # below are not sure
        subset_index = np.zeros(ninst, dtype=bool)
        subset_index[idx] = True
    elif fileindexed:
        print("-> Using a subset of instances.")
        subset_index = np.zeros(ninst, dtype=bool)
        aux = pd.read_csv(model.opts.selvars.file_idx, header=None).values.flatten()
        aux = aux[aux < ninst]
        subset_index[aux] = True
    elif bydensity:
        print("-> Creating a small scale experiment for validation based on density.")
        subset_index, _, _ = filter_by_us(
            model.data.x,
            model.data.y,
            model.data.y_bin,
            model.opts.selvars,
        )
        subset_index = ~subset_index
        print(f"-> Percentage of instances retained: \
              {round(100 * np.mean(subset_index), 2)}%")
    else:
        print("-> Using the complete set of the instances.")
        subset_index = np.ones(ninst, dtype=bool)

    if fileindexed or fractional or bydensity:
        if bydensity:
            model.data_dense = model.data

        model.data.x = model.data.x[subset_index, :]
        model.data.y = model.data.y[subset_index, :]
        model.data.x_raw = model.data.x_raw[subset_index, :]
        model.data.y_raw = model.data.y_raw[subset_index, :]
        model.data.y_bin = model.data.y_bin[subset_index, :]
        model.data.beta = model.data.beta[subset_index, :]
        model.data.num_good_algos = model.data.num_good_algos[subset_index, :]
        model.data.y_best = model.data.y_best[subset_index, :]
        model.data.p = model.data.p[subset_index, :]
        model.data.inst_labels = model.data.inst_labels[subset_index, :]

        if hasattr(model.data, "S"):
            model.data.S = model.data.S[subset_index, :]
    return model.data.x.shape[1]  # nfeats
