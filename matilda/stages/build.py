"""Perform instance space analysis on given dataset and configuration.

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

import numpy as np

from matilda.data.metadata import Metadata
from matilda.data.model import Data
from matilda.data.option import Options


def _preprocess_input(metadata: Metadata, options: Options) -> Data:
    raise NotImplementedError


def select_features_and_algorithms(data: Data, opts: Options) -> Data:
    """Select features and algorithms based on options provided in opts.

    Remove instances with too many missing values.

    Parameters
    ----------
    data
        the Data class that contains the content of instances, with
        algorithm and feature labels
    opts
        the Option class that contains setting for analysis.

    :return Data: the Data class that has been modified based on the settings
    """
    popout = data
    print("---------------------------------------------------")
    if (opts.selvars is not None) and \
            (opts.selvars.feats is not None):

        selected_features = [feat for feat in data.feat_labels
                             if feat in opts.selvars.feats]

        # if something were chosen, based on the logic index,
        # rather than the name string
        if selected_features:
            print(f"-> Using the following features: "
                  f"{' '.join(selected_features)}")

            # based on manually selected feature to update the data.x
            is_selected_feature = [data.feat_labels.index(feat)
                                   for feat in selected_features]
            popout.x = data.x[:, is_selected_feature]
            popout.feat_labels = selected_features
        else:
            print("No features were specified in opts.selvars."
                  "feats or it was an empty list.")

    print("---------------------------------------------------")
    if (opts.selvars is not None) and \
            (opts.selvars.algos is not None):
        selected_algorithms = [algo for algo in data.algo_labels
                               if algo in opts.selvars.algos]

        if selected_algorithms:
            print(f"-> Using the following algorithms: "
                  f"{' '.join(selected_algorithms)}")

            is_selected_algo = [data.algo_labels.index(algo)
                                for algo in selected_algorithms]
            popout.y = data.y[:, is_selected_algo]
            popout.algo_labels = selected_algorithms
        else:
            print("No algorithms were specified in opts.selvars."
                  "algos or it was an empty list.")
    return popout


def remove_instances_with_many_missing_values(data: Data) -> Data:
    """Remove rows (instances) and features (X columns).

    Parameters
    ----------
    data
        the Data class that contains the content of instances, with
        algorithm and feature labels

    :return Data: the Data class that has been modified based on the Washing criterion

     Washing criterion:
        1. For any row, if that row in both X and Y are NaN, remove
        2. For X columns, if that column's 20% grids are filled with NaN, remove
    """
    popout = data
    # Identify rows where all elements are NaN in X or Y
    idx = np.all(np.isnan(data.x), axis=1) | \
          np.all(np.isnan(data.y), axis=1)
    if np.any(idx):
        print("-> There are instances with too many missing values. "
              "They are being removed to increase speed.")
        # Remove instances (rows) where all values are NaN
        popout.x = data.x[~idx]
        popout.y = data.y[~idx]

        popout.inst_labels = data.inst_labels[~idx]

        if data.s is not None:
            popout.s = data.s[~idx]

    # Check for features(column) with more than 20% missing values
    threshold = 0.20
    idx = np.mean(np.isnan(data.x), axis=0) >= threshold

    if np.any(idx):
        print("-> There are features with too many missing values. "
              "They are being removed to increase speed.")
        popout.x = data.x[:, ~idx]
        popout.feat_labels = [label for label,
        keep in zip(data.feat_labels, ~idx) if keep]

    ninst = popout.x.shape[0]
    nuinst = len(np.unique(popout.x, axis=0))
    # check if there are too many repeated instances
    max_duplic_ratio = 0.5
    if nuinst / ninst < max_duplic_ratio:
        print("-> There are too many repeated instances. "
              "It is unlikely that this run will produce good results.",
              )
    return popout
