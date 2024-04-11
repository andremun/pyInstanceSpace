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

import sys
import warnings

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from matilda.data.model import Model

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


def data_processing(idx: NDArray[np.bool_], model: Model) -> int:  # fix late
    """
    Process data for instance space analysis.

    :param idx: A boolean array indicating features to be retained or removed.
    """
    if idx is not None:
        # Remove instances with too many missing values
        warnings.warn(
            "There are features with too many missing values. They are\
being removed to increase speed."
        )
        model.data.x = model.data.x.loc[:, ~idx]
        model.data.feat_labels = [
            label for label, keep in zip(model.data.feat_labels, ~idx) if keep
        ]

    # get the number of instances and unique instances
    ninst = model.data.x.shape[0]
    nuinst = len(model.data.x.drop_duplicates())

    # check if there are too many repeated instances
    if nuinst / ninst < MAX_DUPLICATES_RATIO:
        warnings.warn(
            "-> There are too many repeated instances. It is unlikely\
that this run will produce good results."
        )

    # Storing the raw data for further processing, e.g., graphs
    return 1
