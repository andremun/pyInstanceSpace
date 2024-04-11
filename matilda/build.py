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

import numpy as np

from matilda.data.model import Model


def build_instance_space(rootdir: str) -> Model:
    """
    Construct and return a Model object after instance space analysis.

    :param rootdir: The root directory containing the data and configuration files
    :return: A Model object representing the built instance space.
    """
    # TODO: Rewrite buildIS logic in Python
    raise NotImplementedError


def select_features_and_algorithms(model: Model, opts: Model.opts):
    """
    Select features and algorithms based on options provided in opts.
    Remove instances with too many missing values.
    """
    print("-------------------------------------------------------------------------")
    if (getattr(opts, 'selvars', None) is not None) and (getattr(opts.selvars, 'feats', None) is not None):

        # assume that model.data.feat_labels and opts.selvars.feats are list of string
        selected_features = [feat for feat in model.data.feat_labels if feat in opts.selvars.feats]

        # if something were chosen, based on the logic index, rather than the name string
        if selected_features:
            print(f"-> Using the following features: {' '.join(selected_features)}")

            # 根据选中的特征更新数据
            is_selected_feature = [model.data.feat_labels.index(feat) for feat in selected_features]
            model.data.x = model.data.x[:, is_selected_feature]
            model.data.feat_labels = selected_features
        else:

            print("No features were specified in opts.selvars.feats or it was an empty list.")

    print("-------------------------------------------------------------------------")
    # 选择特定的算法
    if (getattr(opts, 'selvars', None) is not None) and (getattr(opts.selvars, 'algos', None) is not None):
        selected_algorithms = [algo for algo in model.data.algo_labels if algo in opts.selvars.algos]
        print(f"-> Using the following algorithms: {' '.join(selected_algorithms)}")

        # 更新模型数据
        is_selected_algo = [model.data.algo_labels.index(algo) for algo in selected_algorithms]
        model.data.y = model.data.y[:, is_selected_algo]
        model.data.algo_labels = selected_algorithms




if __name__ == "__main__":
    rootdir = sys.argv[1]
    build_instance_space(rootdir)
