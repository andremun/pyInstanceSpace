"""Process the input data before running the main analysis."""

import numpy as np
import pandas as pd
from numpy._typing import NDArray

from matilda.data.metadata import Metadata
from matilda.data.model import (
    PreprocessingDataChanged,
    PreprocessingOut,
)
from matilda.data.options import InstanceSpaceOptions


class Preprocessing:
    """See file docstring."""

    def __init__(
        self,
        matadata: Metadata,
        opts: InstanceSpaceOptions,
    ) -> None:
        """Initialize the preprocessing stage.

        Args
        ----
            matadata: A Metadata object constructed from the parsed CSV data
            opts (Options): Configuration options.
        """
        self.matadata = matadata
        self.opts = opts

    @staticmethod
    def run(
        matadata: Metadata,
        opts: InstanceSpaceOptions,
    ) -> tuple[PreprocessingDataChanged, PreprocessingOut]:
        """Perform preliminary processing on the input data 'x' and 'y'.

        Args
            matadata: An object of data class that contains data
                from CSV file.
            opts: An object of type Options containing options for
                processing.

        Returns
        -------
            A tuple containing the processed data (as 'PreprocessingDataChanged' object)
            and dummy object for further implementation (as 'PreprocessingOut' object).
        """
        new_x, new_y, new_feat_labels, new_algo_labels = (
            Preprocessing.select_features_and_algorithms(
                matadata.features,
                matadata.algorithms,
                matadata.feature_names,
                matadata.algorithm_names,
                opts,
            )
        )

        after_washing = Preprocessing.remove_instances_with_many_missing_values(
            new_x,
            new_y,
            matadata.instance_sources,
            new_feat_labels,
            matadata.instance_labels,
        )

        # From here return the tuple[PreprocessingDataChanged, PreprocessingOut(dummy)]

        pre_data_changed = PreprocessingDataChanged(
            inst_labels=after_washing.inst_labels,
            feat_labels=after_washing.feat_labels,
            algo_labels=after_washing.algo_labels,
            x=after_washing.x,
            y=after_washing.y,
            s=after_washing.s,
        )

        preprocess_out = PreprocessingOut()

        # these stuff will be moved into PRILIM
        """after_process, prelim_opts = Preprocessing.process_data(after_washing, opts)
        prelim_data, prelim_out = Prelim.run(
            after_process.x,
            after_process.y,
            prelim_opts,
        )

        # These should be a part of FILTRER, leave it not delete

        bad_instances_removed = Preprocessing.remove_bad_instances(
            prelim_data.merge_with(data),
        )
        # Preprocessing.split_data(bad_instances_removed, opts, model)
        """

        return pre_data_changed, preprocess_out

    @staticmethod
    def select_features_and_algorithms(
        x: NDArray[np.double],
        y: NDArray[np.double],
        feat_labels: list[str],
        algo_labels: list[str],
        opts: InstanceSpaceOptions,
    ) -> tuple[NDArray[np.double], NDArray[np.double], list[str], list[str]]:
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
        print("---------------------------------------------------")
        new_x = x
        new_feat_labels = feat_labels
        new_y = y
        new_algo_labels = algo_labels
        if opts.selvars.feats is not None:
            selected_features = [
                feat for feat in feat_labels if feat in opts.selvars.feats
            ]

            # if something were chosen, based on the logic index,
            # rather than the name string
            if selected_features:
                print(
                    f"-> Using the following features: "
                    f"{' '.join(selected_features)}",
                )

                # based on manually selected feature to update the data.x
                is_selected_feature = [
                    feat_labels.index(feat) for feat in selected_features
                ]
                new_x = x[:, is_selected_feature]
                new_feat_labels = selected_features
            else:
                print(
                    "No features were specified in opts.selvars."
                    "feats or it was an empty list.",
                )

        print("---------------------------------------------------")
        if opts.selvars.algos is not None:
            selected_algorithms = [
                algo for algo in algo_labels if algo in opts.selvars.algos
            ]

            if selected_algorithms:
                print(
                    f"-> Using the following algorithms: "
                    f"{' '.join(selected_algorithms)}",
                )

                is_selected_algo = [
                    algo_labels.index(algo) for algo in selected_algorithms
                ]
                new_y = y[:, is_selected_algo]
                new_algo_labels = selected_algorithms
            else:
                print(
                    "No algorithms were specified in opts.selvars."
                    "algos or it was an empty list.",
                )
        return new_x, new_y, new_feat_labels, new_algo_labels

    @staticmethod
    def remove_instances_with_many_missing_values(
        x: NDArray[np.double],
        y: NDArray[np.double],
        s: pd.Series | None,  # type: ignore[type-arg]
        feat_labels: list[str],
        inst_labels: pd.Series,  # type: ignore[type-arg]
    ) -> tuple[NDArray[np.double], NDArray[np.double], pd.Series, list[str], pd.Series]:  # type: ignore[type-arg]
        """Remove rows (instances) and features (X columns).

        Parameters
        ----------
        data
            the Data class that contains the content of instances, with
            algorithm and feature labels

        :return Data: Data class that has been updated based on the Washing criterion

         Washing criterion:
            1. For any row, if that row in both X and Y are NaN, remove
            2. For X columns, if that column's 20% grids are filled with NaN, remove
        """
        new_x = x
        new_y = y
        new_inst_labels = inst_labels
        new_s = s
        new_feat_labels = feat_labels
        # Identify rows where all elements are NaN in X or Y
        idx = np.all(np.isnan(x), axis=1) | np.all(np.isnan(y), axis=1)
        if np.any(idx):
            print(
                "-> There are instances with too many missing values. "
                "They are being removed to increase speed.",
            )
            # Remove instances (rows) where all values are NaN
            new_x = x[~idx]
            new_y = y[~idx]

            new_inst_labels = inst_labels[~idx]

            if s is not None:
                new_s = s[~idx]

        # Check for features(column) with more than 20% missing values
        threshold = 0.20
        idx = np.mean(np.isnan(new_x), axis=0) >= threshold

        if np.any(idx):
            print(
                "-> There are features with too many missing values. "
                "They are being removed to increase speed.",
            )
            new_x = new_x[:, ~idx]
            new_feat_labels = [label for label, keep in zip(feat_labels, ~idx) if keep]

        ninst = new_x.shape[0]
        nuinst = len(np.unique(new_x, axis=0))
        # check if there are too many repeated instances
        max_duplic_ratio = 0.5
        if nuinst / ninst < max_duplic_ratio:
            print(
                "-> There are too many repeated instances. "
                "It is unlikely that this run will produce good results.",
            )
        return new_x, new_y, new_inst_labels, new_feat_labels, new_s
