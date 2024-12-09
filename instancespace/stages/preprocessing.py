"""Preprocessing Stage Module.

This module defines the classes and methods for the preprocessing stage
of a machine learning pipeline. It filters data rows based on provided
options, and removes instances or features with too many missing values.

The preprocessing stage outputs a cleaned and filtered dataset that can be
used for further modeling or analysis.

Classes
-------
PreprocessingInput : NamedTuple
    Defines the input data structure for the preprocessing stage.
PreprocessingOutput : NamedTuple
    Defines the output data structure for the preprocessing stage.
PreprocessingStage : Stage
    Class that executes the preprocessing stage.

"""

from typing import NamedTuple

import numpy as np
import pandas as pd
from numpy._typing import NDArray

from instancespace.data.options import (
    SelvarsOptions,
)
from instancespace.stages.stage import Stage


class PreprocessingInput(NamedTuple):
    """Inputs for the Preprocessing stage.

    Attributes
    ----------
    feature_names : list[str]
        List of feature names in the dataset.
    algorithm_names : list[str]
        List of algorithm names in the dataset.
    instance_labels : pd.Series
        Labels for each instance (row) in the dataset.
    instance_sources : pd.Series | None
        Sources for each instance, optional.
    features : NDArray[np.double]
        Feature matrix (instances x features) as a 2D numpy array.
    algorithms : NDArray[np.double]
        Algorithm matrix (instances y algorithms) as a 2D numpy array.
    selvars_options : SelvarsOptions
        Options for selecting variables (features and algorithms).
    """

    feature_names: list[str]
    algorithm_names: list[str]
    instance_labels: pd.Series  # type: ignore[type-arg]
    instance_sources: pd.Series | None  # type: ignore[type-arg]
    features: NDArray[np.double]
    algorithms: NDArray[np.double]
    selvars_options: SelvarsOptions


class PreprocessingOutput(NamedTuple):
    """Outputs for the Preprocessing stage.

    Attributes
    ----------
    inst_labels : pd.Series
        Series containing labels for each instance after preprocessing.
    feat_labels : list[str]
        List of labels corresponding to the selected features.
    algo_labels : list[str]
        List of labels corresponding to the selected algorithms.
    x : NDArray[np.double]
        Preprocessed feature matrix (instances x selected features).
    y : NDArray[np.double]
        Preprocessed algorithm matrix (instances y selected algorithms).
    s : pd.Series | None
        Optional series containing the source of instances after preprocessing.
    x_raw : NDArray[np.double]
        Original feature matrix before any modifications.
    y_raw : NDArray[np.double]
        Original algorithm matrix before any modifications.

    """

    inst_labels: pd.Series  # type: ignore[type-arg]
    feat_labels: list[str]
    algo_labels: list[str]
    x: NDArray[np.double]
    y: NDArray[np.double]
    s: pd.Series | None  # type: ignore[type-arg]
    x_raw: NDArray[np.double]
    y_raw: NDArray[np.double]


class PreprocessingStage(Stage[PreprocessingInput, PreprocessingOutput]):
    """Class for handling the preprocessing stage of the pipeline.

    This stage includes tasks such as feature selection, algorithm selection,
    and removing instances or features with too many missing values.

    Methods
    -------
    select_features_and_algorithms(x, y, feat_labels, algo_labels, selvars)
        Selects features and algorithms from the dataset based on user options.
    remove_instances_with_many_missing_values(x, y, s, feat_labels, inst_labels)
        Removes instances (rows) and features (columns) with excessive missing values.
    """

    def __init__(
        self,
        feature_names: list[str],
        algorithm_names: list[str],
        instance_labels: pd.Series,  # type: ignore[type-arg]
        instance_sources: pd.Series | None,  # type: ignore[type-arg]
        features: NDArray[np.double],
        algorithms: NDArray[np.double],
        selvars: SelvarsOptions,
    ) -> None:
        """Initialize the Preprocessing stage."""
        self.feature_names = feature_names
        self.algorithm_names = algorithm_names
        self.instance_labels = instance_labels
        self.instance_sources = instance_sources
        self.features = features
        self.algorithms = algorithms
        self.selvars = selvars

    @staticmethod
    def _inputs() -> type[PreprocessingInput]:
        return PreprocessingInput

    @staticmethod
    def _outputs() -> type[PreprocessingOutput]:
        return PreprocessingOutput

    @staticmethod
    def _run(inputs: PreprocessingInput) -> PreprocessingOutput:
        """Perform preliminary processing on the input data 'x' and 'y'.

        Args
        -------
        inputs : PreprocessingInput
            Inputs for the cloister stage.

        Returns
        -------
        PreprocessingOutput
            Output of the Preprocessing stage.
        """
        (
            new_x,
            new_y,
            new_feat_labels,
            new_algo_labels,
        ) = PreprocessingStage.select_features_and_algorithms(
            inputs.features,
            inputs.algorithms,
            inputs.feature_names,
            inputs.algorithm_names,
            inputs.selvars_options,
        )

        (
            updated_x,
            updated_y,
            updated_inst_labels,
            updated_feat_labels,
            updated_s,
        ) = PreprocessingStage.remove_instances_with_many_missing_values(
            new_x,
            new_y,
            inputs.instance_sources,
            new_feat_labels,
            inputs.instance_labels,
        )

        return PreprocessingOutput(
            updated_inst_labels,
            updated_feat_labels,
            new_algo_labels,
            updated_x,
            updated_y,
            updated_s,
            updated_x,
            updated_y,
        )

    @staticmethod
    def select_features_and_algorithms(
        x: NDArray[np.double],
        y: NDArray[np.double],
        feat_labels: list[str],
        algo_labels: list[str],
        selvars: SelvarsOptions,
    ) -> tuple[NDArray[np.double], NDArray[np.double], list[str], list[str]]:
        """Select features and algorithms from the dataset.

        Based on the user's configuration, this method filters the features
        and algorithms that should be used in subsequent stages.

        Args
        ----------
        x : NDArray[np.double]
            2D numpy array representing the feature matrix (instances x features).
        y : NDArray[np.double]
            2D numpy array representing the algorithm matrix (instances y algorithms).
        feat_labels : list[str]
            List of labels corresponding to the features in 'x'.
        algo_labels : list[str]
            List of labels corresponding to the algorithms in 'y'.
        selvars : SelvarsOptions
            An instance of SelvarsOptions that contains settings of the prefered
            algorithms and instances.

        Returns
        -------
        tuple[NDArray[np.double], NDArray[np.double], list[str], list[str]]
            A tuple containing:
            - Modified feature matrix after feature selection and instance removal.
            - Modified algorithm matrix after algorithm selection and instance removal.
            - List of selected feature labels.
            - List of selected algorithm labels.
        """
        print("---------------------------------------------------")
        new_x = x
        new_feat_labels = feat_labels
        new_y = y
        new_algo_labels = algo_labels
        if selvars.feats is not None:
            selected_features = [feat for feat in feat_labels if feat in selvars.feats]

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
        if selvars.algos is not None:
            selected_algorithms = [
                algo for algo in algo_labels if algo in selvars.algos
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
    ) -> tuple[  # type: ignore[type-arg]
        NDArray[np.double],
        NDArray[np.double],
        pd.Series,
        list[str],
        pd.Series | None,
    ]:
        """Remove instances and features with excessive missing values.

        Instances (rows) with too many missing values are removed. Additionally,
        features (columns) that exceed a missing value threshold are also removed.
        Washing criterion:
            1. For any row, if that row in both X and Y are NaN, remove
            2. For X columns, if that column's 20% grids are filled with NaN, remove

        Args
        ----------
        x : NDArray[np.double]
            2D numpy array representing the feature matrix (instances x features).
        y : NDArray[np.double]
            2D numpy array representing the algorithm matrix (instances y algorithms).

        s : pd.Series | None
            Optional series containing the source of instances.
        feat_labels : list[str]
            List of labels corresponding to the features in 'x'.
        inst_labels : pd.Series
            Series containing labels for each instance.

        Returns
        -------
        tuple[NDArray[np.double], NDArray[np.double],
        pd.Series, list[str], pd.Series | None]
            A tuple containing the modified feature matrix 'x',
            the modified algorithm matrix 'y',updated instance labels,
            list of feature labels that remain after removal, and optionally
            modified series 's' if provided.
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
