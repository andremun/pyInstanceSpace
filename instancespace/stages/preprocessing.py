# SPDX-License-Identifier: LicenseRef-PolyForm-Noncommercial-1.0.0
# Copyright (c) 2024-2026 Mario Andrés Muñoz
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
from loguru import logger
from numpy._typing import NDArray

from instancespace.data.default_options import DEFAULT_PRELIM_NAN_THRESHOLD
from instancespace.data.options import (
    PrelimOptions,
    SelvarsOptions,
)
from instancespace.stages.stage import Stage

MATRIX_DIMENSIONS = 2
MIN_FEATURES = 3


def validate_viable_dimensions(
    x: NDArray[np.double],
    y: NDArray[np.double],
    *,
    require_algorithms: bool = True,
    context: str = "Input data",
) -> None:
    """Validate matrix shape and the minimum viable instance-space dimensions."""
    x_array = np.asarray(x)
    y_array = np.asarray(y)
    for name, matrix in (("feature", x_array), ("algorithm", y_array)):
        is_real_numeric = np.issubdtype(
            matrix.dtype,
            np.number,
        ) and not np.issubdtype(matrix.dtype, np.complexfloating)
        if matrix.dtype == np.bool_ or not is_real_numeric:
            msg = f"{context} {name} values must be numeric and non-Boolean."
            raise ValueError(msg)
        if matrix.ndim != MATRIX_DIMENSIONS:
            msg = f"{context} {name} values must be a two-dimensional matrix."
            raise ValueError(msg)
        if np.isinf(matrix).any():
            msg = f"{context} {name} values must be finite or NaN."
            raise ValueError(msg)

    if x_array.shape[0] != y_array.shape[0]:
        msg = f"{context} feature and algorithm matrices must have equal row counts."
        raise ValueError(msg)
    if x_array.shape[0] < 1:
        msg = f"{context} must contain at least one instance."
        raise ValueError(msg)
    if x_array.shape[1] < MIN_FEATURES:
        msg = f"{context} must contain at least three features."
        raise ValueError(msg)
    if require_algorithms and y_array.shape[1] < 1:
        msg = f"{context} must contain at least one algorithm."
        raise ValueError(msg)


def _selected_indices(
    labels: list[str],
    requested: list[str] | None,
    *,
    prefix: str,
    kind: str,
) -> list[int]:
    """Resolve a manual selection while preserving dataset order."""
    if not requested:
        return list(range(len(labels)))
    if not all(isinstance(value, str) for value in requested):
        msg = f"Manual {kind} selections must contain only strings."
        raise ValueError(msg)

    lowered_labels = [label.casefold() for label in labels]
    matched_requests: set[int] = set()
    selected: list[int] = []
    for label_index, lowered_label in enumerate(lowered_labels):
        label_selected = False
        for request_index, request in enumerate(requested):
            lowered_request = request.casefold()
            exact_match = lowered_request == lowered_label
            prefixed_match = lowered_request.startswith(prefix) and (
                lowered_request[len(prefix) :] == lowered_label
            )
            if exact_match or prefixed_match:
                label_selected = True
                matched_requests.add(request_index)
        if label_selected:
            selected.append(label_index)

    unknown = [
        request
        for request_index, request in enumerate(requested)
        if request_index not in matched_requests
    ]
    if not selected:
        requested_text = ", ".join(requested)
        available_text = ", ".join(labels)
        msg = (
            f"Manual {kind} selection matched no columns. "
            f"Requested: {requested_text}. Available: {available_text}."
        )
        raise ValueError(msg)
    if unknown:
        logger.warning(
            f"[PREPROCESSING] Unknown {kind} selections were ignored: "
            f"{', '.join(unknown)}.",
        )
    return selected


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
    prelim_options : PrelimOptions
        Composed preliminary-processing options. Legacy direct construction uses
        the same defaults as the aggregate configuration.
    """

    feature_names: list[str]
    algorithm_names: list[str]
    instance_labels: pd.Series  # type: ignore[type-arg]
    instance_sources: pd.Series | None  # type: ignore[type-arg]
    features: NDArray[np.double]
    algorithms: NDArray[np.double]
    selvars_options: SelvarsOptions
    prelim_options: PrelimOptions = PrelimOptions.default()


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
            inputs.prelim_options.nan_threshold,
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
        logger.info("[PREPROCESSING] " + "-" * 65)
        if x.ndim != MATRIX_DIMENSIONS or x.shape[1] != len(feat_labels):
            msg = "Feature labels must match the feature matrix columns."
            raise ValueError(msg)
        if y.ndim != MATRIX_DIMENSIONS or y.shape[1] != len(algo_labels):
            msg = "Algorithm labels must match the algorithm matrix columns."
            raise ValueError(msg)
        if x.shape[0] != y.shape[0]:
            msg = "Feature and algorithm matrices must have equal row counts."
            raise ValueError(msg)

        feature_indices = _selected_indices(
            feat_labels,
            selvars.feats,
            prefix="feature_",
            kind="feature",
        )
        algorithm_indices = _selected_indices(
            algo_labels,
            selvars.algos,
            prefix="algo_",
            kind="algorithm",
        )
        new_x = x[:, feature_indices]
        new_feat_labels = [feat_labels[index] for index in feature_indices]
        new_y = y[:, algorithm_indices]
        new_algo_labels = [algo_labels[index] for index in algorithm_indices]

        if selvars.feats:
            logger.info(
                "[PREPROCESSING] -> Using the following features: "
                f"{' '.join(new_feat_labels)}",
            )

        logger.info("[PREPROCESSING] " + "-" * 65)
        if selvars.algos:
            logger.info(
                "[PREPROCESSING] -> Using the following algorithms: "
                f"{' '.join(new_algo_labels)}",
            )
        return new_x, new_y, new_feat_labels, new_algo_labels

    @staticmethod
    def remove_instances_with_many_missing_values(
        x: NDArray[np.double],
        y: NDArray[np.double],
        s: pd.Series | None,  # type: ignore[type-arg]
        feat_labels: list[str],
        inst_labels: pd.Series,  # type: ignore[type-arg]
        nan_threshold: float = DEFAULT_PRELIM_NAN_THRESHOLD,
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
        nan_threshold : float
            Fraction of missing values at which a feature is removed.

        Returns
        -------
        tuple[NDArray[np.double], NDArray[np.double],
        pd.Series, list[str], pd.Series | None]
            A tuple containing the modified feature matrix 'x',
            the modified algorithm matrix 'y',updated instance labels,
            list of feature labels that remain after removal, and optionally
            modified series 's' if provided.
        """
        validate_viable_dimensions(x, y, context="Preprocessing input")
        if len(feat_labels) != x.shape[1]:
            msg = "Feature labels must match the feature matrix columns."
            raise ValueError(msg)
        if len(inst_labels) != x.shape[0]:
            msg = "Instance labels must match the preprocessing rows."
            raise ValueError(msg)
        if s is not None and len(s) != x.shape[0]:
            msg = "Instance sources must match the preprocessing rows."
            raise ValueError(msg)

        new_x = x
        new_y = y
        new_inst_labels = inst_labels
        new_s = s
        new_feat_labels = feat_labels
        # Identify rows where all elements are NaN in X or Y
        idx = np.all(np.isnan(x), axis=1) | np.all(np.isnan(y), axis=1)
        if np.any(idx):
            logger.info(
                "[PREPROCESSING] -> There are instances with too many missing values. "
                "They are being removed to increase speed.",
            )
            # Remove instances (rows) where all values are NaN
            new_x = x[~idx]
            new_y = y[~idx]

            new_inst_labels = inst_labels[~idx]

            if s is not None:
                new_s = s[~idx]

        if new_x.shape[0] == 0:
            msg = "Data washing removed every instance."
            raise ValueError(msg)

        # Remove feature columns at or above MATLAB's configured NaN fraction.
        idx = np.mean(np.isnan(new_x), axis=0) >= nan_threshold

        if np.any(idx):
            logger.info(
                "[PREPROCESSING] -> There are features with too many missing values. "
                "They are being removed to increase speed.",
            )
            new_x = new_x[:, ~idx]
            new_feat_labels = [label for label, keep in zip(feat_labels, ~idx) if keep]

        validate_viable_dimensions(new_x, new_y, context="Washed data")

        ninst = new_x.shape[0]
        nuinst = len(np.unique(new_x, axis=0))
        # check if there are too many repeated instances
        max_duplic_ratio = 0.5
        if nuinst / ninst < max_duplic_ratio:
            logger.info(
                "[PREPROCESSING] -> There are too many repeated instances. "
                "It is unlikely that this run will produce good results.",
            )
        return new_x, new_y, new_inst_labels, new_feat_labels, new_s
