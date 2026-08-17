# SPDX-License-Identifier: LicenseRef-PolyForm-Noncommercial-1.0.0
# Copyright (c) 2024-2026 Mario Andrés Muñoz
"""Defines data types for metadata.

These classes define types for problem instances found in the metadata.csv file.
"""

from __future__ import annotations

import csv
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from numpy.typing import NDArray
from pandas import DataFrame

MATRIX_DIMENSIONS = 2


@dataclass(frozen=True)
class Metadata:
    # TODO: Ask someone for a better description of what metadata is
    """Metadata for problem instances."""

    feature_names: list[str]
    algorithm_names: list[str]
    instance_labels: pd.Series  # type: ignore[type-arg]
    instance_sources: pd.Series | None  # type: ignore[type-arg]
    features: NDArray[np.double]
    algorithms: NDArray[np.double]

    def __post_init__(self) -> None:
        """Validate direct construction and normalize numeric matrices."""
        _validate_names("feature", self.feature_names, require_one=True)
        _validate_names("algorithm", self.algorithm_names, require_one=False)

        features = _as_real_matrix("features", self.features)
        algorithms = _as_real_matrix("algorithms", self.algorithms)
        object.__setattr__(self, "features", features)
        object.__setattr__(self, "algorithms", algorithms)

        if features.shape[0] != algorithms.shape[0]:
            msg = "Feature and algorithm matrices must have the same number of rows."
            raise ValueError(msg)
        if features.shape[1] != len(self.feature_names):
            msg = "Feature names must match the feature matrix columns."
            raise ValueError(msg)
        if algorithms.shape[1] != len(self.algorithm_names):
            msg = "Algorithm names must match the algorithm matrix columns."
            raise ValueError(msg)
        if not isinstance(self.instance_labels, pd.Series):
            msg = "Instance labels must be a pandas Series."
            raise ValueError(msg)
        if len(self.instance_labels) != features.shape[0]:
            msg = "Instance labels must match the metadata rows."
            raise ValueError(msg)
        if self.instance_sources is not None:
            if not isinstance(self.instance_sources, pd.Series):
                msg = "Instance sources must be a pandas Series or None."
                raise ValueError(msg)
            if len(self.instance_sources) != features.shape[0]:
                msg = "Instance sources must match the metadata rows."
                raise ValueError(msg)

    @staticmethod
    def from_data_frame(data: DataFrame) -> Metadata:
        """Parse metadata from a file, and construct a Metadata object.

        Args
        ----------
        data
            The content of a csv file containing the metadata.

        Returns
        -------
        Metadata
            A Metadata object.
        """
        column_names = _validate_column_headers(data.columns.tolist())
        lowered_names = pd.Index(name.casefold() for name in column_names)
        is_name = lowered_names == "instances"
        is_feat = lowered_names.str.startswith("feature_")
        is_algo = lowered_names.str.startswith("algo_")
        is_source = lowered_names == "source"

        instance_labels = data.loc[:, is_name].iloc[:, 0].copy()

        if pd.api.types.is_numeric_dtype(instance_labels):
            instance_labels = instance_labels.astype(str)

        source_column = None
        if is_source.any():
            source_column = data.loc[:, is_source].iloc[:, 0].copy()

        features_raw = data.loc[:, is_feat]
        algo_raw = data.loc[:, is_algo]

        # Strip the "feature_"/"algo_" column-naming convention so labels,
        # graphs, and exported CSVs show the actual feature/algorithm name
        # (matching MATLAB), not the raw CSV column name.
        feature_names = [_remove_prefix(str(name), "feature_") for name in features_raw]
        algorithm_names = [_remove_prefix(str(name), "algo_") for name in algo_raw]

        features = _data_frame_to_matrix("feature", features_raw)
        algorithms = _data_frame_to_matrix("algorithm", algo_raw)

        return Metadata(
            feature_names=feature_names,
            algorithm_names=algorithm_names,
            features=features,
            algorithms=algorithms,
            instance_sources=source_column,
            instance_labels=instance_labels,
        )


def _remove_prefix(name: str, prefix: str) -> str:
    """Remove a known case-insensitive metadata prefix."""
    return name[len(prefix) :]


def _validate_column_headers(columns: Sequence[object]) -> list[str]:
    """Validate metadata column names before pandas can alter duplicates."""
    if not all(isinstance(column, str) for column in columns):
        msg = "Metadata column names must be strings."
        raise ValueError(msg)

    names = [str(column) for column in columns]
    lowered = [name.casefold() for name in names]
    instance_count = lowered.count("instances")
    source_count = lowered.count("source")
    if instance_count != 1:
        msg = "Metadata must contain exactly one 'instances' column."
        raise ValueError(msg)
    if source_count > 1:
        msg = "Metadata must contain at most one 'source' column."
        raise ValueError(msg)

    feature_names = [
        _remove_prefix(name, "feature_")
        for name, lowered_name in zip(names, lowered)
        if lowered_name.startswith("feature_")
    ]
    algorithm_names = [
        _remove_prefix(name, "algo_")
        for name, lowered_name in zip(names, lowered)
        if lowered_name.startswith("algo_")
    ]
    _validate_names("feature", feature_names, require_one=True)
    _validate_names("algorithm", algorithm_names, require_one=False)
    return names


def _validate_names(kind: str, names: list[str], *, require_one: bool) -> None:
    """Validate stripped feature or algorithm names."""
    if require_one and not names:
        msg = f"Metadata must contain at least one {kind} column."
        raise ValueError(msg)
    if not all(isinstance(name, str) and name for name in names):
        msg = f"Metadata {kind} names must be nonempty strings."
        raise ValueError(msg)

    normalized = [name.casefold() for name in names]
    duplicates = sorted({name for name in normalized if normalized.count(name) > 1})
    if duplicates:
        duplicate_text = ", ".join(duplicates)
        msg = (
            f"Metadata {kind} names must be unique after prefix removal. "
            f"Duplicates: {duplicate_text}."
        )
        raise ValueError(msg)


def _as_real_matrix(name: str, values: object) -> NDArray[np.double]:
    """Return a validated finite-or-missing real matrix."""
    array = np.asarray(values)
    is_real_numeric = np.issubdtype(array.dtype, np.number) and not np.issubdtype(
        array.dtype,
        np.complexfloating,
    )
    if array.dtype == np.bool_ or not is_real_numeric:
        msg = f"Metadata {name} must be numeric and non-Boolean."
        raise ValueError(msg)
    if array.ndim != MATRIX_DIMENSIONS:
        msg = f"Metadata {name} must be a two-dimensional matrix."
        raise ValueError(msg)

    matrix = array.astype(np.double, copy=False)
    if np.isinf(matrix).any():
        msg = f"Metadata {name} must contain only finite values or NaN."
        raise ValueError(msg)
    return matrix


def _data_frame_to_matrix(kind: str, data: DataFrame) -> NDArray[np.double]:
    """Convert validated metadata columns to a double matrix."""
    invalid_columns = [
        str(column)
        for column in data.columns
        if pd.api.types.is_bool_dtype(data[column].dtype)
        or not pd.api.types.is_numeric_dtype(data[column].dtype)
    ]
    if invalid_columns:
        invalid_text = ", ".join(invalid_columns)
        msg = (
            f"Metadata {kind} columns must be numeric and non-Boolean: {invalid_text}."
        )
        raise ValueError(msg)
    matrix = data.to_numpy(dtype=np.double, na_value=np.nan)
    if np.isinf(matrix).any():
        msg = f"Metadata {kind} columns must contain only finite values or NaN."
        raise ValueError(msg)
    return matrix


def from_csv_file(file_path: Path | str) -> Metadata | None:
    """Parse metadata from a CSV file and construct a Metadata object.

    Args
    ----------
    file_path : Path | str
        The path to the CSV file containing the metadata.

    Returns
    -------
    Metadata or None
        A Metadata object constructed from the parsed CSV data, or None if an
        error occurred during file reading or parsing.

    Raises
    ------
    FileNotFoundError
        If the specified file does not exist.
    pandas.errors.EmptyDataError
        If the specified file is empty.
    pandas.errors.ParserError
        If the specified file is not a valid CSV file.
    """
    if isinstance(file_path, str):
        file_path = Path(file_path)

    try:
        with file_path.open(encoding="utf-8-sig", newline="") as csv_file:
            header = next(csv.reader(csv_file))
        _validate_column_headers(header)
        csv_df = pd.read_csv(file_path)
        return Metadata.from_data_frame(csv_df)
    except (
        FileNotFoundError,
        OSError,
        csv.Error,
        pd.errors.ParserError,
        ValueError,
    ) as e:
        logger.error(f"{file_path}: {e!s}")
        return None
    except (pd.errors.EmptyDataError, StopIteration) as err:
        logger.error(f"{file_path}: {err!s}")
        logger.error(f"The file '{file_path}' is empty.")
        return None
