"""Defines data types for metadata.

These classes define types for problem instances found in the metadata.csv file.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pandas import DataFrame


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
        var_labels = data.columns
        is_name = var_labels.str.lower() == "instances"
        is_feat = var_labels.str.lower().str.startswith("feature_")
        is_algo = var_labels.str.lower().str.startswith("algo_")
        is_source = var_labels.str.lower() == "source"

        instance_labels = data.loc[:, is_name].squeeze()

        if pd.api.types.is_numeric_dtype(instance_labels):
            instance_labels = instance_labels.astype(str)

        source_column = None
        if is_source.any():
            source_column = data.loc[:, is_source].squeeze()

        features_raw = data.loc[:, is_feat]
        algo_raw = data.loc[:, is_algo]

        feature_names = features_raw.columns.tolist()
        algorithm_names = algo_raw.columns.tolist()

        return Metadata(
            feature_names=feature_names,
            algorithm_names=algorithm_names,
            features=features_raw.to_numpy(),
            algorithms=algo_raw.to_numpy(),
            instance_sources=source_column,
            instance_labels=instance_labels,
        )

    def to_file(self) -> str:
        """Store metadata in a file from a Metadata object.

        Returns
        -------
        The metadata object serialised into a string.
        """
        raise NotImplementedError


def from_csv_file(file_path: Path) -> Metadata | None:
    """Parse metadata from a CSV file and construct a Metadata object.

    Args
    ----------
    file_path : Path
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
    try:
        csv_df = pd.read_csv(file_path)
    except (FileNotFoundError, OSError, pd.errors.ParserError) as e:
        print(f"{file_path}: {e!s}")
        return None
    except pd.errors.EmptyDataError:
        print(f"The file '{file_path}' is empty.")
        return None

    return Metadata.from_data_frame(csv_df)
