"""Defines data types for metadata.

These classes define types for problem instances found in the metadata.csv file.
"""

from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray


@dataclass(frozen=True)
class Metadata:
    # TODO: Ask someone for a better description of what metadata is
    """Metadata for problem instances."""

    feature_names: list[str]
    algorithm_names: list[str]
    instance_labels: pd.Series
    instance_sources: pd.Series | None
    features: NDArray[np.double]
    algorithms: NDArray[np.double]

    @staticmethod
    def from_file(file_contents: str) -> Metadata:
        """Parse metadata from a file, and construct a Metadata object.

        Args
        ----------
        file_contents
            The content of a csv file containing the metadata.

        Returns
        -------
        Metadata
            A Metadata object.
        """
        csv_df = pd.read_csv(io.StringIO(file_contents))

        var_labels = csv_df.columns
        is_name = var_labels.str.lower() == "instances"
        is_feat = var_labels.str.lower().str.startswith("feature_")
        is_algo = var_labels.str.lower().str.startswith("algo_")
        is_source = var_labels.str.lower() == "source"

        instance_labels = csv_df.loc[:, is_name].squeeze()

        if pd.api.types.is_numeric_dtype(instance_labels):
            instance_labels = instance_labels.astype(str)

        source_column = None
        if is_source.any():
            source_column = csv_df.loc[:, is_source].squeeze()

        features_raw = csv_df.loc[:, is_feat]
        algo_raw = csv_df.loc[:, is_algo]

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
