"""Defines data types for metadata.

These classes define types for problem instances found in the metadata.csv file.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

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
    def from_file(file_contents: Path) -> Metadata:
        """Parse metadata from a file, and construct a Metadata object.

        Parameters
        ----------
        file_contents
            The path of a csv file containing the metadata.

        Returns
        -------
        Metadata
            A Metadata object.
        """
        if not file_contents.is_file():
            raise FileNotFoundError(f"Please place the metadata.csv in the directory"
                                    f" '{file_contents.parent}'")

        print("-------------------------------------------------------------------------")
        print("-> Loading the data.")
        xbar = pd.read_csv(file_contents)

        varlabels = xbar.columns
        is_name = varlabels.str.lower() == "instances"
        isfeat = varlabels.str.lower().str.startswith("feature_")
        isalgo = varlabels.str.lower().str.startswith("algo_")
        issource = varlabels.str.lower() == "source"

        instlabels = xbar.loc[:, is_name].squeeze()

        if pd.api.types.is_numeric_dtype(instlabels):
            instlabels = instlabels.astype(str)

        s = None
        if issource.any():
            s = xbar.loc[:, issource].squeeze()

        x = xbar.loc[:, isfeat]
        y = xbar.loc[:, isalgo]

        feature_names = x.columns.tolist()
        algorithm_names = y.columns.tolist()

        return Metadata(
            feature_names=feature_names,
            algorithm_names=algorithm_names,
            features=x.to_numpy(),
            algorithms=y.to_numpy(),
            instance_sources=s,
            instance_labels=instlabels,
        )

    def to_file(self) -> str:
        """Store metadata in a file from a Metadata object.

        Returns
        -------
        The metadata object serialised into a string.
        """
        raise NotImplementedError
