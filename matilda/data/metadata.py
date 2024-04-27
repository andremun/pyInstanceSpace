"""
Defines data types for metadata.

These classes define types for problem instances found in the metadata.csv file.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Self

import numpy as np
import pandas as pd
from numpy.typing import NDArray


@dataclass
class Metadata:
    # TODO: Ask someone for a better description of what metadata is
    """Metadata for problem instances."""

    feature_names: list[str]
    algorithm_names: list[str]
    features: NDArray[np.double]
    algorithms: NDArray[np.double]
    # newly added:
    inst_labels: pd.Series
    s: pd.Series | None

    @staticmethod
    def from_file(filepath: Path) -> Metadata:
        """
        Parse metadata from a file, and construct a Metadata object.

        :param filepath: The path of a csv file containing the metadata.
        :return: A Metadata object.
        """
        if not filepath.is_file():
            raise FileNotFoundError(f"Please place the metadata.csv in the directory"
                                    f" '{filepath.parent}'")

        print("-------------------------------------------------------------------------")
        print("-> Loading the data.")
        xbar = pd.read_csv(filepath)

        varlabels = xbar.columns
        isname = varlabels.str.lower() == "instances"
        isfeat = varlabels.str.lower().str.startswith("feature_")
        isalgo = varlabels.str.lower().str.startswith("algo_")
        issource = varlabels.str.lower() == "source"

        instlabels = xbar.loc[:, isname].squeeze()

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
            s=s,
            inst_labels=instlabels,
        )

    def to_file(self: Self, filepath: Path) -> None:
        """
        Store metadata in a file from a Metadata object.

        :param filepath: The path of the resulting csv file containing the metadata.
        """
        raise NotImplementedError
