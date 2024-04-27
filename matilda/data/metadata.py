"""
Defines data types for metadata.

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
    instance_sources: pd.Series
    features: NDArray[np.double]
    algorithms: NDArray[np.double]


    @staticmethod
    def from_file(filepath: Path) -> Metadata:
        """
        Parse metadata from a file, and construct a Metadata object.

        :param filepath: The path of a csv file containing the metadata.
        :return: A Metadata object.
        """
        raise NotImplementedError

    def to_file(self, filepath: Path) -> None:
        """
        Store metadata in a file from a Metadata object.

        :param filepath: The path of the resulting csv file containing the metadata.
        """
        raise NotImplementedError

