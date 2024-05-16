"""Defines data types for metadata.

These classes define types for problem instances found in the metadata.csv file.
"""

from __future__ import annotations

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
    instance_labels: pd.Series # type: ignore[type-arg]
    instance_sources: pd.Series # type: ignore[type-arg]
    features: NDArray[np.double]
    algorithms: NDArray[np.double]


    @staticmethod
    def from_file(file_contents: str) -> Metadata:
        """Parse metadata from a file, and construct a Metadata object.

        Args
        ----
        file_contents (str): The contents of a csv file containing the metadata.

        Returns
        -------
        A Metadata object.
        """
        raise NotImplementedError

    def to_file(self) -> str:
        """Store metadata in a file from a Metadata object.

        Returns
        -------
        The metadata object serialised into a string.
        """
        raise NotImplementedError

