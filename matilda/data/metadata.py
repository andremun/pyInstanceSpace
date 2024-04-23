"""
Defines data types for metadata.

These classes define types for problem instances found in the metadata.csv file.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Self


@dataclass
class ProblemInstance:
    # TODO: Ask someone for a better description of what a problem instance is
    """A description of a problem instance."""

    identifier: str
    source: str | None

    # Features and algorithm performance should have type double. Pythons float type
    # has double precision. An alternative would be to use numpy's double128.
    features: dict[str, float]
    algorithms: dict[str, float]

class Metadata:
    # TODO: Ask someone for a better description of what metadata is
    """Metadata for problem instances."""

    feature_names: list[str]
    algorithm_names: list[str]
    problem_instances: list[ProblemInstance]


    @staticmethod
    def from_file(filepath: str) -> Metadata:
        """
        Parse metadata from a file, and construct a Metadata object.

        :param filepath: The path of a csv file containing the metadata.
        :return: A Metadata object.
        """
        raise NotImplementedError

    def to_file(self: Self, filepath: str) -> None:
        """
        Store metadata in a file from a Metadata object.

        :param filepath: The path of the resulting csv file containing the metadata.
        """
        raise NotImplementedError

