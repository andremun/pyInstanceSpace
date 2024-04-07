"""
Defines data types for metadata.

These classes define types for problem instances found in the metadata.csv file.
"""

from dataclasses import dataclass


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
