"""
Defines a collection of data classes that represent configuration options.

These classes provide a structured way to specify and manage settings for different
aspects of the model's execution and behaviour.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Self

import pandas as pd


@dataclass
class ParallelOptions:
    """Configuration options for parallel computing."""

    flag: bool
    n_cores: int


@dataclass
class PerformanceOptions:
    """Options related to performance thresholds and criteria for model evaluation."""

    max_perf: bool
    abs_perf: bool
    epsilon: float
    beta_threshold: float


@dataclass
class AutoOptions:
    """Options for automatic processing steps in the model pipeline."""

    preproc: bool


@dataclass
class BoundOptions:
    """Options for applying bounds in the model calculations or evaluations."""

    flag: bool


@dataclass
class NormOptions:
    """Options to control normalization processes within the model."""

    flag: bool


@dataclass
class SelvarsOptions:
    """Options for selecting variables, including criteria and file indices."""

    small_scale_flag: bool
    small_scale: float
    file_idx_flag: bool
    file_idx: str
    feats: pd.DataFrame
    algos: pd.DataFrame
    type: str
    min_distance: float
    density_flag: bool


@dataclass
class SiftedOptions:
    """Options specific to the sifting process in data analysis."""

    flag: bool
    rho: float
    k: int
    n_trees: int
    max_iter: int
    replicates: int


@dataclass
class PilotOptions:
    """Options for pilot studies or preliminary analysis phases."""

    analytic: bool
    n_tries: int


@dataclass
class CloisterOptions:
    """Options for cloistering in the model."""

    p_val: float
    c_thres: float


@dataclass
class PythiaOptions:
    """Configuration for the Pythia component of the model."""

    cv_folds: int
    is_poly_krnl: bool
    use_weights: bool
    use_lib_svm: bool


@dataclass
class TraceOptions:
    """Options for trace analysis in the model."""

    use_sim: bool
    PI: float


@dataclass
class OutputOptions:
    """Options for controlling the output format."""

    csv: bool
    web: bool
    png: bool


@dataclass
class GeneralOptions:
    """General options that affect the whole system."""

    beta_threshold: float


@dataclass
class Options:
    """Aggregates all options into a single configuration object for the model."""

    parallel: ParallelOptions
    perf: PerformanceOptions
    auto: AutoOptions
    bound: BoundOptions
    norm: NormOptions
    selvars: SelvarsOptions
    sifted: SiftedOptions
    pilot: PilotOptions
    cloister: CloisterOptions
    pythia: PythiaOptions
    trace: TraceOptions
    outputs: OutputOptions
    general: GeneralOptions


    @staticmethod
    def from_file(filepath: Path) -> Options:
        """
        Parse options from a file, and construct an Options object.

        :param filepath: The path of a json file containing the options.
        :return: An Options object.
        """
        raise NotImplementedError

    def to_file(self: Self, filepath: Path) -> None:
        """
        Store options in a file from an Options object.

        :param filepath: The path of the resulting json file containing the options.
        """
        raise NotImplementedError
