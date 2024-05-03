"""Defines a collection of data classes that represent configuration options.

These classes provide a structured way to specify and manage settings for different
aspects of the model's execution and behaviour.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Self, TypeVar

import pandas as pd


@dataclass(frozen=True)
class ParallelOptions:
    """Configuration options for parallel computing."""

    flag: bool
    n_cores: int


@dataclass(frozen=True)
class PerformanceOptions:
    """Options related to performance thresholds and criteria for model evaluation."""

    max_perf: bool
    abs_perf: bool
    epsilon: float
    beta_threshold: float


@dataclass(frozen=True)
class PrelimOptions:
    """Options for running PRELIM."""

    max_perf: bool
    abs_perf: bool
    epsilon: float
    beta_threshold: float
    bound: bool
    norm: bool


@dataclass(frozen=True)
class AutoOptions:
    """Options for automatic processing steps in the model pipeline."""

    preproc: bool


@dataclass(frozen=True)
class BoundOptions:
    """Options for applying bounds in the model calculations or evaluations."""

    flag: bool


@dataclass(frozen=True)
class NormOptions:
    """Options to control normalization processes within the model."""

    flag: bool


@dataclass(frozen=True)
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


@dataclass(frozen=True)
class SiftedOptions:
    """Options specific to the sifting process in data analysis."""

    flag: bool
    rho: float
    k: int
    n_trees: int
    max_iter: int
    replicates: int


@dataclass(frozen=True)
class PilotOptions:
    """Options for pilot studies or preliminary analysis phases."""

    analytic: bool
    n_tries: int


@dataclass(frozen=True)
class CloisterOptions:
    """Options for cloistering in the model."""

    p_val: float
    c_thres: float


@dataclass(frozen=True)
class PythiaOptions:
    """Configuration for the Pythia component of the model."""

    cv_folds: int
    is_poly_krnl: bool
    use_weights: bool
    use_lib_svm: bool


@dataclass(frozen=True)
class TraceOptions:
    """Options for trace analysis in the model."""

    use_sim: bool
    PI: float


@dataclass(frozen=True)
class OutputOptions:
    """Options for controlling the output format."""

    csv: bool
    web: bool
    png: bool


@dataclass(frozen=True)
class GeneralOptions:
    """General options that affect the whole system."""

    beta_threshold: float


@dataclass(frozen=True)
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
    def from_file(file_contents: dict) -> Options:
        """Load configuration options from a JSON file into an Options object.

        This function reads a JSON file from `filepath`, checks for expected
        top-level fields as defined in Options, initializes each part of the
        Options with data from the file, and sets missing optional fields to None.

        Args
        ----------
        file_contents
            Content of the JSON file with configuration options.

        Returns
        -------
        unknown
            Options object populated with data from the file.

        Raises
        ------
        FileNotFoundError
            If the JSON file is not found at filepath.
        ValueError
            If the JSON file contains undefined fields.
        """

        # Validate if the top-level fields match those in the Options class
        options_fields = {f.name for f in fields(Options)}
        extra_fields = set(opts_dict.keys()) - options_fields
        if extra_fields:
            raise ValueError(f"Extra fields in JSON are not defined in Options:"
                             f" {extra_fields}")

        # Initialize each part of Options

        return Options(
            parallel=Options._load_dataclass(ParallelOptions, opts_dict["parallel"]),
            perf=Options._load_dataclass(PerformanceOptions, opts_dict["perf"]),
            auto=Options._load_dataclass(AutoOptions, opts_dict["auto"]),
            bound=Options._load_dataclass(BoundOptions, opts_dict["bound"]),
            norm=Options._load_dataclass(NormOptions, opts_dict["norm"]),
            selvars=Options._load_dataclass(SelvarsOptions, opts_dict["selvars"]),
            sifted=Options._load_dataclass(SiftedOptions, opts_dict["sifted"])
            if "sifted" in opts_dict else None,
            pilot=Options._load_dataclass(PilotOptions, opts_dict["pilot"])
            if "pilot" in opts_dict else None,
            cloister=Options._load_dataclass(CloisterOptions, opts_dict["cloister"])
            if "cloister" in opts_dict else None,
            pythia=Options._load_dataclass(PythiaOptions, opts_dict["pythia"])
            if "pythia" in opts_dict else None,
            trace=Options._load_dataclass(TraceOptions, opts_dict["trace"])
            if "trace" in opts_dict else None,
            outputs=Options._load_dataclass(OutputOptions, opts_dict["outputs"])
            if "outputs" in opts_dict else None,
            general=Options._load_dataclass(GeneralOptions, opts_dict["general"])
            if "general" in opts_dict else None,
        )

    def to_file(self: Self, filepath: Path) -> None:
        """Store options in a file from an Options object.

        Returns
        -------
        The options object serialised into a string.
        """
        raise NotImplementedError

    T = TypeVar("T", ParallelOptions, PerformanceOptions,
                AutoOptions, BoundOptions, NormOptions, SelvarsOptions,
                SiftedOptions, PilotOptions, CloisterOptions, PythiaOptions,
                TraceOptions, OutputOptions, GeneralOptions)

    @staticmethod
    def _validate_fields(data_class: type[T], data: dict) -> None:
        """
        Validate all keys in the provided dictionary are valid fields in dataclass.

        Args
        ----------
        data_class : type[T]
            The dataclass type to validate against.
        data : dict
            The dictionary whose keys are to be validated.

        Raises
        ------
        ValueError
            If an undefined field is found in the dictionary.
        """
        # Get all defined fields in the data class
        known_fields = {f.name for f in fields(data_class)}
        # Check if all fields in the JSON are defined in the data class
        for key in data:
            if key not in known_fields:
                raise ValueError(f"Field '{key}' in JSON is not defined "
                                 f"in the dataclass '{data_class.__name__}'")

    @staticmethod
    def _load_dataclass(data_class: type[T], data: dict) -> T:
        """Load data into a dataclass from a dictionary.

        Ensures all dictionary keys match dataclass fields and fills in fields
        with available data or None.

        Args
        ----------
        data_class : type[T]
            The dataclass type to populate.
        data : dict
            Dictionary containing data to load into the dataclass.

        Returns
        -------
        T
            An instance of the dataclass populated with data.
        """
        Options._validate_fields(data_class, data)
        # for every subfield, fill in the attribute with the content,
        # return None if can't find the attribute content in the JSON
        init_args = {f.name: data.get(f.name, None) for f in fields(data_class)}
        return data_class(**init_args)


opts_dict = json.loads(file_contents)