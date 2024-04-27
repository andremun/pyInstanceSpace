"""
Defines a collection of data classes that represent configuration options.

These classes provide a structured way to specify and manage settings for different
aspects of the model's execution and behaviour.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Self, TypeVar

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
class PrelimOptions:
    """Options for running PRELIM."""

    max_perf: bool
    abs_perf: bool
    epsilon: float
    beta_threshold: float
    bound: bool
    norm: bool


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
class Options:
    """Aggregates all options into a single configuration object for the model."""

    parallel: ParallelOptions | None
    perf: PerformanceOptions | None
    auto: AutoOptions | None
    bound: BoundOptions | None
    norm: NormOptions | None
    selvars: SelvarsOptions | None
    sifted: SiftedOptions | None
    pilot: PilotOptions | None
    cloister: CloisterOptions | None
    pythia: PythiaOptions | None
    trace: TraceOptions | None
    outputs: OutputOptions | None
    "we probably need to have this 'general' field"  # general: GeneralOptions


    @staticmethod
    def from_file(filepath: Path) -> Options:
        """
        Load configuration options from a JSON file into an Options object.

        This function reads a JSON file from `filepath`, checks for expected
        top-level fields as defined in Options, initializes each part of the
        Options with data from the file, and sets missing optional fields to None.

        :param filepath: Path to the JSON file with configuration options.
        :return: Options object populated with data from the file.
        :raises FileNotFoundError: If the JSON file is not found at filepath.
        :raises ValueError: If the JSON file contains undefined fields.
        """
        if not filepath.is_file():
            raise FileNotFoundError(f"Please place the options.json in the directory '"
                                    f"{filepath.parent}'")

        with Path.open(filepath) as file:
            opts_dict = json.load(file)

        # Validate if the top-level fields match those in the Options class
        options_fields = {f.name for f in fields(Options)}
        extra_fields = set(opts_dict.keys()) - options_fields
        if extra_fields:
            raise ValueError(f"Extra fields in JSON not defined in Options:"
                             f" {extra_fields}")

        # Initialize each part of Options
        options = Options(
            parallel=load_dataclass(ParallelOptions, opts_dict["parallel"])
            if "parallel" in opts_dict else None,
            perf=load_dataclass(PerformanceOptions, opts_dict["perf"])
            if "perf" in opts_dict else None,
            auto=load_dataclass(AutoOptions, opts_dict["auto"])
            if "auto" in opts_dict else None,
            bound=load_dataclass(BoundOptions, opts_dict["bound"])
            if "bound" in opts_dict else None,
            norm=load_dataclass(NormOptions, opts_dict["norm"])
            if "norm" in opts_dict else None,
            selvars=load_dataclass(SelvarsOptions, opts_dict["selvars"])
            if "selvars" in opts_dict else None,
            sifted=load_dataclass(SiftedOptions, opts_dict["sifted"])
            if "sifted" in opts_dict else None,
            pilot=load_dataclass(PilotOptions, opts_dict["pilot"])
            if "pilot" in opts_dict else None,
            cloister=load_dataclass(CloisterOptions, opts_dict["cloister"])
            if "cloister" in opts_dict else None,
            pythia=load_dataclass(PythiaOptions, opts_dict["pythia"])
            if "pythia" in opts_dict else None,
            trace=load_dataclass(TraceOptions, opts_dict["trace"])
            if "trace" in opts_dict else None,
            outputs=load_dataclass(OutputOptions, opts_dict["outputs"])
            if "outputs" in opts_dict else None,
        )

        print("-------------------------------------------------------------------------")
        print("-> Listing options to be used:")
        for field_name in fields(Options):
            field_value = getattr(options, field_name.name)
            print(f"{field_name.name}: {field_value}")

        return options

    def to_file(self: Self, filepath: Path) -> None:
        """
        Store options in a file from an Options object.

        :param filepath: The path of the resulting json file containing the options.
        """
        raise NotImplementedError


T = TypeVar("T", ParallelOptions, PerformanceOptions,
            AutoOptions, BoundOptions, NormOptions, SelvarsOptions,
            SiftedOptions, PilotOptions, CloisterOptions, PythiaOptions,
            TraceOptions, OutputOptions)


def validate_fields(data_class: type[T], data: dict) -> None:
    """
    Validate all keys in the provided dictionary are valid fields in dataclass.

    Raises an error if a key is found that is not a field of the dataclass.
    :param data_class: The dataclass type to validate against.
    :param data: The dictionary whose keys are to be validated.
    :raises ValueError: If an undefined field is found in the dictionary.
    """
    # Get all defined fields in the data class
    known_fields = {f.name for f in fields(data_class)}
    # Check if all fields in the JSON are defined in the data class
    for key in data:
        if key not in known_fields:
            raise ValueError(f"Field '{key}' in JSON is not defined in the dataclass '"
                             f"{data_class.__name__}'")


def load_dataclass(data_class: type[T], data: dict) -> T:
    """
    Load data into a dataclass from a dictionary.

    Ensures all dictionary keys match dataclass fields
    and fills in fields with available data or None.
    :param data_class: The dataclass type to populate.
    :param data: Dictionary containing data to load into the dataclass.
    :return: An instance of the dataclass populated with data.
    """
    validate_fields(data_class, data)
    # for every subfield, fill in the attribute with the content,
    # return None if can't find the attribute content in the JSON
    init_args = {f.name: data.get(f.name, None) for f in fields(data_class)}
    return data_class(**init_args)
