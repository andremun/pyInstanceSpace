"""Defines a collection of data classes that represent configuration options.

These classes provide a structured way to specify and manage settings for different
aspects of the model's execution and behaviour.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Self, TypeVar

import pandas as pd

from matilda.data.default_options import (
    DEFAULT_AUTO_PREPROC,
    DEFAULT_BOUND_FLAG,
    DEFAULT_CLOISTER_C_THRES,
    DEFAULT_CLOISTER_P_VAL,
    DEFAULT_NORM_FLAG,
    DEFAULT_OUTPUTS_CSV,
    DEFAULT_OUTPUTS_PNG,
    DEFAULT_OUTPUTS_WEB,
    DEFAULT_PARALLEL_FLAG,
    DEFAULT_PARALLEL_N_CORES,
    DEFAULT_PERFORMANCE_ABS_PERF,
    DEFAULT_PERFORMANCE_BETA_THRESHOLD,
    DEFAULT_PERFORMANCE_EPSILON,
    DEFAULT_PERFORMANCE_MAX_PERF,
    DEFAULT_PILOT_ANALYTICS,
    DEFAULT_PILOT_N_TRIES,
    DEFAULT_PYTHIA_CV_FOLDS,
    DEFAULT_PYTHIA_IS_POLY_KRNL,
    DEFAULT_PYTHIA_USE_LIB_SVM,
    DEFAULT_PYTHIA_USE_WEIGHTS,
    DEFAULT_SELVARS_DENSITY_FLAG,
    DEFAULT_SELVARS_FILE_IDX,
    DEFAULT_SELVARS_FILE_IDX_FLAG,
    DEFAULT_SELVARS_MIN_DISTANCE,
    DEFAULT_SELVARS_SMALL_SCALE,
    DEFAULT_SELVARS_SMALL_SCALE_FLAG,
    DEFAULT_SELVARS_TYPE,
    DEFAULT_SIFTED_FLAG,
    DEFAULT_SIFTED_K,
    DEFAULT_SIFTED_MAX_ITER,
    DEFAULT_SIFTED_NTREES,
    DEFAULT_SIFTED_REPLICATES,
    DEFAULT_SIFTED_RHO,
    DEFAULT_TRACE_PI,
    DEFAULT_TRACE_USE_SIM,
)


class MissingOptionsError(Exception):
    """A required option wasn't set.

    An error raised when a stage is ran that requires an option to be set, and the
    option isn't present.
    """

    pass

@dataclass(frozen=True)
class ParallelOptions:
    """Configuration options for parallel computing."""

    flag: bool
    n_cores: int

    @staticmethod
    def default(
        flag: bool = DEFAULT_PARALLEL_FLAG,
        n_cores: int = DEFAULT_PARALLEL_N_CORES,
    ) -> ParallelOptions:
        """Instantiate with default values."""
        return ParallelOptions(
            flag=flag,
            n_cores=n_cores,
        )


@dataclass(frozen=True)
class PerformanceOptions:
    """Options related to performance thresholds and criteria for model evaluation."""

    max_perf: bool
    abs_perf: bool
    epsilon: float
    beta_threshold: float

    @staticmethod
    def default(
        max_perf: bool = DEFAULT_PERFORMANCE_MAX_PERF,
        abs_perf: bool = DEFAULT_PERFORMANCE_ABS_PERF,
        epsilon: float = DEFAULT_PERFORMANCE_EPSILON,
        beta_threshold: float = DEFAULT_PERFORMANCE_BETA_THRESHOLD,
    ) -> PerformanceOptions:
        """Instantiate with default values."""
        return PerformanceOptions(
            max_perf=max_perf,
            abs_perf=abs_perf,
            epsilon=epsilon,
            beta_threshold=beta_threshold,
        )


@dataclass(frozen=True)
class AutoOptions:
    """Options for automatic processing steps in the model pipeline."""

    preproc: bool

    @staticmethod
    def default(
        preproc: bool = DEFAULT_AUTO_PREPROC,
    ) -> AutoOptions:
        """Instantiate with default values."""
        return AutoOptions(
            preproc=preproc,
        )


@dataclass(frozen=True)
class BoundOptions:
    """Options for applying bounds in the model calculations or evaluations."""

    flag: bool

    @staticmethod
    def default(
        flag: bool = DEFAULT_BOUND_FLAG,
    ) -> BoundOptions:
        """Instantiate with default values."""
        return BoundOptions(
            flag=flag,
        )


@dataclass(frozen=True)
class NormOptions:
    """Options to control normalization processes within the model."""

    flag: bool

    @staticmethod
    def default(
        flag: bool = DEFAULT_NORM_FLAG,
    ) -> NormOptions:
        """Instantiate with default values."""
        return NormOptions(
            flag=flag,
        )


@dataclass(frozen=True)
class SelvarsOptions:
    """Options for selecting variables, including criteria and file indices."""

    small_scale_flag: bool
    small_scale: float
    file_idx_flag: bool
    file_idx: str
    feats: pd.DataFrame | None
    algos: pd.DataFrame | None
    selvars_type: str
    min_distance: float
    density_flag: bool

    @staticmethod
    def default(
        small_scale_flag: bool = DEFAULT_SELVARS_SMALL_SCALE_FLAG,
        small_scale: float = DEFAULT_SELVARS_SMALL_SCALE,
        file_idx_flag: bool = DEFAULT_SELVARS_FILE_IDX_FLAG,
        file_idx: str = DEFAULT_SELVARS_FILE_IDX,
        feats: pd.DataFrame | None = None,
        algos: pd.DataFrame | None = None,
        selvars_type: str = DEFAULT_SELVARS_TYPE,
        min_distance: float = DEFAULT_SELVARS_MIN_DISTANCE,
        density_flag: bool = DEFAULT_SELVARS_DENSITY_FLAG,
    ) -> SelvarsOptions:
        """Instantiate with default values."""
        return SelvarsOptions(
            small_scale_flag=small_scale_flag,
            small_scale=small_scale,
            file_idx_flag=file_idx_flag,
            file_idx=file_idx,
            feats=feats,
            algos=algos,
            selvars_type=selvars_type,
            min_distance=min_distance,
            density_flag=density_flag,
        )


@dataclass(frozen=True)
class SiftedOptions:
    """Options specific to the sifting process in data analysis."""

    flag: bool
    rho: float
    k: int
    n_trees: int
    max_iter: int
    replicates: int

    @staticmethod
    def default(
        flag: bool = DEFAULT_SIFTED_FLAG,
        rho: float = DEFAULT_SIFTED_RHO,
        k: int = DEFAULT_SIFTED_K,
        n_trees: int = DEFAULT_SIFTED_NTREES,
        max_iter: int = DEFAULT_SIFTED_MAX_ITER,
        replicates: int = DEFAULT_SIFTED_REPLICATES,
    ) -> SiftedOptions:
        """Instantiate with default values."""
        return SiftedOptions(
            flag=flag,
            rho=rho,
            k=k,
            n_trees=n_trees,
            max_iter=max_iter,
            replicates=replicates,
        )


@dataclass(frozen=True)
class PilotOptions:
    """Options for pilot studies or preliminary analysis phases."""

    analytic: bool
    n_tries: int

    @staticmethod
    def default(
        analytic: bool = DEFAULT_PILOT_ANALYTICS,
        n_tries: int = DEFAULT_PILOT_N_TRIES,
    ) -> PilotOptions:
        """Instantiate with default values."""
        return PilotOptions(
            analytic=analytic,
            n_tries=n_tries,
        )


@dataclass(frozen=True)
class CloisterOptions:
    """Options for cloistering in the model."""

    p_val: float
    c_thres: float

    @staticmethod
    def default(
        p_val: float = DEFAULT_CLOISTER_P_VAL,
        c_thres: float = DEFAULT_CLOISTER_C_THRES,
    ) -> CloisterOptions:
        """Instantiate with default values."""
        return CloisterOptions(
            p_val=p_val,
            c_thres=c_thres,
        )


@dataclass(frozen=True)
class PythiaOptions:
    """Configuration for the Pythia component of the model."""

    cv_folds: int
    is_poly_krnl: bool
    use_weights: bool
    use_lib_svm: bool

    @staticmethod
    def default(
        cv_folds: int = DEFAULT_PYTHIA_CV_FOLDS,
        is_poly_krnl: bool = DEFAULT_PYTHIA_IS_POLY_KRNL,
        use_weights: bool = DEFAULT_PYTHIA_USE_WEIGHTS,
        use_lib_svm: bool = DEFAULT_PYTHIA_USE_LIB_SVM,
    ) -> PythiaOptions:
        """Instantiate with default values."""
        return PythiaOptions(
            cv_folds=cv_folds,
            is_poly_krnl=is_poly_krnl,
            use_weights=use_weights,
            use_lib_svm=use_lib_svm,
        )


@dataclass(frozen=True)
class TraceOptions:
    """Options for trace analysis in the model."""

    use_sim: bool
    pi: float

    @staticmethod
    def default(
        use_sim: bool = DEFAULT_TRACE_USE_SIM,
        pi: float = DEFAULT_TRACE_PI,
    ) -> TraceOptions:
        """Instantiate with default values."""
        return TraceOptions(
            use_sim=use_sim,
            pi=pi,
        )


@dataclass(frozen=True)
class OutputOptions:
    """Options for controlling the output format."""

    csv: bool
    web: bool
    png: bool

    @staticmethod
    def default(
        csv: bool = DEFAULT_OUTPUTS_CSV,
        web: bool = DEFAULT_OUTPUTS_WEB,
        png: bool = DEFAULT_OUTPUTS_PNG,
    ) -> OutputOptions:
        """Instantiate with default values."""
        return OutputOptions(
            csv=csv,
            web=web,
            png=png,
        )


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

    @staticmethod
    def from_dict(file_contents: dict[str, Any]) -> Options:
        """Load configuration options from a JSON file into an Options object.

        This function reads a JSON file from `filepath`, checks for expected
        top-level fields as defined in Options, initializes each part of the
        Options with data from the file, and sets missing optional fields using
        their default values.

        Args
        ----------
        file_contents
            Content of the dict with configuration options.

        Returns
        -------
        Options
            Options object populated with data from the file.

        Raises
        ------
        ValueError
            If the JSON file contains undefined sub options.
        """
        # Validate if the top-level fields match those in the Options class
        options_fields = {f.name for f in fields(Options)}
        extra_fields = set(file_contents.keys()) - options_fields

        if extra_fields:
            raise ValueError(f"Extra fields in JSON are not defined in Options:"
                             f" {extra_fields}")

        # Initialize each part of Options, using default values for missing fields
        return Options(
            parallel=Options._load_dataclass(ParallelOptions,
                                             file_contents.get("parallel", {})),
            perf=Options._load_dataclass(PerformanceOptions,
                                         file_contents.get("perf", {})),
            auto=Options._load_dataclass(AutoOptions, file_contents.get("auto", {})),
            bound=Options._load_dataclass(BoundOptions, file_contents.get("bound", {})),
            norm=Options._load_dataclass(NormOptions, file_contents.get("norm", {})),
            selvars=Options._load_dataclass(SelvarsOptions,
                                            file_contents.get("selvars", {})),
            sifted=Options._load_dataclass(SiftedOptions,
                                           file_contents.get("sifted", {})),
            pilot=Options._load_dataclass(PilotOptions, file_contents.get("pilot", {})),
            cloister=Options._load_dataclass(CloisterOptions,
                                             file_contents.get("cloister", {})),
            pythia=Options._load_dataclass(PythiaOptions,
                                           file_contents.get("pythia", {})),
            trace=Options._load_dataclass(TraceOptions, file_contents.get("trace", {})),
            outputs=Options._load_dataclass(OutputOptions,
                                            file_contents.get("outputs", {})),
        )

    def to_file(self: Self, filepath: Path) -> None:
        """Store options in a file from an Options object.

        Returns
        -------
        The options object serialised into a string.
        """
        raise NotImplementedError

    @staticmethod
    def default(
        parallel: ParallelOptions | None,
        perf: PerformanceOptions | None,
        auto: AutoOptions | None,
        bound: BoundOptions | None,
        norm: NormOptions | None,
        selvars: SelvarsOptions | None,
        sifted: SiftedOptions | None,
        pilot: PilotOptions | None,
        cloister: CloisterOptions | None,
        pythia: PythiaOptions | None,
        trace: TraceOptions | None,
        outputs: OutputOptions | None,
    ) -> Options:
        """Instantiate with default values."""
        return Options(
            parallel= parallel or ParallelOptions.default(),
            perf= perf or PerformanceOptions.default(),
            auto= auto or AutoOptions.default(),
            bound= bound or BoundOptions.default(),
            norm= norm or NormOptions.default(),
            selvars= selvars or SelvarsOptions.default(),
            sifted= sifted or SiftedOptions.default(),
            pilot= pilot or PilotOptions.default(),
            cloister= cloister or CloisterOptions.default(),
            pythia= pythia or PythiaOptions.default(),
            trace= trace or TraceOptions.default(),
            outputs= outputs or OutputOptions.default(),
        )

    T = TypeVar("T", ParallelOptions, PerformanceOptions,
                AutoOptions, BoundOptions, NormOptions, SelvarsOptions,
                SiftedOptions, PilotOptions, CloisterOptions, PythiaOptions,
                TraceOptions, OutputOptions)

    @staticmethod
    def _validate_fields(data_class: type[T], data: dict[str, Any]) -> None:
        """Validate all keys in the provided dictionary are valid fields in dataclass.

        Args
        ----------
        data_class : type[T]
            The dataclass type to validate against.
        data : dict
            The dictionary whose keys are to be validated.

        Raises
        ------
        ValueError
            If an undefined field is found in the dictionary or

        """
        # Get all defined fields in the data class
        known_fields = {f.name for f in fields(data_class)}

        # Check if all fields in the JSON are defined in the data class
        extra_fields = set(data.keys()) - known_fields
        if extra_fields:
            raise ValueError(
                f"Field(s) '{extra_fields}' in JSON are not "
                f"defined in the data class '{data_class.__name__}'.")

    @staticmethod
    def _load_dataclass(data_class: type[T], data: dict[str, Any]) -> T:
        """Load data into a dataclass from a dictionary.

        Ensures all dictionary keys match dataclass fields and fills in fields
        with available data. If a field is missing in the dictionary, the default
        value from the dataclass is used.

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

        Raises
        ------
        ValueError
            If the dictionary contains keys that are not valid fields in the dataclass.
        """
        # Get the default values for the dataclass fields
        default_values = {
            f.name: getattr(data_class.default(), f.name) for f in fields(data_class)
        }

        Options._validate_fields(data_class, data)

        # Update the default values with the provided data
        init_args = {**default_values, **data}

        return data_class(**init_args)


# Options not part of the main Options class

@dataclass(frozen=True)
class PrelimOptions:
    """Options for running PRELIM."""

    max_perf: bool
    abs_perf: bool
    epsilon: float
    beta_threshold: float
    bound: bool
    norm: bool

    @staticmethod
    def from_options(options: Options) -> PrelimOptions:
        """Get a prelim options object from an existing Options object."""
        return PrelimOptions(
            max_perf=options.perf.max_perf,
            abs_perf=options.perf.abs_perf,
            epsilon=options.perf.epsilon,
            beta_threshold=options.perf.beta_threshold,
            bound=options.bound.flag,
            norm=options.norm.flag,
        )


def from_json_file(file_path: Path) -> Options | None:
    """Parse options from a JSON file and construct an Options object.

    Args
    ----
    file_path : Path
        The path to the JSON file containing the options.

    Returns
    -------
    Options or None
        An Options object constructed from the parsed JSON data, or None if an
        error occurred during file reading or parsing.

    Raises
    ------
    FileNotFoundError
        If the specified file does not exist.
    json.JSONDecodeError
        If the specified file contains invalid JSON.
    OSError
        If an I/O error occurred while reading the file.
    ValueError
        If the parsed JSON data contains invalid options.
    """
    try:
        with file_path.open() as o:
            options_contents = o.read()
        opts_dict = json.loads(options_contents)

        return Options.from_dict(opts_dict)
    except (FileNotFoundError, json.JSONDecodeError, OSError) as e:
        print(f"{file_path}: {e!s}")
        return None
    except ValueError as e:
        print(f"Error: Invalid options data in the file '{file_path}'.")
        print(f"Error details: {e!s}")
        return None
