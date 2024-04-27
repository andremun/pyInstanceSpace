"""
Contains modules for instance space analysis.

The module consists of various algorithms to perform instance space analysis.
    - build: Perform instance space analysis on given dataset and configuration.
    - prelim: Performing preliminary data processing.
    - sifted: Perform feature selection and optimization in data analysis.
    - pythia: Perform algorithm selection and performance evaluation using SVM.
    - cloister: Perform correlation analysis to estimate a boundary for the space.
    - pilot: Obtaining a two-dimensional projection.
    - trace: Calculating the algorithm footprints.
    - example: IPython Notebook to run analysis on local machine

Perform instance space analysis on given dataset and configuration.

Construct an instance space from data and configuration files located in a specified
directory. The instance space is represented as a Model object, which encapsulates the
analytical results and metadata of the instance space analysis.

The main function in this module, `build_instance_space`, reads the necessary
data from the provided directory, performs instance space analysis, and then
constructs a Model object that represents this analysis. This Model object can
then be used for further analysis, visualization, or processing within the
larger framework of the Matilda data analysis suite.

Functions:
    build_instance_space(rootdir: str) -> Model:
        Construct and return a Model object after performing instance space analysis
        on the data and configurations found in the specified root directory.

Example usage:
    python your_module_name.py /path/to/data
"""

from __future__ import annotations

from collections import defaultdict
from enum import Enum
from pathlib import Path

from matilda.cloister import Cloister
from matilda.data.metadata import Metadata
from matilda.data.model import (
    CloisterOut,
    Data,
    Model,
    PilotOut,
    PrelimOut,
    PythiaOut,
    SiftedOut,
    TraceOut,
)
from matilda.data.option import Options
from matilda.pilot import Pilot
from matilda.prelim import Prelim
from matilda.pythia import Pythia
from matilda.sifted import Sifted
from matilda.trace import Trace


class _Stage(Enum):
    PRELIM = "prelim"
    SIFTED = "sifted"
    PILOT = "pilot"
    CLOISTER = "cloister"
    TRACE = "trace"
    PYTHIA = "pythia"

class StageError(Exception):
    """
    Prerequisite stages haven't been ran.

    An error raised when a user attempts to run a stage without first running any
    prerequisite stages.
    """

    pass

class InstanceSpace:
    """TODO: Describe what an instance space IS."""

    _stages: dict[_Stage, bool]
    _metadata: Metadata
    _options: Options

    _data: Data | None

    _prelim_out: PrelimOut | None
    _sifted_out: SiftedOut | None
    _pilot_out: PilotOut | None
    _cloister_out: CloisterOut | None
    _trace_out: TraceOut | None
    _pythia_out: PythiaOut | None

    _model: Model | None


    def __init__(self, metadata: Metadata, options: Options) -> None:
        """
        Create a new InstanceSpace object.

        TODO: Fill in the docstring here. This will be the most enduser visible version
        of this so it needs to be informative.

        Args:
        ----
            metadata (Metadata): _description_
            options (Options): _description_

        """
        self._stages = defaultdict(lambda: False)
        self._metadata = metadata
        self._options = options

        _data = None

        _prelim_out = None
        _sifted_out = None
        _pilot_out = None
        _cloister_out = None
        _trace_out = None
        _pythia_out = None

        _model = None


    def build(self) -> Model:
        """
        Construct and return a Model object after instance space analysis.

        This runs all stages.

        TODO: Fill in the docstring here. This will be the most enduser visible version
        of this so it needs to be informative.

        Returns:
        -------
            model: A Model object representing the built instance space.

        """
        raise NotImplementedError


    def prelim(self) -> PrelimOut:
        """
        Run the prelim stage.

        TODO: Fill in the docstring here. This will be the most enduser visible version
        of this so it needs to be informative.

        Returns:
        -------
            prelim_out: The return of the prelim stage.

        """
        self._stages[_Stage.PRELIM] = True

        self._data, self._prelim_out = Prelim.run(
            self._metadata.features,
            self._metadata.algorithms,
            self._options,
        )

        return self._prelim_out


    def sifted(self) -> SiftedOut:
        """
        Run the sifted stage.

        TODO: Fill in the docstring here. This will be the most enduser visible version
        of this so it needs to be informative.

        Returns:
        -------
            sifted_out: The return of the sifted stage.

        """
        if not self._stages[_Stage.PRELIM] or self._data is None:
            raise StageError

        self._stages[_Stage.SIFTED] = True

        something, self._sifted_out = Sifted.run(
            self._data.x,
            self._data.y,
            self._data.y_bin,
            self._options.sifted,
        )

        return self._sifted_out


    def pilot(self) -> PilotOut:
        """
        Run the pilot stage.

        TODO: Fill in the docstring here. This will be the most enduser visible version
        of this so it needs to be informative.

        Returns:
        -------
            pilot_out: The return of the pilot stage.

        """
        if not self._stages[_Stage.SIFTED] or self._data is None:
            raise StageError

        self._stages[_Stage.PILOT] = True

        self._pilot_out = Pilot.run(
            self._data.x,
            self._data.y,
            self._data.feat_labels,
            self._options.pilot,
        )

        return self._pilot_out


    def cloister(self) -> CloisterOut:
        """
        Run the cloister stage.

        TODO: Fill in the docstring here. This will be the most enduser visible version
        of this so it needs to be informative.

        Returns:
        -------
            cloister_out: The return of the cloister stage.

        """
        if (
            not self._stages[_Stage.PILOT] or self._data is None
            or self._pilot_out is None
        ):
            raise StageError

        self._stages[_Stage.CLOISTER] = True

        self._cloister_out = Cloister.run(
            self._data.x,
            self._pilot_out.a,
            self._options,
        )

        return self._cloister_out


    def trace(self) -> TraceOut:
        """
        Run the trace stage.

        TODO: Fill in the docstring here. This will be the most enduser visible version
        of this so it needs to be informative.

        Returns:
        -------
            trace_out: The return of the trace stage.

        """
        if (
            not self._stages[_Stage.PILOT] or self._data is None
            or self._pilot_out is None
        ):
            raise StageError

        self._stages[_Stage.TRACE] = True

        self._trace_out = Trace.run(
            self._pilot_out.z,
            self._data.y_bin,
            self._data.p,
            self._data.beta,
            self._data.algo_labels,
            self._options.trace,
        )

        return self._trace_out


    def pythia(self) -> PythiaOut:
        """
        Run the pythia stage.

        TODO: Fill in the docstring here. This will be the most enduser visible version
        of this so it needs to be informative.

        Returns:
        -------
            pythia_out: The return of the pythia stage.

        """
        if (
            not self._stages[_Stage.PILOT] or self._data is None
            or self._pilot_out is None
        ):
            raise StageError

        self._stages[_Stage.PYTHIA] = True

        self._pythia_out = Pythia.run(
            self._pilot_out.z,
            self._data.y_raw,
            self._data.y_bin,
            self._data.y_best,
            self._data.algo_labels,
            self._options,
        )

        return self._pythia_out


def from_files(metadata_filepath: Path, options_filepath: Path) -> InstanceSpace:
    """
    Construct an instance space object from 2 files.

    Args:
    ----
        metadata_filepath (Path): Path to the metadata csv file.
        options_filepath (Path): Path to the options json file.

    Returns:
    -------
        instance_space: A new instance space object instantiated with metadata and
        options from the specified files.

    """
    metadata = Metadata.from_file(metadata_filepath)
    options = Options.from_file(options_filepath)

    return InstanceSpace(metadata, options)


def from_directory(directory: Path) -> InstanceSpace:
    """
    Construct an instance space object from 2 files.

    Args:
    ----
        directory (str): Path to correctly formatted directory.

    Returns:
    -------
        instance_space: A new instance space object instantiated with metadata and
        options from the specified directory.

    """
    metadata = Metadata.from_file(directory / "metadata.csv")
    options = Options.from_file(directory / "options.json")

    return InstanceSpace(metadata, options)
