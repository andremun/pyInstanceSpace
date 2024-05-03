"""TODO: document instance space module."""

from collections import defaultdict
from dataclasses import fields
from enum import Enum
from pathlib import Path

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
from matilda.stages.cloister import Cloister
from matilda.stages.pilot import Pilot
from matilda.stages.prelim import Prelim
from matilda.stages.pythia import Pythia
from matilda.stages.sifted import Sifted
from matilda.stages.trace import Trace


class _Stage(Enum):
    PRELIM = "prelim"
    SIFTED = "sifted"
    PILOT = "pilot"
    CLOISTER = "cloister"
    TRACE = "trace"
    PYTHIA = "pythia"


class StageError(Exception):
    """Prerequisite stages haven't been ran.

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
        """Create a new InstanceSpace object.

        TODO: Fill in the docstring here. This will be the most enduser visible version
        of this so it needs to be informative.

        Args
        ----
            metadata (Metadata): _description_
            options (Options): _description_
        """
        self._stages = defaultdict(lambda: False)
        self._metadata = metadata
        self._options = options

        self._data = None

        self._prelim_out = None
        self._sifted_out = None
        self._pilot_out = None
        self._cloister_out = None
        self._trace_out = None
        self._pythia_out = None

        self._model = None

    def build(self) -> Model:
        """Construct and return a Model object after instance space analysis.

        This runs all stages.

        TODO: Fill in the docstring here. This will be the most enduser visible version
        of this so it needs to be informative.

        Returns
        -------
            model: A Model object representing the built instance space.
        """
        raise NotImplementedError

    def prelim(self) -> PrelimOut:
        """Run the prelim stage.

        TODO: Fill in the docstring here. This will be the most enduser visible version
        of this so it needs to be informative.

        Returns
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
        """Run the sifted stage.

        TODO: Fill in the docstring here. This will be the most enduser visible version
        of this so it needs to be informative.

        Returns
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
        """Run the pilot stage.

        TODO: Fill in the docstring here. This will be the most enduser visible version
        of this so it needs to be informative.

        Returns
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
        """Run the cloister stage.

        TODO: Fill in the docstring here. This will be the most enduser visible version
        of this so it needs to be informative.

        Returns
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
        """Run the trace stage.

        TODO: Fill in the docstring here. This will be the most enduser visible version
        of this so it needs to be informative.

        Returns
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
        """Run the pythia stage.

        TODO: Fill in the docstring here. This will be the most enduser visible version
        of this so it needs to be informative.

        Returns
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

    def get_options(self) -> Options:
        """Get the options for test cases.

        Returns
        -------
        Options
            The options object associated with this instance space.
        """
        return self._options

    def get_metadata(self) -> Metadata:
        """Get the metadata for test cases.

        Returns
        -------
        Metadata
            The metadata object associated with this instance space.
        """
        return self._metadata


def instance_space_from_files(
    metadata_filepath: Path,
    options_filepath: Path,
) -> InstanceSpace:
    """Construct an instance space object from 2 files.

    Args
    ----
        metadata_filepath (Path): Path to the metadata csv file.
        options_filepath (Path): Path to the options json file.

    Returns
    -------
        instance_space: A new instance space object instantiated with metadata and
        options from the specified files.

    """
    if not metadata_filepath.is_file():
        raise FileNotFoundError(f"Please place the metadata.csv in the directory"
                                f" '{metadata_filepath.parent}'")
    print("-------------------------------------------------------------------------")
    print("-> Loading the data.")
    with metadata_filepath.open() as f:
        metadata_contents = f.read()
    metadata = metadata.from_file(metadata_contents)

    if not options_filepath.is_file():
        raise FileNotFoundError(f"Please place the options.json in the directory '"
                                f"{options_filepath.parent}'")
    with options_filepath.open() as o:
        options_contents = o.read()
    options = Options.from_file(options_contents)
    print("-------------------------------------------------------------------------")
    print("-> Listing options to be used:")
    for field_name in fields(Options):
        field_value = getattr(options, field_name.name)
        print(f"{field_name.name}: {field_value}")

    return InstanceSpace(metadata, options)


def instance_space_from_directory(directory: Path) -> InstanceSpace:
    """Construct an instance space object from 2 files.

    Args
    ----
        directory (str): Path to correctly formatted directory.

    Returns
    -------
        instance_space (InstanceSpace): A new instance space object instantiated with
        metadata and options from the specified directory.

    """
    metadata = Metadata.from_file(directory / "metadata.csv")

    options = Options.from_file()

    return InstanceSpace(metadata, options)
