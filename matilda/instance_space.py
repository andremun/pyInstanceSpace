"""TODO: document instance space module."""

from collections import defaultdict
from dataclasses import fields
from enum import Enum
from pathlib import Path

from matilda._serializers import (
    save_instance_space_for_web,
    save_instance_space_to_csv,
)
from matilda.data.metadata import Metadata, from_csv_file
from matilda.data.model import (
    CloisterOut,
    Data,
    PilotOut,
    PrelimOut,
    PythiaOut,
    SiftedOut,
    StageState,
    TraceOut,
)
from matilda.data.options import (
    InstanceSpaceOptions,
    PrelimOptions,
    from_json_file,
)
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
    _options: InstanceSpaceOptions

    _data: Data | None

    _prelim_state: StageState[PrelimOut] | None
    _sifted_state: StageState[SiftedOut] | None
    _pilot_state: StageState[PilotOut] | None
    _cloister_state: StageState[CloisterOut] | None
    _trace_state: StageState[TraceOut] | None
    _pythia_state: StageState[PythiaOut] | None

    def __init__(self, metadata: Metadata, options: InstanceSpaceOptions) -> None:
        """Create a new InstanceSpace object.

        TODO: Fill in the docstring here. This will be the most enduser visible version
        of this so it needs to be informative.

        Args
        ----
            metadata (Metadata): _description_
            options (InstanceSpaceOptions): _description_
        """
        self._stages = defaultdict(lambda: False)
        self._metadata = metadata
        self._options = options

        self._data = None

        self._prelim_state = None
        self._sifted_state = None
        self._pilot_state = None
        self._cloister_state = None
        self._trace_state = None
        self._pythia_state = None

        self._model = None

    @property
    def metadata(self) -> Metadata:
        """Get metadata."""
        return self._metadata

    @property
    def options(self) -> InstanceSpaceOptions:
        """Get options."""
        return self._options

    def build(self) -> None:
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

        self._clear_stages_after_prelim()

        data_changed, prelim_out = Prelim.run(
            self._metadata.features,
            self._metadata.algorithms,
            PrelimOptions.from_options(self._options),
            self._options.selvars,
        )

        if self._data is None:
            raise NotImplementedError

        self._prelim_state = StageState[PrelimOut](
            data_changed.merge_with(self._data),
            prelim_out,
        )

        return prelim_out

    def _clear_stages_after_prelim(self) -> None:
        self._sifted_state = None
        self._stages[_Stage.SIFTED] = False
        self._clear_stages_after_sifted()

    def sifted(self) -> SiftedOut:
        """Run the sifted stage.

        TODO: Fill in the docstring here. This will be the most enduser visible version
        of this so it needs to be informative.

        Returns
        -------
            sifted_out: The return of the sifted stage.
        """
        if not self._stages[_Stage.PRELIM] or self._prelim_state is None:
            raise StageError

        self._stages[_Stage.SIFTED] = True

        self._clear_stages_after_sifted()

        data_changed, sifted_out = Sifted.run(
            self._prelim_state.data.x,
            self._prelim_state.data.y,
            self._prelim_state.data.y_bin,
            self._options.sifted,
        )

        self._sifted_state = StageState[SiftedOut](
            data_changed.merge_with(self._prelim_state.data),
            sifted_out,
        )

        return sifted_out

    def _clear_stages_after_sifted(self) -> None:
        self._pilot_state = None
        self._stages[_Stage.PILOT] = False
        self._clear_stages_after_pilot()

    def pilot(self) -> PilotOut:
        """Run the pilot stage.

        TODO: Fill in the docstring here. This will be the most enduser visible version
        of this so it needs to be informative.

        Returns
        -------
            pilot_out: The return of the pilot stage.
        """
        if not self._stages[_Stage.SIFTED] or self._sifted_state is None:
            raise StageError

        self._stages[_Stage.PILOT] = True

        self._clear_stages_after_pilot()

        data_changed, pilot_out = Pilot.run(
            self._sifted_state.data.x,
            self._sifted_state.data.y,
            self._sifted_state.data.feat_labels,
            self._options.pilot,
        )

        self._pilot_state = StageState[PilotOut](
            data_changed.merge_with(self._sifted_state.data),
            pilot_out,
        )

        return pilot_out

    def _clear_stages_after_pilot(self) -> None:
        self._cloister_state = None
        self._trace_state = None
        self._pythia_state = None
        self._stages[_Stage.CLOISTER] = False
        self._stages[_Stage.TRACE] = False
        self._stages[_Stage.PYTHIA] = False
        self._clear_stages_after_cloister()
        self._clear_stages_after_trace()
        self._clear_stages_after_pythia()

    def cloister(self) -> CloisterOut:
        """Run the cloister stage.

        TODO: Fill in the docstring here. This will be the most enduser visible version
        of this so it needs to be informative.

        Returns
        -------
            cloister_out: The return of the cloister stage.
        """
        if not self._stages[_Stage.PILOT] or self._pilot_state is None:
            raise StageError

        self._stages[_Stage.CLOISTER] = True

        self._clear_stages_after_cloister()

        data_changed, cloister_out = Cloister.run(
            self._pilot_state.data.x,
            self._pilot_state.out.a,
            self._options.cloister,
        )

        self._cloister_state = StageState[CloisterOut](
            data_changed.merge_with(self._pilot_state.data),
            cloister_out,
        )

        return cloister_out

    def _clear_stages_after_cloister(self) -> None:
        pass

    def trace(self) -> TraceOut:
        """Run the trace stage.

        TODO: Fill in the docstring here. This will be the most enduser visible version
        of this so it needs to be informative.

        Returns
        -------
            trace_out: The return of the trace stage.
        """
        if not self._stages[_Stage.PILOT] or self._pilot_state is None:
            raise StageError

        self._stages[_Stage.TRACE] = True

        self._clear_stages_after_trace()

        data_changed, trace_out = Trace.run(
            self._pilot_state.out.z,
            self._pilot_state.data.y_bin,
            self._pilot_state.data.p,
            self._pilot_state.data.beta,
            self._pilot_state.data.algo_labels,
            self._options.trace,
        )

        self._trace_state = StageState[TraceOut](
            data_changed.merge_with(self._pilot_state.data),
            trace_out,
        )

        return trace_out

    def _clear_stages_after_trace(self) -> None:
        pass

    def pythia(self) -> PythiaOut:
        """Run the pythia stage.

        TODO: Fill in the docstring here. This will be the most enduser visible version
        of this so it needs to be informative.

        Returns
        -------
            pythia_out: The return of the pythia stage.
        """
        if not self._stages[_Stage.PILOT] or self._pilot_state is None:
            raise StageError

        self._stages[_Stage.PYTHIA] = True

        self._clear_stages_after_pythia()

        data_changed, pythia_out = Pythia.run(
            self._pilot_state.out.z,
            self._pilot_state.data.y_raw,
            self._pilot_state.data.y_bin,
            self._pilot_state.data.y_best,
            self._pilot_state.data.algo_labels,
            self._options.pythia,
        )

        self._pythia_state = StageState[PythiaOut](
            data_changed.merge_with(self._pilot_state.data),
            pythia_out,
        )

        return pythia_out

    def _clear_stages_after_pythia(self) -> None:
        pass

    def save_to_csv(self, output_directory: Path) -> None:
        """Save csv outputs to a directory."""
        print(
            "=========================================================================",
        )
        print("-> Writing the data on CSV files for posterior analysis.")

        if (
            not self._stages[_Stage.CLOISTER]
            or not self._stages[_Stage.TRACE]
            or not self._stages[_Stage.PYTHIA]
        ):
            raise StageError

        if (
            self._trace_state is None
            or self._pilot_state is None
            or self._prelim_state is None
            or self._pythia_state is None
        ):
            raise StageError

        # TODO: Placeholder, need to work out how to get the most relevant data
        # Conductor branch would solve this, needs more thought
        if self._data is None:
            raise StageError

        if (
            self._trace_state is None
            or self._pilot_state is None
            or self._sifted_state is None
            or self._pythia_state is None
            or self._cloister_state is None
        ):
            raise StageError

        save_instance_space_to_csv(
            output_directory,
            self._data,
            self._sifted_state,
            self._trace_state,
            self._pilot_state,
            self._cloister_state,
            self._pythia_state,
        )

    def save_for_web(self, output_directory: Path) -> None:
        """Save csv outputs used for the web frontend to a directory."""
        print(
            "=========================================================================",
        )
        print("-> Writing the data for the web interface.")

        if (
            not self._stages[_Stage.CLOISTER]
            or not self._stages[_Stage.TRACE]
            or not self._stages[_Stage.PYTHIA]
            or not self._stages[_Stage.SIFTED]
        ):
            raise StageError

        if self._prelim_state is None or self._sifted_state is None:
            raise StageError

        # TODO: Placeholder, need to work out how to get the most relevant data
        # Conductor branch would solve this, needs more thought
        if self._data is None:
            raise StageError

        save_instance_space_for_web(
            output_directory,
            self._prelim_state,
            self._sifted_state,
        )


def instance_space_from_files(
    metadata_filepath: Path,
    options_filepath: Path,
) -> InstanceSpace | None:
    """Construct an instance space object from 2 files.

    Args
    ----
        metadata_filepath (Path): Path to the metadata csv file.
        options_filepath (Path): Path to the options json file.

    Returns
    -------
        InstanceSpace | None: A new instance space object instantiated
        with metadata and options from the specified files, or None
        if the initialization fails.

    """
    print("-------------------------------------------------------------------------")
    print("-> Loading the data.")

    metadata = from_csv_file(metadata_filepath)

    if metadata is None:
        print("Failed to initialize metadata")
        return None

    print("-> Successfully loaded the data.")
    print("-------------------------------------------------------------------------")
    print("-> Loading the options.")

    options = from_json_file(options_filepath)

    if options is None:
        print("Failed to initialize options")
        return None

    print("-> Successfully loaded the options.")

    print("-> Listing options to be used:")
    for field_name in fields(InstanceSpaceOptions):
        field_value = getattr(options, field_name.name)
        print(f"{field_name.name}: {field_value}")

    return InstanceSpace(metadata, options)


def instance_space_from_directory(directory: Path) -> InstanceSpace | None:
    """Construct an instance space object from 2 files.

    Args
    ----
        directory (str): Path to correctly formatted directory,
        where the .csv file is metadata.csv, and .json file is
        options.json

    Returns
    -------
        InstanceSpace | None: A new instance space
        object instantiated with metadata and options from
        the specified directory, or None if the initialization fails.

    """
    metadata_path = Path(directory / "metadata.csv")
    options_path = Path(directory / "options.json")

    return instance_space_from_files(metadata_path, options_path)
