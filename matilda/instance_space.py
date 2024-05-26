"""TODO: document instance space module."""
from collections import defaultdict
import csv
from dataclasses import fields
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

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
from matilda.data.option import Options, PrelimOptions, from_json_file
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

    _prelim_state: StageState[PrelimOut] | None
    _sifted_state: StageState[SiftedOut] | None
    _pilot_state: StageState[PilotOut] | None
    _cloister_state: StageState[CloisterOut] | None
    _trace_state: StageState[TraceOut] | None
    _pythia_state: StageState[PythiaOut] | None

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
    def options(self) -> Options:
        """Get options."""
        return self._options

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

        self._clear_stages_after_prelim()

        data_changed, prelim_out = Prelim.run(
            self._metadata.features,
            self._metadata.algorithms,
            PrelimOptions.from_options(self._options),
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

    def save_csv(self, output_directory: Path) -> None:
        """Save csv outputs to a directory."""
        print("=========================================================================")
        print("-> Writing the data on CSV files for posterior analysis.")

        if not output_directory.is_dir():
            raise ValueError("output_directory isn't a directory.")

        if (
            not self._stages[_Stage.CLOISTER] or
            not self._stages[_Stage.TRACE] or
            not self._stages[_Stage.PYTHIA]
        ):
            raise StageError

        # TODO: Placeholder, need to work out how to get the most relevant data
        # Conductor branch would solve this, needs more thought
        if self._data is None:
            raise StageError

        if (
            self._trace_state is None or
            self._pilot_state is None or
            self._prelim_state is None or
            self._pythia_state is None
        ):
            raise StageError

        num_algorithms = self._data.y.shape[1]

        for i in range(num_algorithms):

            best = self._trace_state.out.best[i]
            if best is not None and best.polygon is not None:
                best = self._trace_state.out.best[i]
                algorithm_labels = self._trace_state.data.algo_labels[i]
                InstanceSpace._write_array_to_csv(
                    best.polygon.vertices,
                    pd.Series(["z_1", "z_2"]),
                    InstanceSpace._make_bind_labels(best.polygon.vertices),
                    output_directory / f"footprint_{algorithm_labels}_good.csv",
                )

            good = self._trace_state.out.good[i]
            if good is not None and good.polygon is not None:
                algorithm_labels = self._trace_state.data.algo_labels[i]
                InstanceSpace._write_array_to_csv(
                    good.polygon.vertices,
                    pd.Series(["z_1", "z_2"]),
                    InstanceSpace._make_bind_labels(good.polygon.vertices),
                    output_directory / f"footprint_{algorithm_labels}_good.csv",
                )

        InstanceSpace._write_array_to_csv(
            self._pilot_state.out.z,
            pd.Series(["z_1", "z_2"]),
            self._pilot_state.data.inst_labels,
            output_directory / "coordinates.csv",
        )

        if self._cloister_state is not None:
            InstanceSpace._write_array_to_csv(
                self._cloister_state.out.z_edge,
                pd.Series(["z_1", "z_2"]),
                InstanceSpace._make_bind_labels(self._cloister_state.out.z_edge),
                output_directory / "bounds.csv",
            )
            InstanceSpace._write_array_to_csv(
                self._cloister_state.out.z_ecorr,
                pd.Series(["z_1", "z_2"]),
                InstanceSpace._make_bind_labels(self._cloister_state.out.z_ecorr),
                output_directory / "bounds_prunned.csv",
            )

        InstanceSpace._write_array_to_csv(
            self._data.x_raw[:, self._prelim_state.out.idx],
            pd.Series(self._data.feat_labels),
            self._data.inst_labels,
            output_directory / "feature_raw.csv",
        )
        InstanceSpace._write_array_to_csv(
            self._data.x,
            pd.Series(self._data.feat_labels),
            self._data.inst_labels,
            output_directory / "feature_process.csv",
        )
        InstanceSpace._write_array_to_csv(
            self._data.y_raw,
            pd.Series(self._data.feat_labels),
            self._data.inst_labels,
            output_directory / "algorithm_raw.csv",
        )
        InstanceSpace._write_array_to_csv(
            self._data.y,
            pd.Series(self._data.feat_labels),
            self._data.inst_labels,
            output_directory / "algorithm_process.csv",
        )
        InstanceSpace._write_array_to_csv(
            self._data.y_bin,
            pd.Series(self._data.feat_labels),
            self._data.inst_labels,
            output_directory / "algorithm_bin.csv",
        )
        InstanceSpace._write_array_to_csv(
            self._data.num_good_algos,
            pd.Series(["NumGoodAlgos"]),
            self._data.inst_labels,
            output_directory / "good_algos.csv",
        )
        InstanceSpace._write_array_to_csv(
            self._data.beta,
            pd.Series(["IsBetaEasy"]),
            self._data.inst_labels,
            output_directory / "beta_easy.csv",
        )
        InstanceSpace._write_array_to_csv(
            self._data.p,
            pd.Series(["Best_Algorithm"]),
            self._data.inst_labels,
            output_directory / "portfolio.csv",
        )
        InstanceSpace._write_array_to_csv(
            self._pythia_state.out.y_hat,
            pd.Series(self._data.algo_labels),
            self._data.inst_labels,
            output_directory / "algorithm_svm.csv",
        )
        InstanceSpace._write_array_to_csv(
            self._pythia_state.out.selection0,
            pd.Series(["Best_Algorithm"]),
            self._data.inst_labels,
            output_directory / "portfolio_svm.csv",
        )
        InstanceSpace._write_cell_to_csv(
            self._trace_state.out.summary[2:, [3, 5, 6, 8, 10, 11]],
            self._trace_state.out.summary[1, [3, 5, 6, 8, 10, 11]],
            self._trace_state.out.summary[2:, 1],
            output_directory / "footprint_performance.csv",
        )
        if self._pilot_state.out.summary is not None:
            InstanceSpace._write_cell_to_csv(
                self._pilot_state.out.summary[2:, 2:],
                self._pilot_state.out.summary[1, 2:],
                self._pilot_state.out.summary[2:, 1],
                output_directory / "footprint_performance.csv",
            )
        InstanceSpace._write_cell_to_csv(
            self._pythia_state.out.summary[2:, 2:],
            self._pythia_state.out.summary[1, 2:],
            self._pythia_state.out.summary[2:, 1],
            output_directory / "svm_table.csv",
        )

    @staticmethod
    def _write_array_to_csv(
        data: NDArray[Any], # TODO: Try to unify these
        column_names: pd.Series, # TODO: Try to unify these
        row_names: pd.Series, # type: ignore[type-arg]
        filename: Path,
    ) -> None:
        pd.DataFrame(data, index=row_names, columns=column_names).to_csv(filename)


    @staticmethod
    def _write_cell_to_csv(
        data: pd.Series, # TODO: Try to unify these
        column_names: pd.Series, # TODO: Try to unify these
        row_names: pd.Series, # type: ignore[type-arg]
        filename: Path,
    ) -> None:
        pd.DataFrame(data, index=row_names, columns=column_names).to_csv(filename)

    @staticmethod
    def _make_bind_labels(
        data: NDArray[Any],
    ) -> pd.Series:
        return pd.Series([f"bnd_pnt_{i}" for i in range(data.shape[0])])


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
    for field_name in fields(Options):
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
