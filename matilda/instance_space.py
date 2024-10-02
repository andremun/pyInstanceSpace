"""TODO: document instance space module."""

from collections.abc import Generator
from dataclasses import fields
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from typing_extensions import TypeVar

from matilda.data.metadata import Metadata, from_csv_file
from matilda.data.options import (
    AutoOptions,
    BoundOptions,
    CloisterOptions,
    InstanceSpaceOptions,
    NormOptions,
    OutputOptions,
    ParallelOptions,
    PerformanceOptions,
    PilotOptions,
    PrelimOptions,
    PythiaOptions,
    SelvarsOptions,
    SiftedOptions,
    TraceOptions,
    from_json_file,
)
from matilda.model import Model
from matilda.stage_builder import StageBuilder
from matilda.stage_runner import (
    AnnotatedStageOutput,
    StageRunner,
    StageRunningError,
)
from matilda.stages.cloister import CloisterStage
from matilda.stages.pilot import PilotStage
from matilda.stages.prelim import PrelimStage
from matilda.stages.preprocessing import PreprocessingStage
from matilda.stages.pythia import PythiaStage
from matilda.stages.sifted import SiftedStage
from matilda.stages.stage import IN, OUT, Stage, StageClass
from matilda.stages.trace import TraceStage

T = TypeVar("T", bound="_InstanceSpaceInputs")
class _InstanceSpaceInputs(NamedTuple):
    feature_names: list[str]
    algorithm_names: list[str]
    instance_labels: pd.Series  # type: ignore[type-arg]
    instance_sources: pd.Series | None  # type: ignore[type-arg]
    features: NDArray[np.double]
    algorithms: NDArray[np.double]
    parallel_options: ParallelOptions
    perf_options: PerformanceOptions
    auto_options: AutoOptions
    bound_options: BoundOptions
    norm_options: NormOptions
    selvars_options: SelvarsOptions
    sifted_options: SiftedOptions
    pilot_options: PilotOptions
    cloister_options: CloisterOptions
    pythia_options: PythiaOptions
    trace_options: TraceOptions
    outputs_options: OutputOptions
    prelim_options: PrelimOptions

    @classmethod
    def from_metadata_and_options(
        cls: type[T],
        metadata: Metadata,
        options: InstanceSpaceOptions,
    ) -> T:
        return cls(
            feature_names=metadata.feature_names,
            algorithm_names=metadata.algorithm_names,
            instance_labels=metadata.instance_labels,
            instance_sources=metadata.instance_sources,
            features=metadata.features,
            algorithms=metadata.algorithms,
            parallel_options=options.parallel,
            perf_options=options.perf,
            auto_options=options.auto,
            bound_options=options.bound,
            norm_options=options.norm,
            selvars_options=options.selvars,
            sifted_options=options.sifted,
            pilot_options=options.pilot,
            cloister_options=options.cloister,
            pythia_options=options.pythia,
            trace_options=options.trace,
            outputs_options=options.outputs,
            prelim_options=PrelimOptions.from_options(options),
        )


class InstanceSpace:
    """TODO: Describe what an instance space IS."""

    _runner: StageRunner
    _stages: list[StageClass]

    _metadata: Metadata
    _options: InstanceSpaceOptions

    _model: Model | None
    _final_output: dict[str, Any] | None

    def __init__(
        self,
        metadata: Metadata,
        options: InstanceSpaceOptions,
        stages: list[StageClass] = [
            PreprocessingStage,
            PrelimStage,
            SiftedStage,
            PilotStage,
            PythiaStage,
            CloisterStage,
            TraceStage,
        ],
        additional_initial_inputs_type: type[NamedTuple] | None = None,
    ) -> None:
        """Initialise the InstanceSpace.

        Args:
            stages list[StageClass], optional: A list of stages to be ran.
        """
        self._metadata = metadata
        self._options = options
        self._stages = stages

        self._model = None

        stage_builder = StageBuilder()

        for stage in stages:
            stage_builder.add_stage(stage)


        annotations = stage_builder._named_tuple_to_stage_arguments(  # noqa: SLF001
            _InstanceSpaceInputs,
        )

        if additional_initial_inputs_type is not None:
            annotations |= (
                stage_builder._named_tuple_to_stage_arguments(  # noqa: SLF001
                    additional_initial_inputs_type,
                )
            )

        self._runner = stage_builder.build(annotations)

    @property
    def metadata(self) -> Metadata:
        """Get metadata."""
        return self._metadata

    @property
    def options(self) -> InstanceSpaceOptions:
        """Get options."""
        return self._options

    @property
    def model(self) -> Model:
        """Get model.

        Raises
        ------
            StageRunningError: If the InstanceSpace hasn't been built, will raise a
                StageRunningError.

        Returns
        -------
            Model: _description_
        """
        if not self._model:
            if not self._final_output:
                raise StageRunningError("InstanceSpace has not been completely ran.")

            self._model = Model.from_stage_runner_output(
                self._final_output,
                self._options,
            )

        return self._model

    def build(
        self,
        metadata: Metadata,
        options: InstanceSpaceOptions,
        **_arguments: Any,  # noqa: ANN401 # TODO: make this work
    ) -> Model:
        """Build the instance space.

        Options will be broken down to sub fields to be passed to stages. You can
        override inputs to stages.

        Args
        ----
            metadata Metadata: _description_
            options InstanceSpaceOptions: _description_

        Returns
        -------
            tuple[Any]: The output of all stages

        """
        inputs = _InstanceSpaceInputs.from_metadata_and_options(metadata, options)
        self._runner.run_all(inputs)

        return self.model

    def run_iter(
        self,
        metadata: Metadata,
        options: InstanceSpaceOptions,
        **_arguments: Any,  # noqa: ANN401 # TODO: make this work
    ) -> Generator[AnnotatedStageOutput, None, None]:
        """Run all stages, yielding between so the data can be examined.

        Args
        ----
            metadata Metadata: _description_
            options InstanceSpaceOptions: _description_

        Yields
        ------
            Generator[None, tuple[Any], None]: _description_
        """
        inputs = _InstanceSpaceInputs.from_metadata_and_options(metadata, options)
        # TODO: split out metadata and options into component fields
        yield from self._runner.run_iter(inputs)

    def run_stage(
        self,
        stage: type[Stage[IN, OUT]],
        **arguments: Any,  # noqa: ANN401
    ) -> OUT:
        """Run a single stage.

        All inputs to the stage must either be present from previously ran stages, or
        be given as arguments to this function. Arguments to this function have
        priority over outputs from previous stages.

        Args
        ----
            stage StageClass: Any inputs to the stage.

        Returns
        -------
            list[Any]: The output of the stage
        """
        # Need to type coerce the output of the stage because generics
        return self._runner.run_stage(stage, **arguments)

    def run_until_stage(
        self,
        stage: StageClass,
        metadata: Metadata,
        options: InstanceSpaceOptions,
        **_arguments: Any,  # noqa: ANN401
    ) -> None:
        """Run all stages until the specified stage, as well as the specified stage.

        Args
        ----
            stage StageClass: The stage to stop running stages after.
            metadata Metadata: _description_
            options InstanceSpaceOptions: _description_
            **arguments dict[str, Any]: if this is the first time stages are ran,
                initial inputs, and overriding inputs for other stages. TODO: rewrite
                this

        Returns
        -------
            list[Any]: _description_
        """
        inputs = _InstanceSpaceInputs.from_metadata_and_options(metadata, options)
        # TODO: split out metadata and options into component fields
        self._runner.run_until_stage(
            stage,
            inputs,
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
