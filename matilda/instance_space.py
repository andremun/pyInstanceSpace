"""TODO: document instance space module."""

from collections.abc import Generator
from dataclasses import fields
from pathlib import Path
from typing import Any, NamedTuple

from matilda.data.metadata import Metadata, from_csv_file
from matilda.data.options import InstanceSpaceOptions, from_json_file
from matilda.model import Model
from matilda.stage_builder import StageArgument, StageBuilder
from matilda.stage_runner import AnnotatedStageOutput, StageRunner, StageRunningError
from matilda.stages.cloister import CloisterStage
from matilda.stages.pilot_stage import PilotStage
from matilda.stages.prelim_stage import PrelimStage
from matilda.stages.preprocessing_stage import PreprocessingStage
from matilda.stages.pythia_stage import PythiaStage
from matilda.stages.sifted_stage import SiftedStage
from matilda.stages.stage import Stage
from matilda.stages.trace_stage import TraceStage


class InstanceSpace:
    """TODO: Describe what an instance space IS."""

    _runner: StageRunner
    _stages: list[type[Stage]]

    _metadata: Metadata
    _options: InstanceSpaceOptions

    _model: Model | None
    _final_output: dict[str, Any] | None

    def __init__(
        self,
        metadata: Metadata,
        options: InstanceSpaceOptions,
        stages: list[type[Stage]] = [
            PreprocessingStage,
            PrelimStage,
            SiftedStage,
            PilotStage,
            PythiaStage,
            CloisterStage,
            TraceStage,
        ],
        additional_initial_inputs: type[NamedTuple] | None = None,
    ) -> None:
        """Initialise the InstanceSpace.

        Args:
            stages list[type[Stage]], optional: A list of stages to be ran.
        """
        self._metadata = metadata
        self._options = options
        self._stages = stages

        self._model = None

        stage_builder = StageBuilder()

        for stage in stages:
            stage_builder.add_stage(stage)

        annotations = {StageArgument(k, v) for k, v in Metadata.__annotations__.items()}
        annotations |= {
            StageArgument(k, v) for k, v in InstanceSpaceOptions.__annotations__.items()
        }
        if additional_initial_inputs is not None:
            annotations |= (
                stage_builder._named_tuple_to_stage_arguments(  # noqa: SLF001
                    additional_initial_inputs,
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
        **arguments: Any,  # noqa: ANN401
    ) -> Model:  # TODO: Replace this with model / get_model
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
        self._runner.run_all(**metadata.__dict__, **options.__dict__, **arguments)

        raise NotImplementedError

    def run_iter(
        self,
        metadata: Metadata,
        options: InstanceSpaceOptions,
        **arguments: Any,  # noqa: ANN401
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
        # TODO: split out metadata and options into component fields
        yield from self._runner.run_iter(
            **metadata.__dict__,
            **options.__dict__,
            **arguments,
        )

    def run_stage(
        self,
        stage: type[Stage],
        **arguments: Any,  # noqa: ANN401
    ) -> tuple[Any]:
        """Run a single stage.

        All inputs to the stage must either be present from previously ran stages, or
        be given as arguments to this function. Arguments to this function have
        priority over outputs from previous stages.

        Args
        ----
            stage type[Stage]: Any inputs to the stage.

        Returns
        -------
            list[Any]: The output of the stage
        """
        return self._runner.run_stage(stage, **arguments)

    def run_until_stage(
        self,
        stage: type[Stage],
        metadata: Metadata,
        options: InstanceSpaceOptions,
        **arguments: Any,  # noqa: ANN401
    ) -> None:
        """Run all stages until the specified stage, as well as the specified stage.

        Args
        ----
            stage type[Stage]: The stage to stop running stages after.
            metadata Metadata: _description_
            options InstanceSpaceOptions: _description_
            **arguments dict[str, Any]: if this is the first time stages are ran,
                initial inputs, and overriding inputs for other stages. TODO: rewrite
                this

        Returns
        -------
            list[Any]: _description_
        """
        # TODO: split out metadata and options into component fields
        self._runner.run_until_stage(
            stage,
            **metadata.__dict__,
            **options.__dict__,
            **arguments,
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
