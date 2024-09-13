"""TODO: document instance space module."""

from collections.abc import Generator
from typing import Any

from matilda.data.metadata import Metadata
from matilda.data.options import InstanceSpaceOptions
from matilda.stage_builder import StageArgument, StageBuilder
from matilda.stage_runner import AnnotatedStageOutput, StageRunner
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

    runner: StageRunner
    _stages: list[type[Stage]]

    def __init__(
        self,
        stages: list[type[Stage]] = [
            PreprocessingStage,
            PrelimStage,
            SiftedStage,
            PilotStage,
            PythiaStage,
            CloisterStage,
            TraceStage,
        ],
    ) -> None:
        """Initialise the InstanceSpace.

        Args:
            stages list[type[Stage]], optional: A list of stages to be ran.
        """
        self._stages = stages

        stage_builder = StageBuilder()

        for stage in stages:
            stage_builder.add_stage(stage)

        # TODO: do this manually, I am just pulling it out of annotations to get it
        # working
        self.runner = stage_builder.build(
            [StageArgument(k, v) for k, v in Metadata.__annotations__.items()]
            + [
                StageArgument(k, v)
                for k, v in InstanceSpaceOptions.__annotations__.items()
            ],
        )

    def build(
        self,
        metadata: Metadata,
        options: InstanceSpaceOptions,
        **arguments: Any,  # noqa: ANN401
    ) -> None:  # TODO: Replace this with model / get_model
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
        # TODO: split out metadata and options into component fields
        self.runner.run_all(**metadata.__dict__, **options.__dict__, **arguments)

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
        yield from self.runner.run_iter(
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
        return self.runner.run_stage(stage, **arguments)

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
        self.runner.run_until_stage(
            stage,
            **metadata.__dict__,
            **options.__dict__,
            **arguments,
        )
