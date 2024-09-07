"""TODO: document instance space module."""

from collections.abc import Generator
from typing import Any

from matilda.data.metadata import Metadata
from matilda.data.options import InstanceSpaceOptions
from matilda.stage_builder import StageBuilder
from matilda.stage_runner import StageRunner
from matilda.stages.cloister_stage import CloisterStage
from matilda.stages.pilot_stage import PilotStage
from matilda.stages.prelim_stage import PrelimStage
from matilda.stages.preprocessing_stage import PreprocessingStage
from matilda.stages.pythia_stage import PythiaStage
from matilda.stages.sifted_stage import SiftedStage
from matilda.stages.stage import Stage
from matilda.stages.trace_stage import TraceStage


class NewInstanceSpace:
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
        """
        Initialise the InstanceSpace.

        Args:
            stages list[type[Stage]], optional: A list of stages to be ran.
        """
        self._stages = stages

        stage_builder = StageBuilder()

        for stage in stages:
            stage_builder.add_stage(
                stage,
                stage._inputs(),  # noqa: SLF001
                stage._outputs(),  # noqa: SLF001
            )

        self.runner = stage_builder.build()

        raise NotImplementedError

    def build(
        self,
        metadata: Metadata,
        options: InstanceSpaceOptions,
        **arguments: Any,  # noqa: ANN401
    ) -> tuple[Any]:  # TODO: Replace this with model / get_model
        """
        Build the instance space.

        Options will be broken down to sub fields to be passed to stages. You can
        override inputs to stages.

        Args
        ----
            metadata (Metadata): _description_
            options (InstanceSpaceOptions): _description_

        Returns
        -------
            tuple[Any]: The output of all stages

        """
        return self.runner.run_all(metadata=metadata, options=options, **arguments)

    def run_iter(
        self,
        metadata: Metadata,
        options: InstanceSpaceOptions,
        **arguments: Any,  # noqa: ANN401
    ) -> Generator[None, tuple[Any], None]:
        """Run all stages, yielding between so the data can be examined.

        Args
        ----
            metadata (Metadata): _description_
            options (InstanceSpaceOptions): _description_

        Yields
        ------
            Generator[None, tuple[Any], None]: _description_
        """
        yield from self.runner.run_iter(metadata=metadata, options=options, **arguments)

    def run_stage(
        self,
        stage: type[Stage],
        **arguments: Any,  # noqa: ANN401
    ) -> tuple[Any]:
        """
        Run a single stage.

        All inputs to the stage must either be present from previously ran stages, or
        be given as arguments to this function. Arguments to this function have
        priority over outputs from previous stages.

        Args
        ----
            stage (type[Stage]): Any inputs to the stage.

        Returns
        -------
            list[Any]: The output of the stage
        """
        return self.runner.run_stage(stage, **arguments)

    def run_until_stage(
        self,
        stage: type[Stage],
        **arguments: Any,  # noqa: ANN401
    ) -> tuple[Any]:
        """
        Run all stages until the specified stage, as well as the specified stage.

        Args
        ----
            stage type[Stage]: The stage to stop running stages after.
            **arguments dict[str, Any]: if this is the first time stages are ran,
                initial inputs, and overriding inputs for other stages. TODO: rewrite
                this

        Returns
        -------
            list[Any]: _description_
        """
        if stage not in self._stages:
            raise KeyError("stage not found in instance space.")

        for intermediate_stage in self._stages:
            stage_arguments = arguments[stage]
            stage_output = self.runner.run_stage(stage, **arguments[stage])

            if intermediate_stage == stage:
                return stage_output

        raise NotImplementedError
