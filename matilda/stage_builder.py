"""A stage builder to resolve a collection of stages."""

from typing import Self

from matilda.stage_runner import StageRunner
from matilda.stages.stage import Stage, StageArgument


class StageBuilder:
    """A stage builder to resolve a collection of stages."""

    def __init__(self) -> None:
        """Initialise a StageBuilder."""
        pass

    def add_stage(
        self,
        stage: type[Stage],
        arguments: list[StageArgument],
        outputs: list[StageArgument],
    ) -> Self:
        """Add a stage to the builder.

        Stages don't need to be added in running order.

        Args
        ----
            stage type[Stage]: A Stage class
            arguments list[StageArgument]: A list of inputs that the stage takes
            outputs list[StageArgument]: A list of outputs the stage produces

        Returns
        -------
            Self
        """
        raise NotImplementedError

    def build(
        self,
    ) -> StageRunner:
        """Resolve the stages, and produce a StageRunner to run them.

        This will check that all stages can be ran with inputs from previous stages, and
        resolve a running order for the stages.

        Returns
        -------
            StageRunner: A StageRunner for the given stages
        """
        raise NotImplementedError
