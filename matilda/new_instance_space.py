"""TODO: document instance space module."""

from typing import Any

from matilda.data.metadata import Metadata
from matilda.data.options import InstanceSpaceOptions
from matilda.stages.stage import Stage


class NewInstanceSpace:
    """TODO: Describe what an instance space IS."""

    def __init__(
        self,
        stages: list[type[Stage]] = [],
    ) -> None:
        """
        Initialise the InstanceSpace.

        Args:
            stages list[type[Stage]], optional: A list of stages to be ran.
        """
        raise NotImplementedError

    def build(
        self,
        metadata: Metadata,
        options: InstanceSpaceOptions,
        **arguments: dict[str, Any],
    ) -> None:
        """
        Build the instance space.

        Options will be broken down to sub fields to be passed to stages. You can
        override inputs to stages.

        Args
        ----
            metadata (Metadata): _description_
            options (InstanceSpaceOptions): _description_

        """
        raise NotImplementedError

    def run_stage(self, stage: type[Stage], **arguments: dict[str, Any]) -> list[Any]:
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
        raise NotImplementedError

    def run_until_stage(
        self,
        stage: type[Stage],
        **arguments: dict[str, Any],
    ) -> list[Any]:
        """
        Run all stages until the specified stage, as well as the specified stage.

        Args
        ----
            stage type[Stage]: Initial inputs, as well as any inputs you want to
            override.

        Returns
        -------
            list[Any]: _description_
        """
        raise NotImplementedError
