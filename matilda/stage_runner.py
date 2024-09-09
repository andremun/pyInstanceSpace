"""A runner to run a list of stages."""

from collections.abc import Generator
from typing import Any

from matilda.stages.stage import Stage, StageArgument


class StageRunner:
    """A runner to run a list of stages."""

    # Data output from stages that can be used as input for future stages
    output_data: dict[str, Any]

    # List of stages to be ran
    # TODO: We could do this as a list[list[...]] and specify stages that can be ran
    # in parallel
    stages: list[list[type[Stage]]]

    # Types and names of inputs and outputs of stages, used for order resolution and
    # dependency injection.
    input_arguments: dict[type[Stage], set[StageArgument]]
    output_arguments: dict[type[Stage], set[StageArgument]]

    def __init__(
        self,
        stages: list[list[type[Stage]]],
        input_arguments: dict[type[Stage], set[StageArgument]],
        output_arguments: dict[type[Stage], set[StageArgument]],
    ) -> None:
        """Create a StageRunner from a preresolved set of stages.

        All stages inputs and outputs are assumed to already be resolved.
        """
        self.stages = stages
        self.input_arguments = input_arguments
        self.output_arguments = output_arguments

        self.output_data = {}

        raise NotImplementedError

        # TODO: Check the inputs are actually resolved, throw if not

    def run_iter(
        self,
        **initial_inputs: Any,  # noqa: ANN401
    ) -> Generator[None, tuple[Any], None]:
        """Run all stages, yielding after every run.

        Yields
        ------
            Generator[None, tuple[Any], None]: _description_
        """
        raise NotImplementedError

    def run_stage(
        self,
        stage: type[Stage],
        **arguments: Any,  # noqa: ANN401
    ) -> tuple[Any]:
        """Run a single stage.

        Errors if prerequisite stages haven't been ran.

        Args
        ----
            stages list[type[Stage]]: A list of stages to run.
            **arguments dict[str, Any]: Inputs for the stage. If inputs aren't provided
                the runner will try to get them from previously ran stages. If they
                still aren't present the stage will raise an error.
        """
        raise NotImplementedError

    def run_many_stages_parallel(
        self,
        stages: list[type[Stage]],
        **initial_inputs: Any,  # noqa: ANN401
    ) -> tuple[tuple[Any]]:
        """Run multiple stages in parallel.

        All prerequisite stages must have already been ran. The stages cannot be a
        prerequisite for other stages being ran at the same time.

        Args
        ----
            stages list[type[Stage]]: A list of stages to run.

        Returns
        -------
            tuple[tuple[Any]]: _description_
        """
        raise NotImplementedError

    def run_all(self, **initial_inputs: Any) -> tuple[Any]:  # noqa: ANN401
        """Run all stages from start to finish.

        Return the entire outputs data object when finished.

        Returns
        -------
            tuple[Any]: _description_
        """
        for initial_input_name, initial_input_data in initial_inputs.items():
            # TODO: Check all inputs are present and correct. Otherwise Throw.
            pass

        # for schedule_item in self.stages:
        #     for stage in schedule_item:
        #         input_data: list[Any] = []
        #         for input_name, input_type in self.input_arguments[stage]:
        #             input_data.append(self.output_data[input_name])
        #             # TODO: Do some check that input is the right type

        #         outputs = stage._run(*input_data)

        #         for output_name, output_value in outputs._asdict().items():
        #             # Do some check that the output is the right type
        #             self.output_data[output_name] = output_value

        raise NotImplementedError

    def run_until_stage(
        self,
        stage: type[Stage],
        **initial_inputs: Any,  # noqa: ANN401
    ) -> tuple[Any]:
        """Run all stages until the specified stage, as well as the specified stage.

        Returns
        -------
            tuple[Any]: _description_
        """
        raise NotImplementedError
