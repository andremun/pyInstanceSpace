"""A runner to run a list of stages."""

from typing import Any

from matilda.stages.stage import Stage, StageArgument


class StageRunner:
    """A runner to run a list of stages."""

    # Data output from stages that can be used as input for future stages
    output_data: dict[str, Any]

    # List of stages to be ran
    # TODO: We could do this as a list[list[...]] and specify stages that can be ran
    # in parallel
    stages: list[type[Stage]]

    # Types and names of inputs and outputs of stages, used for order resolution and
    # dependency injection.
    initial_input_arguments: list[StageArgument]
    input_arguments: dict[type[Stage], list[StageArgument]]
    output_arguments: dict[type[Stage], list[StageArgument]]

    def __init__(
        self,
        stages: list[type[Stage]],
        initial_input_arguments: list[StageArgument],
        input_arguments: dict[type[Stage], list[StageArgument]],
        output_arguments: dict[type[Stage], list[StageArgument]],
    ) -> None:
        """
        Create a StageRunner from a preresolved set of stages.

        All stages inputs and outputs are assumed to already be resolved.
        """
        self.stages = stages
        self.initial_input_arguments = initial_input_arguments
        self.input_arguments = input_arguments
        self.output_arguments = output_arguments

        self.output_data = {}

        # TODO: Check the inputs are actually resolved, throw if not

    def run(self, **initial_inputs: Any) -> tuple[Any]:  # noqa: ANN401
        """
        Run all stages from start to finish.

        Return the entire outputs data object when finished.
        """
        for initial_input_name, initial_input_data in self.initial_input_arguments:
            # TODO: Check all inputs are present and correct. Otherwise Throw.
            pass


        for stage in self.stages:
            input_data: list[Any] = []
            for input_name, input_type in self.input_arguments[stage]:
                input_data.append(self.output_data[input_name])
                # TODO: Do some check that input is the right type

            outputs = stage._run(*input_data)  # noqa: SLF001

            for i in range(len(outputs)):
                output_name, output_type = self.output_arguments[stage][i]
                # Do some check that the output is the right type
                self.output_data[output_name] = outputs[i]

        return # TODO: all of outputs?
