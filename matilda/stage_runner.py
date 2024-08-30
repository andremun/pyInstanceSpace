"""A runner to run a list of stages."""
from typing import Any

from matilda.stage import Stage


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
    input_arguments: dict[type[Stage], tuple[str, type]]
    output_arguments: dict[type[Stage], tuple[str, type]]

    def __init__(
        self,
        stages: list[type[Stage]],
        input_arguments: dict[type[Stage], tuple[str, type]],
        output_arguments: dict[type[Stage], tuple[str, type]],
    ) -> None:
        """
        Create a StageRunner from a preresolved set of stages.

        All stages inputs and outputs are assumed to already be resolved.
        """
        self.stages = stages
        self.input_arguments = input_arguments
        self.output_arguments = output_arguments

        self.output_data = {}

        # TODO: Check the inputs are actually resolved, throw if not

    def run(self, **initial_inputs: dict[str, Any]) -> tuple[Any]:
        """
        Run all stages from start to finish.

        Return the entire outputs data object when finished.
        """
        for input_name, input_data in initial_inputs:
            pass


        for stage in self.stages:
            input_data = []
            for input_name, input_type in self.input_arguments[stage]:
                input_data.append(self.outputs[input_name])
                # Do some check that input is the right type
            outputs = stage.run(*input_data)

            for i in range(len(outputs)):
                output_name, output_type = self.outputs[stage][i]
                # Do some check that the output is the right type
                self.output_data[output_name] = outputs[i]

        return # all of outputs?