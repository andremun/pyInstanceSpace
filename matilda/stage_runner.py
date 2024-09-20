"""A runner to run a list of stages."""

from collections import defaultdict
from collections.abc import Generator
from typing import Any, NamedTuple

from matilda.stages.stage import OUT, Stage, StageClass

StageScheduleElement = list[StageClass]


class StageArgument(NamedTuple):
    """An input or output of a stage."""

    parameter_name: str
    parameter_type: type


class StageRunningError(Exception):
    """An error during stage running."""


class AnnotatedStageOutput(NamedTuple):
    """The yielded output of running a stage."""

    stage: StageClass
    output: NamedTuple


class StageRunner:
    """A runner to run a list of stages."""

    # Data output from stages that can be used as input for future stages. Saved
    # at every stage schedule so you can rerun stages.
    _schedule_output_data: list[dict[str, Any]]

    _available_arguments: dict[str, Any]

    # Cached index for when a stage is going to be ran, calculated in the constructor
    _stage_to_schedule_index: dict[StageClass, int]

    # List of stages to be ran
    _stage_order: list[StageScheduleElement]

    _current_schedule_item: int
    _stages_ran: defaultdict[StageClass, bool]

    def __init__(
        self,
        stages: list[StageScheduleElement],
        input_arguments: dict[StageClass, set[StageArgument]],
        output_arguments: dict[StageClass, set[StageArgument]],
        initial_input_annotations: set[StageArgument],
    ) -> None:
        """Create a StageRunner from a preresolved set of stages.

        All stages inputs and outputs are assumed to already be resolved.
        """
        self._stage_order = stages

        self._schedule_output_data = [{}]
        self._current_schedule_item = 0

        self._available_arguments = {}
        self._stage_to_schedule_index = {}
        self._stages_ran = defaultdict(lambda: False)

        for i, schedule in enumerate(self._stage_order):
            for stage in schedule:
                self._stage_to_schedule_index[stage] = i

        self._check_stage_order_is_runnable(
            stages,
            input_arguments,
            output_arguments,
            initial_input_annotations,
        )

    def run_iter(
        self,
        **additional_arguments: Any,  # noqa: ANN401
    ) -> Generator[AnnotatedStageOutput, None, dict[str, Any]]:
        """Run all stages, yielding after every run.

        Yields
        ------
            Generator[AnnotatedStageOutput, None, dict[str, Any]]: _description_
        """
        self._rollback_to_schedule_index(0)

        self._available_arguments = additional_arguments

        for schedule in self._stage_order:
            for stage in schedule:
                yield AnnotatedStageOutput(stage, self.run_stage(stage))

        return self._available_arguments

    def run_stage(
        self,
        stage: type[Stage[Any, OUT]],
        **additional_arguments: Any,  # noqa: ANN401
    ) -> OUT:
        """Run a single stage.

        Errors if prerequisite stages haven't been ran.

        Args
        ----
            stages list[StageClass]: A list of stages to run.
            **arguments dict[str, Any]: Inputs for the stage. If inputs aren't provided
                the runner will try to get them from previously ran stages. If they
                still aren't present the stage will raise an error.
        """
        # Make sure stage can be ran
        stage_schedule_index = self._stage_to_schedule_index[stage]
        if stage_schedule_index > self._current_schedule_item:
            raise StageRunningError(
                f"{stage} could not be ran, as prerequisite stages have not yet "
                + "been ran",
            )

        # If running an earlier stage again, rollback any changes made after that stages
        # schedule
        if stage_schedule_index != self._current_schedule_item:
            self._rollback_to_schedule_index(stage_schedule_index)

        available_arguments = self._available_arguments.copy()
        for k, v in additional_arguments.items():
            available_arguments[k] = v

        input_arguments = stage._inputs()  # noqa: SLF001

        raw_inputs = {}

        for input_name in input_arguments._fields:
            # TODO: Some sort of type check on the inputs
            raw_inputs[input_name] = available_arguments[input_name]

        # TODO: See if this actually works
        inputs: NamedTuple = input_arguments.__new__(input_arguments, **raw_inputs)

        outputs = stage._run(inputs)  # noqa: SLF001

        for output_name, output_value in outputs._asdict().items():
            self._available_arguments[output_name] = output_value

        self._schedule_output_data[self._current_schedule_item] = (
            self._available_arguments
        )

        self._stages_ran[stage] = True

        self._progress_schedule()

        return outputs

    def run_many_stages_parallel(
        self,
        stages: list[StageClass],
        additional_arguments: NamedTuple,
    ) -> dict[str, Any]:
        """Run multiple stages in parallel.

        All prerequisite stages must have already been ran. The stages cannot be a
        prerequisite for other stages being ran at the same time.

        Args
        ----
            stages list[StageClass]: A list of stages to run.

        Returns
        -------
            tuple[tuple[Any]]: _description_
        """
        raise NotImplementedError

    def run_all(self, additional_arguments: NamedTuple) -> dict[str, Any]:
        """Run all stages from start to finish.

        Return the entire outputs data object when finished.

        Returns
        -------
            tuple[Any]: _description_
        """
        self._rollback_to_schedule_index(0)

        self._available_arguments = additional_arguments._asdict()

        for schedule in self._stage_order:
            for stage in schedule:
                self.run_stage(stage)

        return self._available_arguments

    def run_remaining(self, additional_arguments: NamedTuple) -> dict[str, Any]:
        """Run stages that haven't been ran to finish.

        Return the entire outputs data object when finished.

        Returns
        -------
            tuple[Any]: _description_
        """
        for schedule in self._stage_order:
            for stage in schedule:
                if not self._stages_ran[stage]:
                    self.run_stage(stage, additional_arguments)

        return self._available_arguments

    def run_until_stage(
        self,
        stop_at_stage: StageClass,
        additional_arguments: NamedTuple,
    ) -> dict[str, Any]:
        """Run all stages until the specified stage, as well as the specified stage.

        Returns
        -------
            tuple[Any]: _description_
        """
        self._rollback_to_schedule_index(0)

        self._available_arguments = additional_arguments._asdict()

        for schedule in self._stage_order:
            if stop_at_stage in schedule:
                break

            for stage in schedule:
                self.run_stage(stage)

        return self._available_arguments

        # TODO: Work out what this should return. Maybe just the dict of outputs?

    @staticmethod
    def _check_stage_order_is_runnable(
        stages: list[StageScheduleElement],
        input_arguments: dict[StageClass, set[StageArgument]],
        output_arguments: dict[StageClass, set[StageArgument]],
        initial_input_annotations: set[StageArgument],
    ) -> None:
        available_arguments = initial_input_annotations.copy()

        for schedule_element in stages:
            for stage in schedule_element:
                if len(input_arguments[stage] - available_arguments) > 0:
                    raise StageRunningError(
                        "Stage order was not runnable. Not all inputs were available "
                        + "for a stage at the time of running.",
                    )

            for stage in schedule_element:
                available_arguments |= output_arguments[stage]

    def _rollback_to_schedule_index(
        self,
        index: int,
    ) -> None:
        self._current_schedule_item = index
        self._available_arguments = self._schedule_output_data[index]

        self._schedule_output_data = self._schedule_output_data[: index + 1]

        for schedule_element in self._stage_order[index + 1 :]:
            for stage in schedule_element:
                self._stages_ran[stage] = False

    def _progress_schedule(self) -> None:
        current_schedule_finished = True
        for stage in self._stage_order[self._current_schedule_item]:
            if not self._stages_ran[stage]:
                current_schedule_finished = False
                break

        if current_schedule_finished:
            self._schedule_output_data[self._current_schedule_item] = (
                self._available_arguments
            )

            self._current_schedule_item += 1

            if len(self._schedule_output_data) <= self._current_schedule_item:
                self._schedule_output_data.append({})
