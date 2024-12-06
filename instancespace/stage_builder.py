"""A stage builder to resolve a collection of stages."""

from typing import NamedTuple, Self, get_args

from pyInstanceSpace.stage_runner import StageRunner, StageScheduleElement, _StageArgument
from pyInstanceSpace.stages.stage import RunAfter, RunBefore, StageClass


class _BeforeAfterRestriction(NamedTuple):
    before: StageClass
    after: StageClass


class _StageRestrictions(NamedTuple):
    run_before: set[StageClass]
    run_after: set[StageClass]


class StageResolutionError(Exception):
    """An error during stage resolution."""


class StageBuilder:
    """A stage builder to resolve a collection of stages.

    ##Example:##
    ```python
    stage_builder = StageBuilder()

    # Stages don't need to be added in order, but are here for demonstration purposes.
    stage_builder
        .add(PreprocessingStage)
        .add(PrelimStage)
        .add(SiftedStage)

    stage_runner = stage_builder.build()

    ```

    ##Example:##
    ```python

    # Stages don't need to be added in order, but are here for demonstration purposes.
    stage_runner = StageBuilder()
        .add(PreprocessingStage)
        .add(PrelimStage)
        .build()

    ```

    ## Concepts

    ### Mutating Stage
    A stage that has the exact same argument as an input and an output. This stage will
    be ran immediately after the first time the argument is an output of a previous
    stage, and other stages that have the argument as an input will be run after the
    mutating stage. This behaviour can be overwritten using RunBefore and RunAfter.

    ### RunBefore
    Having this as an input argument for a stage will cause it to be ran before the
    designated stage. If this is not possible an error will be thrown on resolution.
    This input has no effect on the stage itself, and will not be passed to the stage.

    ### RunAfter
    Having this as an input argument for a stage will cause it to be ran after the
    designated stage. If this is not possible an error will be thrown on resolution.
    This input has no effect on the stage itself, and will not be passed to the stage.
    """

    _stages: set[StageClass]
    _stage_inputs: dict[StageClass, set[_StageArgument]]
    _stage_outputs: dict[StageClass, set[_StageArgument]]

    def __init__(self) -> None:
        """Initialise a new StageBuilder.

        @private
        """
        self._stages = set()
        self._stage_inputs = {}
        self._stage_outputs = {}

    def add_stage(
        self,
        stage: StageClass,
    ) -> Self:
        """Add a stage to the builder.

        Stages don't need to be added in running order.

        Args
        ----
            stage StageClass: A Stage class
            inputs list[_StageArgument]: A list of inputs that the stage takes
            outputs list[_StageArgument]: A list of outputs the stage produces

        Returns
        -------
            Self
        """
        if stage in self._stages:
            raise ValueError(
                f"Stage {stage} has already been added, and cannot be added again.",
            )

        inputs = stage._inputs()  # noqa: SLF001
        outputs = stage._outputs()  # noqa: SLF001

        for output_name, output_type in self._named_tuple_to_stage_arguments(outputs):
            if isinstance(output_type, type) and (
                issubclass(output_type, RunBefore) or issubclass(output_type, RunAfter)
            ):
                raise TypeError(
                    f"Argument {output_name} is a {output_type}. "
                    + f"{output_type}s are only allowed as inputs.",
                )

        self._stages.add(stage)
        self._stage_inputs[stage] = self._named_tuple_to_stage_arguments(inputs)
        self._stage_outputs[stage] = self._named_tuple_to_stage_arguments(outputs)

        return self

    def build(
        self,
        initial_input_arguments: type[NamedTuple] | set[_StageArgument],
    ) -> StageRunner:
        """Resolve the stages, and produce a StageRunner to run them.

        This will check that all stages can be ran with inputs from previous stages, and
        resolve a running order for the stages.

        Returns
        -------
            StageRunner: A StageRunner for the given stages
        """
        if isinstance(initial_input_arguments, set):
            initial_input_annotations = initial_input_arguments
        else:
            initial_input_annotations = self._named_tuple_to_stage_arguments(
                initial_input_arguments,
            )

        stage_order = self._resolve_stages(initial_input_annotations)

        return StageRunner(
            stage_order,
            self._stage_inputs,
            self._stage_outputs,
            initial_input_annotations,
        )

    def _resolve_stages(
        self,
        initial_input_annotations: set[_StageArgument],
    ) -> list[StageScheduleElement]:
        resolved_stages: set[StageClass] = set()

        stage_order: list[StageScheduleElement] = []

        available_inputs: set[_StageArgument] = initial_input_annotations

        previous_resolved_stages: set[StageClass] | None = None

        # Find mutating stages
        mutating_stages: dict[StageClass, set[_StageArgument]] = (
            self._get_mutating_stages()
        )

        # Find before and after restrictions
        ordering_restrictions: set[_BeforeAfterRestriction] = (
            self._get_ordering_restrictions()
        )

        while (
            previous_resolved_stages is None
            or len(resolved_stages - previous_resolved_stages) > 0
        ):
            # No stages left to resolve, return the ordering
            if len(self._stages - resolved_stages) == 0:
                return stage_order

            previous_resolved_stages = resolved_stages.copy()

            stages_can_run: set[StageClass] = set()

            # Find stages to run
            for stage in self._stages - resolved_stages:
                if self._stage_resolves(
                    stage,
                    resolved_stages,
                    available_inputs,
                    ordering_restrictions,
                ):
                    stages_can_run.add(stage)

            # Do the mutating stage rule
            stages_can_run_post_mutating_check = self._mutating_stages_check(
                stages_can_run,
                mutating_stages,
            )

            # Check for stages with the same output running at the same time
            for stage in stages_can_run_post_mutating_check:
                for other_stage in stages_can_run_post_mutating_check - {stage}:
                    shared_outputs = (
                        self._stage_outputs[stage] & self._stage_outputs[other_stage]
                    )
                    if len(shared_outputs) > 0:
                        raise StageResolutionError(
                            f"The order {stage} and {other_stage} run is ambiguous. "
                            + "Add a RunBefore to one of them to explicitly specify an "
                            + "ordering for them. Reason: Same output resolved at "
                            + "same time.",
                        )

            # Add outputs of stages that can run to the list of available inputs
            for stage in stages_can_run_post_mutating_check:
                for argument in self._stage_outputs[stage]:
                    available_inputs.add(argument)
                    resolved_stages.add(stage)

            stage_order.append(list(stages_can_run_post_mutating_check))

        # If stage resolution failed, raise a detailed error
        if len(self._stages - resolved_stages) > 0:
            inputs_message = ""
            for stage in self._stages - resolved_stages:
                required_inputs = self._strip_run_restriction_arguments(
                    self._stage_inputs[stage],
                )

                missing_inputs = required_inputs - available_inputs
                inputs_message += f"    [{stage.__name__}]\n"
                inputs_message += "".join(
                    map(
                        lambda x: f"       {x.parameter_name}: {x.parameter_type}\n",
                        missing_inputs,
                    ),
                )

            available_inputs_message = ""
            available_inputs_message += "".join(
                map(
                    lambda x: f"       {x.parameter_name}: {x.parameter_type}\n",
                    available_inputs,
                ),
            )

            raise StageResolutionError(
                "Stages could not be resolved due to missing inputs.\n"
                + "Missing inputs:\n"
                + inputs_message
                + "\n"
                + "Available inputs:\n"
                + available_inputs_message,
            )

        return stage_order

    @staticmethod
    def _strip_run_restriction_arguments(
        arguments: set[_StageArgument],
    ) -> set[_StageArgument]:
        return {
            argument
            for argument in arguments
            if not isinstance(argument.parameter_type, type)
            or not (
                issubclass(argument.parameter_type, RunBefore)
                or issubclass(argument.parameter_type, RunAfter)
            )
        }

    def _get_mutating_stages(self) -> dict[StageClass, set[_StageArgument]]:
        mutating_stages: dict[StageClass, set[_StageArgument]] = {}

        for stage in self._stages:
            mutating_arguments: set[_StageArgument] = set()

            for stage_input in self._stage_inputs[stage]:
                if stage_input in self._stage_outputs[stage]:
                    mutating_arguments.add(stage_input)

            if len(mutating_arguments) > 0:
                mutating_stages[stage] = mutating_arguments

        return mutating_stages

    def _get_ordering_restrictions(self) -> set[_BeforeAfterRestriction]:
        ordering_restrictions: set[_BeforeAfterRestriction] = set()

        for stage in self._stages:
            for argument in self._stage_inputs[stage]:
                if isinstance(argument.parameter_type, type) and issubclass(
                    argument.parameter_type,
                    RunBefore,
                ):
                    before_stage = get_args(argument.parameter_type)[0]
                    ordering_restrictions.add(
                        _BeforeAfterRestriction(stage, before_stage),
                    )

                if isinstance(argument.parameter_type, type) and issubclass(
                    argument.parameter_type,
                    RunAfter,
                ):
                    after_stage = get_args(argument.parameter_type)[0]
                    ordering_restrictions.add(
                        _BeforeAfterRestriction(after_stage, stage),
                    )
        return ordering_restrictions

    @staticmethod
    def _get_restrictions_for_stage(
        ordering_restrictions: set[_BeforeAfterRestriction],
        stage: StageClass,
    ) -> _StageRestrictions:
        run_before: set[StageClass] = set()
        run_after: set[StageClass] = set()

        for before, after in ordering_restrictions:
            if before == stage:
                run_before.add(after)
            if after == stage:
                run_after.add(before)

        return _StageRestrictions(run_before, run_after)

    def _stage_resolves(
        self,
        stage: StageClass,
        resolved_stages: set[StageClass],
        available_inputs: set[_StageArgument],
        ordering_restrictions: set[_BeforeAfterRestriction],
    ) -> bool:
        required_inputs = self._strip_run_restriction_arguments(
            self._stage_inputs[stage],
        )
        _, run_after_stages = self._get_restrictions_for_stage(
            ordering_restrictions,
            stage,
        )

        all_required_inputs_resolved = len(required_inputs - available_inputs) == 0
        restrictions_resolved = len(run_after_stages - resolved_stages) == 0

        return all_required_inputs_resolved and restrictions_resolved

    def _mutating_stages_check(
        self,
        stages_can_run: set[StageClass],
        mutating_stages: dict[StageClass, set[_StageArgument]],
    ) -> set[StageClass]:
        mutating_stages_can_run = set(mutating_stages.keys()) & stages_can_run

        if len(mutating_stages_can_run) == 0:
            return stages_can_run

        # Check for duplicate mutating stages of the same argument
        for stage in mutating_stages_can_run:
            for other_stage in mutating_stages_can_run - {stage}:
                common_mutated_arguments = (
                    mutating_stages[stage] & mutating_stages[other_stage]
                )

                if len(common_mutated_arguments) > 0:
                    raise StageResolutionError(
                        f"The order {stage} and {other_stage} run is ambiguous. "
                        + "Add a RunBefore to one of them to explicitly specify an "
                        + "ordering for them. Reason: Mutating stage with shared "
                        + "mutating argument.",
                    )

        # Find a list of arguments mutated by mutating stages
        mutating_stage_arguments = set()
        for stage in mutating_stages_can_run:
            mutating_stage_arguments |= mutating_stages[stage]

        stages_can_run_post_mutating_check = set()

        # Remove any stages with an argument being mutated by a mutating stage
        for stage in stages_can_run:
            mutating_arguments_in_stage_input = (
                mutating_stage_arguments & self._stage_inputs[stage]
            )

            if stage in mutating_stages:
                mutating_arguments_in_stage_input -= mutating_stages[stage]

            if len(mutating_arguments_in_stage_input) == 0:
                stages_can_run_post_mutating_check.add(stage)

        return stages_can_run_post_mutating_check

    @staticmethod
    def _named_tuple_to_stage_arguments(
        named_tuple: type[NamedTuple],
    ) -> set[_StageArgument]:
        stage_arguments: set[_StageArgument] = set()

        for argument_name, argument_type in named_tuple.__annotations__.items():
            stage_arguments.add(_StageArgument(argument_name, argument_type))

        return stage_arguments


def _format_stage_arguments(stage_arguments: set[_StageArgument]) -> str:
    return "".join(
        map(
            lambda x: f"{x.parameter_name}, {x.parameter_type}\n",
            stage_arguments,
        ),
    )
