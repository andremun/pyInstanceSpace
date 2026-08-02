# SPDX-License-Identifier: LicenseRef-PolyForm-Noncommercial-1.0.0
# Copyright (c) 2024-2026 Mario Andrés Muñoz
"""A runner to run a list of stages, and a builder to attach extra stages to one.

`build_stage_runner()` is the entry point: given a caller-supplied, already
-ordered base schedule plus any extra/plugin stages to attach via
`RunBefore`/`RunAfter`, it produces a ready-to-use `StageRunner`.
"""

from collections import defaultdict
from collections.abc import Generator
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from typing import Any, NamedTuple, get_args, get_origin

from instancespace.stages.stage import OUT, RunAfter, RunBefore, Stage, StageClass

StageScheduleElement = list[StageClass]


def _default_false() -> bool:
    """Return False. A module-level `defaultdict` factory, so `StageRunner` can pickle.

    A lambda can't be pickled (`pickle` looks up callables by qualified name
    at module scope) - anything holding a `StageRunner` (e.g. `InstanceSpace`)
    would fail to serialise otherwise.
    """
    return False


def _is_run_restriction_type(parameter_type: type) -> bool:
    """Check whether a field's type is RunBefore[X]/RunAfter[X].

    These are subscripted generics (e.g. `typing._GenericAlias`), not plain
    classes - get_origin() is the correct way to identify them, not
    isinstance()/issubclass() on the type itself.
    """
    return get_origin(parameter_type) in (RunBefore, RunAfter)


def _deepcopy_stage_inputs(inputs: NamedTuple) -> NamedTuple:
    """Deep-copy a stage's resolved inputs, except any live `ThreadPoolExecutor`.

    `run_stage()` deep-copies inputs so a stage can't mutate the runner's own
    `_available_arguments` state in place. A cached, reused pool (Q6 - see
    `TraceInputs.executor`) must be exempted from that: it isn't
    deepcopy-safe (it holds OS-level thread/queue state `copy.deepcopy`
    can't serialise), and even if it were, copying it would silently create
    a redundant pool instead of submitting to the shared one, defeating the
    entire point of caching it. Pre-seeding `deepcopy`'s memo with the
    executor's id is the standard way to say "this object is already
    copied" and have the original reference returned unchanged.
    """
    memo = {
        id(value): value for value in inputs if isinstance(value, ThreadPoolExecutor)
    }
    return deepcopy(inputs, memo)


class _StageArgument(NamedTuple):
    """An input or output of a stage."""

    parameter_name: str
    parameter_type: type


class StageResolutionError(Exception):
    """An error attaching an extra stage to a base schedule."""


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

    @staticmethod
    def _debug_print(a: Any, do_print: bool) -> None:  # noqa: ANN401
        if do_print:
            print("[DEBUG]: ", end="")
            print(a)

    def __init__(
        self,
        stages: list[StageScheduleElement],
        input_arguments: dict[StageClass, set[_StageArgument]],
        output_arguments: dict[StageClass, set[_StageArgument]],
        initial_input_annotations: set[_StageArgument],
    ) -> None:
        """Create a StageRunner from a preresolved set of stages.

        @private

        All stages inputs and outputs are assumed to already be resolved.
        """
        self._stage_order = stages

        self._schedule_output_data = [{}]
        self._current_schedule_item = 0

        self._available_arguments = {}
        self._stage_to_schedule_index = {}
        self._stages_ran = defaultdict(_default_false)

        for i, schedule in enumerate(self._stage_order):
            for stage in schedule:
                self._stage_to_schedule_index[stage] = i

        self._check_stage_order_is_runnable(
            stages,
            input_arguments,
            output_arguments,
            initial_input_annotations,
        )

    def __getstate__(self) -> dict[str, Any]:
        """Drop the live `ThreadPoolExecutor` (if any) so this can be pickled.

        `_available_arguments`/`_schedule_output_data` may hold a reference
        to `InstanceSpace`'s cached executor (see `TraceInputs.executor`,
        Q6) under the `"executor"` key - a `ThreadPoolExecutor` holds OS
        thread/queue state `pickle` can't serialise. Every caller that runs
        a stage (`InstanceSpace.run_stage()`/`build()`/`run_iter()`/
        `run_until_stage()`) always re-supplies a fresh executor before
        running anything, so dropping the stale reference here is safe -
        nothing downstream ever reads it straight out of a restored
        checkpoint without going through one of those call sites first.
        """
        state = self.__dict__.copy()
        state["_available_arguments"] = {
            key: value
            for key, value in state["_available_arguments"].items()
            if key != "executor"
        }
        state["_schedule_output_data"] = [
            {key: value for key, value in schedule.items() if key != "executor"}
            for schedule in state["_schedule_output_data"]
        ]
        return state

    def run_iter(
        self,
        additional_arguments: NamedTuple,
    ) -> Generator[AnnotatedStageOutput, None, dict[str, Any]]:
        """Run all stages, yielding after every run.

        Yields
        ------
            Generator[AnnotatedStageOutput, None, dict[str, Any]]: _description_
        """
        self._rollback_to_schedule_index(0)

        self._available_arguments = additional_arguments._asdict()

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
        StageRunner._debug_print("running " + stage.__name__, True)
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

        for input_name, input_type in input_arguments.__annotations__.items():
            # RunBefore[X]/RunAfter[X] fields are only read while building the
            # schedule (see build_stage_runner() below) - nothing ever produces
            # them as an output, so they're left unset here and fall back to
            # their NamedTuple field default instead.
            if _is_run_restriction_type(input_type):
                continue
            # TODO: Some sort of type check on the inputs
            raw_inputs[input_name] = available_arguments[input_name]

        inputs: NamedTuple = input_arguments.__new__(input_arguments, **raw_inputs)

        outputs = stage._run(_deepcopy_stage_inputs(inputs))  # noqa: SLF001

        for output_name, output_value in outputs._asdict().items():
            self._available_arguments[output_name] = output_value

        self._schedule_output_data[self._current_schedule_item] = (
            self._available_arguments
        )

        self._stages_ran[stage] = True

        self._progress_schedule()

        return outputs

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
        input_arguments: dict[StageClass, set[_StageArgument]],
        output_arguments: dict[StageClass, set[_StageArgument]],
        initial_input_annotations: set[_StageArgument],
    ) -> None:
        available_arguments = initial_input_annotations.copy()

        for schedule_element in stages:
            for stage in schedule_element:
                required_inputs = {
                    argument
                    for argument in input_arguments[stage]
                    if not _is_run_restriction_type(argument.parameter_type)
                }
                if len(required_inputs - available_arguments) > 0:
                    raise StageRunningError(
                        "Stage order was not runnable. Not all inputs were available "
                        + "for a stage at the time of running. Missing inputs: "
                        + f"{list(required_inputs - available_arguments)}",
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


def named_tuple_to_stage_arguments(
    named_tuple: type[NamedTuple],
) -> set[_StageArgument]:
    """Extract a NamedTuple's fields as a set of `_StageArgument`s."""
    return {
        _StageArgument(name, arg_type)
        for name, arg_type in named_tuple.__annotations__.items()
    }


def build_stage_runner(
    base_order: list[StageScheduleElement],
    extra_stages: list[StageClass],
    initial_input_arguments: type[NamedTuple] | set[_StageArgument],
) -> StageRunner:
    """Attach extra stages to a caller-supplied base order and build a StageRunner.

    `base_order` is a caller-supplied `list[StageScheduleElement]` (a list of
    "waves" - stages within the same wave are order-independent of each
    other). It is taken as given, not inferred: the caller owns their own
    pipeline's shape. This lets `InstanceSpace` hardcode its own known
    7-stage order rather than have it re-derived from scratch on every
    construction.

    Each stage in `extra_stages` must declare where it attaches relative to
    a stage already in `base_order`, via a `RunBefore[X]`/`RunAfter[X]` field
    in its `_inputs()` NamedTuple - it is not placed by matching its
    input/output types against the rest of the pipeline. It is inserted as
    its own new wave immediately before/after X's wave; multiple extras
    resolving to the same attachment point share that new wave.

    ##Example:##
    ```python
    runner = build_stage_runner(
        base_order=[[PrelimStage], [SiftedStage]],
        extra_stages=[MyPlugin],  # MyPlugin's inputs declare RunAfter[SiftedStage]
        initial_input_arguments=initial_input_arguments,
    )
    ```
    """
    stage_inputs: dict[StageClass, set[_StageArgument]] = {}
    stage_outputs: dict[StageClass, set[_StageArgument]] = {}

    def register_stage(stage: StageClass) -> None:
        stage_input_type = stage._inputs()  # noqa: SLF001
        stage_output_type = stage._outputs()  # noqa: SLF001
        stage_inputs[stage] = named_tuple_to_stage_arguments(stage_input_type)
        stage_outputs[stage] = named_tuple_to_stage_arguments(stage_output_type)

    for wave in base_order:
        for stage in wave:
            register_stage(stage)

    for stage in extra_stages:
        if stage in stage_inputs:
            raise ValueError(
                f"Stage {stage} has already been added, and cannot be added again.",
            )
        register_stage(stage)

    stage_order = _attach_extra_stages(base_order, extra_stages, stage_inputs)

    if isinstance(initial_input_arguments, set):
        initial_input_annotations = initial_input_arguments
    else:
        initial_input_annotations = named_tuple_to_stage_arguments(
            initial_input_arguments,
        )

    return StageRunner(
        stage_order,
        stage_inputs,
        stage_outputs,
        initial_input_annotations,
    )


def _attach_extra_stages(
    base_order: list[StageScheduleElement],
    extra_stages: list[StageClass],
    stage_inputs: dict[StageClass, set[_StageArgument]],
) -> list[StageScheduleElement]:
    # Index stages in the *original* base order before any insertions, so
    # every extra's attachment point is computed against the same reference
    # regardless of insertion order.
    base_wave_index: dict[StageClass, int] = {}
    for i, wave in enumerate(base_order):
        for stage in wave:
            base_wave_index[stage] = i

    # Group extras by resolved insertion index, so extras attaching at the
    # same point share one new wave (mirroring how e.g. CloisterStage and
    # PythiaStage already share a wave in the built-in order).
    insertions: dict[int, list[StageClass]] = {}
    for stage in extra_stages:
        insertions.setdefault(
            _resolve_attachment_index(stage, stage_inputs, base_wave_index),
            [],
        ).append(stage)

    stage_order = [list(wave) for wave in base_order]
    for index in sorted(insertions, reverse=True):
        stage_order.insert(index, insertions[index])

    return stage_order


def _resolve_attachment_index(
    stage: StageClass,
    stage_inputs: dict[StageClass, set[_StageArgument]],
    base_wave_index: dict[StageClass, int],
) -> int:
    run_after_targets: list[StageClass] = []
    run_before_targets: list[StageClass] = []

    for argument in stage_inputs[stage]:
        # RunBefore[X]/RunAfter[X] are subscripted generics (e.g.
        # `typing._GenericAlias`), not plain classes - get_origin() is the
        # correct way to identify them, not isinstance()/issubclass().
        origin = get_origin(argument.parameter_type)
        if origin is RunAfter:
            run_after_targets.append(get_args(argument.parameter_type)[0])
        elif origin is RunBefore:
            run_before_targets.append(get_args(argument.parameter_type)[0])

    if not run_after_targets and not run_before_targets:
        raise StageResolutionError(
            f"{stage} was added as an extra stage but declares no "
            "RunBefore[X]/RunAfter[X] input field naming a stage in the "
            "base order. Extra stages must explicitly declare where they "
            "attach - their position is no longer inferred from matching "
            "input/output types.",
        )

    for target in run_after_targets + run_before_targets:
        if target not in base_wave_index:
            raise StageResolutionError(
                f"{stage} declares RunBefore/RunAfter {target}, but "
                f"{target} is not in this builder's base order.",
            )

    after_index = (
        max(base_wave_index[t] for t in run_after_targets) + 1
        if run_after_targets
        else None
    )
    before_index = (
        min(base_wave_index[t] for t in run_before_targets)
        if run_before_targets
        else None
    )

    if after_index is not None and before_index is not None:
        if after_index > before_index:
            raise StageResolutionError(
                f"{stage}'s RunAfter and RunBefore targets conflict - it "
                "would need to run both after and before the same point "
                "in the base order.",
            )
        return after_index

    if after_index is not None:
        return after_index

    assert before_index is not None  # guaranteed by the earlier raise
    return before_index
