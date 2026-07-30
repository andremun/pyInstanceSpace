"""Test StageRunner and build_stage_runner (formerly StageBuilder, folded in)."""

import pickle
from concurrent.futures import ThreadPoolExecutor
from typing import NamedTuple

import pytest

from instancespace.stage_runner import StageResolutionError, build_stage_runner
from instancespace.stages.stage import RunAfter, RunBefore, Stage


class InitialArguments(NamedTuple):
    """Initial arguments for very basic test."""

    a: int


class _StageAInput(NamedTuple):
    a: int


class _StageAOutput(NamedTuple):
    b: str


class StageA(Stage[_StageAInput, _StageAOutput]):
    """Basic stage for tests."""

    @staticmethod
    def _inputs() -> type[NamedTuple]:
        return _StageAInput

    @staticmethod
    def _outputs() -> type[NamedTuple]:
        return _StageAOutput

    @staticmethod
    def _run(inputs: _StageAInput) -> _StageAOutput:
        return _StageAOutput(inputs.a.__str__())


class _StageBInput(NamedTuple):
    b: str


class _StageBOutput(NamedTuple):
    c: str


class StageB(Stage[_StageBInput, _StageBOutput]):
    """Basic stage for tests."""

    @staticmethod
    def _inputs() -> type[NamedTuple]:
        return _StageBInput

    @staticmethod
    def _outputs() -> type[NamedTuple]:
        return _StageBOutput

    @staticmethod
    def _run(inputs: _StageBInput) -> _StageBOutput:
        return _StageBOutput(inputs.b.__str__() + " 2")


class _StageCInput(NamedTuple):
    b: str
    run_after: RunAfter[StageA] = RunAfter()


class _StageCOutput(NamedTuple):
    d: str


class StageC(Stage[_StageCInput, _StageCOutput]):
    """An extra stage attached via RunAfter."""

    @staticmethod
    def _inputs() -> type[NamedTuple]:
        return _StageCInput

    @staticmethod
    def _outputs() -> type[NamedTuple]:
        return _StageCOutput

    @staticmethod
    def _run(inputs: _StageCInput) -> _StageCOutput:
        return _StageCOutput(inputs.b.__str__() + " c")


class _StageDInput(NamedTuple):
    a: int
    run_before: RunBefore[StageB] = RunBefore()


class _StageDOutput(NamedTuple):
    e: str


class StageD(Stage[_StageDInput, _StageDOutput]):
    """An extra stage attached via RunBefore."""

    @staticmethod
    def _inputs() -> type[NamedTuple]:
        return _StageDInput

    @staticmethod
    def _outputs() -> type[NamedTuple]:
        return _StageDOutput

    @staticmethod
    def _run(inputs: _StageDInput) -> _StageDOutput:
        return _StageDOutput(inputs.a.__str__() + " d")


class _StageEInput(NamedTuple):
    a: int


class _StageEOutput(NamedTuple):
    f: str


class StageE(Stage[_StageEInput, _StageEOutput]):
    """An extra stage that forgets to declare an attachment point."""

    @staticmethod
    def _inputs() -> type[NamedTuple]:
        return _StageEInput

    @staticmethod
    def _outputs() -> type[NamedTuple]:
        return _StageEOutput

    @staticmethod
    def _run(inputs: _StageEInput) -> _StageEOutput:
        return _StageEOutput(inputs.a.__str__())


class _StageFInput(NamedTuple):
    a: int
    executor: ThreadPoolExecutor | None = None


class _StageFOutput(NamedTuple):
    received_executor: ThreadPoolExecutor | None


class StageF(Stage[_StageFInput, _StageFOutput]):
    """A stage whose input carries a live, non-deepcopy-safe resource (Q6-style)."""

    @staticmethod
    def _inputs() -> type[NamedTuple]:
        return _StageFInput

    @staticmethod
    def _outputs() -> type[NamedTuple]:
        return _StageFOutput

    @staticmethod
    def _run(inputs: _StageFInput) -> _StageFOutput:
        return _StageFOutput(inputs.executor)


class InitialArgumentsWithExecutor(NamedTuple):
    """Initial arguments for the executor-passthrough test."""

    a: int
    executor: ThreadPoolExecutor | None = None


def test_run_stage_does_not_deepcopy_a_live_executor() -> None:
    """A ThreadPoolExecutor-typed input passes through by reference, not copied.

    `run_stage()` deep-copies its resolved inputs to protect the runner's own
    state - but `ThreadPoolExecutor` isn't deepcopy-safe (it holds OS-level
    thread/queue state `copy.deepcopy` can't handle) and, even if it were,
    copying it would silently create a redundant pool instead of reusing the
    shared one, defeating Q6's entire purpose. Regression test for a real
    `TypeError: cannot pickle '_queue.SimpleQueue' object` crash found via
    the T2 end-to-end integration test.
    """
    executor = ThreadPoolExecutor(max_workers=1)
    try:
        stage_runner = build_stage_runner(
            [[StageF]],
            [],
            InitialArgumentsWithExecutor,
        )

        output = stage_runner.run_stage(StageF, a=1, executor=executor)

        assert output.received_executor is executor
    finally:
        executor.shutdown(wait=True)


def test_running_basic_example() -> None:
    """Make sure running a caller-supplied explicit order works."""
    stage_runner = build_stage_runner([[StageA], [StageB]], [], InitialArguments)

    initial_arguments = InitialArguments(1)

    output = stage_runner.run_all(initial_arguments)

    assert output == {
        "a": 1,
        "b": "1",
        "c": "1 2",
    }


def test_rerunning_earlier_stage() -> None:
    """Make sure re-running an earlier stage rolls back and re-runs later ones."""
    stage_runner = build_stage_runner([[StageA], [StageB]], [], InitialArguments)

    initial_arguments = InitialArguments(1)
    output = stage_runner.run_all(initial_arguments)

    assert output == {
        "a": 1,
        "b": "1",
        "c": "1 2",
    }

    stage_runner.run_stage(StageA, a=2)
    stage_b_output = stage_runner.run_stage(StageB)

    assert stage_b_output._asdict() == {
        "c": "2 2",
    }


def test_extra_stage_run_after() -> None:
    """An extra stage declaring RunAfter attaches immediately after its target."""
    stage_runner = build_stage_runner(
        [[StageA], [StageB]],
        [StageC],
        InitialArguments,
    )

    assert stage_runner._stage_order == [[StageA], [StageC], [StageB]]  # noqa: SLF001

    output = stage_runner.run_all(InitialArguments(1))

    assert output == {
        "a": 1,
        "b": "1",
        "c": "1 2",
        "d": "1 c",
    }


def test_extra_stage_run_before() -> None:
    """An extra stage declaring RunBefore attaches immediately before its target."""
    stage_runner = build_stage_runner(
        [[StageA], [StageB]],
        [StageD],
        InitialArguments,
    )

    assert stage_runner._stage_order == [[StageA], [StageD], [StageB]]  # noqa: SLF001

    output = stage_runner.run_all(InitialArguments(1))

    assert output == {
        "a": 1,
        "b": "1",
        "c": "1 2",
        "e": "1 d",
    }


def test_multiple_extra_stages_share_a_wave() -> None:
    """Two extras resolving to the same attachment point share one new wave."""
    stage_runner = build_stage_runner(
        [[StageA], [StageB]],
        [StageC, StageD],
        InitialArguments,
    )

    assert stage_runner._stage_order == [  # noqa: SLF001
        [StageA],
        [StageC, StageD],
        [StageB],
    ]


def test_extra_stage_without_attachment_point_raises() -> None:
    """An extra stage with no RunBefore/RunAfter field is rejected clearly."""
    with pytest.raises(
        StageResolutionError,
        match="no RunBefore\\[X\\]/RunAfter\\[X\\]",
    ):
        build_stage_runner([[StageA], [StageB]], [StageE], InitialArguments)


def test_stage_runner_is_picklable_after_running_a_stage() -> None:
    """Regression test for #293: a live `_stages_ran` must not block pickling.

    `defaultdict(lambda: False)` can't be pickled - `StageRunner` (and
    therefore anything holding one, e.g. `InstanceSpace`) would fail to
    serialise once any stage had run and populated `_stages_ran`.
    """
    stage_runner = build_stage_runner([[StageA], [StageB]], [], InitialArguments)
    stage_runner.run_all(InitialArguments(1))

    restored = pickle.loads(pickle.dumps(stage_runner))

    assert restored._stages_ran == stage_runner._stages_ran  # noqa: SLF001
    restored_args = restored._available_arguments  # noqa: SLF001
    stage_runner_args = stage_runner._available_arguments  # noqa: SLF001
    assert restored_args == stage_runner_args
    # The restored defaultdict's factory must still work, not just its
    # already-populated entries.
    assert restored._stages_ran[StageC] is False  # noqa: SLF001


def test_stage_runner_is_picklable_with_a_live_executor_in_arguments() -> None:
    """`StageRunner.__getstate__()` must strip a live executor before pickling.

    Unlike `test_run_stage_does_not_deepcopy_a_live_executor` (which only needs
    pass-through-by-reference within one process), a `ThreadPoolExecutor` can't
    be pickled at all once it lands in `_available_arguments`/
    `_schedule_output_data` (Q6-style, see `InitialArgumentsWithExecutor`) - this
    is what checkpointing an `InstanceSpace` mid-pipeline (`InstanceSpace.save()`)
    relies on not crashing on. `run_all()` (rather than a direct `run_stage()`
    call) is what actually seeds `"executor"` into `_available_arguments` - it
    comes from the *initial* arguments NamedTuple, not from any stage's output.
    Uses `StageA` (which never reads or outputs `"executor"`) rather than
    `StageF`, since `_InstanceSpaceInputs.executor` similarly sits unread by
    most real stages - only the ones that actually declare it as an input
    (e.g. `TraceInputs.executor`) ever touch it.
    """
    executor = ThreadPoolExecutor(max_workers=1)
    try:
        stage_runner = build_stage_runner(
            [[StageA]],
            [],
            InitialArgumentsWithExecutor,
        )
        stage_runner.run_all(InitialArgumentsWithExecutor(a=1, executor=executor))

        restored = pickle.loads(pickle.dumps(stage_runner))

        assert "executor" not in restored._available_arguments  # noqa: SLF001
        assert all(
            "executor" not in schedule
            for schedule in restored._schedule_output_data  # noqa: SLF001
        )
    finally:
        executor.shutdown(wait=True)
