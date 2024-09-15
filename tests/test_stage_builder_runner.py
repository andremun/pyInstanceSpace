"""Test the StageBuilder and StageRunner."""

from typing import NamedTuple

from matilda.stage_builder import StageBuilder
from matilda.stages.stage import Stage


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


def test_basic_stage_resolution() -> None:
    """Test a very basic stage resolution case."""
    stage_builder = StageBuilder()
    stage_builder.add_stage(StageB)
    stage_builder.add_stage(StageA)

    stage_runner = stage_builder.build(InitialArguments)

    stage_ordering = stage_runner._stage_order  # noqa: SLF001

    assert stage_ordering == [[StageA], [StageB]]


def test_running_basic_example() -> None:
    """Make sure running a very simple example works."""
    stage_builder = StageBuilder()
    stage_builder.add_stage(StageB)
    stage_builder.add_stage(StageA)

    stage_runner = stage_builder.build(InitialArguments)

    initial_arguments = InitialArguments(1)

    stage_runner.run_all(initial_arguments)
