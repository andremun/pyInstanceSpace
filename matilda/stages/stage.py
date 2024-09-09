"""Generic stage."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Generic, NamedTuple, TypeVar


class StageArgument(NamedTuple):
    """An input or output of a stage."""

    parameter_name: str
    parameter_type: type


class Stage(ABC):
    """Generic stage."""

    @staticmethod
    @abstractmethod
    def _inputs() -> list[StageArgument]:
        """Return inputs of the STAGE (run method)."""
        pass

    @staticmethod
    @abstractmethod
    def _outputs() -> list[StageArgument]:
        """Return outputs of the STAGE (run method)."""
        pass

    @abstractmethod
    def _run(*args: Any) -> NamedTuple:  # noqa: ANN401
        """Run the stage."""
        pass


T = TypeVar("T", bound=type[Stage])


class RunBefore(Generic[T]):
    """Marks that a stage should be run before another stage."""


class RunAfter(Generic[T]):
    """Marks that a stage should be run after another stage."""

    pass
