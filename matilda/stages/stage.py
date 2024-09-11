"""Generic stage."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, NamedTuple, TypeVar

IN = TypeVar("IN", bound=NamedTuple)


class Stage(ABC, Generic[IN]):
    """Generic stage."""

    @staticmethod
    @abstractmethod
    def _inputs() -> type[NamedTuple]:
        """Return inputs of the STAGE (run method)."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def _outputs() -> type[NamedTuple]:
        """Return outputs of the STAGE (run method)."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def _run(inputs: IN) -> NamedTuple:
        """Run the stage."""
        raise NotImplementedError


T = TypeVar("T", bound=type[Stage])


class RunBefore(Generic[T]):
    """Marks that a stage should be run before another stage."""


class RunAfter(Generic[T]):
    """Marks that a stage should be run after another stage."""

    pass
