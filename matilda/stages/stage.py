"""Generic stage."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Generic, NamedTuple, TypeVar

IN = TypeVar("IN", bound=NamedTuple)
OUT = TypeVar("OUT", bound=NamedTuple)


class Stage(ABC, Generic[IN, OUT]):
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
    def _run(inputs: IN) -> OUT:
        """Run the stage."""
        raise NotImplementedError


StageClass = type[Stage[Any, Any]]
"""The class of a stage.

Used to annotate type when referencing a stage generically.

Usage::

    list_of_classes: list[StageClass] = [PrelimStage, CloisterStage]
"""

T = TypeVar("T", bound=StageClass)


class RunBefore(Generic[T]):
    """Marks that a stage should be run before another stage."""


class RunAfter(Generic[T]):
    """Marks that a stage should be run after another stage."""
