"""Generic stage."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, NamedTuple


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
    def _run(*args: Any) -> tuple[Any]:
        """Run the stage."""
        pass
