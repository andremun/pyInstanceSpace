"""An abstract stage."""

from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar, TypeVarTuple

IN = TypeVarTuple("IN")
OUT = TypeVar("OUT", bound=tuple[Any, ...])


class Stage(ABC, Generic[*IN, OUT]):
    """An abstract stage that can be run by the stage runner."""

    _label: str

    def __init__(self, run_input: list[type[Any]]) -> None:
        """Set the label on initialisation of the class."""
        self._label = self.__class__.__name__
        self._input = run_input

    @staticmethod
    @abstractmethod
    def _inputs() -> list[type[Any]]:
        pass

    @staticmethod
    @abstractmethod
    def run(*args: *IN) -> OUT:
        """Run the stage."""
        pass
