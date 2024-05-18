"""An abstract stage."""

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

IN = TypeVar("IN")
OUT = TypeVar("OUT")

class Stage(ABC, Generic[IN, OUT]):
    """An abstract stage that can be run by the stage runner."""

    _label: str

    def __init__(self) -> None:
        """Set the label on initialisation of the class."""
        self._label = self.__class__.__name__

    @staticmethod
    @abstractmethod
    def run(stage_input: IN) -> OUT:
        """Run the stage."""
        pass
