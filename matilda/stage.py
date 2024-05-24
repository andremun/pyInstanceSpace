"""An abstract stage."""

from abc import ABC, abstractmethod
from typing import Any, Generic, ParamSpec, TypeVar, TypeVarTuple

# T = TypeVar("T")
# class StageParameter(Generic[T]):
#     """An input or output for a stage."""

#     _parameter: T

#     @property
#     def parameter(self) -> T:
#         """Get the parameter."""
#         return self._parameter

#     def __init__(self, parameter: T) -> None:
#         """Initialise a stage parameter."""
#         self._parameter = parameter

IN = TypeVarTuple("IN")
OUT = TypeVar("OUT", bound=tuple)

class Stage(ABC, Generic[*IN, OUT]):
    """An abstract stage that can be run by the stage runner."""

    _label: str

    def __init__(self) -> None:
        """Set the label on initialisation of the class."""
        self._label = self.__class__.__name__

    @staticmethod
    @abstractmethod
    def run(*args: *IN) -> OUT:
        """Run the stage."""
        pass
