"""Generic stage."""

from abc import ABC, abstractmethod
from typing import Any


class Stage(ABC):
    """Generic stage."""

    @staticmethod
    @abstractmethod
    def _inputs() -> list[tuple[str, type]]:
        """Return inputs of the STAGE (run method)."""
        pass

    @staticmethod
    @abstractmethod
    def _outputs() -> list[tuple[str, type]]:
        """Return outputs of the STAGE (run method)."""
        pass


    @abstractmethod
    def _run(*args: Any) -> tuple[Any]:
        """Run the stage."""
        pass
