from abc import ABC
from collections import abc

class Stage(ABC):
    @staticmethod
    @abc
    def _inputs() -> list[tuple[str, type]]:
        """Return inputs of the STAGE (run method)."""
        pass

    @staticmethod
    @abc
    def _outputs() -> list[tuple[str, type]]:
        """Return outputs of the STAGE (run method)."""
        pass

    @staticmethod
    @abc
    def _run(*args: tuple[any]) -> tuple[any]:
        """Run the stage."""
        pass