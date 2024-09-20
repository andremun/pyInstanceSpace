"""Generic stage."""

from abc import ABC, abstractmethod
from types import UnionType


class Stage(ABC):
    """Generic stage."""

    @staticmethod
    @abstractmethod
    def _inputs() -> list[tuple[str, type | UnionType]]:
        """Return inputs of the STAGE (run method)."""
        pass

    @staticmethod
    @abstractmethod
    def _outputs() -> list[tuple[str, type | UnionType]]:
        """Return outputs of the STAGE (run method)."""
        pass
