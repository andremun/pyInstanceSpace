"""Generic stage."""

from abc import ABC, abstractmethod
from types import UnionType


class Stage(ABC):
    """Generic stage."""

    def __init__(self) -> None:
        """Initialize the stage."""
        return

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

    @abstractmethod
    def _run(self, *args: Any) -> tuple[Any]:
        """Run the stage."""
        pass
