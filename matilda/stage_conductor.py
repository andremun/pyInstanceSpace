"""TODO: Write this."""

from typing import Any

from matilda.conductor_types import Stage


class StageConductor:
    """TODO: write this."""

    def __init__(self, stages: list[Stage[Any]]) -> None:
        """Init."""
        self.stages = stages
