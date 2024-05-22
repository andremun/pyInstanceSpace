"""A generic stage manager."""

from abc import ABC
from typing import Any

from matilda.stage import Stage


class StageResolutionError(Exception):
    """The conductor could not be initialised.

    This happens when a conductor is initialised with stages that can't resolve its
    inputs. This can happen if no other stage or the initial input return one of its
    inputs, or if there is a cyclic dependency.
    """

    pass

# S = TypeVar("S")
class Conductor(ABC):
    """A generic stage manager."""

    _stage_schedule: list[set[type[Stage]]]

    def __init__(self, stages: list[type[Stage]], *inputs: list[Any]) -> None:
        """Initialise a conductor."""
        # Stages that have been successfully added to the conductors stage schedule
        resolved_stages: list[set[type[Stage]]] = []

        # Inputs that are available for future stages that haven't been resolved
        available_input_types: set[type] = {type(i) for i in inputs}

        # The resolved stages at the last iteration of resolving. Used to check that
        # resolution hasn't stalled with no possible answer.
        last_resolved_stages = stages.copy()

        while len(stages) > 0:
            new_stages: set[type] = set()
            for stage in stages:
                pass

            # The resolved stages haven't changed since last loop, so it will never
            # resolve.
            if len(last_resolved_stages) == len(stages) and len(stages) > 0:
                raise StageResolutionError

    def run(self) -> None:
        """Run all the stages."""
        raise NotImplementedError()


    def run_stage(self) -> None:
        """Run the specified stage."""
        raise NotImplementedError()
