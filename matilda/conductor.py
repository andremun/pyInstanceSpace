"""A generic stage manager."""

from abc import ABC
from typing import Any, Generic, TypeVar

from matilda.stage import Stage


class StageResolutionError(Exception):
    """The conductor could not be initialised.

    This happens when a conductor is initialised with stages that can't resolve its
    inputs. This can happen if no other stage or the initial input return one of its
    inputs, or if there is a cyclic dependency.
    """

    pass


S = TypeVar("S", bound=Stage)
class Conductor(ABC, Generic[S]):
    """A generic stage manager."""

    _stage_schedule: list[set[type[S]]]

    _inputs: list[Any]

    def __init__(self, stages: list[type[S]], *inputs: list[Any]) -> None:
        """Initialise a conductor."""
        # Stages that have been successfully added to the conductors stage schedule
        resolved_stages: list[set[type[S]]] = []

        # Inputs that are available for future stages that haven't been resolved
        available_input_types: set[type] = {type(i) for i in inputs}

        # The resolved stages at the last iteration of resolving. Used to check that
        # resolution hasn't stalled with no possible answer.
        last_resolved_stages = stages.copy()

        while len(stages) > 0:
            new_stages: set[type[S]] = set()
            for stage in stages:
                # If all of a stages inputs are available, it can be run at this time
                if (
                    len(
                        set(stage._inputs()).difference(  # noqa: SLF001
                            available_input_types,
                        ),
                    )
                    == 0
                ):
                    new_stages.add(stage)
                    stages.remove(stage)

            # The resolved stages haven't changed since last loop, so it will never
            # resolve.
            if len(last_resolved_stages) == len(stages) and len(stages) > 0:
                raise StageResolutionError

            resolved_stages.append(new_stages)

        self._stage_schedule = resolved_stages

    def print_schedule(self) -> None:
        """Print a representation of each stage."""
        for i, step in enumerate(self._stage_schedule):
            print(f"Step: {i}")

            for stage in step:
                print(f"    {stage.__name__}")

    def run(self) -> None:
        """Run all the stages."""
        raise NotImplementedError()

    def run_stage(self) -> None:
        """Run the specified stage."""
        raise NotImplementedError()
