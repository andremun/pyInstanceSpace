"""test."""

from abc import ABC
from typing import Any, TypeVar

T = TypeVar("T")


def _test_stage_decorator(base_class: type[T]) -> type[T]:
    return base_class


@_test_stage_decorator
class Stage:
    """An abstract stage that can be run by the stage runner."""

    pass

type StageInput = tuple[str, type]


class Conductor:
    """A generic stage manager."""

    _stage_schedule: list[set[type]]

    _inputs: list[StageInput]
