# SPDX-License-Identifier: LicenseRef-PolyForm-Noncommercial-1.0.0
# Copyright (c) 2024-2026 Mario Andrés Muñoz
"""Generic stage."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Generic, NamedTuple, TypeVar

IN = TypeVar("IN", bound=NamedTuple)
OUT = TypeVar("OUT", bound=NamedTuple)
PREDICT_IN = TypeVar("PREDICT_IN")
FITTED = TypeVar("FITTED")
PREDICT_OUT = TypeVar("PREDICT_OUT")


class Stage(ABC, Generic[IN, OUT]):
    """Generic stage."""

    @staticmethod
    @abstractmethod
    def _inputs() -> type[NamedTuple]:
        """Return inputs of the STAGE (run method)."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def _outputs() -> type[NamedTuple]:
        """Return outputs of the STAGE (run method)."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def _run(inputs: IN) -> OUT:
        """Run the stage."""
        raise NotImplementedError


class PredictiveStage(ABC, Generic[PREDICT_IN, FITTED, PREDICT_OUT]):
    """Optional contract for applying a fitted stage to unseen data.

    This is separate from :class:`Stage` so build-only stages and plugins do not
    need to implement inference, and so the build-only ``StageRunner`` contract
    remains unchanged.
    """

    @staticmethod
    @abstractmethod
    def predict(inputs: PREDICT_IN, fitted: FITTED) -> PREDICT_OUT:
        """Apply fitted state without training or mutating it."""
        raise NotImplementedError


StageClass = type[Stage[Any, Any]]
"""The class of a stage.

Used to annotate type when referencing a stage generically.

Usage::

    list_of_classes: list[StageClass] = [PrelimStage, CloisterStage]
"""

T = TypeVar("T", bound=Stage[Any, Any])


class RunBefore(Generic[T]):
    """Marks that a stage should be run before another stage.

    Usage::

        class MyInput(NamedTuple):
            run_before: RunBefore[SomeStage] = RunBefore()
            ...
    """


class RunAfter(Generic[T]):
    """Marks that a stage should be run after another stage.

    Usage::

        class MyInput(NamedTuple):
            run_after: RunAfter[SomeStage] = RunAfter()
            ...
    """
