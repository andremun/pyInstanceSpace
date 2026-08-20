# SPDX-License-Identifier: LicenseRef-PolyForm-Noncommercial-1.0.0
# Copyright (c) 2024-2026 Mario Andrés Muñoz
"""Compatibility tests for the optional fitted-stage inference contract."""

from inspect import isabstract
from typing import NamedTuple

import pytest

from instancespace.stages import PredictiveStage as PublicPredictiveStage
from instancespace.stages.pilot import PilotStage
from instancespace.stages.prelim import PrelimStage
from instancespace.stages.pythia import PythiaStage
from instancespace.stages.sifted import SiftedStage
from instancespace.stages.stage import PredictiveStage, Stage
from instancespace.stages.trace import TraceStage


class _BuildInput(NamedTuple):
    value: int


class _BuildOutput(NamedTuple):
    result: int


class _BuildOnlyStage(Stage[_BuildInput, _BuildOutput]):
    @staticmethod
    def _inputs() -> type[_BuildInput]:
        return _BuildInput

    @staticmethod
    def _outputs() -> type[_BuildOutput]:
        return _BuildOutput

    @staticmethod
    def _run(inputs: _BuildInput) -> _BuildOutput:
        return _BuildOutput(inputs.value + 1)


class _InferenceOnlyStage(PredictiveStage[int, int, int]):
    @staticmethod
    def predict(inputs: int, fitted: int) -> int:
        return inputs + fitted


def test_build_only_stage_does_not_need_predict() -> None:
    """Existing build stages and plugins remain concrete without inference."""
    assert not isabstract(_BuildOnlyStage)
    assert _BuildOnlyStage._run(_BuildInput(1)) == _BuildOutput(2)


def test_predictive_stage_applies_separate_fitted_state() -> None:
    """Inference accepts types independent of the build runner's NamedTuples."""
    expected = 5
    assert _InferenceOnlyStage.predict(2, 3) == expected


def test_predictive_stage_is_exported_from_the_stages_package() -> None:
    """Expose the optional contract without changing the build-only Stage API."""
    assert PublicPredictiveStage is PredictiveStage


@pytest.mark.parametrize(
    "stage",
    [PrelimStage, SiftedStage, PilotStage, PythiaStage, TraceStage],
)
def test_each_inference_capable_stage_implements_the_optional_contract(
    stage: type[object],
) -> None:
    """Keep the five built-in inference owners explicit and concrete."""
    assert issubclass(stage, PredictiveStage)
    assert not isabstract(stage)
