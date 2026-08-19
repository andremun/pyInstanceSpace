# SPDX-License-Identifier: LicenseRef-PolyForm-Noncommercial-1.0.0
# Copyright (c) 2024-2026 Mario Andrés Muñoz
"""Tests for InstanceSpace checkpointing (`save()`/`load()`) and progress reporting.

Two groups:

- Real, but fast, round-trip tests against `PreprocessingStage`/`PrelimStage`
  only (not the full 7-stage pipeline, which is genuinely slow - see
  `test_build_integration.py`'s ~8.5 minute T2 test) - these exercise the
  actual `joblib`/HMAC round trip and confirm a checkpoint really can resume
  and produce the same result as an uninterrupted run.
- Bare, stubbed-`_runner` unit tests (mirroring `test_instance_space_executor.py`'s
  style) for `run_stage()`/`build()`'s progress-reporting call pattern, so they
  don't need a real pipeline to check who got called when.
"""

from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from instancespace.data.metadata import Metadata
from instancespace.data.options import InstanceSpaceOptions
from instancespace.instance_space import InstanceSpace, instance_space_from_files
from instancespace.model import Model, ModelSignatureError
from instancespace.progress_reporter import NullProgressReporter
from instancespace.stage_runner import StageRunningError
from instancespace.stages.cloister import CloisterStage
from instancespace.stages.pilot import PilotStage
from instancespace.stages.prelim import PrelimStage
from instancespace.stages.preprocessing import PreprocessingStage
from instancespace.stages.pythia import PythiaStage
from instancespace.stages.sifted import SiftedStage
from instancespace.stages.trace import TraceStage

# ---------------------------------------------------------------------------
# Real, fast round-trip tests
# ---------------------------------------------------------------------------


@pytest.fixture()
def two_stage_instance_space() -> InstanceSpace:
    """A real InstanceSpace with Preprocessing+Prelim already run (fast: no GA/SVM)."""
    script_dir = Path(__file__).resolve().parent
    metadata_path = script_dir / "test_data/preprocessing/metadata.csv"
    options_path = script_dir / "test_data/preprocessing/options.json"

    instance_space = instance_space_from_files(metadata_path, options_path)
    assert instance_space is not None
    instance_space.run_stage(PreprocessingStage)
    instance_space.run_stage(PrelimStage)
    return instance_space


def test_save_with_parallelism_disabled_loads_without_an_executor(
    two_stage_instance_space: InstanceSpace,
    tmp_path: Path,
) -> None:
    # This fixture's options set parallel.flag=false, so staged execution must
    # not create a pool merely for checkpointing.
    assert two_stage_instance_space._executor is None  # noqa: SLF001
    path = tmp_path / "checkpoint.joblib"

    two_stage_instance_space.save(path)
    loaded = InstanceSpace.load(path)

    assert loaded._executor is None  # noqa: SLF001
    assert loaded._executor_workers is None  # noqa: SLF001


def test_loaded_checkpoint_resumes_and_matches_an_uninterrupted_run(
    two_stage_instance_space: InstanceSpace,
    tmp_path: Path,
) -> None:
    path = tmp_path / "checkpoint.joblib"
    two_stage_instance_space.save(path)
    loaded = InstanceSpace.load(path)

    loaded.run_stage(SiftedStage)
    two_stage_instance_space.run_stage(SiftedStage)

    loaded_runner = loaded._runner  # noqa: SLF001
    uninterrupted_runner = two_stage_instance_space._runner  # noqa: SLF001
    loaded_selvars = loaded_runner._available_arguments["selvars"]  # noqa: SLF001
    uninterrupted_selvars = uninterrupted_runner._available_arguments[  # noqa: SLF001
        "selvars"
    ]
    assert np.array_equal(loaded_selvars, uninterrupted_selvars)


def test_signed_round_trip_and_tamper_detection(
    two_stage_instance_space: InstanceSpace,
    tmp_path: Path,
) -> None:
    path = tmp_path / "checkpoint.joblib"
    secret_key = b"a-server-managed-secret"

    two_stage_instance_space.save(path, secret_key=secret_key)
    loaded = InstanceSpace.load(path, secret_key=secret_key)
    assert loaded._executor is None  # noqa: SLF001

    with pytest.raises(ModelSignatureError):
        InstanceSpace.load(path, secret_key=b"wrong-key")

    with pytest.raises(ModelSignatureError):
        InstanceSpace.load(path)  # downgrade-attack guard


# ---------------------------------------------------------------------------
# Bare, stubbed-runner progress-reporting tests
# ---------------------------------------------------------------------------


def _bare_instance_space(reporter: Any = None) -> InstanceSpace:  # noqa: ANN401
    metadata = Metadata(
        feature_names=["f1"],
        algorithm_names=["a1"],
        instance_labels=pd.Series(["i1"]),
        instance_sources=None,
        features=np.zeros((1, 1)),
        algorithms=np.zeros((1, 1)),
    )
    options = InstanceSpaceOptions.default(*([None] * 12))

    space = InstanceSpace.__new__(InstanceSpace)
    space._metadata = metadata
    space._options = options
    space._progress_reporter = reporter or NullProgressReporter()
    space._model = None
    space._final_output = None
    space._stages = [
        PreprocessingStage,
        PrelimStage,
        SiftedStage,
        PilotStage,
        PythiaStage,
        CloisterStage,
        TraceStage,
    ]
    space._executor = None
    space._executor_workers = None
    return space


class FakeStage:
    """Stand-in `StageClass` - only `.__name__` is read by `_stage_report_name()`."""

    __name__ = "FakeStage"


def test_run_stage_reports_completion_but_not_job_completed_mid_schedule() -> None:
    reporter = MagicMock()
    space = _bare_instance_space(reporter)
    space._runner = cast(
        Any,
        SimpleNamespace(
            run_stage=lambda stage, **kwargs: "output",  # noqa: ARG005
            _available_arguments={"already": "seeded"},
            _current_schedule_item=1,
            _stage_order=[["wave0"], ["wave1"], ["wave2"]],
        ),
    )

    result: str = space.run_stage(FakeStage)  # type: ignore[arg-type]

    assert result == "output"
    reporter.report_stage_completed.assert_called_once()
    args, kwargs = reporter.report_stage_completed.call_args
    assert args[0] == "fake"
    assert kwargs["instance_space"] is space
    reporter.report_job_completed.assert_not_called()
    space.close()


def test_run_stage_reports_job_completed_when_schedule_finishes() -> None:
    reporter = MagicMock()
    space = _bare_instance_space(reporter)
    space._runner = cast(
        Any,
        SimpleNamespace(
            run_stage=lambda stage, **kwargs: "output",  # noqa: ARG005
            _available_arguments={"already": "seeded"},
            _current_schedule_item=2,
            _stage_order=[["wave0"], ["wave1"]],
        ),
    )

    space.run_stage(FakeStage)  # type: ignore[arg-type]

    reporter.report_job_completed.assert_called_once_with(instance_space=space)
    space.close()


def test_run_stage_reports_failure_and_reraises() -> None:
    reporter = MagicMock()
    space = _bare_instance_space(reporter)

    def failing_run_stage(stage: object, **kwargs: object) -> None:
        raise RuntimeError("boom")

    space._runner = cast(
        Any,
        SimpleNamespace(
            run_stage=failing_run_stage,
            _available_arguments={"already": "seeded"},
        ),
    )

    with pytest.raises(RuntimeError, match="boom"):
        space.run_stage(FakeStage)  # type: ignore[arg-type]

    reporter.report_stage_failed.assert_called_once_with("fake", "boom")
    reporter.report_job_failed.assert_called_once_with("boom")
    space.close()


def test_run_stage_seeds_initial_inputs_on_a_truly_fresh_runner() -> None:
    """The very first `run_stage()` call must seed inputs itself (no build() call).

    This is what lets a SLURM job that only ever calls `run_stage()` (never
    `build()`/`run_until_stage()`) bootstrap the very first stage of a fresh
    pipeline, not just resume a checkpoint.
    """
    space = _bare_instance_space()
    captured: dict[str, Any] = {}

    def fake_run_stage(stage: object, **kwargs: object) -> str:
        captured.update(space._runner._available_arguments)  # noqa: SLF001
        return "output"

    space._runner = cast(
        Any,
        SimpleNamespace(
            run_stage=fake_run_stage,
            _available_arguments={},  # truly fresh: nothing seeded yet
            _current_schedule_item=0,
            _stage_order=[["wave0"]],
        ),
    )

    space.run_stage(FakeStage)  # type: ignore[arg-type]

    assert captured["feature_names"] == ["f1"]
    assert captured["algorithm_names"] == ["a1"]
    space.close()


def test_run_stage_does_not_reseed_a_partially_completed_runner() -> None:
    """A checkpoint restored partway through must not be treated as fresh."""
    space = _bare_instance_space()
    previously_seeded = {"feature_names": ["already-there"], "some_output": 42}

    def fake_run_stage(stage: object, **kwargs: object) -> str:
        return "output"

    space._runner = cast(
        Any,
        SimpleNamespace(
            run_stage=fake_run_stage,
            _available_arguments=previously_seeded,
            _current_schedule_item=1,
            _stage_order=[["wave0"], ["wave1"]],
        ),
    )

    space.run_stage(FakeStage)  # type: ignore[arg-type]

    assert space._runner._available_arguments is previously_seeded  # noqa: SLF001
    space.close()


def test_build_reports_each_stage_and_job_completed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reporter = MagicMock()
    space = _bare_instance_space(reporter)

    class FakeStageOutput:
        def __init__(self, stage: type) -> None:
            self.stage = stage

    def fake_run_iter(_inputs: object) -> Iterator[FakeStageOutput]:
        yield FakeStageOutput(PreprocessingStage)
        yield FakeStageOutput(PrelimStage)

    space._runner = cast(
        Any,
        SimpleNamespace(
            run_iter=fake_run_iter,
            _available_arguments={"done": True},
            _current_schedule_item=2,
            _stage_order=[[PreprocessingStage], [PrelimStage]],
        ),
    )
    space._model = "stale-model"  # type: ignore[assignment]
    monkeypatch.setattr(
        Model,
        "from_stage_runner_output",
        classmethod(lambda cls, output, options: "fresh-model"),  # noqa: ARG005
    )

    result = space.build()

    assert cast(str, result) == "fresh-model"
    assert reporter.report_stage_completed.call_count == 2  # noqa: PLR2004
    reported_names = [
        call.args[0] for call in reporter.report_stage_completed.call_args_list
    ]
    assert reported_names == ["preprocessing", "prelim"]
    reporter.report_job_completed.assert_called_once_with(instance_space=space)
    space.close()


def test_build_reports_job_failed_and_reraises() -> None:
    reporter = MagicMock()
    space = _bare_instance_space(reporter)

    def failing_run_iter(_inputs: object) -> Iterator[object]:
        raise RuntimeError("build blew up")
        yield  # pragma: no cover - makes this a generator function

    space._runner = cast(Any, SimpleNamespace(run_iter=failing_run_iter))

    with pytest.raises(RuntimeError, match="build blew up"):
        space.build()

    reporter.report_job_failed.assert_called_once_with("build blew up")
    space.close()


def test_run_iter_finalizes_only_after_full_exhaustion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A partially consumed iterator cannot expose either old or partial models."""
    space = _bare_instance_space()

    class FakeStageOutput:
        stage = PreprocessingStage
        output = "partial"

    runner = SimpleNamespace(
        _available_arguments={"old": True},
        _current_schedule_item=0,
        _stage_order=[[PreprocessingStage]],
    )

    def fake_run_iter(_inputs: object) -> Iterator[FakeStageOutput]:
        yield FakeStageOutput()
        runner._available_arguments = {"fresh": True}
        runner._current_schedule_item = 1

    runner.run_iter = fake_run_iter
    space._runner = cast(Any, runner)
    space._model = "stale-model"  # type: ignore[assignment]
    space._final_output = {"old": True}
    monkeypatch.setattr(
        Model,
        "from_stage_runner_output",
        classmethod(lambda cls, output, options: "fresh-model"),  # noqa: ARG005
    )

    outputs = space.run_iter()
    first = next(outputs)

    assert cast(str, first.output) == "partial"
    assert space._model is None  # noqa: SLF001
    assert space._final_output is None  # noqa: SLF001
    with pytest.raises(StageRunningError, match="not been completely"):
        _ = space.model

    with pytest.raises(StopIteration):
        next(outputs)

    assert space._final_output == {"fresh": True}  # noqa: SLF001
    assert cast(str, space.model) == "fresh-model"
    space.close()


def test_run_until_stage_forwards_overrides_without_exposing_partial_model() -> None:
    """Partial scheduling updates state but keeps ``model`` unavailable."""
    space = _bare_instance_space()
    captured: dict[str, Any] = {}

    def fake_run_until_stage(
        stage: object,
        inputs: object,
        **arguments: object,
    ) -> dict[str, Any]:
        captured["stage"] = stage
        captured["inputs"] = inputs
        captured.update(arguments)
        return {"partial": True}

    space._runner = cast(
        Any,
        SimpleNamespace(
            run_until_stage=fake_run_until_stage,
            _current_schedule_item=1,
            _stage_order=[[PreprocessingStage], [PrelimStage]],
        ),
    )
    space._model = "stale-model"  # type: ignore[assignment]
    space._final_output = {"old": True}

    threshold = 0.25
    output = space.run_until_stage(PreprocessingStage, threshold=threshold)

    assert output == {"partial": True}
    assert captured["threshold"] == threshold
    assert space._model is None  # noqa: SLF001
    assert space._final_output is None  # noqa: SLF001
    with pytest.raises(StageRunningError, match="not been completely"):
        _ = space.model
    space.close()


def test_completed_run_stage_replaces_a_stale_cached_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Rerunning a completed stage never returns the previously cached model."""
    space = _bare_instance_space()
    space._runner = cast(
        Any,
        SimpleNamespace(
            run_stage=lambda stage, **kwargs: "output",  # noqa: ARG005
            _available_arguments={"fresh": True},
            _current_schedule_item=1,
            _stage_order=[[FakeStage]],
        ),
    )
    space._model = "stale-model"  # type: ignore[assignment]
    space._final_output = {"old": True}
    monkeypatch.setattr(
        Model,
        "from_stage_runner_output",
        classmethod(lambda cls, output, options: "fresh-model"),  # noqa: ARG005
    )

    space.run_stage(FakeStage)  # type: ignore[arg-type]

    assert cast(str, space.model) == "fresh-model"
    assert space._final_output == {"fresh": True}  # noqa: SLF001
    space.close()


def test_getstate_drops_a_live_executor() -> None:
    space = _bare_instance_space()
    space._executor = ThreadPoolExecutor(max_workers=1)
    space._executor_workers = 1

    try:
        state = space.__getstate__()
        assert state["_executor"] is None
        assert state["_executor_workers"] is None
    finally:
        space.close()


def test_getstate_strips_executor_from_aliased_final_output() -> None:
    """Legacy/directly assigned final output is still scrubbed when aliased."""
    space = _bare_instance_space()
    executor = ThreadPoolExecutor(max_workers=1)
    try:
        shared_dict = {"executor": executor, "other": 1}
        space._final_output = shared_dict

        state = space.__getstate__()

        assert "executor" not in state["_final_output"]
        assert state["_final_output"]["other"] == 1
        # The live object itself must be untouched (still usable in-process).
        assert shared_dict["executor"] is executor
    finally:
        executor.shutdown(wait=True)
        space.close()
