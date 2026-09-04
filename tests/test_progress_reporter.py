# SPDX-License-Identifier: LicenseRef-PolyForm-Noncommercial-1.0.0
# Copyright (c) 2024-2026 Mario Andrés Muñoz
"""Unit tests for `progress_reporter.py`.

Exercises the reporters directly, against small stand-ins for `InstanceSpace`
rather than a real pipeline - `test_instance_space_checkpoint.py` covers the
real wiring (via `InstanceSpace.build()`/`run_stage()`) end to end.
"""

import json
import urllib.error
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

from instancespace.progress_reporter import (
    CompositeProgressReporter,
    FileProgressReporter,
    HttpProgressReporter,
    NullProgressReporter,
    OutputDetail,
    StageStatus,
    serialize_stage_output,
)


def _fake_instance_space(available_arguments: dict[str, Any]) -> Any:
    runner = SimpleNamespace(_available_arguments=available_arguments)
    return SimpleNamespace(_runner=runner)


def test_serialize_stage_output_none_detail_returns_empty_dict() -> None:
    instance_space = _fake_instance_space({"x": 1})
    assert serialize_stage_output(instance_space, "prelim", OutputDetail.NONE) == {}


def test_serialize_stage_output_metadata_reports_shape_and_length() -> None:
    import numpy as np

    instance_space = _fake_instance_space(
        {"x": np.zeros((3, 2)), "names": ["a", "b"]},
    )

    result = serialize_stage_output(instance_space, "prelim", OutputDetail.METADATA)

    assert result["stage_name"] == "prelim"
    assert result["outputs"]["x"]["shape"] == [3, 2]
    assert result["outputs"]["names"]["length"] == 2
    assert "pickle_data" not in result


def test_serialize_stage_output_full_includes_pickle_data() -> None:
    instance_space = _fake_instance_space({"x": 1})

    result = serialize_stage_output(instance_space, "prelim", OutputDetail.FULL)

    assert "pickle_data" in result
    assert result["pickle_size"] > 0


def test_serialize_stage_output_swallows_errors_into_the_result() -> None:
    # No `_runner` attribute at all - must not raise.
    result = serialize_stage_output(object(), "prelim", OutputDetail.METADATA)
    assert result == {"stage_name": "prelim"}


class TestFileProgressReporter:
    def test_writes_progress_json_through_the_stage_lifecycle(
        self,
        tmp_path: Path,
    ) -> None:
        reporter = FileProgressReporter(tmp_path / "progress.json")

        reporter.report_stage_started("prelim")
        reporter.report_stage_completed("prelim")
        reporter.report_job_completed()

        progress = json.loads((tmp_path / "progress.json").read_text())
        assert progress["stages"]["prelim"]["status"] == StageStatus.COMPLETED.value
        assert progress["completed"] is True

    def test_stage_failure_is_recorded(self, tmp_path: Path) -> None:
        reporter = FileProgressReporter(tmp_path / "progress.json")

        reporter.report_stage_failed("prelim", "boom")

        progress = json.loads((tmp_path / "progress.json").read_text())
        assert progress["stages"]["prelim"]["status"] == StageStatus.FAILED.value
        assert progress["failed"] is True
        assert progress["error"] == "boom"

    def test_stages_dir_saves_a_pickle_snapshot_per_stage(
        self,
        tmp_path: Path,
    ) -> None:
        reporter = FileProgressReporter(
            tmp_path / "progress.json",
            stages_dir=tmp_path / "stages",
        )
        instance_space = _fake_instance_space({"x": 1})

        reporter.report_stage_completed("prelim", instance_space=instance_space)

        assert (tmp_path / "stages" / "prelim.pkl").exists()

    def test_job_completed_saves_a_final_pickle(self, tmp_path: Path) -> None:
        reporter = FileProgressReporter(
            tmp_path / "progress.json",
            stages_dir=tmp_path / "stages",
        )
        instance_space = _fake_instance_space({"x": 1})

        reporter.report_job_completed(instance_space=instance_space)

        assert (tmp_path / "stages" / "final.pkl").exists()

    def test_no_stages_dir_means_no_pickle_is_written(self, tmp_path: Path) -> None:
        reporter = FileProgressReporter(tmp_path / "progress.json")
        instance_space = _fake_instance_space({"x": 1})

        reporter.report_stage_completed("prelim", instance_space=instance_space)

        assert list(tmp_path.iterdir()) == [tmp_path / "progress.json"]


class TestHttpProgressReporter:
    def test_report_stage_completed_posts_expected_payload(self) -> None:
        reporter = HttpProgressReporter(
            callback_url="http://backend.example/callback",
            job_id=42,
            auth_token="secret-token",
        )

        response = MagicMock()
        response.status = 200
        response.__enter__ = MagicMock(return_value=response)
        response.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=response) as mock_urlopen:
            reporter.report_stage_completed("prelim", duration_seconds=1.5)

        request = mock_urlopen.call_args[0][0]
        assert request.headers["Authorization"] == "Bearer secret-token"
        payload = json.loads(request.data.decode("utf-8"))
        assert payload["job_id"] == 42
        assert payload["event"] == "stage_completed"
        assert payload["stage_name"] == "prelim"
        assert payload["duration_seconds"] == 1.5

    def test_network_failure_is_swallowed_not_raised(self) -> None:
        reporter = HttpProgressReporter(
            callback_url="http://backend.example/callback",
            job_id=1,
        )

        with patch(
            "urllib.request.urlopen",
            side_effect=urllib.error.URLError("connection refused"),
        ):
            reporter.report_job_failed("boom")  # must not raise


class TestCompositeProgressReporter:
    def test_fans_out_to_every_reporter(self) -> None:
        first = MagicMock()
        second = MagicMock()
        composite = CompositeProgressReporter([first, second])

        composite.report_stage_completed("prelim", instance_space=None)

        first.report_stage_completed.assert_called_once_with("prelim", None, None)
        second.report_stage_completed.assert_called_once_with("prelim", None, None)

    def test_one_reporter_failing_does_not_stop_the_others(self) -> None:
        failing = MagicMock()
        failing.report_job_completed.side_effect = RuntimeError("boom")
        healthy = MagicMock()
        composite = CompositeProgressReporter([failing, healthy])

        composite.report_job_completed()  # must not raise

        healthy.report_job_completed.assert_called_once()


class TestNullProgressReporter:
    def test_every_method_is_a_no_op(self) -> None:
        reporter = NullProgressReporter()

        reporter.report_stage_started("prelim")
        reporter.report_stage_completed("prelim")
        reporter.report_stage_failed("prelim", "boom")
        reporter.report_job_completed()
        reporter.report_job_failed("boom")
