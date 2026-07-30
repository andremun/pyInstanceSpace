# SPDX-License-Identifier: LicenseRef-PolyForm-Noncommercial-1.0.0
# Copyright (c) 2024-2026 Mario Andrés Muñoz
"""Progress reporting for instance space analysis.

Reports build()-time stage progress to external systems via HTTP callbacks
or file output. This is what lets a pipeline that is split across multiple
processes (e.g. one SLURM job per stage, triggered by a scheduler rather
than a single long-lived Python process looping over every stage) tell an
orchestrator what happened without that orchestrator polling for it.
"""

import base64
import contextlib
import json
import pickle
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any

from loguru import logger


class OutputDetail(str, Enum):
    """Level of detail to include in progress reports."""

    NONE = "none"  # No output data
    METADATA = "metadata"  # Just sizes and type info
    FULL = "full"  # Include base64-encoded pickle data


class StageStatus(str, Enum):
    """Status of a pipeline stage."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


def serialize_stage_output(
    instance_space: Any,  # noqa: ANN401
    stage_name: str,
    detail_level: OutputDetail = OutputDetail.METADATA,
) -> dict[str, Any]:
    """Serialize stage output for inclusion in progress reports.

    Args
    ----
        instance_space : Any
            The InstanceSpace object after stage completion.
        stage_name : str
            Name of the completed stage.
        detail_level : OutputDetail
            How much detail to include.

    Returns
    -------
        dict[str, Any]: Dictionary with stage output information.
    """
    if detail_level == OutputDetail.NONE:
        return {}

    result: dict[str, Any] = {
        "stage_name": stage_name,
    }

    try:
        # Get the runner's available arguments (outputs from all stages so far)
        if hasattr(instance_space, "_runner") and hasattr(
            instance_space._runner,  # noqa: SLF001
            "_available_arguments",
        ):
            args = instance_space._runner._available_arguments  # noqa: SLF001

            # Collect metadata about outputs
            output_info = {}
            for key, value in args.items():
                info: dict[str, Any] = {
                    "type": type(value).__name__,
                }

                # Add size info for arrays/dataframes
                if hasattr(value, "shape"):
                    info["shape"] = list(value.shape)
                elif hasattr(value, "__len__"):
                    info["length"] = len(value)

                # Add memory size estimate
                if hasattr(value, "nbytes"):
                    info["bytes"] = value.nbytes
                elif hasattr(value, "memory_usage"):
                    with contextlib.suppress(Exception):
                        info["bytes"] = int(value.memory_usage(deep=True))

                output_info[key] = info

            result["outputs"] = output_info

            # Include full pickle data if requested. Relies on InstanceSpace's
            # own __getstate__ to drop the live, unpicklable ThreadPoolExecutor
            # (see instance_space.py) - without that this raises for any
            # InstanceSpace that has run at least one stage.
            if detail_level == OutputDetail.FULL:
                try:
                    pickled = pickle.dumps(instance_space)
                    result["pickle_data"] = base64.b64encode(pickled).decode("ascii")
                    result["pickle_size"] = len(pickled)
                except Exception as e:
                    result["pickle_error"] = str(e)

    except Exception as e:
        result["serialization_error"] = str(e)

    return result


@dataclass
class StageProgress:
    """Progress information for a single stage."""

    stage_name: str
    status: StageStatus
    timestamp: str
    error: str | None = None
    duration_seconds: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result: dict[str, Any] = {
            "stage_name": self.stage_name,
            "status": self.status.value,
            "timestamp": self.timestamp,
        }
        if self.error:
            result["error"] = self.error
        if self.duration_seconds is not None:
            result["duration_seconds"] = self.duration_seconds
        return result


class ProgressReporter(ABC):
    """Abstract base class for progress reporters.

    Progress reporters are responsible for communicating stage execution
    status to external systems. They can include varying levels of detail
    about stage outputs in their reports.
    """

    # Default output detail level - can be overridden by subclasses
    output_detail: OutputDetail = OutputDetail.METADATA

    @abstractmethod
    def report_stage_started(self, stage_name: str) -> None:
        """Report that a stage has started."""

    @abstractmethod
    def report_stage_completed(
        self,
        stage_name: str,
        duration_seconds: float | None = None,
        instance_space: Any = None,  # noqa: ANN401
    ) -> None:
        """Report that a stage has completed successfully.

        Args
        ----
            stage_name : str
                Name of the completed stage.
            duration_seconds : float | None
                How long the stage took.
            instance_space : Any
                The InstanceSpace object (for saving intermediate state
                and including output data in the report).
        """

    @abstractmethod
    def report_stage_failed(self, stage_name: str, error: str) -> None:
        """Report that a stage has failed."""

    @abstractmethod
    def report_job_completed(
        self,
        instance_space: Any = None,  # noqa: ANN401
    ) -> None:
        """Report that the entire job has completed.

        Args
        ----
            instance_space : Any
                The final InstanceSpace object (for including final output
                data in the report).
        """

    @abstractmethod
    def report_job_failed(self, error: str) -> None:
        """Report that the job has failed."""


class HttpProgressReporter(ProgressReporter):
    """Reports progress via HTTP POST to a callback URL.

    Example callback URL: http://backend:8081/internal/jobs/123/stage-callback

    The reporter can include different levels of detail in its callbacks:
    - NONE: Just status updates
    - METADATA: Status + output type/size information
    - FULL: Status + metadata + base64-encoded pickle of InstanceSpace
    """

    def __init__(
        self,
        callback_url: str,
        job_id: int,
        auth_token: str | None = None,
        timeout_seconds: int = 10,
        output_detail: OutputDetail = OutputDetail.METADATA,
        include_pickle_on_completion: bool = False,
    ) -> None:
        """Initialize the HTTP progress reporter.

        Args
        ----
            callback_url : str
                URL to POST progress updates to.
            job_id : int
                Job ID for the callback payload.
            auth_token : str | None
                Optional bearer token for authentication.
            timeout_seconds : int
                HTTP request timeout.
            output_detail : OutputDetail
                Level of detail to include for stage outputs.
            include_pickle_on_completion : bool
                Whether to include full pickle on job completion (overrides
                output_detail for the final callback).
        """
        self.callback_url = callback_url
        self.job_id = job_id
        self.auth_token = auth_token
        self.timeout_seconds = timeout_seconds
        self.output_detail = output_detail
        self.include_pickle_on_completion = include_pickle_on_completion
        self._start_times: dict[str, datetime] = {}

    def _send_callback(self, payload: dict[str, Any]) -> bool:
        """Send a callback to the configured URL.

        Args
        ----
            payload : dict[str, Any]
                JSON payload to send.

        Returns
        -------
            bool: True if callback was successful, False otherwise.
        """
        try:
            data = json.dumps(payload, default=str).encode("utf-8")
            headers = {"Content-Type": "application/json"}
            if self.auth_token:
                headers["Authorization"] = f"Bearer {self.auth_token}"

            req = urllib.request.Request(
                self.callback_url,
                data=data,
                headers=headers,
                method="POST",
            )

            with urllib.request.urlopen(
                req,
                timeout=self.timeout_seconds,
            ) as response:
                if response.status == 200:  # noqa: PLR2004
                    logger.debug(f"Progress callback sent successfully: {payload}")
                    return True
                logger.warning(f"Progress callback returned status {response.status}")
                return False

        except urllib.error.URLError as e:
            logger.warning(f"Failed to send progress callback: {e}")
            return False
        except Exception as e:
            logger.warning(f"Unexpected error sending progress callback: {e}")
            return False

    def report_stage_started(self, stage_name: str) -> None:
        """Report that a stage has started."""
        self._start_times[stage_name] = datetime.now(UTC)
        payload = {
            "job_id": self.job_id,
            "event": "stage_started",
            "stage_name": stage_name,
            "timestamp": datetime.now(UTC).isoformat(),
        }
        self._send_callback(payload)

    def report_stage_completed(
        self,
        stage_name: str,
        duration_seconds: float | None = None,
        instance_space: Any = None,  # noqa: ANN401
    ) -> None:
        """Report that a stage has completed successfully."""
        if duration_seconds is None and stage_name in self._start_times:
            start = self._start_times[stage_name]
            duration_seconds = (datetime.now(UTC) - start).total_seconds()

        payload: dict[str, Any] = {
            "job_id": self.job_id,
            "event": "stage_completed",
            "stage_name": stage_name,
            "timestamp": datetime.now(UTC).isoformat(),
        }
        if duration_seconds is not None:
            payload["duration_seconds"] = duration_seconds

        # Include output details if configured and instance_space provided
        if instance_space is not None and self.output_detail != OutputDetail.NONE:
            output_data = serialize_stage_output(
                instance_space,
                stage_name,
                self.output_detail,
            )
            if output_data:
                payload["output"] = output_data

        self._send_callback(payload)

    def report_stage_failed(self, stage_name: str, error: str) -> None:
        """Report that a stage has failed."""
        payload = {
            "job_id": self.job_id,
            "event": "stage_failed",
            "stage_name": stage_name,
            "error": error,
            "timestamp": datetime.now(UTC).isoformat(),
        }
        self._send_callback(payload)

    def report_job_completed(
        self,
        instance_space: Any = None,  # noqa: ANN401
    ) -> None:
        """Report that the entire job has completed."""
        payload: dict[str, Any] = {
            "job_id": self.job_id,
            "event": "job_completed",
            "timestamp": datetime.now(UTC).isoformat(),
        }

        # Include final output if configured
        if instance_space is not None:
            # Use FULL detail for job completion if include_pickle_on_completion
            detail = (
                OutputDetail.FULL
                if self.include_pickle_on_completion
                else self.output_detail
            )
            if detail != OutputDetail.NONE:
                output_data = serialize_stage_output(
                    instance_space,
                    "final",
                    detail,
                )
                if output_data:
                    payload["output"] = output_data

        self._send_callback(payload)

    def report_job_failed(self, error: str) -> None:
        """Report that the job has failed."""
        payload = {
            "job_id": self.job_id,
            "event": "job_failed",
            "error": error,
            "timestamp": datetime.now(UTC).isoformat(),
        }
        self._send_callback(payload)


class FileProgressReporter(ProgressReporter):
    """Reports progress by writing to a JSON file.

    Useful as a fallback when HTTP callbacks aren't available, or for
    debugging/testing. If `stages_dir` is given, this also writes an
    unsigned pickle snapshot of the InstanceSpace after every stage - a
    convenient debugging artifact, but not the supported checkpoint/resume
    mechanism (see `InstanceSpace.save()`/`InstanceSpace.load()`, which sign
    the file the same way `Model.save()`/`Model.load()` do).
    """

    def __init__(
        self,
        progress_file: Path,
        stages_dir: Path | None = None,
    ) -> None:
        """Initialize the file progress reporter.

        Args
        ----
            progress_file : Path
                Path to the progress.json file.
            stages_dir : Path | None
                Optional directory to save intermediate stage pickles.
        """
        self.progress_file = Path(progress_file)
        self.stages_dir = Path(stages_dir) if stages_dir else None
        self._start_times: dict[str, datetime] = {}
        self._progress: dict[str, Any] = {"stages": {}}

    def _save_progress(self) -> None:
        """Write current progress to file."""
        try:
            self._progress["last_updated"] = datetime.now(UTC).isoformat()
            self.progress_file.write_text(json.dumps(self._progress, indent=2))
        except OSError as e:
            logger.error(f"Failed to write progress file: {e}")

    def report_stage_started(self, stage_name: str) -> None:
        """Report that a stage has started."""
        self._start_times[stage_name] = datetime.now(UTC)
        self._progress["stages"][stage_name] = {
            "status": StageStatus.RUNNING.value,
            "started_at": datetime.now(UTC).isoformat(),
        }
        self._progress["current_stage"] = stage_name
        self._save_progress()

    def report_stage_completed(
        self,
        stage_name: str,
        duration_seconds: float | None = None,
        instance_space: Any = None,  # noqa: ANN401
    ) -> None:
        """Report that a stage has completed successfully."""
        if duration_seconds is None and stage_name in self._start_times:
            start = self._start_times[stage_name]
            duration_seconds = (datetime.now(UTC) - start).total_seconds()

        stage_entry: dict[str, Any] = {
            "status": StageStatus.COMPLETED.value,
            "completed_at": datetime.now(UTC).isoformat(),
        }
        if duration_seconds is not None:
            stage_entry["duration_seconds"] = duration_seconds
        self._progress["stages"][stage_name] = stage_entry

        self._save_progress()

        # Save intermediate pickle if configured
        if self.stages_dir is not None and instance_space is not None:
            try:
                self.stages_dir.mkdir(parents=True, exist_ok=True)
                stage_file = self.stages_dir / f"{stage_name}.pkl"
                with stage_file.open("wb") as f:
                    pickle.dump(instance_space, f)
            except (OSError, pickle.PicklingError) as e:
                logger.warning(f"Failed to save stage pickle: {e}")

    def report_stage_failed(self, stage_name: str, error: str) -> None:
        """Report that a stage has failed."""
        self._progress["stages"][stage_name] = {
            "status": StageStatus.FAILED.value,
            "failed_at": datetime.now(UTC).isoformat(),
            "error": error,
        }
        self._progress["failed"] = True
        self._progress["error"] = error
        self._save_progress()

    def report_job_completed(
        self,
        instance_space: Any = None,  # noqa: ANN401
    ) -> None:
        """Report that the entire job has completed."""
        self._progress["completed"] = True
        self._progress["completed_at"] = datetime.now(UTC).isoformat()
        self._save_progress()

        # Save final model pickle if stages_dir is configured
        if self.stages_dir is not None and instance_space is not None:
            try:
                self.stages_dir.mkdir(parents=True, exist_ok=True)
                final_file = self.stages_dir / "final.pkl"
                with final_file.open("wb") as f:
                    pickle.dump(instance_space, f)
            except (OSError, pickle.PicklingError) as e:
                logger.warning(f"Failed to save final pickle: {e}")

    def report_job_failed(self, error: str) -> None:
        """Report that the job has failed."""
        self._progress["failed"] = True
        self._progress["error"] = error
        self._progress["failed_at"] = datetime.now(UTC).isoformat()
        self._save_progress()


class CompositeProgressReporter(ProgressReporter):
    """Combines multiple progress reporters.

    Useful for reporting to both HTTP and file simultaneously.
    """

    def __init__(self, reporters: list[ProgressReporter]) -> None:
        """Initialize with a list of reporters."""
        self.reporters = reporters

    def report_stage_started(self, stage_name: str) -> None:
        """Report that a stage has started, to every configured reporter."""
        for reporter in self.reporters:
            try:
                reporter.report_stage_started(stage_name)
            except Exception as e:
                logger.warning(f"Reporter {type(reporter).__name__} failed: {e}")

    def report_stage_completed(
        self,
        stage_name: str,
        duration_seconds: float | None = None,
        instance_space: Any = None,  # noqa: ANN401
    ) -> None:
        """Report that a stage has completed, to every configured reporter."""
        for reporter in self.reporters:
            try:
                reporter.report_stage_completed(
                    stage_name,
                    duration_seconds,
                    instance_space,
                )
            except Exception as e:
                logger.warning(f"Reporter {type(reporter).__name__} failed: {e}")

    def report_stage_failed(self, stage_name: str, error: str) -> None:
        """Report that a stage has failed, to every configured reporter."""
        for reporter in self.reporters:
            try:
                reporter.report_stage_failed(stage_name, error)
            except Exception as e:
                logger.warning(f"Reporter {type(reporter).__name__} failed: {e}")

    def report_job_completed(
        self,
        instance_space: Any = None,  # noqa: ANN401
    ) -> None:
        """Report that the job has completed, to every configured reporter."""
        for reporter in self.reporters:
            try:
                reporter.report_job_completed(instance_space)
            except Exception as e:
                logger.warning(f"Reporter {type(reporter).__name__} failed: {e}")

    def report_job_failed(self, error: str) -> None:
        """Report that the job has failed, to every configured reporter."""
        for reporter in self.reporters:
            try:
                reporter.report_job_failed(error)
            except Exception as e:
                logger.warning(f"Reporter {type(reporter).__name__} failed: {e}")


class NullProgressReporter(ProgressReporter):
    """A no-op reporter for when progress reporting is disabled (the default)."""

    def report_stage_started(self, stage_name: str) -> None:
        """Do nothing."""

    def report_stage_completed(
        self,
        stage_name: str,
        duration_seconds: float | None = None,
        instance_space: Any = None,  # noqa: ANN401
    ) -> None:
        """Do nothing."""

    def report_stage_failed(self, stage_name: str, error: str) -> None:
        """Do nothing."""

    def report_job_completed(
        self,
        instance_space: Any = None,  # noqa: ANN401
    ) -> None:
        """Do nothing."""

    def report_job_failed(self, error: str) -> None:
        """Do nothing."""
