# SPDX-License-Identifier: LicenseRef-PolyForm-Noncommercial-1.0.0
# Copyright (c) 2024-2026 Mario Andrés Muñoz
"""Perform instance space analysis on given dataset and configuration.

Construct an instance space from data and configuration files located in a specified
directory. The instance space is represented as a Model object, which encapsulates the
analytical results and metadata of the instance space analysis.
"""

import hashlib
import hmac
import multiprocessing
import time
from collections.abc import Generator
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any, NamedTuple, TypeVar

import joblib
import numpy as np
import pandas as pd
from loguru import logger
from matplotlib.axes import Axes
from numpy.typing import NDArray

from instancespace.data.metadata import Metadata, from_csv_file
from instancespace.data.model import ExploreResult, TraceOut
from instancespace.data.options import (
    AutoOptions,
    BoundOptions,
    CloisterOptions,
    GeneralOptions,
    InstanceSpaceOptions,
    NormOptions,
    OutputOptions,
    ParallelOptions,
    PerformanceOptions,
    PilotOptions,
    PrelimOptions,
    PythiaOptions,
    SelvarsOptions,
    SiftedOptions,
    TraceOptions,
    from_json_file,
)
from instancespace.model import Model, ModelSignatureError
from instancespace.plotting import (
    plot_footprint,
    plot_good,
    plot_portfolio,
    plot_sources,
)
from instancespace.progress_reporter import NullProgressReporter, ProgressReporter
from instancespace.stage_runner import (
    AnnotatedStageOutput,
    StageRunner,
    StageRunningError,
    StageScheduleElement,
    build_stage_runner,
    named_tuple_to_stage_arguments,
)
from instancespace.stages.cloister import CloisterStage
from instancespace.stages.pilot import PilotPredictInput, PilotStage
from instancespace.stages.prelim import (
    PrelimPredictInput,
    PrelimStage,
    compute_binary_performance,
)
from instancespace.stages.preprocessing import (
    PreprocessingStage,
    validate_viable_dimensions,
)
from instancespace.stages.pythia import (
    PythiaEvaluateInput,
    PythiaPredictInput,
    PythiaStage,
)
from instancespace.stages.sifted import SiftedPredictInput, SiftedStage
from instancespace.stages.stage import IN, OUT, Stage, StageClass
from instancespace.stages.trace import TracePredictInput, TraceStage
from instancespace.utils.print_options import format_options

T = TypeVar("T", bound="_InstanceSpaceInputs")


class _InstanceSpaceInputs(NamedTuple):
    feature_names: list[str]
    algorithm_names: list[str]
    instance_labels: pd.Series  # type: ignore[type-arg]
    instance_sources: pd.Series | None  # type: ignore[type-arg]
    features: NDArray[np.double]
    algorithms: NDArray[np.double]
    parallel_options: ParallelOptions
    perf_options: PerformanceOptions
    auto_options: AutoOptions
    bound_options: BoundOptions
    norm_options: NormOptions
    selvars_options: SelvarsOptions
    sifted_options: SiftedOptions
    pilot_options: PilotOptions
    cloister_options: CloisterOptions
    pythia_options: PythiaOptions
    trace_options: TraceOptions
    outputs_options: OutputOptions
    prelim_options: PrelimOptions
    general_options: GeneralOptions
    executor: ThreadPoolExecutor | None = None

    @classmethod
    def from_metadata_and_options(
        cls: type[T],
        metadata: Metadata,
        options: InstanceSpaceOptions,
    ) -> T:
        return cls(
            feature_names=metadata.feature_names,
            algorithm_names=metadata.algorithm_names,
            instance_labels=metadata.instance_labels,
            instance_sources=metadata.instance_sources,
            features=metadata.features,
            algorithms=metadata.algorithms,
            parallel_options=options.parallel,
            perf_options=options.perf,
            auto_options=options.auto,
            bound_options=options.bound,
            norm_options=options.norm,
            selvars_options=options.selvars,
            sifted_options=options.sifted,
            pilot_options=options.pilot,
            cloister_options=options.cloister,
            pythia_options=options.pythia,
            trace_options=options.trace,
            outputs_options=options.outputs,
            prelim_options=PrelimOptions.from_options(options),
            general_options=options.general,
        )


# The fixed execution order for the built-in pipeline (S2). PythiaStage and
# CloisterStage share a wave - both depend only on PilotStage's output, and
# nothing downstream depends on CloisterStage's output at all - matching the
# schedule the DAG auto-resolver produced for this exact pipeline before S2
# removed it. Any stage passed to InstanceSpace() that isn't one of these is
# treated as an extra/plugin stage and must declare where it attaches via a
# RunBefore[X]/RunAfter[X] input field (see build_stage_runner() in stage_runner.py).
_BUILTIN_STAGE_ORDER: list[StageScheduleElement] = [
    [PreprocessingStage],
    [PrelimStage],
    [SiftedStage],
    [PilotStage],
    [PythiaStage, CloisterStage],
    [TraceStage],
]


class ExploreStage(Enum):
    """One step of the explore()-time inference pipeline.

    These lightweight identifiers describe the lazy orchestration order.
    Scientific inference is owned by the corresponding predictive stage;
    metadata reconciliation and conditional evaluation remain coordinated by
    :class:`InstanceSpace`.
    """

    PRELIM = "prelim"
    SIFTED = "sifted"
    PILOT = "pilot"
    PYTHIA = "pythia"
    TRACE = "trace"
    EVALUATION = "evaluation"


class AnnotatedExploreOutput(NamedTuple):
    """The yielded output of running one explore-time step.

    Mirrors `stage_runner.AnnotatedStageOutput`'s (stage, output) shape.
    """

    stage: ExploreStage
    output: Any


class _EvaluationResult(NamedTuple):
    """PYTHIA-vs-ground-truth evaluation fields (F9).

    Yielded for `ExploreStage.EVALUATION`. All per-algorithm fields are
    width `n_trained_algorithms + n_new_algos` (full MATLAB parity) - see
    `InstanceSpace._explore_evaluate`.
    """

    y_actual: NDArray[np.bool_]
    y_best_actual: NDArray[np.double]
    p_actual: NDArray[np.int_]
    beta_actual: NDArray[np.bool_]
    accuracy_actual: NDArray[np.double]
    precision_actual: NDArray[np.double]
    recall_actual: NDArray[np.double]
    cvcmat_actual: NDArray[np.double]
    algo_labels: list[str]
    trace_out: TraceOut | None = None


class InstanceSpace:
    """The main instance space class.

    ## Basic Example:
    ```python

        from instancespace import *

        metadata = metadata.from_csv_file('./metadata.csv')
        options = InstanceSpaceOptions.default()
        # options = options.from_json_file('./options.json')

        instance_space = InstanceSpace(metadata, options)

        model = instance_space.build()

        model.save_to_csv('./output/')
        model.save_graphs('./output/')
    ```
    """

    _runner: StageRunner
    _stages: list[StageClass]

    _metadata: Metadata
    _options: InstanceSpaceOptions
    _progress_reporter: ProgressReporter

    _model: Model | None
    _final_output: dict[str, Any] | None

    # When parallel execution is enabled, lazily created and reused across
    # staged calls (Q6). Mirrors MATLAB's ensurePool() "rightSize" check.
    _executor: ThreadPoolExecutor | None
    _executor_workers: int | None

    def __init__(
        self,
        metadata: Metadata,
        options: InstanceSpaceOptions,
        stages: list[StageClass] = [
            PreprocessingStage,
            PrelimStage,
            SiftedStage,
            PilotStage,
            PythiaStage,
            CloisterStage,
            TraceStage,
        ],
        additional_initial_inputs_type: type[NamedTuple] | None = None,
        progress_reporter: ProgressReporter | None = None,
    ) -> None:
        """Initialise the InstanceSpace.

        Args
        ----
            metadata : Metadata
                TODO THIS
            options : InstanceSpaceOptions
                Options to build the instance space.
            stages : list[StageClass], optional
                A list of stages to be ran.
            additional_initial_inputs_type : type[NamedTuple] | None, optional
                Extra initial inputs used by plugins.
            progress_reporter : ProgressReporter | None, optional
                Reporter for tracking build()/run_stage() progress - e.g.
                `HttpProgressReporter` for a SLURM-triggered job to call back
                to an orchestrator after each stage. Defaults to a no-op
                reporter, so omitting this changes nothing about existing
                behaviour.
        """
        validate_viable_dimensions(
            metadata.features,
            metadata.algorithms,
            context="Build metadata",
        )
        self._metadata = metadata
        self._options = options
        self._stages = stages
        self._progress_reporter = progress_reporter or NullProgressReporter()

        self._model: Model | None = None
        self._final_output: dict[str, Any] | None = None
        self._explore_results: list[ExploreResult] = []
        self._executor = None
        self._executor_workers = None

        requested_stages = set(stages)
        base_order = [
            [stage for stage in wave if stage in requested_stages]
            for wave in _BUILTIN_STAGE_ORDER
        ]
        base_order = [wave for wave in base_order if wave]

        known_stages = {stage for wave in _BUILTIN_STAGE_ORDER for stage in wave}
        extra_stages = [stage for stage in stages if stage not in known_stages]

        annotations = named_tuple_to_stage_arguments(_InstanceSpaceInputs)

        if additional_initial_inputs_type is not None:
            annotations |= named_tuple_to_stage_arguments(
                additional_initial_inputs_type,
            )

        self._runner = build_stage_runner(base_order, extra_stages, annotations)

    @property
    def metadata(self) -> Metadata:
        """Get metadata."""
        return self._metadata

    @property
    def options(self) -> InstanceSpaceOptions:
        """Get options."""
        return self._options

    @property
    def model(self) -> Model:
        """Get model.

        Raises
        ------
            StageRunningError: If the InstanceSpace hasn't been built, will raise a
                StageRunningError.

        Returns
        -------
            Model: The output of building the instance space.
        """
        if self._model is None:
            if (
                self._final_output is None
                or not self._runner_is_complete()
                or not self._can_build_complete_model()
            ):
                raise StageRunningError("InstanceSpace has not been completely ran.")

            self._model = Model.from_stage_runner_output(
                self._final_output,
                self._options,
            )

        return self._model

    def __getstate__(self) -> dict[str, Any]:
        """Drop the live `ThreadPoolExecutor` (if any) so this can be checkpointed.

        A `ThreadPoolExecutor` holds OS thread/queue state that `pickle`
        can't serialise, so it can't survive a `save()`/`load()` round trip
        (or a progress reporter's `OutputDetail.FULL` snapshot, see
        `progress_reporter.serialize_stage_output()`) as-is. Dropping it here
        is safe: `_get_executor()` recreates it lazily on the next
        `build()`/`run_stage()`/etc. call (Q6), the same as after `close()`.

        `_final_output` needs the same treatment as `_runner`'s own state
        (see `StageRunner.__getstate__()`), because a completed run's output
        snapshot can carry the live executor under the `"executor"` key.
        """
        state = self.__dict__.copy()
        state["_executor"] = None
        state["_executor_workers"] = None
        final_output = state["_final_output"]
        if final_output is not None:
            state["_final_output"] = {
                key: value for key, value in final_output.items() if key != "executor"
            }
        return state

    def save(self, path: Path | str, secret_key: bytes | None = None) -> None:
        """Checkpoint this InstanceSpace to `path` via signed `joblib`.

        Captures the entire pipeline state - not just a finished `Model` -
        so a partially-built InstanceSpace (e.g. one stage run per SLURM
        job, with the next stage triggered by a separate, later job) can be
        reconstructed by `load()` and continue from exactly where it left
        off via `run_stage()`. Signing follows the same scheme as
        `Model.save()`/`Model.load()`: see that method's docstring for the
        exact `secret_key` semantics and the production invariant (always
        sign, never accept a user-supplied path server-side).
        """
        if isinstance(path, str):
            path = Path(path)

        joblib.dump(self, path)

        sig_path = path.with_name(path.name + ".sig")
        if secret_key is not None:
            signature = hmac.new(secret_key, path.read_bytes(), hashlib.sha256)
            sig_path.write_bytes(signature.digest())
        elif sig_path.exists():
            sig_path.unlink()

    @classmethod
    def load(
        cls: type["InstanceSpace"],
        path: Path | str,
        secret_key: bytes | None = None,
    ) -> "InstanceSpace":
        """Restore an InstanceSpace checkpoint previously written by `save()`.

        Same four verification cases as `Model.load()` (matched/mismatched/
        missing/unexpected signature) - see that method's docstring.
        """
        if isinstance(path, str):
            path = Path(path)

        sig_path = path.with_name(path.name + ".sig")
        sig_exists = sig_path.exists()

        if secret_key is not None and not sig_exists:
            raise ModelSignatureError(
                f"secret_key was given but no signature file exists at "
                f"{sig_path}; refusing to load an unverifiable file.",
            )
        if secret_key is None and sig_exists:
            raise ModelSignatureError(
                f"A signature file exists at {sig_path} but no secret_key "
                "was given; refusing to load a signed file without "
                "verification.",
            )

        if secret_key is not None:
            expected_signature = sig_path.read_bytes()
            actual_signature = hmac.new(
                secret_key,
                path.read_bytes(),
                hashlib.sha256,
            ).digest()
            if not hmac.compare_digest(actual_signature, expected_signature):
                raise ModelSignatureError(
                    f"Signature verification failed for {path}; refusing to "
                    "deserialise.",
                )

        instance_space = joblib.load(path)
        if not isinstance(instance_space, cls):
            raise TypeError(
                f"{path} does not contain an {cls.__name__} (got "
                f"{type(instance_space).__name__!r}).",
            )
        return instance_space

    def _get_executor(self) -> ThreadPoolExecutor | None:
        """Return the cached executor when parallel execution is enabled.

        When ``ParallelOptions.flag`` is false, no pool is created and any
        cached pool is closed. Otherwise the pool is recreated only if the
        worker count changes (mirrors MATLAB's ``ensurePool()``
        ``rightSize`` check).
        """
        if not self._options.parallel.flag:
            self.close()
            return None

        worker_count = min(
            self._options.parallel.n_cores,
            multiprocessing.cpu_count(),
        )
        if self._executor is None or self._executor_workers != worker_count:
            if self._executor is not None:
                self._executor.shutdown(wait=True)
            self._executor = ThreadPoolExecutor(max_workers=worker_count)
            self._executor_workers = worker_count
        return self._executor

    def _invalidate_model_state(self) -> None:
        """Invalidate cached results before mutating runner state."""
        self._model = None
        self._final_output = None

    def _runner_is_complete(self) -> bool:
        """Return whether every scheduled stage wave has completed."""
        current_item = self._runner._current_schedule_item  # noqa: SLF001
        stage_order = self._runner._stage_order  # noqa: SLF001
        return current_item >= len(stage_order)

    def _can_build_complete_model(self) -> bool:
        """Return whether this instance includes every model-producing stage."""
        required_stages = {stage for wave in _BUILTIN_STAGE_ORDER for stage in wave}
        return required_stages.issubset(self._stages)

    def _finalize_model_state(self, output: dict[str, Any]) -> None:
        """Record a stable final snapshot only for a complete model pipeline."""
        if self._runner_is_complete() and self._can_build_complete_model():
            self._final_output = output.copy()

    def close(self) -> None:
        """Release resources held across staged calls (currently: the TRACE pool).

        Safe to call even if nothing has been built yet. A subsequent staged
        call recreates the pool lazily when parallel execution is enabled.
        """
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None
            self._executor_workers = None

    def plot_sources(self, ax: Axes | None = None) -> Axes:
        """Scatter training instances in the 2D instance space, coloured by source.

        See ``instancespace.plotting.plot_sources``.
        """
        return plot_sources(self.model, ax=ax)

    def plot_portfolio(self, ax: Axes | None = None) -> Axes:
        """Scatter instances coloured by their best-performing algorithm.

        See ``instancespace.plotting.plot_portfolio``.
        """
        return plot_portfolio(self.model, ax=ax)

    def plot_good(self, algo: str | int, ax: Axes | None = None) -> Axes:
        """Scatter instances coloured by PYTHIA's good/bad prediction for one algorithm.

        See ``instancespace.plotting.plot_good``.
        """
        return plot_good(self.model, algo, ax=ax)

    def plot_footprint(
        self,
        algo: str | int,
        kind: str = "good",
        ax: Axes | None = None,
    ) -> Axes:
        """Draw one algorithm's trained footprint polygon(s) over training instances.

        See ``instancespace.plotting.plot_footprint``.
        """
        return plot_footprint(self.model, algo, kind=kind, ax=ax)

    @staticmethod
    def _stage_report_name(stage: StageClass) -> str:
        """Report name for a stage, e.g. `PrelimStage` -> `"prelim"`.

        `getattr(..., "__name__", ...)` rather than `stage.__name__` directly:
        existing tests (e.g. `test_instance_space_executor.py`) stub `stage`
        as a plain string rather than a real `StageClass` when they only care
        about argument pass-through, not stage identity.
        """
        name = getattr(stage, "__name__", str(stage))
        return name.replace("Stage", "").lower()

    def build(
        self,
    ) -> Model:
        """Build the instance space, in one call, start to finish.

        Options will be broken down to sub fields to be passed to stages. You can
        override inputs to stages. Progress is reported via the configured
        `progress_reporter` (a no-op by default - see `__init__`), the same as
        `run_stage()`.

        Returns
        -------
            Model: The output of all stages.

        """
        self._invalidate_model_state()
        try:
            inputs = _InstanceSpaceInputs.from_metadata_and_options(
                self.metadata,
                self.options,
            )._replace(executor=self._get_executor())

            for stage_output in self._runner.run_iter(inputs):
                self._progress_reporter.report_stage_completed(
                    self._stage_report_name(stage_output.stage),
                    instance_space=self,
                )

            self._finalize_model_state(
                self._runner._available_arguments,  # noqa: SLF001
            )

            self._progress_reporter.report_job_completed(instance_space=self)

            return self.model

        except Exception as e:
            self._progress_reporter.report_job_failed(str(e))
            raise

    def run_iter(
        self,
    ) -> Generator[AnnotatedStageOutput, None, None]:
        """Run all stages, yielding between so the data can be examined.

        Yields
        ------
            Generator[AnnotatedStageOutput, None]: The output of each stage, annotated
                with what stage was ran, as multiple stages ran in the same schedule can
                be ran in any order.
        """
        self._invalidate_model_state()
        inputs = _InstanceSpaceInputs.from_metadata_and_options(
            self.metadata,
            self.options,
        )._replace(executor=self._get_executor())
        yield from self._runner.run_iter(inputs)
        self._finalize_model_state(
            self._runner._available_arguments,  # noqa: SLF001
        )

    def run_stage(
        self,
        stage: type[Stage[IN, OUT]],
        **arguments: Any,  # noqa: ANN401
    ) -> OUT:
        """Run a single stage.

        All inputs to the stage must either be present from previously ran stages, or
        be given as arguments to this function. Arguments to this function have
        priority over outputs from previous stages.

        Progress is reported via the configured `progress_reporter` (a no-op
        by default - see `__init__`). This is the entry point a SLURM job
        that runs one stage per invocation should call: load a checkpoint
        with `InstanceSpace.load()` (or construct fresh for the first
        stage), call `run_stage()` for the stage this invocation was told to
        run, then `save()` the result so a later, separately-triggered job
        can resume. `self.model` becomes available once the last stage in
        the schedule has been run this way, exactly as after `build()`.

        Args
        ----
            stage : StageClass
                The stage to be ran.

            **arguments : Any
                Any additional inputs to the stage. Outputs from previous stages will
                be used if not provided.

        Returns
        -------
            list[Any]: The output of the stage.
        """
        self._invalidate_model_state()
        arguments.setdefault("executor", self._get_executor())
        stage_name = self._stage_report_name(stage)
        start = time.monotonic()

        # A completely fresh InstanceSpace (as opposed to one restored via
        # load()) has never had its initial inputs seeded - build()/run_iter()/
        # run_until_stage() all do this themselves before running anything,
        # but run_stage() is meant to be usable as the sole per-process entry
        # point for a schedule split across separate invocations (e.g. one
        # SLURM job per stage), so it has to do the same the first time it's
        # called. A checkpoint loaded partway through never hits this branch:
        # every prior stage's outputs (plus the original seed) are already in
        # `_available_arguments`.
        if not self._runner._available_arguments:  # noqa: SLF001
            seed = _InstanceSpaceInputs.from_metadata_and_options(
                self.metadata,
                self.options,
            )._replace(executor=self._get_executor())
            self._runner._available_arguments = seed._asdict()  # noqa: SLF001

        try:
            output = self._runner.run_stage(stage, **arguments)
        except Exception as e:
            self._progress_reporter.report_stage_failed(stage_name, str(e))
            self._progress_reporter.report_job_failed(str(e))
            raise

        self._finalize_model_state(
            self._runner._available_arguments,  # noqa: SLF001
        )
        duration_seconds = time.monotonic() - start

        self._progress_reporter.report_stage_completed(
            stage_name,
            duration_seconds=duration_seconds,
            instance_space=self,
        )

        if self._runner_is_complete():
            self._progress_reporter.report_job_completed(instance_space=self)

        return output

    def run_until_stage(
        self,
        stage: StageClass,
        **arguments: Any,  # noqa: ANN401
    ) -> dict[str, Any]:
        """Run all stages until the specified stage, as well as the specified stage.

        Args
        ----
            stage : StageClass
                A stage in the last wave to execute.
            **arguments : Any
                Per-run input overrides. Successful overrides remain available
                to every downstream stage.

        Returns
        -------
            dict[str, Any]: The raw output dict of all ran stages.
        """
        self._invalidate_model_state()
        inputs = _InstanceSpaceInputs.from_metadata_and_options(
            self.metadata,
            self.options,
        )._replace(executor=self._get_executor())
        output = self._runner.run_until_stage(
            stage,
            inputs,
            **arguments,
        )
        self._finalize_model_state(output)
        return output

    @property
    def explore_results(self) -> list[ExploreResult]:
        """Get list of explore results from previous explore() calls.

        Returns
        -------
            list[ExploreResult]: List of explore results, in order of execution.
        """
        return self._explore_results

    def explore(
        self,
        test_metadata: Metadata,
        *,
        dataset_id: str | None = None,
    ) -> ExploreResult:
        """Apply trained instance space model to new test data.

        This method applies the transformations and models learned during build()
        to new test instances. It performs:
        1. Feature preprocessing (PRELIM normalization)
        2. Feature selection (SIFTED)
        3. Dimensionality reduction to the trained 2D/3D PILOT projection
        4. Algorithm performance prediction (PYTHIA SVMs)
        5. Footprint membership analysis (TRACE)

        Args
        ----
            test_metadata : Metadata
                New instances with the same feature columns as training data. Feature
                columns are matched by name, not position, so they may be supplied in
                any order (this is deliberate, permanent behaviour, not a stricter
                order check like MATLAB's `featureOrderMismatch`).
            dataset_id : str | None, optional
                Identifier for this test dataset. If not provided, a timestamp-based
                ID will be generated.

        Returns
        -------
            ExploreResult
                Contains projected coordinates, algorithm predictions, and
                footprint membership for the test instances.

        Raises
        ------
            RuntimeError
                If build() has not been called before explore().
            ValueError
                If test_metadata features do not match training features, or trained
                and explored projection dimensions differ.
        """
        # Run every inference stage, then assemble the result from each stage's output
        stages = {
            annotated.stage: annotated.output
            for annotated in self.explore_stage_iter(test_metadata)
        }

        if dataset_id is None:
            dataset_id = f"explore_{datetime.now(tz=UTC).strftime('%Y%m%d_%H%M%S')}"

        inst_labels = self._extract_instance_labels(test_metadata)
        pythia_result = stages[ExploreStage.PYTHIA]
        trace_result = stages[ExploreStage.TRACE]
        evaluation_result = stages.get(ExploreStage.EVALUATION)

        result = ExploreResult(
            dataset_id=dataset_id,
            timestamp=datetime.now(tz=UTC),
            x=stages[ExploreStage.SIFTED],
            z=stages[ExploreStage.PILOT],
            y_hat=pythia_result[0] if pythia_result else None,
            pr0_hat=pythia_result[1] if pythia_result else None,
            selection0=pythia_result[2] if pythia_result else None,
            in_good=trace_result[0] if trace_result else None,
            in_best=trace_result[1] if trace_result else None,
            inst_labels=inst_labels,
            y_actual=evaluation_result.y_actual if evaluation_result else None,
            y_best_actual=(
                evaluation_result.y_best_actual if evaluation_result else None
            ),
            p_actual=evaluation_result.p_actual if evaluation_result else None,
            beta_actual=evaluation_result.beta_actual if evaluation_result else None,
            accuracy_actual=(
                evaluation_result.accuracy_actual if evaluation_result else None
            ),
            precision_actual=(
                evaluation_result.precision_actual if evaluation_result else None
            ),
            recall_actual=(
                evaluation_result.recall_actual if evaluation_result else None
            ),
            cvcmat_actual=(
                evaluation_result.cvcmat_actual if evaluation_result else None
            ),
            algo_labels=evaluation_result.algo_labels if evaluation_result else None,
            trace_out=evaluation_result.trace_out if evaluation_result else None,
        )

        self._explore_results.append(result)
        return result

    def explore_stage_iter(
        self,
        test_metadata: Metadata,
    ) -> Generator[AnnotatedExploreOutput, None, None]:
        """Run explore() one stage at a time, yielding each stage's output.

        Same computation and trained model as ``explore()``, but instead of
        returning a single ``ExploreResult`` it yields an ``AnnotatedExploreOutput``
        (an ``(ExploreStage, output)`` pair - the explore-time counterpart of
        ``run_iter()``'s ``AnnotatedStageOutput``) after each inference stage, so the
        intermediate result of every stage can be inspected or plotted before the next
        one runs. In order: ``ExploreStage.PRELIM``/``SIFTED``/``PILOT`` yield the
        transformed feature or coordinate array, ``PYTHIA`` yields ``(y_hat, pr0_hat,
        selection0)``, ``TRACE`` yields ``(in_good, in_best)``, and - only when
        ``test_metadata`` carries algorithm performance columns (F9) -
        ``EVALUATION`` yields the ground-truth-vs-prediction evaluation fields
        as an ``_EvaluationResult``. ``EVALUATION`` is omitted entirely (not
        yielded at all) when the test set has no ground truth, matching
        ``explore()``'s own conditional field population.

        When the test set introduces algorithms absent from training (full
        MATLAB parity, F9), ``PYTHIA``'s and ``TRACE``'s yielded arrays are
        widened to include a column per new algorithm (``False``/``0``
        placeholders - no trained classifier or footprint exists for them),
        matching MATLAB's own ``evaluateTestSet`` reconciliation happening
        *before* ``PYTHIA``/``TRACE`` run, not after.

        Args
        ----
            test_metadata : Metadata
                New instances with the same feature columns as training data.

        Yields
        ------
            AnnotatedExploreOutput
                The stage that just ran and its output.

        Raises
        ------
            RuntimeError
                If build() has not been called before explore().
            ValueError
                If test_metadata features do not match training features. A fitted
                TRACE/projection dimension mismatch is deliberately checked by
                :meth:`TraceStage.predict` only when the lazy iterator advances to
                TRACE, after PYTHIA has been yielded.
        """
        self._validate_for_explore(test_metadata)

        has_ground_truth = len(test_metadata.algorithm_names) > 0
        new_algo_labels: list[str] = []
        if has_ground_truth:
            algo_labels = self._require_model().data.algo_labels
            new_algo_labels = self._find_new_algorithms(test_metadata, algo_labels)

        x = self._explore_prelim(self._extract_features(test_metadata))
        yield AnnotatedExploreOutput(ExploreStage.PRELIM, x)
        x = self._explore_sifted(x)
        yield AnnotatedExploreOutput(ExploreStage.SIFTED, x)
        z = self._explore_pilot(x)
        yield AnnotatedExploreOutput(ExploreStage.PILOT, z)
        pythia_result = self._explore_pythia(z, len(new_algo_labels))
        yield AnnotatedExploreOutput(ExploreStage.PYTHIA, pythia_result)

        yield AnnotatedExploreOutput(
            ExploreStage.TRACE,
            self._explore_trace(z, len(new_algo_labels)),
        )

        if has_ground_truth:
            y_hat = pythia_result[0]
            evaluation_result = self._explore_evaluate(
                test_metadata,
                y_hat,
                new_algo_labels,
            )
            rescored_trace = TraceStage.rescore(
                self._require_model().trace,
                z,
                evaluation_result.y_actual,
                evaluation_result.p_actual,
                evaluation_result.beta_actual,
                evaluation_result.algo_labels,
            )
            yield AnnotatedExploreOutput(
                ExploreStage.EVALUATION,
                evaluation_result._replace(trace_out=rescored_trace),
            )

    def _require_model(self) -> Model:
        """Return the trained model, raising if build() hasn't been called yet."""
        try:
            return self.model
        except StageRunningError as exc:
            raise RuntimeError(
                "Must call build() before explore(). "
                "The instance space model must be trained first.",
            ) from exc

    def _validate_for_explore(self, metadata: Metadata) -> None:
        """Validate that the instance space is ready for explore and metadata is valid.

        Args
        ----
            metadata : Metadata
                Test metadata to validate.

        Raises
        ------
            RuntimeError
                If build() has not been called.
            ValueError
                If test metadata features don't match training features.
        """
        self._require_model()
        validate_viable_dimensions(
            metadata.features,
            metadata.algorithms,
            require_algorithms=False,
            context="Explore metadata",
        )

        # Training feature names, pre-SIFTED: build() overwrites the model's own
        # feat_labels with the post-SIFTED subset, so the original metadata is the
        # source of truth for what explore()'s feature extraction needs to select.
        training_features = set(self._metadata.feature_names)
        test_features = set(metadata.feature_names)

        # Check that test data has all required features
        missing_features = training_features - test_features
        if missing_features:
            raise ValueError(
                f"Test metadata is missing features required by training: "
                f"{sorted(missing_features)}",
            )

        # Warn about extra features (they will be ignored)
        extra_features = test_features - training_features
        if extra_features:
            logger.warning(
                f"Test metadata has extra features that will be ignored: "
                f"{sorted(extra_features)}",
            )

    def _extract_features(self, metadata: Metadata) -> NDArray[np.double]:
        """Extract feature matrix from metadata, matching training format.

        Extracts features in the same order as training data and handles
        any reordering needed.

        Args
        ----
            metadata : Metadata
                Metadata containing features to extract.

        Returns
        -------
            NDArray[np.double]
                Feature matrix with shape (n_instances, n_features).
        """
        # Get the feature order from training (pre-SIFTED, see _validate_for_explore)
        training_feature_names = self._metadata.feature_names

        # Build feature matrix in training order
        test_feature_dict = dict(
            zip(metadata.feature_names, range(len(metadata.feature_names))),
        )

        # Reorder test features to match training order
        feature_indices = [test_feature_dict[name] for name in training_feature_names]
        x = metadata.features[:, feature_indices]

        return x.astype(np.double)

    def _extract_instance_labels(self, metadata: Metadata) -> pd.Series:  # type: ignore[type-arg]
        """Extract instance labels from metadata.

        Args
        ----
            metadata : Metadata
                Metadata containing instance labels.

        Returns
        -------
            pd.Series
                Series of instance labels.
        """
        return metadata.instance_labels

    def _explore_prelim(self, x: NDArray[np.double]) -> NDArray[np.double]:
        """Compatibility wrapper for :meth:`PrelimStage.predict`."""
        return PrelimStage.predict(
            PrelimPredictInput(
                x,
                self._options.auto.preproc,
                self._options.bound.flag,
                self._options.norm.flag,
            ),
            self._require_model().prelim,
        )

    def _explore_sifted(self, x: NDArray[np.double]) -> NDArray[np.double]:
        """Compatibility wrapper for :meth:`SiftedStage.predict`."""
        return SiftedStage.predict(
            SiftedPredictInput(x),
            self._require_model().sifted,
        )

    def _explore_pilot(self, x: NDArray[np.double]) -> NDArray[np.double]:
        """Compatibility wrapper for :meth:`PilotStage.predict`."""
        return PilotStage.predict(
            PilotPredictInput(x),
            self._require_model().pilot,
        )

    def _explore_pythia(
        self,
        z: NDArray[np.double],
        n_new_algos: int = 0,
    ) -> tuple[NDArray[np.bool_], NDArray[np.double], NDArray[np.int_]]:
        """Compatibility wrapper for :meth:`PythiaStage.predict`."""
        predicted = PythiaStage.predict(
            PythiaPredictInput(z, n_new_algos),
            self._require_model().pythia,
        )
        return predicted[0], predicted[1], predicted[2]

    def _explore_trace(
        self,
        z: NDArray[np.double],
        n_new_algos: int = 0,
    ) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
        """Compatibility wrapper for :meth:`TraceStage.predict`."""
        predicted = TraceStage.predict(
            TracePredictInput(z, n_new_algos),
            self._require_model().trace,
        )
        return predicted[0], predicted[1]

    def _find_new_algorithms(
        self,
        test_metadata: Metadata,
        algo_labels: list[str],
    ) -> list[str]:
        """Find test-set algorithm names absent from the trained model (F9).

        Case-insensitive match (mirrors MATLAB's ``strcmpi``, `InstanceSpace.
        evaluateTestSet`'s reconciliation step). Order matches first
        appearance in `test_metadata.algorithm_names`, duplicates collapsed -
        matching MATLAB's own append-in-encounter-order `Yaux`/`lblaux`
        widening.

        Args
        ----
            test_metadata : Metadata
                Test metadata, possibly carrying `algo_*` performance columns.
            algo_labels : list[str]
                The trained algorithm order (`Model.data.algo_labels`).

        Returns
        -------
            list[str]
                Test-set algorithm names not present in `algo_labels`.
        """
        trained_lower = {label.lower() for label in algo_labels}
        seen: set[str] = set()
        new_algos: list[str] = []
        for name in test_metadata.algorithm_names:
            lower = name.lower()
            if lower not in trained_lower and lower not in seen:
                new_algos.append(name)
                seen.add(lower)
        return new_algos

    def _build_test_algo_matrix(
        self,
        test_metadata: Metadata,
        algo_labels: list[str],
        new_algo_labels: list[str],
    ) -> tuple[NDArray[np.double], NDArray[np.bool_]]:
        """Reindex test_metadata's raw performance to [trained | new] algo order.

        Case-insensitive name matching (mirrors MATLAB's ``strcmpi``). An
        algorithm present in training but absent from the test set becomes an
        all-NaN column - `compute_binary_performance`'s existing
        `max_perf`/`abs_perf` NaN handling then treats it as never the best
        for any instance (matching MATLAB's convention for missing ground
        truth), so no separate branch is needed for that case there. The
        returned mask records which of the *trained* columns have real ground
        truth for callers that need that metadata. MATLAB-compatible PYTHIA
        evaluation deliberately scores every non-empty trained classifier; an
        absent column therefore remains the reconciled all-false truth vector.

        Algorithms in `new_algo_labels` (present in the test set, absent from
        training - see `_find_new_algorithms`) are appended as extra columns
        with their real test-set performance, matching MATLAB's `Yaux`
        widening in `evaluateTestSet` - they participate as full candidates
        in the binary-performance comparison (`compute_binary_performance`
        can pick a new algorithm as "best" for an instance) even though no
        classifier/footprint exists for them elsewhere in `explore()`.

        Args
        ----
            test_metadata : Metadata
                Test metadata, possibly carrying `algo_*` performance columns.
            algo_labels : list[str]
                The trained algorithm order (`Model.data.algo_labels`).
            new_algo_labels : list[str]
                Test-set-only algorithm names, from `_find_new_algorithms`.

        Returns
        -------
            tuple[NDArray[np.double], NDArray[np.bool_]]
                - y_raw_test: (n_instances, n_trained + n_new) raw performance,
                  reindexed to `algo_labels + new_algo_labels`' order, NaN
                  where a trained algorithm is absent from the test set.
                - has_ground_truth: (n_trained,) mask of which *trained*
                  columns have real ground truth (new algorithms always do,
                  by construction - not included in this mask).
        """
        test_cols = {
            name.lower(): i for i, name in enumerate(test_metadata.algorithm_names)
        }
        ninst = test_metadata.algorithms.shape[0]
        n_trained = len(algo_labels)
        n_new = len(new_algo_labels)
        y_raw_test = np.full((ninst, n_trained + n_new), np.nan, dtype=np.double)
        has_ground_truth = np.zeros(n_trained, dtype=np.bool_)

        for i, label in enumerate(algo_labels):
            col = test_cols.get(label.lower())
            if col is not None:
                y_raw_test[:, i] = test_metadata.algorithms[:, col]
                has_ground_truth[i] = True

        for j, label in enumerate(new_algo_labels):
            col = test_cols[label.lower()]
            y_raw_test[:, n_trained + j] = test_metadata.algorithms[:, col]

        return y_raw_test, has_ground_truth

    def _explore_evaluate(
        self,
        test_metadata: Metadata,
        y_hat: NDArray[np.bool_],
        new_algo_labels: list[str],
    ) -> _EvaluationResult:
        """Evaluate PYTHIA's predictions against ground truth (F9).

        Ports MATLAB's `InstanceSpace.evaluateTestSet` (`InstanceSpace.m:736`),
        which calls the training-time `PYTHIA()` function itself, switched
        into `PYTHIAevalMode` by a 7th (trained-model) argument: computes the
        same PRELIM-equivalent ground-truth fields (`y_bin`/`y_best`/`p`/
        `beta`) for the test set via `compute_binary_performance` (F9's
        extraction, shared with `PrelimStage._prelim()`), then derives
        per-algorithm `accuracy`/`precision`/`recall`/confusion-matrix from
        the already-trained classifiers' predictions (`y_hat`, already
        computed by `_explore_pythia` - not recomputed here) against that
        ground truth, matching MATLAB's exact formulas (`tp/(tp+fp)`,
        `tp/(tp+fn)`, `(tp+tn)/ninst`, `core/PYTHIA.m:379-381`).

        A trained algorithm absent from the test metadata retains its
        reconciled all-false truth column and is scored when its classifier is
        non-empty, exactly as MATLAB does. Algorithms in `new_algo_labels`
        always have real ground truth by construction but no trained-model
        slot, so their rates stay `NaN` while their confusion rows remain zero.
        They still participate as full candidates in `y_best_actual`/
        `p_actual`/`beta_actual` through the widened performance calculation.

        Args
        ----
            test_metadata : Metadata
                Test metadata carrying `algo_*` performance columns (the
                caller - `explore_stage_iter` - only calls this when at least
                one is present).
            y_hat : NDArray[np.bool_]
                PYTHIA's binary predictions, already computed by
                `_explore_pythia` and already widened to include
                `new_algo_labels`' columns (all `False`).
                Shape: (n_instances, n_trained + n_new).
            new_algo_labels : list[str]
                Test-set-only algorithm names, from `_find_new_algorithms`.

        Returns
        -------
            _EvaluationResult
                The ground-truth-vs-prediction evaluation fields, all
                per-algorithm fields width `n_trained + n_new`.
        """
        model = self._require_model()
        algo_labels = model.data.algo_labels
        y_raw_test, _ = self._build_test_algo_matrix(
            test_metadata,
            algo_labels,
            new_algo_labels,
        )

        perf = compute_binary_performance(
            y_raw_test,
            self._options.perf,
            self._options.general,
            log_prefix="EXPLORE",
        )

        evaluated = PythiaStage.evaluate(
            PythiaEvaluateInput(
                perf.y_bin,
                y_hat,
            ),
            model.pythia,
        )

        return _EvaluationResult(
            y_actual=perf.y_bin,
            y_best_actual=perf.y_best,
            p_actual=perf.p,
            beta_actual=perf.beta,
            accuracy_actual=evaluated.accuracy,
            precision_actual=evaluated.precision,
            recall_actual=evaluated.recall,
            cvcmat_actual=evaluated.cvcmat,
            algo_labels=[*algo_labels, *new_algo_labels],
        )


def instance_space_from_files(
    metadata_filepath: Path,
    options_filepath: Path,
) -> InstanceSpace | None:
    """Construct an instance space object from 2 files.

    Args
    ----
        metadata_filepath (Path): Path to the metadata csv file.
        options_filepath (Path): Path to the options json file.

    Returns
    -------
        InstanceSpace | None: A new instance space object instantiated
        with metadata and options from the specified files, or None
        if the initialization fails.

    """
    logger.info(
        "-------------------------------------------------------------------------",
    )
    logger.info("-> Loading the data.")

    metadata = from_csv_file(metadata_filepath)

    if metadata is None:
        logger.error("Failed to initialize metadata")
        return None

    logger.info("-> Successfully loaded the data.")
    logger.info(
        "-------------------------------------------------------------------------",
    )
    logger.info("-> Loading the options.")

    options = from_json_file(options_filepath)

    if options is None:
        logger.error("Failed to initialize options")
        return None

    logger.info("-> Successfully loaded the options.")

    if options.general.verbose:
        logger.debug("-> Listing options to be used:")
        for line in format_options(options):
            logger.debug(line)

    return InstanceSpace(metadata, options)


def instance_space_from_directory(directory: Path) -> InstanceSpace | None:
    """Construct an instance space object from 2 files.

    Args
    ----
        directory (str): Path to correctly formatted directory,
        where the .csv file is metadata.csv, and .json file is
        options.json

    Returns
    -------
        InstanceSpace | None: A new instance space
        object instantiated with metadata and options from
        the specified directory, or None if the initialization fails.

    """
    metadata_path = Path(directory / "metadata.csv")
    options_path = Path(directory / "options.json")

    return instance_space_from_files(metadata_path, options_path)
