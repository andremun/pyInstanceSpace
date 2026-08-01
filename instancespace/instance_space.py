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
from shapely.geometry import Point

from instancespace.data.metadata import Metadata, from_csv_file
from instancespace.data.model import ExploreResult
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
from instancespace.stages.pilot import PilotStage
from instancespace.stages.prelim import PrelimStage
from instancespace.stages.preprocessing import PreprocessingStage
from instancespace.stages.pythia import PythiaStage
from instancespace.stages.sifted import SiftedStage
from instancespace.stages.stage import IN, OUT, Stage, StageClass
from instancespace.stages.trace import TraceStage
from instancespace.utils.print_options import format_options

# Fraction of explore()-time instances clipped to the training PRELIM bounds above
# which an out-of-distribution warning fires (matches MATLAB's exploreIS.m).
_OOD_CLIPPED_FRACTION_THRESHOLD = 0.05

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

    The explore-time counterpart of a build-time `StageClass`: `explore()`
    reuses `build()`'s trained parameters rather than running the real
    `Stage` subclasses in a predict mode (see roadmap item F8), so these
    are lightweight identifiers, not `Stage` subclasses themselves.
    """

    PRELIM = "prelim"
    SIFTED = "sifted"
    PILOT = "pilot"
    PYTHIA = "pythia"
    TRACE = "trace"


class AnnotatedExploreOutput(NamedTuple):
    """The yielded output of running one explore-time step.

    Mirrors `stage_runner.AnnotatedStageOutput`'s (stage, output) shape.
    """

    stage: ExploreStage
    output: Any


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

    # Lazily created, reused across staged calls (Q6) - mirrors MATLAB's
    # ensurePool()'s "rightSize" check: recreated only if the worker count
    # changes, not on every stage call. Currently backs TraceStage's
    # footprint computation.
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
            if self._final_output is None:
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
        (see `StageRunner.__getstate__()`): `build()`/`run_stage()` set it to
        the *same* dict object as `_runner._available_arguments` (not a
        copy), so it carries the same stale `"executor"` entry.
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

    def _get_executor(self) -> ThreadPoolExecutor:
        """Return a cached ThreadPoolExecutor, reused across staged calls (Q6).

        Recreated only if the worker count changes (mirrors MATLAB's
        `ensurePool()`'s "rightSize" check) rather than on every call.
        """
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

    def close(self) -> None:
        """Release resources held across staged calls (currently: the TRACE pool).

        Safe to call even if nothing has been built yet. A subsequent
        `build()`/`run_stage()`/etc. call recreates the pool lazily.
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

            self._final_output = self._runner._available_arguments  # noqa: SLF001

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
        inputs = _InstanceSpaceInputs.from_metadata_and_options(
            self.metadata,
            self.options,
        )._replace(executor=self._get_executor())
        yield from self._runner.run_iter(inputs)

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

        self._final_output = self._runner._available_arguments  # noqa: SLF001
        duration_seconds = time.monotonic() - start

        self._progress_reporter.report_stage_completed(
            stage_name,
            duration_seconds=duration_seconds,
            instance_space=self,
        )

        current_item = self._runner._current_schedule_item  # noqa: SLF001
        stage_order = self._runner._stage_order  # noqa: SLF001
        schedule_complete = current_item >= len(stage_order)
        if schedule_complete:
            self._progress_reporter.report_job_completed(instance_space=self)

        return output

    def run_until_stage(
        self,
        stage: StageClass,
        **_arguments: Any,  # noqa: ANN401
    ) -> dict[str, Any]:
        """Run all stages until the specified stage, as well as the specified stage.

        Args
        ----
            stage StageClass: The stage to stop running stages after.
            metadata Metadata: _description_
            options InstanceSpaceOptions: _description_
            **arguments dict[str, Any]: if this is the first time stages are ran the
                initial inputs, and overriding inputs for other stages.

        Returns
        -------
            dict[str, Any]: The raw output dict of all ran stages.
        """
        inputs = _InstanceSpaceInputs.from_metadata_and_options(
            self.metadata,
            self.options,
        )._replace(executor=self._get_executor())
        return self._runner.run_until_stage(
            stage,
            inputs,
        )

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
        3. Dimensionality reduction to 2D (PILOT projection)
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
                If test_metadata features don't match training features.
        """
        # Run every inference stage, then assemble the result from each stage's output
        stages = {
            annotated.stage: annotated.output
            for annotated in self.explore_stage_iter(test_metadata)
        }

        if dataset_id is None:
            dataset_id = (
                f"explore_{datetime.now(tz=UTC).strftime('%Y%m%d_%H%M%S')}"
            )

        inst_labels = self._extract_instance_labels(test_metadata)
        pythia_result = stages[ExploreStage.PYTHIA]
        trace_result = stages[ExploreStage.TRACE]

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
        selection0)`` and ``TRACE`` yields ``(in_good, in_best)``.

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
                If test_metadata features don't match training features.
        """
        self._validate_for_explore(test_metadata)

        x = self._explore_prelim(self._extract_features(test_metadata))
        yield AnnotatedExploreOutput(ExploreStage.PRELIM, x)
        x = self._explore_sifted(x)
        yield AnnotatedExploreOutput(ExploreStage.SIFTED, x)
        z = self._explore_pilot(x)
        yield AnnotatedExploreOutput(ExploreStage.PILOT, z)
        yield AnnotatedExploreOutput(ExploreStage.PYTHIA, self._explore_pythia(z))
        yield AnnotatedExploreOutput(ExploreStage.TRACE, self._explore_trace(z))

    def _require_model(self) -> Model:
        """Return the trained model, raising if build() hasn't been called yet."""
        if self._model is None:
            raise RuntimeError(
                "Must call build() before explore(). "
                "The instance space model must be trained first.",
            )
        return self._model

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
        """Apply PRELIM transformations to features.

        Applies bounding, Box-Cox transformation, and z-score normalization
        using parameters learned during training.

        Args
        ----
            x : NDArray[np.double]
                Raw feature matrix with shape (n_instances, n_features).

        Returns
        -------
            NDArray[np.double]
                Transformed feature matrix with shape (n_instances, n_features).
        """
        from scipy import stats

        prelim = self._require_model().prelim

        clipped = np.any(
            (x < prelim.lo_bound) | (x > prelim.hi_bound),
            axis=1,
        )
        frac_clipped = np.mean(clipped)
        if frac_clipped > _OOD_CLIPPED_FRACTION_THRESHOLD:
            logger.warning(
                f"explore(): {frac_clipped:.1%} of test instances have at least one "
                "feature outside the training bounds and were clipped to them. This "
                "suggests the test set may not be well represented by the trained "
                "instance space; consider retraining with a combined dataset.",
            )

        # Create a copy to avoid modifying input
        x_transformed = x.copy()
        n_features = x.shape[1]

        for i in range(n_features):
            x_transformed[:, i] = np.clip(
                x_transformed[:, i],
                prelim.lo_bound[i],
                prelim.hi_bound[i],
            )

            x_transformed[:, i] = x_transformed[:, i] - prelim.min_x[i] + 1

            idx_valid = ~np.isnan(x_transformed[:, i])
            if np.any(idx_valid):
                x_transformed[idx_valid, i] = stats.boxcox(
                    x_transformed[idx_valid, i],
                    prelim.lambda_x[i],
                )

            if np.any(idx_valid):
                x_transformed[idx_valid, i] = (
                    x_transformed[idx_valid, i] - prelim.mu_x[i]
                ) / prelim.sigma_x[i]

        return x_transformed

    def _explore_sifted(self, x: NDArray[np.double]) -> NDArray[np.double]:
        """Apply feature selection from SIFTED stage.

        Args
        ----
            x : NDArray[np.double]
                Feature matrix with shape (n_instances, n_features).

        Returns
        -------
            NDArray[np.double]
                Feature matrix with selected features only.
                Shape: (n_instances, n_selected_features).
        """
        sifted = self._require_model().sifted
        selected_indices = sifted.selvars
        x_selected = x[:, selected_indices]

        return x_selected

    def _explore_pilot(self, x: NDArray[np.double]) -> NDArray[np.double]:
        """Project features to 2D instance space using PILOT.

        Args
        ----
            x : NDArray[np.double]
                Feature matrix with shape (n_instances, n_selected_features).

        Returns
        -------
            NDArray[np.double]
                2D coordinates with shape (n_instances, 2).
        """
        a = self._require_model().pilot.a
        return x @ a.T

    def _explore_pythia(
        self,
        z: NDArray[np.double],
    ) -> tuple[NDArray[np.bool_], NDArray[np.double], NDArray[np.int_]]:
        """Get algorithm predictions using PYTHIA's trained classifiers.

        Ports MATLAB PYTHIAtest.m: z-score normalises the 2D coordinates using the
        training projection's own mean/std, recomputed here from ``model.pilot.z``
        rather than read back from the stored ``PythiaOutput.mu``/``sigma`` (the
        two are equal - both are simple mean/std of the same raw training
        coordinates - this just avoids the extra indirection), applies each
        per-algorithm classifier natively via scikit-learn's own ``predict``/
        ``predict_proba`` (an ``SVC`` unless ``PythiaOptions.classifier`` selected
        a different registered type - S1 made this classifier-agnostic before F1
        added the registry, so no change was needed here), and picks the algorithm
        with the highest precision-weighted positive prediction per instance.

        Args
        ----
            z : NDArray[np.double]
                2D coordinates with shape (n_instances, 2).

        Returns
        -------
            tuple[NDArray[np.bool_], NDArray[np.double], NDArray[np.int_]]
                - y_hat: binary good/bad predictions, shape (n_instances, n_algorithms)
                - pr0_hat: posterior probability of the "bad" class, shape
                  (n_instances, n_algorithms)
                - selection0: recommended algorithm index (0-based) per instance,
                  or -1 when no algorithm was predicted good. Shape (n_instances,)
        """
        model = self._require_model()
        pythia = model.pythia
        train_z = np.asarray(model.pilot.z, dtype=np.double)
        mu = train_z.mean(axis=0)
        sigma = train_z.std(axis=0, ddof=1)
        precision = np.asarray(pythia.precision, dtype=np.double)
        svms = pythia.svm

        z_norm = (z - mu) / sigma
        n_inst = z_norm.shape[0]
        n_algos = len(svms)

        y_hat = np.zeros((n_inst, n_algos), dtype=np.bool_)
        pr0_hat = np.zeros((n_inst, n_algos), dtype=np.double)

        for i, svc in enumerate(svms):
            proba = svc.predict_proba(z_norm)
            bad_idx = int(np.where(~svc.classes_)[0][0])
            pr0_hat[:, i] = proba[:, bad_idx]
            y_hat[:, i] = svc.predict(z_norm)

        weighted = y_hat.astype(np.double) * precision
        best = weighted.max(axis=1)
        selection0 = weighted.argmax(axis=1).astype(np.int_)
        selection0[best <= 0] = -1

        return y_hat, pr0_hat, selection0

    def _explore_trace(
        self,
        z: NDArray[np.double],
    ) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
        """Check footprint membership using TRACE polygons.

        Ports the per-instance equivalent of MATLAB TRACEtest: for each test
        point and each algorithm, check whether the point lies inside the
        algorithm's good and best footprints. MATLAB's ``inpolygon`` treats
        boundary points as inside; ``polygon.covers`` matches that semantics
        (closed set), whereas ``polygon.contains`` would exclude the boundary.

        ``in_space`` is intentionally omitted: ``exploreIS.m`` does not compute
        it, and the value in ``step5_trace_membership.csv`` is sourced from
        CLOISTER (a build-time stage outside this port's scope).

        Args
        ----
            z : NDArray[np.double]
                2D coordinates with shape (n_instances, 2).

        Returns
        -------
            tuple[NDArray[np.bool_], NDArray[np.bool_]]
                - in_good: (n_instances, n_algorithms) bool array
                - in_best: (n_instances, n_algorithms) bool array
        """
        trace = self._require_model().trace
        n = z.shape[0]
        n_algos = len(trace.good)
        points = [Point(z[i, 0], z[i, 1]) for i in range(n)]

        in_good = np.zeros((n, n_algos), dtype=np.bool_)
        in_best = np.zeros((n, n_algos), dtype=np.bool_)

        for j in range(n_algos):
            good_poly = trace.good[j].polygon
            best_poly = trace.best[j].polygon
            if good_poly is not None:
                in_good[:, j] = [good_poly.covers(p) for p in points]
            if best_poly is not None:
                in_best[:, j] = [best_poly.covers(p) for p in points]

        return in_good, in_best


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
