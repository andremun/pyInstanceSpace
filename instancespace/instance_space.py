"""Perform instance space analysis on given dataset and configuration.

Construct an instance space from data and configuration files located in a specified
directory. The instance space is represented as a Model object, which encapsulates the
analytical results and metadata of the instance space analysis.
"""

from collections.abc import Generator
from dataclasses import fields
from datetime import datetime
from pathlib import Path
from typing import Any, NamedTuple, TypeVar

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from instancespace.data.metadata import Metadata, from_csv_file
from instancespace.data.model import ExploreResult
from instancespace.data.options import (
    AutoOptions,
    BoundOptions,
    CloisterOptions,
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
from instancespace.model import Model
from instancespace.stage_builder import StageBuilder
from instancespace.stage_runner import (
    AnnotatedStageOutput,
    StageRunner,
    StageRunningError,
)
from instancespace.stages.cloister import CloisterStage
from instancespace.stages.pilot import PilotStage
from instancespace.stages.prelim import PrelimStage
from instancespace.stages.preprocessing import PreprocessingStage
from instancespace.stages.pythia import PythiaStage
from instancespace.stages.sifted import SiftedStage
from instancespace.stages.stage import IN, OUT, Stage, StageClass
from instancespace.stages.trace import TraceStage

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
        )


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

    _model: Model | None
    _final_output: dict[str, Any] | None

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
        """
        self._metadata = metadata
        self._options = options
        self._stages = stages

        self._model: Model | None = None
        self._final_output: dict[str, Any] | None = None
        self._explore_results: list[ExploreResult] = []

        stage_builder = StageBuilder()

        for stage in stages:
            stage_builder.add_stage(stage)

        annotations = stage_builder._named_tuple_to_stage_arguments(  # noqa: SLF001
            _InstanceSpaceInputs,
        )

        if additional_initial_inputs_type is not None:
            annotations |= (
                stage_builder._named_tuple_to_stage_arguments(  # noqa: SLF001
                    additional_initial_inputs_type,
                )
            )

        self._runner = stage_builder.build(annotations)

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

    def build(
        self,
    ) -> Model:
        """Build the instance space.

        Options will be broken down to sub fields to be passed to stages. You can
        override inputs to stages.

        Returns
        -------
            tuple[Any]: The output of all stages

        """
        inputs = _InstanceSpaceInputs.from_metadata_and_options(
            self.metadata,
            self.options,
        )
        self._final_output = self._runner.run_all(inputs)

        return self.model

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
        )
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
        return self._runner.run_stage(stage, **arguments)

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
        )
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
                New instances with the same feature columns as training data.
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
        # Validate that training has been completed
        self._validate_for_explore(test_metadata)

        # Generate dataset_id if not provided
        if dataset_id is None:
            dataset_id = f"explore_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Extract raw features and instance labels from test metadata
        x_raw = self._extract_features(test_metadata)
        inst_labels = self._extract_instance_labels(test_metadata)

        # Apply stages in order
        x = self._explore_prelim(x_raw)
        x = self._explore_sifted(x)
        z = self._explore_pilot(x)

        # Get algorithm predictions and footprint membership
        pythia_result = self._explore_pythia(z)
        trace_result = self._explore_trace(z)

        # Build result
        result = ExploreResult(
            dataset_id=dataset_id,
            timestamp=datetime.now(),
            x=x,
            z=z,
            y_hat=pythia_result[0] if pythia_result else None,
            pr0_hat=pythia_result[1] if pythia_result else None,
            selection0=pythia_result[2] if pythia_result else None,
            in_good=trace_result[0] if trace_result else None,
            in_best=trace_result[1] if trace_result else None,
            in_space=trace_result[2] if trace_result else None,
            inst_labels=inst_labels,
        )

        self._explore_results.append(result)
        return result

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
        if self._model is None:
            raise RuntimeError(
                "Must call build() before explore(). "
                "The instance space model must be trained first."
            )

        # Get training feature names from the model's data
        training_features = set(self._model.data.feat_labels)
        test_features = set(metadata.feature_names)

        # Check that test data has all required features
        missing_features = training_features - test_features
        if missing_features:
            raise ValueError(
                f"Test metadata is missing features required by training: "
                f"{sorted(missing_features)}"
            )

        # Warn about extra features (they will be ignored)
        extra_features = test_features - training_features
        if extra_features:
            print(
                f"Warning: Test metadata has extra features that will be ignored: "
                f"{sorted(extra_features)}"
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
        # Get the feature order from training
        training_feature_names = self._model.data.feat_labels  # type: ignore[union-attr]

        # Build feature matrix in training order
        test_feature_dict = dict(zip(metadata.feature_names, range(len(metadata.feature_names))))

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

        prelim = self._model.prelim  # type: ignore[union-attr]

        # Create a copy to avoid modifying input
        x_transformed = x.copy()
        n_features = x.shape[1]

        for i in range(n_features):
            x_transformed[:, i] = np.clip(
                x_transformed[:, i],
                prelim.lo_bound[i],
                prelim.hi_bound[i]
            )

            x_transformed[:, i] = x_transformed[:, i] - prelim.min_x[i] + 1

            idx_valid = ~np.isnan(x_transformed[:, i])
            if np.any(idx_valid):
                x_transformed[idx_valid, i] = stats.boxcox(
                    x_transformed[idx_valid, i],
                    prelim.lambda_x[i]
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
        sifted = self._model.sifted  # type: ignore[union-attr]
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

        Note
        ----
            PLACEHOLDER - Currently returns zeros.
        """
        # TODO: Implement projection using self._model.pilot.a
        return np.zeros((x.shape[0], 2), dtype=np.double)

    def _explore_pythia(
        self,
        z: NDArray[np.double],
    ) -> tuple[NDArray[np.bool_], NDArray[np.double], NDArray[np.int_]] | None:
        """Get algorithm predictions using PYTHIA SVMs.

        Args
        ----
            z : NDArray[np.double]
                2D coordinates with shape (n_instances, 2).

        Returns
        -------
            tuple[NDArray[np.bool_], NDArray[np.double], NDArray[np.int_]] | None
                Tuple of (y_hat, pr0_hat, selection0) or None.
                - y_hat: Binary predictions (n_instances, n_algorithms)
                - pr0_hat: Probability predictions (n_instances, n_algorithms)
                - selection0: Recommended algorithm indices (n_instances,)

        Note
        ----
            PLACEHOLDER - Currently returns None.
        """
        # TODO: Implement SVM predictions using self._model.pythia
        return None

    def _explore_trace(
        self,
        z: NDArray[np.double],
    ) -> tuple[NDArray[np.bool_], NDArray[np.bool_], NDArray[np.bool_]] | None:
        """Check footprint membership using TRACE polygons.

        Args
        ----
            z : NDArray[np.double]
                2D coordinates with shape (n_instances, 2).

        Returns
        -------
            tuple[NDArray[np.bool_], NDArray[np.bool_], NDArray[np.bool_]] | None
                Tuple of (in_good, in_best, in_space) or None.
                - in_good: Inside "good" footprint (n_instances, n_algorithms)
                - in_best: Inside "best" footprint (n_instances, n_algorithms)
                - in_space: Inside overall space boundary (n_instances,)

        Note
        ----
            PLACEHOLDER - Currently returns None.
        """
        # TODO: Implement footprint containment using self._model.trace
        return None


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
    print("-------------------------------------------------------------------------")
    print("-> Loading the data.")

    metadata = from_csv_file(metadata_filepath)

    if metadata is None:
        print("Failed to initialize metadata")
        return None

    print("-> Successfully loaded the data.")
    print("-------------------------------------------------------------------------")
    print("-> Loading the options.")

    options = from_json_file(options_filepath)

    if options is None:
        print("Failed to initialize options")
        return None

    print("-> Successfully loaded the options.")

    print("-> Listing options to be used:")
    for field_name in fields(InstanceSpaceOptions):
        field_value = getattr(options, field_name.name)
        print(f"{field_name.name}: {field_value}")

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
