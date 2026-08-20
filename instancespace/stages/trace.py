# SPDX-License-Identifier: LicenseRef-PolyForm-Noncommercial-1.0.0
# Copyright (c) 2024-2026 Mario Andrés Muñoz
"""TRACE Stage Module for Performance-Based Footprint Estimation.

This module implements the TRACE stage, which analyzes the performance of multiple
algorithms by generating geometric footprints. These footprints represent the areas
of good, best, and beta performance based on the clustering of instance data. The
footprints are further evaluated for their density and purity in relation to the
performance metrics of the algorithms.

This module provides the historical DBSCAN-based TRACE implementation and MATLAB's
current two- or three-dimensional TRACE3 alpha-shape implementation.

The TRACE stage has several key steps:
1. Cluster the instance data using DBSCAN to identify regions of interest.
2. Generate geometric footprints representing algorithm performance.
3. Detect and resolve contradictions between algorithm footprints.
4. Compute performance metrics such as area, density, and purity for each footprint.
5. Optionally smoothen the polygonal boundaries for more refined footprint shapes.

This module is structured around the `Trace` class, which encapsulates the entire
process of footprint estimation and performance evaluation. Methods are provided
to cluster data, generate polygons, resolve contradictions between footprints, and
compute statistical metrics.

Dependencies:
- multiprocessing
- numpy
- pandas
- scipy
- shapely
- sklearn

Classes
-------
Trace :
    The primary class that implements the TRACE stage, providing methods to generate
    footprints and compute performance-based metrics.

Footprint :
    A dataclass representing a footprint with geometric and statistical properties.

Functions
---------
from_polygon(polygon, z, y_bin, smoothen=False):
    A function to create a Footprint object from a given polygon and corresponding
    instance data, optionally smoothing the polygon borders.
"""

import multiprocessing
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import NamedTuple

import numpy as np
import pandas as pd
from loguru import logger
from numpy.typing import NDArray
from scipy.spatial import ConvexHull, QhullError
from scipy.special import gamma
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import triangulate, unary_union

from instancespace.data.model import Footprint, TraceOut, pointwise_covers
from instancespace.data.options import (
    GeneralOptions,
    ParallelOptions,
    PythiaOptions,
    TraceOptions,
)
from instancespace.stages.stage import Stage
from instancespace.utils.alpha_shape import (
    AlphaShape2D,
    AlphaShape3D,
    TetrahedralMesh,
    legacy_alpha_shape,
)

POLYGON_MIN_POINT_REQUIREMENT = 3
TRACE_ARRAY_DIMENSIONS = 2
TRACE_2D_COORDINATES = 2
TRACE_3D_COORDINATES = 3
TRACE_COORDINATE_COUNTS = (TRACE_2D_COORDINATES, TRACE_3D_COORDINATES)
MIN_ALPHA_SPECTRUM_SIZE = 2

type TraceGeometry = Polygon | MultiPolygon | TetrahedralMesh


class TraceInputs(NamedTuple):
    """A named tuple to encapsulate the inputs required for the TRACE analysis.

    Attributes:
    ----------
    z : NDArray[np.double]
        The space of instances, represented as an array of data points (features).
    selection0 : NDArray[np.int_]
        PYTHIA selections as zero-based algorithm indices; ``-1`` means that no
        algorithm was selected.
    p : NDArray[np.int_]
        PRELIM selections as MATLAB-compatible one-based algorithm indices.
    beta : NDArray[np.bool_]
        A binary array indicating specific beta thresholds for the footprint.
    algo_labels : list[str]
        A list of labels for each algorithm, represented as strings.
    y_hat : NDArray[np.bool_]
        A binary array indicating performance of the Pythia algorithm,
        where each column corresponds to an algorithm's performance.
    y_bin : NDArray[np.bool_]
        A binary array indicating performance of the data-driven approach,
        where each column corresponds to an algorithm's performance.
    trace_options : TraceOptions
        Configuration options for the TRACE analysis, determining specific behaviour
        for footprint construction and evaluation.
    general_options : GeneralOptions
        General options (e.g. verbosity), not specific to any one stage.
    """

    z: NDArray[np.double]
    selection0: NDArray[np.int_]
    p: NDArray[np.int_]
    beta: NDArray[np.bool_]
    algo_labels: list[str]
    y_hat: NDArray[np.bool_]
    y_bin: NDArray[np.bool_]
    trace_options: TraceOptions
    parallel_options: ParallelOptions
    general_options: GeneralOptions
    executor: ThreadPoolExecutor | None = None
    pythia_options: PythiaOptions = PythiaOptions.default()


class TraceOutputs(NamedTuple):
    """A named tuple to encapsulate the outputs of the TRACE analysis.

    Attributes:
    ----------
    space : Footprint
        The footprint representing the entire space of instances.
    good : list[Footprint]
        A list of footprints for the regions of good performance for each algorithm.
    best : list[Footprint]
        A list of footprints for the regions of best performance for each algorithm.
    hard : Footprint
        The footprint representing the region that fails to meet the beta threshold.
    summary : pd.DataFrame
        A pandas DataFrame containing the summary of the footprint analysis, including
        metrics such as area, density, and purity for both good and best performance
        regions.
    """

    space: Footprint
    good: list[Footprint]
    best: list[Footprint]
    hard: Footprint
    trace_summary: pd.DataFrame


class TraceStage(Stage[TraceInputs, TraceOutputs]):
    """A class to manage the TRACE analysis process for performance footprints.

    The TRACE class is designed to analyze the performance of different algorithms by
    generating geometric footprints that represent areas of good, best, and beta
    performance. The footprints are constructed based on clustering of instance data
    and are evaluated for their density and purity relative to specific algorithmic
    performance metrics.

    Attributes:
    ----------
    z : NDArray[np.double]
    The space of instances, represented as an array of data points (features).
    y_bin : NDArray[np.bool_]
    Binary indicators of performance, where each column corresponds to an
    algorithm's performance.
    p : NDArray[np.int_]
    Performance metrics for algorithms, represented as integers where each value
    corresponds to the index of an algorithm.
    beta : NDArray[np.bool_]
    Specific binary thresholds for footprint calculation.
    algo_labels : list[str]
    List of labels for each algorithm.
    opts : TraceOptions
    Configuration options for TRACE and its subroutines, controlling the behavior
    of the analysis.

    Methods:
    -------
    __init__(self) -> None:
    Initializes the Trace class without any parameters.

    run(self, z: NDArray[np.double], y_bin: NDArray[np.bool_], p: NDArray[np.int_],
    beta: NDArray[np.bool_], algo_labels: list[str], opts: TraceOptions)
    -> tuple[TraceDataChanged, TraceOut]:
    Performs the TRACE footprint analysis and returns the results, including
    footprints and a summary.

    build(self, y_bin: NDArray[np.bool_]) -> Footprint:
    Constructs a footprint polygon using DBSCAN clustering based on the provided
    binary indicators.

    contra(self, base: Footprint, test: Footprint, y_base: NDArray[np.bool_],
       y_test: NDArray[np.bool_]) -> tuple[Footprint, Footprint]:
    Detects and resolves contradictions between two footprint polygons.

    tight(self, polygon: Polygon | MultiPolygon, y_bin: NDArray[np.bool_])
    -> Polygon | None:
    Refines an existing polygon by removing slivers and improving its shape.

    fit_poly(self, polydata: NDArray[np.double], y_bin: NDArray[np.bool_])
    -> Polygon | None:
    Fits a polygon to the given data points, ensuring it adheres to purity constraints.

    summary(self, footprint: Footprint, space_area: float, space_density: float)
    -> list[float]:
    Summarizes the footprint metrics, returning a list of values such as area,
    normalized area, density, normalized density, and purity.

    throw(self) -> Footprint:
    Generates an empty footprint with default values, indicating insufficient data.

    run_dbscan(self, y_bin: NDArray[np.bool_], data: NDArray[np.double])
    -> NDArray[np.int_]:
    Performs DBSCAN clustering on the dataset and returns an array of cluster labels.

    process_algorithm(self, i: int) -> tuple[int, Footprint, Footprint]:
    Processes a single algorithm to calculate its good and best performance footprints.

    parallel_processing(self, n_workers: int, n_algos: int) -> tuple[list[Footprint],
    list[Footprint]]:
    Performs parallel processing to calculate footprints for multiple algorithms.
    """

    def __init__(
        self,
        z: NDArray[np.double],
        y_bin: NDArray[np.bool_],
        p: NDArray[np.int_],
        beta: NDArray[np.bool_],
        algo_labels: list[str],
        trace_opts: TraceOptions,
        parallel_opts: ParallelOptions,
        general_opts: GeneralOptions,
        executor: ThreadPoolExecutor | None = None,
        y_hat: NDArray[np.bool_] | None = None,
        pythia_skipped: bool = False,
    ) -> None:
        """Initialise the Trace analysis with provided data and options.

        Parameters:
        ----------
        z : NDArray[np.double]
            The space of instances, represented as an array of data points (features).
        y_bin : NDArray[np.bool_]
            Binary indicators of performance for each algorithm.
        p : NDArray[np.int_]
            Performance metrics for algorithms, where each value corresponds to
            the index of an algorithm.
        beta : NDArray[np.bool_]
            Specific binary thresholds for footprint calculation.
        algo_labels : list[str]
            List of labels for each algorithm.
        trace_opts : TraceOptions
            Configuration options for TRACE and its subroutines.
        parallel_opts : ParallelOptions
            Configuration options for parallel processing in Matilda.
        general_opts : GeneralOptions
            General options (e.g. verbosity), not specific to any one stage.
        executor : ThreadPoolExecutor | None
            A caller-owned, already-running pool to submit footprint work to
            instead of creating and tearing down a fresh one. ``None`` (the
            default) preserves the previous per-call pool behaviour.
        y_hat : NDArray[np.bool_] | None
            Optional PYTHIA predictions used only by TRACE3.
        pythia_skipped : bool
            Whether PYTHIA intentionally returned placeholder predictions.
        """
        self.z = z
        self.y_bin = y_bin
        self.p = p
        self.beta = beta
        self.algo_labels = algo_labels
        self.opts = trace_opts
        self.parallel_opts = parallel_opts
        self.general_opts = general_opts
        self.executor = executor
        self.y_hat = y_hat
        self.pythia_skipped = pythia_skipped
        self._trace3_space_area = 0.0

    def _log(self, msg: str) -> None:
        """Log a top-level, always-shown stage message."""
        logger.info(f"[TRACE] {msg}")

    def _log_detail(self, msg: str) -> None:
        """Log per-trial/per-iteration detail, only shown when general.verbose."""
        if self.general_opts.verbose:
            logger.debug(f"[TRACE] {msg}")

    @staticmethod
    def _inputs() -> type[TraceInputs]:
        """Use the method for determining the inputs for trace.

        Args
        ----

        Returns
        -------
            list[tuple[str, type]]
                List of inputs for the stage
        """
        return TraceInputs

    @staticmethod
    def _outputs() -> type[TraceOutputs]:
        """Use the method for determining the outputs for trace.

        Args
        ----

        Returns
        -------
            list[tuple[str, type]]
                List of outputs for the stage
        """
        return TraceOutputs

    @staticmethod
    def _run(inputs: TraceInputs) -> TraceOutputs:
        """Use the method for running the trace stage as well as surrounding buildIS.

        Args
        ----
            options (TraceOptions): Configuration options for TRACE and its subroutines

        Returns
        -------
            tuple[Footprint, list[Footprint], list[Footprint], Footprint, pd.DataFrame]
                The results of the trace stage
        """
        logger.info(
            "[TRACE] ========================================================"
            "================",
        )
        logger.info("[TRACE] -> Calling TRACE to perform the footprint analysis.")
        logger.info(
            "[TRACE] ========================================================"
            "================",
        )

        if TraceStage._uses_trace3(inputs.z, inputs.trace_options):
            logger.info(
                "[TRACE]   -> TRACE3 will use true performance labels and "
                "experimental portfolio selections.",
            )
            experimental_selection = TraceStage._experimental_portfolio_indices(
                inputs.p,
                n_instances=inputs.z.shape[0],
                n_algorithms=inputs.y_bin.shape[1],
            )
            return TraceStage.trace(
                inputs.z,
                inputs.y_bin,
                experimental_selection,
                inputs.beta,
                inputs.algo_labels,
                inputs.trace_options,
                inputs.parallel_options,
                inputs.general_options,
                inputs.executor,
                y_hat=inputs.y_hat,
                pythia_skipped=inputs.pythia_options.skip,
            )

        if inputs.trace_options.use_sim:
            logger.info(
                "[TRACE]   -> TRACE will use PYTHIA's results to calculate the"
                " footprints.",
            )
            return TraceStage.trace(
                inputs.z,
                inputs.y_hat,
                inputs.selection0,
                inputs.beta,
                inputs.algo_labels,
                inputs.trace_options,
                inputs.parallel_options,
                inputs.general_options,
                inputs.executor,
            )
        logger.info(
            "[TRACE]   -> TRACE will use experimental data to calculate the"
            " footprints.",
        )
        experimental_selection = TraceStage._experimental_portfolio_indices(
            inputs.p,
            n_instances=inputs.z.shape[0],
            n_algorithms=inputs.y_bin.shape[1],
        )
        return TraceStage.trace(
            inputs.z,
            inputs.y_bin,
            experimental_selection,
            inputs.beta,
            inputs.algo_labels,
            inputs.trace_options,
            inputs.parallel_options,
            inputs.general_options,
            inputs.executor,
        )

    @staticmethod
    def _experimental_portfolio_indices(
        p: NDArray[np.int_],
        *,
        n_instances: int,
        n_algorithms: int,
    ) -> NDArray[np.int_]:
        """Validate PRELIM's one-based portfolio and convert it for TRACE.

        ``Data.p`` deliberately preserves MATLAB's ``1..n_algorithms`` indexing.
        TRACE's common implementation uses Python's ``0..n_algorithms-1`` indexing,
        as does PYTHIA's ``selection0`` (with ``-1`` for no selection).  This method
        is the sole conversion boundary for experimental TRACE input.
        """
        portfolio = np.asarray(p)
        if portfolio.ndim != 1 or portfolio.shape[0] != n_instances:
            msg = (
                "Experimental portfolio p must be a one-dimensional array with "
                "one entry per instance."
            )
            raise ValueError(msg)
        if not (
            np.issubdtype(portfolio.dtype, np.integer)
            or np.issubdtype(portfolio.dtype, np.floating)
        ):
            msg = "Experimental portfolio p must contain numeric algorithm indices."
            raise ValueError(msg)
        if not np.all(np.isfinite(portfolio)) or not np.all(
            portfolio == np.floor(portfolio),
        ):
            msg = "Experimental portfolio p must contain finite integer indices."
            raise ValueError(msg)
        if np.any(portfolio < 1) or np.any(portfolio > n_algorithms):
            msg = (
                "Experimental portfolio p must use one-based algorithm indices in "
                f"the range 1..{n_algorithms}."
            )
            raise ValueError(msg)
        return portfolio.astype(np.int_, copy=False) - 1

    @staticmethod
    def trace(
        z: NDArray[np.double],
        y_bin: NDArray[np.bool_],
        p: NDArray[np.int_],
        beta: NDArray[np.bool_],
        algo_labels: list[str],
        trace_opts: TraceOptions,
        parallel_opts: ParallelOptions,
        general_opts: GeneralOptions,
        executor: ThreadPoolExecutor | None = None,
        *,
        y_hat: NDArray[np.bool_] | None = None,
        pythia_skipped: bool = False,
    ) -> TraceOutputs:
        """Perform the TRACE footprint analysis.

        Parameters:
        ----------
        z : NDArray[np.double]
            The space of instances.
        y_bin : NDArray[np.bool_]
            Binary indicators of performance.
        p : NDArray[np.int_]
            Performance metrics for algorithms.
        beta : NDArray[np.bool_]
            Specific beta threshold for footprint calculation.
        algo_labels : list[str]
            Labels for each algorithm.
        trace_opts : TraceOptions
            Configuration options for TRACE and its subroutines.
        parallel_opts : ParallelOptions
            Configuration options for parallel processing in Matilda.
        general_opts : GeneralOptions
            General options (e.g. verbosity), not specific to any one stage.
        executor : ThreadPoolExecutor | None
            A caller-owned pool to reuse instead of creating a fresh one.
        y_hat : NDArray[np.bool_] | None
            Optional PYTHIA predictions used only by TRACE3.
        pythia_skipped : bool
            Whether PYTHIA intentionally returned placeholder predictions.

        Returns:
        -------
        TraceDataChanged:
            Should be Empty
        TraceOut:
            An instance of TraceOut containing the analysis results, including
            the calculated footprints and summary statistics.
        """
        trace = TraceStage(
            z,
            y_bin,
            p,
            beta,
            algo_labels,
            trace_opts,
            parallel_opts,
            general_opts,
            executor,
            y_hat,
            pythia_skipped,
        )
        return trace._trace()  # noqa: SLF001

    def _trace(self) -> TraceOutputs:
        """Perform the TRACE footprint analysis.

        Parameters:
        ----------
        z : NDArray[np.double]
            The space of instances.
        y_bin : NDArray[np.bool_]
            Binary indicators of performance.
        p : NDArray[np.int_]
            Performance metrics for algorithms.
        beta : NDArray[np.bool_]
            Specific beta threshold for footprint calculation.
        algo_labels : list[str]
            Labels for each algorithm.
        opts : TraceOptions
            Configuration options for TRACE and its subroutines.

        Returns:
        -------
        TraceDataChanged:
            Should be Empty
        TraceOut:
            An instance of TraceOut containing the analysis results, including
            the calculated footprints and summary statistics.
        """
        if self.opts.method == "trace3":
            return self._trace3()
        if self._uses_trace3(self.z, self.opts):
            logger.warning(
                "[TRACE] Legacy TRACE is two-dimensional; dispatching the "
                "three-dimensional projection to TRACE3.",
            )
            return self._trace3()
        if self.opts.method != "legacy":
            msg = f"Unsupported TRACE method: {self.opts.method!r}."
            raise ValueError(msg)
        return self._trace_legacy()

    def _trace_legacy(self) -> TraceOutputs:
        """Run the historical DBSCAN-based TRACE implementation."""
        # Create a boolean array to calculate the space footprint

        true_array: NDArray[np.bool_] = np.array(
            [True for _ in self.y_bin],
            dtype=np.bool_,
        )

        # Calculate the space footprint (area and density)
        self._log("  -> TRACE is calculating the space area and density.")
        space = self.build(true_array)  # Build the footprint for the entire space
        self._log(f"    -> Space area: {space.area} | Space density: {space.density}")

        # Prepare to calculate footprints for each algorithm's
        # good and best performance
        self._log(
            "------------------------------------------------------------------------",
        )
        self._log("  -> TRACE is calculating the algorithm footprints.")

        # Calculate the good and best performance footprints for all algorithms
        # Determine the number of algorithms being analyzed
        n_algos = self.y_bin.shape[1]
        good, best = self.compute_algorithm_qualities(n_algos)

        # Detect and resolve contradictions between the best performance footprints
        if self.opts.contra:
            self._remove_contradictions(best, n_algos)
        else:
            self._log(
                "  -> TRACE is skipping contradiction removal (trace.contra=False).",
            )

        # Calculate the footprint for the beta threshold,
        # which is a stricter performance threshold
        self._log(
            "------------------------------------------------------------------------",
        )
        self._log("  -> TRACE is calculating the beta-footprint.")
        hard = self.build(
            ~self.beta,
        )  # Build the footprint for instances not meeting the beta threshold

        # Prepare the summary table for all algorithms,
        # which includes various performance metrics
        self._log(
            "------------------------------------------------------------------------",
        )
        self._log("  -> TRACE is preparing the summary table.")

        final_df = self._summary_table(
            good,
            best,
            self.algo_labels,
            space,
            round_values=False,
        )
        # Print the completed summary of the TRACE analysis
        self._log("  -> TRACE has completed. Footprint analysis results:")
        self._log(f"\n{final_df}")

        # Return the results as a TraceOut dataclass instance
        return TraceOutputs(
            space=space,
            good=good,
            best=best,
            hard=hard,
            trace_summary=final_df,
        )

    def _trace3(self) -> TraceOutputs:
        """Run MATLAB's dimension-generic TRACE3 footprint algorithm."""
        self._validate_trace3_inputs()
        space = self._trace3_space_footprint()
        self._trace3_space_area = space.area

        self._log("  -> TRACE3 is calculating the algorithm footprints.")
        if self.y_hat is None or self.pythia_skipped:
            self._log(
                "  -> PYTHIA predictions are unavailable; TRACE3 will use true "
                "labels without a prediction filter.",
            )
        n_algorithms = self.y_bin.shape[1]
        good, best = self.compute_algorithm_qualities(n_algorithms)

        self._log("  -> TRACE3 is calculating the beta-footprint.")
        hard = self._build_trace3(
            ~self.beta,
            None,
            self._trace3_space_area,
        )
        summary = self._summary_table(
            good,
            best,
            self.algo_labels,
            space,
            round_values=True,
        )
        return TraceOutputs(space, good, best, hard, summary)

    def _validate_trace3_inputs(self) -> None:
        """Reject array shapes that cannot satisfy TRACE3's 2D/3D contract."""
        n_instances = self.z.shape[0] if self.z.ndim == TRACE_ARRAY_DIMENSIONS else -1
        n_algorithms = (
            self.y_bin.shape[1] if self.y_bin.ndim == TRACE_ARRAY_DIMENSIONS else -1
        )
        if (
            self.z.ndim != TRACE_ARRAY_DIMENSIONS
            or self.z.shape[1] not in TRACE_COORDINATE_COUNTS
        ):
            msg = "TRACE3 requires a Z matrix with two or three coordinates."
            raise ValueError(msg)
        if self.y_bin.shape != (n_instances, len(self.algo_labels)):
            msg = "TRACE3 Ybin must have one row per instance and column per algorithm."
            raise ValueError(msg)
        if self.p.shape != (n_instances,):
            msg = "TRACE3 portfolio selections must have one entry per instance."
            raise ValueError(msg)
        if self.beta.shape != (n_instances,):
            msg = "TRACE3 beta must have one entry per instance."
            raise ValueError(msg)
        if n_algorithms < 1:
            msg = "TRACE3 requires at least one algorithm."
            raise ValueError(msg)
        if self.y_hat is not None and self.y_hat.shape != self.y_bin.shape:
            msg = "TRACE3 Yhat must have the same shape as Ybin."
            raise ValueError(msg)

    def _trace3_space_footprint(self) -> Footprint:
        """Create MATLAB's convex-hull space metrics without storing geometry."""
        try:
            area = float(ConvexHull(self.z).volume)
        except QhullError:
            area = 0.0
        n_instances = self.z.shape[0]
        density = float(n_instances / area) if area > 0 else 0.0
        return Footprint(
            polygon=None,
            area=area,
            elements=n_instances,
            good_elements=n_instances,
            density=density,
            purity=1.0,
            dimension=self.z.shape[1],
        )

    def _build_trace3(
        self,
        y_bin: NDArray[np.bool_],
        y_hat: NDArray[np.bool_] | None,
        space_area: float,
    ) -> Footprint:
        """Build one TRACE3 footprint from truth and an optional PYTHIA filter."""
        alpha_shape = self._trace3_alpha_shape(y_bin, y_hat)
        if alpha_shape is None:
            return self.throw()

        geometry = alpha_shape.geometry(alpha_shape.critical_radius)
        footprint, valid = self._trace3_metrics(
            geometry,
            y_bin,
            alpha_shape.critical_radius,
        )
        if not valid or footprint.area < self.opts.min_area_frac * space_area:
            return self.throw()

        purity_threshold = self.opts.purity
        if (
            footprint.purity >= purity_threshold
            or alpha_shape.spectrum.size < MIN_ALPHA_SPECTRUM_SIZE
        ):
            return footprint

        previous_region_threshold = 0.0
        radii = np.linspace(
            alpha_shape.critical_radius,
            float(np.min(alpha_shape.spectrum)),
            101,
            dtype=np.double,
        )[1:]
        for radius in radii:
            before_threshold_update = alpha_shape.geometry(
                float(radius),
                region_threshold=previous_region_threshold,
            )
            pre_measure = (
                self._geometry_measure(before_threshold_update)
                if before_threshold_update is not None
                else 0.0
            )
            previous_region_threshold = pre_measure / 20.0
            geometry = alpha_shape.geometry(
                float(radius),
                region_threshold=previous_region_threshold,
            )
            footprint, valid = self._trace3_metrics(geometry, y_bin, float(radius))
            if not valid or footprint.area < self.opts.min_area_frac * space_area:
                return self.throw()
            if footprint.purity >= purity_threshold:
                return footprint

        return footprint

    def _trace3_alpha_shape(
        self,
        y_bin: NDArray[np.bool_],
        y_hat: NDArray[np.bool_] | None,
    ) -> AlphaShape2D | AlphaShape3D | None:
        """Create reusable alpha data when enough unique support exists."""
        support = y_bin if y_hat is None else np.logical_and(y_bin, y_hat)
        supporting_points = np.unique(self.z[support], axis=0)
        if supporting_points.shape[0] <= self.opts.min_instances:
            return None
        if self.z.shape[1] == TRACE_2D_COORDINATES:
            return AlphaShape2D.from_points(supporting_points)
        return AlphaShape3D.from_points(supporting_points)

    def _trace3_metrics(
        self,
        polygon: TraceGeometry | None,
        y_bin: NDArray[np.bool_],
        alpha_radius: float,
    ) -> tuple[Footprint, bool]:
        """Calculate TRACE3 metrics and report whether the geometry is usable."""
        if polygon is None or polygon.is_empty or not np.isfinite(alpha_radius):
            return Footprint(None, 0, 0, 0, 0, 0, self.z.shape[1]), False
        footprint = Footprint.from_polygon(polygon, self.z, y_bin)
        valid = (
            footprint.polygon is not None
            and np.isfinite(footprint.area)
            and footprint.area > 0
            and footprint.elements > 0
        )
        return footprint, bool(valid)

    @staticmethod
    def _geometry_measure(geometry: TraceGeometry) -> float:
        """Return area for polygons or volume for retained tetrahedral meshes."""
        return (
            float(geometry.volume)
            if isinstance(geometry, TetrahedralMesh)
            else float(geometry.area)
        )

    @staticmethod
    def _uses_trace3(z: NDArray[np.double], options: TraceOptions) -> bool:
        """Return whether the configured method must use TRACE3 geometry."""
        return bool(
            options.method == "trace3"
            or (
                options.method == "legacy"
                and z.ndim == TRACE_ARRAY_DIMENSIONS
                and z.shape[1] == TRACE_3D_COORDINATES
            ),
        )

    @staticmethod
    def _summary_table(
        good: list[Footprint],
        best: list[Footprint],
        algo_labels: list[str],
        space: Footprint,
        *,
        round_values: bool,
    ) -> pd.DataFrame:
        """Build the stable Python TRACE summary schema."""
        if len(good) != len(best) or len(good) != len(algo_labels):
            msg = (
                "TRACE summary requires matching good, best, and algorithm-label "
                "counts."
            )
            raise ValueError(msg)
        measure_label = (
            "Volume"
            if TraceStage._footprint_dimension(space) == TRACE_3D_COORDINATES
            else "Area"
        )
        columns = [
            f"{measure_label}_Good",
            f"{measure_label}_Good_Normalised",
            "Density_Good",
            "Density_Good_Normalised",
            "Purity_Good",
            f"{measure_label}_Best",
            f"{measure_label}_Best_Normalised",
            "Density_Best",
            "Density_Best_Normalised",
            "Purity_Best",
        ]
        rows: list[list[float]] = []
        for good_footprint, best_footprint in zip(good, best, strict=True):
            row = TraceStage.summary(good_footprint, space.area, space.density)
            row.extend(TraceStage.summary(best_footprint, space.area, space.density))
            rows.append(row)
        numeric = pd.DataFrame(rows, columns=columns)
        if round_values:
            numeric = numeric.round(3)
        return pd.concat(
            [pd.DataFrame(algo_labels, columns=["Algorithm"]), numeric],
            axis=1,
        )

    @staticmethod
    def rescore(
        trained: TraceOut,
        z: NDArray[np.double],
        y_bin: NDArray[np.bool_],
        p: NDArray[np.int_],
        beta: NDArray[np.bool_],
        algo_labels: list[str],
    ) -> TraceOut:
        """Re-evaluate trained geometry against new truth without rebuilding it."""
        n_instances = z.shape[0] if z.ndim == TRACE_ARRAY_DIMENSIONS else -1
        n_algorithms = y_bin.shape[1] if y_bin.ndim == TRACE_ARRAY_DIMENSIONS else -1
        if (
            z.ndim != TRACE_ARRAY_DIMENSIONS
            or z.shape[1] not in TRACE_COORDINATE_COUNTS
        ):
            msg = "TRACE rescore requires a Z matrix with two or three coordinates."
            raise ValueError(msg)
        trained_dimension = TraceStage._trace_dimension(trained)
        if z.shape[1] != trained_dimension:
            msg = (
                "TRACE rescore coordinate mismatch: trained geometry has "
                f"{trained_dimension} dimensions but explored Z has {z.shape[1]}."
            )
            raise ValueError(msg)
        if y_bin.shape != (n_instances, len(algo_labels)):
            msg = "TRACE rescore Ybin must match instances and algorithm labels."
            raise ValueError(msg)
        portfolio = TraceStage._experimental_portfolio_indices(
            p,
            n_instances=n_instances,
            n_algorithms=n_algorithms,
        )
        if beta.shape != (n_instances,):
            msg = "TRACE rescore beta must have one entry per instance."
            raise ValueError(msg)

        good = [
            (
                TraceStage._rescore_footprint(trained.good[i], z, y_bin[:, i])
                if i < len(trained.good)
                else Footprint(None, 0, 0, 0, 0, 0, trained_dimension)
            )
            for i in range(n_algorithms)
        ]
        best = [
            (
                TraceStage._rescore_footprint(trained.best[i], z, portfolio == i)
                if i < len(trained.best)
                else Footprint(None, 0, 0, 0, 0, 0, trained_dimension)
            )
            for i in range(n_algorithms)
        ]
        hard = TraceStage._rescore_footprint(trained.hard, z, ~beta)
        summary = TraceStage._summary_table(
            good,
            best,
            algo_labels,
            trained.space,
            round_values=True,
        )
        return TraceOut(trained.space, good, best, hard, summary)

    @staticmethod
    def _rescore_footprint(
        trained: Footprint,
        z: NDArray[np.double],
        y_bin: NDArray[np.bool_],
    ) -> Footprint:
        """Update only evidence metrics for one trained footprint."""
        polygon = trained.polygon
        if polygon is None or polygon.is_empty:
            return Footprint(
                polygon,
                trained.area,
                0,
                0,
                0,
                0,
                TraceStage._footprint_dimension(trained),
            )
        inside = pointwise_covers(polygon, z)
        elements = int(np.sum(inside))
        good_elements = int(np.sum(np.logical_and(inside, y_bin)))
        if elements == 0 or trained.area == 0:
            density = 0.0
            purity = 0.0
        else:
            density = float(elements / trained.area)
            purity = float(good_elements / elements)
        return Footprint(
            polygon=polygon,
            area=trained.area,
            elements=elements,
            good_elements=good_elements,
            density=density,
            purity=purity,
            dimension=TraceStage._footprint_dimension(trained),
        )

    @staticmethod
    def _footprint_dimension(footprint: Footprint) -> int:
        """Infer dimensionality while remaining compatible with old joblib data."""
        geometry = footprint.polygon
        if isinstance(geometry, TetrahedralMesh):
            return TRACE_3D_COORDINATES
        return int(getattr(footprint, "dimension", TRACE_2D_COORDINATES))

    @staticmethod
    def _trace_dimension(trained: TraceOut) -> int:
        """Find a trained TRACE model's geometry coordinate count."""
        space_dimension = TraceStage._footprint_dimension(trained.space)
        if space_dimension == TRACE_3D_COORDINATES:
            return space_dimension
        for footprint in [*trained.good, *trained.best, trained.hard]:
            if isinstance(footprint.polygon, TetrahedralMesh):
                return TRACE_3D_COORDINATES
        return TRACE_2D_COORDINATES

    def _remove_contradictions(
        self,
        best: list[Footprint],
        n_algos: int,
    ) -> None:
        """Detect and resolve contradictions between all pairs of best footprints.

        Mutates `best` in place. Split out of `_trace()` so the `contra`
        option (F11 - matches MATLAB legacy TRACE's `contra`, default `True`)
        can skip this step entirely rather than branching around it inline.
        """
        self._log(
            "------------------------------------------------------------------------",
        )
        self._log(
            "  -> TRACE is detecting and removing contradictory"
            " sections of the footprints.",
        )
        for i in range(n_algos):
            self._log(f"  -> Base algorithm '{self.algo_labels[i]}'")
            start_base = (
                time.time()
            )  # Track the start time for processing this base algorithm

            algo_1: NDArray[np.bool_] = np.array(
                [int(v) == i for v in self.p],
                dtype=np.bool_,
            )

            for j in range(i + 1, n_algos):
                self._log_detail(
                    f"      -> TRACE is comparing '"
                    f"{self.algo_labels[i]}' with '{self.algo_labels[j]}'",
                )
                start_test = time.time()  # Track the start time for the comparison

                # Create boolean arrays indicating which points correspond
                # to each algorithm's best performance

                algo_2: NDArray[np.bool_] = np.array(
                    [int(v) == j for v in self.p],
                    dtype=np.bool_,
                )

                # Resolve contradictions between the compared algorithms'  footprints
                best[i], best[j] = self.contra(best[i], best[j], algo_1, algo_2)

                # Print the elapsed time for the comparison
                elapsed_test = time.time() - start_test
                self._log_detail(
                    f"      -> Test algorithm '{self.algo_labels[j]}' completed. "
                    f"Elapsed time: {elapsed_test:.2f}s",
                )

            # Print the elapsed time for processing this base algorithm
            elapsed_base = time.time() - start_base
            self._log(
                f"  -> Base algorithm '{self.algo_labels[i]}' completed. Elapsed time:"
                f" {elapsed_base:.2f}s",
            )

    def build(self, y_bin: NDArray[np.bool_]) -> Footprint:
        """Construct a footprint polygon using DBSCAN clustering.

        Parameters:
        ----------
        y_bin : NDArray[np.bool_]
            Binary indicator vector indicating which data points are of interest.

        Returns:
        -------
        Footprint:
            The constructed footprint with calculated area, density, and purity.
        """
        # Extract rows where y_bin is True
        filtered_z = self.z[y_bin]

        # Find unique rows
        unique_rows = np.unique(filtered_z, axis=0)

        # Check the number of unique rows
        if unique_rows.shape[0] < POLYGON_MIN_POINT_REQUIREMENT:
            return self.throw()

        labels = self.run_dbscan(y_bin, unique_rows)
        flag = False
        polygon_body: Polygon | MultiPolygon = Polygon()
        for i in range(1, int(np.max(labels)) + 1):
            polydata = unique_rows[labels == i]

            aux = self.fit_poly(polydata, y_bin)
            if aux is not None and not aux.is_empty:
                if not flag:
                    polygon_body = aux
                    flag = True
                else:
                    polygon_body = polygon_body.union(aux)

        if not flag or polygon_body.is_empty:
            return self.throw()

        return Footprint.from_polygon(
            polygon=polygon_body,
            z=self.z,
            y_bin=y_bin,
            smoothen=True,
        )

    def contra(
        self,
        base: Footprint,
        test: Footprint,
        y_base: NDArray[np.bool_],
        y_test: NDArray[np.bool_],
    ) -> tuple[Footprint, Footprint]:
        """Detect and resolve contradictions between two footprint polygons.

        Parameters:
        ----------
        base : Footprint
            The base footprint polygon.
        test : Footprint
            The test footprint polygon.
        y_base : NDArray[np.bool_]
            Binary array indicating the points corresponding to the base footprint.
        y_test : NDArray[np.bool_]
            Binary array indicating the points corresponding to the test footprint.

        Returns:
        -------
        tuple:
            Updated base and test footprints after resolving contradictions.
        """
        if base.polygon is None or test.polygon is None:
            return base, test
        if not isinstance(base.polygon, Polygon | MultiPolygon) or not isinstance(
            test.polygon,
            Polygon | MultiPolygon,
        ):
            msg = "Legacy TRACE contradiction removal requires 2D polygons."
            raise ValueError(msg)

        base_polygon = base.polygon
        test_polygon = test.polygon

        max_tries = 3
        num_tries = 1
        contradiction = base_polygon.intersection(test_polygon)

        while not contradiction.is_empty and num_tries <= max_tries:
            num_elements = np.sum(pointwise_covers(contradiction, self.z))
            if num_elements == 0:
                self._log_detail(
                    "        -> The contradicting area contains no instances; "
                    "leaving both footprints unchanged.",
                )
                break

            num_good_elements_base = np.sum(
                pointwise_covers(contradiction, self.z[y_base]),
            )
            num_good_elements_test = np.sum(
                pointwise_covers(contradiction, self.z[y_test]),
            )

            purity_base = num_good_elements_base / num_elements
            purity_test = num_good_elements_test / num_elements

            if purity_base > purity_test:
                c_area = contradiction.area / test_polygon.area
                self._log_detail(
                    f"        -> {round(100 * c_area, 1)}% of the test footprint "
                    "is contradictory.",
                )
                test_polygon = test_polygon.difference(contradiction)
                if num_tries < max_tries:
                    test_polygon = self.tight(test_polygon, y_test)
            elif purity_test > purity_base:
                c_area = contradiction.area / base_polygon.area
                self._log_detail(
                    f"        -> {round(100 * c_area, 1)}% of the base footprint "
                    "is contradictory.",
                )
                base_polygon = base_polygon.difference(contradiction)
                if num_tries < max_tries:
                    base_polygon = self.tight(base_polygon, y_base)
            else:
                self._log_detail(
                    "        -> Purity of the contradicting areas is equal for both "
                    "footprints.",
                )
                self._log_detail("        -> Ignoring the contradicting area.")
                break

            if base_polygon.is_empty or test_polygon.is_empty:
                break

            contradiction = base_polygon.intersection(test_polygon)

            num_tries += 1

        base = Footprint.from_polygon(polygon=base_polygon, z=self.z, y_bin=y_base)
        test = Footprint.from_polygon(polygon=test_polygon, z=self.z, y_bin=y_test)

        return base, test

    def tight(
        self,
        polygon: Polygon | MultiPolygon,
        y_bin: NDArray[np.bool_],
    ) -> Polygon | MultiPolygon:
        """Refine an existing polygon by removing slivers and improving its shape.

        Parameters:
        ----------
        polygon : Polygon | MultiPolygon
            The polygon or multipolygon to be refined.
        y_bin : NDArray[np.bool_]
            Binary array indicating which data points belong to the polygon.

        Returns:
        -------
        Polygon | MultiPolygon:
            The refined polygon, or an empty polygon if refinement fails.
        """
        splits = (
            [item for item in polygon.geoms]
            if isinstance(polygon, MultiPolygon)
            else [polygon]
        )
        n_polygons = len(splits)
        refined_polygons: list[Polygon | MultiPolygon] = []

        for i in range(n_polygons):
            criteria = np.logical_and(
                pointwise_covers(splits[i], self.z),
                y_bin,
            )
            polydata = self.z[criteria]

            if polydata.shape[0] < POLYGON_MIN_POINT_REQUIREMENT:
                continue

            aux = self.fit_poly(polydata, y_bin)

            if aux is not None and not aux.is_empty:
                refined_polygons.append(aux)

        if refined_polygons:
            return unary_union(refined_polygons)
        return Polygon()

    def fit_poly(
        self,
        polydata: NDArray[np.double],
        y_bin: NDArray[np.bool_],
    ) -> Polygon | MultiPolygon | None:
        """Fit a polygon to the given data points, following the purity constraints.

        Parameters:
        ----------
        polydata : NDArray[np.double]
            The data points to fit the polygon to.
        y_bin : NDArray[np.bool_]
            Binary array indicating which data points should be considered
            for the polygon.

        Returns:
        -------
        Polygon | MultiPolygon | None:
            The fitted polygon, or None if the fitting fails.
        """
        if polydata.shape[0] < POLYGON_MIN_POINT_REQUIREMENT:
            return None

        polygon = legacy_alpha_shape(polydata, 2.15).simplify(0.05)

        if not np.all(y_bin):
            if polygon.is_empty:
                return None
            tri = triangulate(polygon)
            for piece in tri:
                elements = np.sum(pointwise_covers(piece.convex_hull, self.z))
                good_elements = np.sum(
                    pointwise_covers(piece.convex_hull, self.z[y_bin]),
                )
                if elements == 0 or (good_elements / elements) < self.opts.purity:
                    polygon = polygon.difference(piece)

        return polygon

    @staticmethod
    def summary(
        footprint: Footprint,
        space_area: float,
        space_density: float,
    ) -> list[float]:
        """Summarize the footprint metrics.

        Parameters:
        ----------
        footprint : Footprint
            The footprint to summarize.
        space_area : float
            The total area of the space being analyzed.
        space_density : float
            The density of the entire space.

        Returns:
        -------
        list:
            A list containing summarized metrics such as area, normalized area,
            density, normalized density, and purity.
        """
        area = footprint.area if footprint.area is not None else 0
        normalised_area = (
            float(area / space_area)
            if ((space_area is not None) and (space_area != 0))
            else float(area)
        )
        density = footprint.density if footprint.density is not None else 0
        normalised_density = (
            float(density / space_density)
            if ((space_density is not None) and (space_density != 0))
            else float(footprint.density)
        )
        purity = float(footprint.purity)

        out = [area, normalised_area, density, normalised_density, purity]
        return [
            element if ((element is not None) and (not np.isnan(element))) else 0
            for element in out
        ]

    def throw(self) -> Footprint:
        """Generate a footprint with default values, indicating insufficient data.

        Returns:
        -------
        Footprint:
            An instance of Footprint with default values.
        """
        self._log_detail(
            "        -> There are not enough instances to calculate a footprint.",
        )
        self._log_detail("        -> The subset of instances used is too small.")
        z = getattr(self, "z", None)
        dimension = (
            z.shape[1]
            if isinstance(z, np.ndarray)
            and z.ndim == TRACE_ARRAY_DIMENSIONS
            and z.shape[1] in TRACE_COORDINATE_COUNTS
            else TRACE_2D_COORDINATES
        )
        return Footprint(None, 0, 0, 0, 0, 0, dimension)

    @staticmethod
    def run_dbscan(
        y_bin: NDArray[np.bool_],
        data: NDArray[np.double],
    ) -> NDArray[np.int_]:
        """Perform DBSCAN clustering on the dataset.

        Parameters:
        ----------
        y_bin : NDArray[np.bool_]
            Binary indicator vector to filter the data points.
        data : NDArray[np.double]
            The dataset to cluster.

        Returns:
        -------
        NDArray[np.int_]:
            Array of cluster labels for each data point.
        """
        nn = int(max(min(np.ceil(np.sum(y_bin) / 20), 50), 3))
        # Compute Eps
        eps = TraceStage.epsilon(data, nn)
        return TraceStage.dbscan(data, nn, eps)

    @staticmethod
    def epsilon(x: NDArray[np.double], k: int) -> float:
        """Analytical way of estimating neighborhood radius for DBSCAN.

        Parameters:
        ----------
        x: NDArray[np.double]
            data matrix (m, n); m-objects, n-variables
        k: int
            number of objects in a neighborhood of an object
            (minimal number of objects considered as a cluster)

        Returns:
        -------
        Eps: float
            Estimated neighborhood radius
        """
        m, n = x.shape
        ranges = np.max(x, axis=0) - np.min(x, axis=0)
        numerator = np.prod(ranges) * k * gamma(0.5 * n + 1)
        denominator = m * np.sqrt(np.pi**n)
        return float((numerator / denominator) ** (1.0 / n))

    @staticmethod
    def dist(
        i: NDArray[np.double],
        x: NDArray[np.double],
    ) -> NDArray[np.double]:
        """Calculate the Euclidean distances between objects.

        Parameters:
        ----------
        i: NDArray[np.double]
            an object (1, n)
        x: NDArray[np.double]
            data matrix (m, n); m-objects, n-variables

        Returns:
        -------
        D: float
            Euclidean distance (m,)
        """
        _, n = x.shape

        if n == 1:
            return np.asarray(np.abs(x[:, 0] - i[0]), dtype=np.double)
        return np.asarray(np.sqrt(np.sum((x - i) ** 2, axis=1)), dtype=np.double)

    @staticmethod
    def dbscan(x: NDArray[np.double], k: int, eps: float) -> NDArray[np.int_]:
        """Density-Based Spatial Clustering of Applications with Noise (DBSCAN).

        Parameters:
        ----------
        x: NDArray[np.double]
           data matrix (m, n); m-objects, n-variables
        k: int
            minimum number of points to form a cluster
        eps: float
            neighborhood radius; if None, it will be estimated using the epsilon
            function

        Returns:
        -------
        class_: NDArray[np.int_]
            Cluster assignments for each point (-1 for noise)
        """
        m, n = x.shape
        if eps is None:
            eps = TraceStage.epsilon(x, k)
        # Augment x with indices
        x_with_index = np.hstack((np.arange(m).reshape(m, 1), x))
        type_ = np.zeros(m, dtype=np.int_)  # 1: core, 0: border, -1: noise
        no = 1  # Cluster label
        touched = np.zeros(m, dtype=np.bool_)
        classes = np.zeros(m, dtype=np.int_)  # Cluster assignment
        for i in range(m):
            if touched[i] == 0:
                ob = x_with_index[i, :]
                d = TraceStage.dist(ob[1:], x_with_index[:, 1:])
                ind = np.where(d <= eps)[0]
                if 1 < len(ind) < k + 1:
                    type_[i] = 0  # Border point
                    classes[i] = 0
                if len(ind) == 1:
                    type_[i] = -1  # Noise point
                    classes[i] = -1
                    touched[i] = 1
                if len(ind) >= k + 1:
                    type_[i] = 1  # Core point
                    classes[ind] = no
                    ind_list = list(ind)
                    while len(ind_list) > 0:
                        current_index = ind_list[0]
                        ob = x_with_index[current_index, :]
                        touched[current_index] = 1
                        ind_list.pop(0)
                        d = TraceStage.dist(ob[1:], x_with_index[:, 1:])
                        i1 = np.where(d <= eps)[0]
                        if len(i1) > 1:
                            classes[i1] = no
                            if len(i1) >= k + 1:
                                type_[int(ob[0])] = 1
                            else:
                                type_[int(ob[0])] = 0
                            for j in i1:
                                if touched[j] == 0:
                                    touched[j] = 1
                                    ind_list.append(j)
                                    classes[j] = no
                    no += 1
        i1 = np.where(classes == 0)[0]
        classes[i1] = -1
        type_[i1] = -1
        return classes

    def process_algorithm(self, i: int) -> tuple[int, Footprint, Footprint]:
        """Process an algorithm to calculate its good and best performance footprints.

        Parameters:
        ----------
        i : int
            Index of the algorithm to process.

        Returns:
        -------
        tuple[int, Footprint, Footprint]:
            The index of the algorithm, and its good and best performance footprints.
        """
        if self._uses_trace3(self.z, self.opts):
            return self._process_algorithm_trace3(i)

        start_time = time.time()
        self._log(f"    -> Good performance footprint for '{self.algo_labels[i]}'")
        good_performance = self.build(self.y_bin[:, i])

        self._log(f"    -> Best performance footprint for '{self.algo_labels[i]}'")
        bool_array: NDArray[np.bool_] = np.array(
            [int(v) == i for v in self.p],
            dtype=np.bool_,
        )
        best_performance = self.build(bool_array)

        elapsed_time = time.time() - start_time
        self._log(
            f"    -> Algorithm '{self.algo_labels[i]}' completed. "
            f"Elapsed time: {elapsed_time:.2f}s",
        )

        return i, good_performance, best_performance

    def _process_algorithm_trace3(self, i: int) -> tuple[int, Footprint, Footprint]:
        """Build one algorithm's TRACE3 good and best footprints."""
        prediction = (
            None if self.y_hat is None or self.pythia_skipped else self.y_hat[:, i]
        )
        good = self._build_trace3(
            self.y_bin[:, i],
            prediction,
            self._trace3_space_area,
        )
        best = self._build_trace3(
            self.p == i,
            prediction,
            self._trace3_space_area,
        )
        return i, good, best

    def compute_algorithm_qualities(
        self,
        n_algos: int,
    ) -> tuple[list[Footprint], list[Footprint]]:
        """Perform parallel processing to calculate footprints for multiple algorithms.

        Parameters:
        ----------
        n_workers : int
            Number of worker threads to use.
        n_algos : int
            Number of algorithms to process.

        Returns:
        -------
        tuple[list[Footprint], list[Footprint]]:
            Lists of good and best performance footprints for each algorithm.
        """
        z = getattr(self, "z", None)
        dimension = (
            z.shape[1]
            if isinstance(z, np.ndarray)
            and z.ndim == TRACE_ARRAY_DIMENSIONS
            and z.shape[1] in TRACE_COORDINATE_COUNTS
            else TRACE_2D_COORDINATES
        )
        good: list[Footprint] = [
            Footprint(None, 0, 0, 0, 0, 0, dimension) for _ in range(n_algos)
        ]
        best: list[Footprint] = [
            Footprint(None, 0, 0, 0, 0, 0, dimension) for _ in range(n_algos)
        ]

        if not self.parallel_opts.flag:
            for i in range(n_algos):
                _, good[i], best[i] = self.process_algorithm(i)
        elif self.executor is not None:
            self._submit_algorithm_futures(self.executor, n_algos, good, best)
        else:
            worker_count = min(self.parallel_opts.n_cores, multiprocessing.cpu_count())
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                self._submit_algorithm_futures(executor, n_algos, good, best)

        return good, best

    def _submit_algorithm_futures(
        self,
        executor: ThreadPoolExecutor,
        n_algos: int,
        good: list[Footprint],
        best: list[Footprint],
    ) -> None:
        """Submit each algorithm's footprint computation to `executor` and gather it.

        `good`/`best` are filled in place, indexed by algorithm.
        """
        futures = [executor.submit(self.process_algorithm, i) for i in range(n_algos)]
        for future in as_completed(futures):
            i: int
            good_performance: Footprint
            best_performance: Footprint
            i, good_performance, best_performance = future.result()
            good[i] = good_performance
            best[i] = best_performance
