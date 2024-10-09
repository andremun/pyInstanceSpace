"""TRACE Stage Module for Performance-Based Footprint Estimation.

This module implements the TRACE stage, which analyzes the performance of multiple
algorithms by generating geometric footprints. These footprints represent the areas
of good, best, and beta performance based on the clustering of instance data. The
footprints are further evaluated for their density and purity in relation to the
performance metrics of the algorithms.

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
- alphashape
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

import math
import multiprocessing
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import NamedTuple

import alphashape
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.special import gamma
from shapely.geometry import MultiPoint, MultiPolygon, Polygon
from shapely.ops import triangulate, unary_union
from sklearn.cluster import DBSCAN

from matilda.data.model import Footprint
from matilda.data.options import ParallelOptions, TraceOptions
from matilda.stages.stage import Stage

POLYGON_MIN_POINT_REQUIREMENT = 3


class TraceInputs(NamedTuple):
    """A named tuple to encapsulate the inputs required for the TRACE analysis.

    Attributes:
    ----------
    z : NDArray[np.double]
        The space of instances, represented as an array of data points (features).
    selection0 : NDArray[np.int_]
        Performance metrics from the Pythia algorithm, represented as an array.
    p : NDArray[np.int_]
        Performance metrics from the data source, represented as an array of values.
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
    summary: pd.DataFrame


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
        opts : TraceOptions
            Configuration options for TRACE and its subroutines.
        """
        self.z = z
        self.y_bin = y_bin
        self.p = p
        self.beta = beta
        self.algo_labels = algo_labels
        self.opts = trace_opts
        self.parallel_opts = parallel_opts

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
        print(
            "========================================================================",
        )
        print("-> Calling TRACE to perform the footprint analysis.")
        print(
            "========================================================================",
        )

        if inputs.trace_options.use_sim:
            print("  -> TRACE will use PYTHIA's results to calculate the footprints.")
            return TraceStage.trace(
                inputs.z,
                inputs.y_hat,
                inputs.selection0,
                inputs.beta,
                inputs.algo_labels,
                inputs.trace_options,
                inputs.parallel_options,
            )
        print("  -> TRACE will use experimental data to calculate the footprints.")
        return TraceStage.trace(
            inputs.z,
            inputs.y_bin,
            inputs.p,
            inputs.beta,
            inputs.algo_labels,
            inputs.trace_options,
            inputs.parallel_options,
        )

    @staticmethod
    def trace(
        z: NDArray[np.double],
        y_bin: NDArray[np.bool_],
        p: NDArray[np.int_],
        beta: NDArray[np.bool_],
        algo_labels: list[str],
        trace_opts: TraceOptions,
        parallel_opts: ParallelOptions,
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
        trace = TraceStage(z, y_bin, p, beta, algo_labels, trace_opts, parallel_opts)
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
        # Create a boolean array to calculate the space footprint

        true_array: NDArray[np.bool_] = np.array(
            [True for _ in self.y_bin],
            dtype=np.bool_,
        )

        # Calculate the space footprint (area and density)
        print("  -> TRACE is calculating the space area and density.")
        space = self.build(true_array)  # Build the footprint for the entire space
        print(f"    -> Space area: {space.area} | Space density: {space.density}")

        # Prepare to calculate footprints for each algorithm's
        # good and best performance
        print(
            "------------------------------------------------------------------------",
        )
        print("  -> TRACE is calculating the algorithm footprints.")

        # Calculate the good and best performance footprints for all algorithms
        # Determine the number of algorithms being analyzed
        n_algos = self.y_bin.shape[1]
        good, best = self.compute_algorithm_qualities(n_algos)

        # Detect and resolve contradictions between the best performance footprints
        print(
            "------------------------------------------------------------------------",
        )
        print(
            "  -> TRACE is detecting and removing contradictory"
            " sections of the footprints.",
        )
        for i in range(n_algos):
            print(f"  -> Base algorithm '{self.algo_labels[i]}'")
            start_base = (
                time.time()
            )  # Track the start time for processing this base algorithm

            algo_1: NDArray[np.bool_] = np.array(
                [int(v) == i for v in self.p],
                dtype=np.bool_,
            )

            for j in range(i + 1, n_algos):
                print(
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
                print(
                    f"      -> Test algorithm '{self.algo_labels[j]}' completed. "
                    f"Elapsed time: {elapsed_test:.2f}s",
                )

            # Print the elapsed time for processing this base algorithm
            elapsed_base = time.time() - start_base
            print(
                f"  -> Base algorithm '{self.algo_labels[i]}' completed. Elapsed time:"
                f" {elapsed_base:.2f}s",
            )

        # Calculate the footprint for the beta threshold,
        # which is a stricter performance threshold
        print(
            "------------------------------------------------------------------------",
        )
        print("  -> TRACE is calculating the beta-footprint.")
        hard = self.build(
            ~self.beta,
        )  # Build the footprint for instances not meeting the beta threshold

        # Prepare the summary table for all algorithms,
        # which includes various performance metrics
        print(
            "------------------------------------------------------------------------",
        )
        print("  -> TRACE is preparing the summary table.")

        # Create a pandas DataFrame and name the column "Algorithms"
        algorithm_names_df = pd.DataFrame(self.algo_labels, columns=["Algorithm"])

        data_labels = [
            "Area_Good",
            "Area_Good_Normalised",
            "Density_Good",
            "Density_Good_Normalised",
            "Purity_Good",
            "Area_Best",
            "Area_Best_Normalised",
            "Density_Best",
            "Density_Best_Normalised",
            "Purity_Best",
        ]

        # Populate the summary table with metrics for each algorithm's
        # good and best footprints
        summary_data = []

        for i, _ in enumerate(self.algo_labels):
            summary_row = self.summary(good[i], space.area, space.density)
            # Add good performance metrics
            summary_row.extend(
                self.summary(
                    best[i],
                    space.area,
                    space.density,
                ),
            )  # Add the best performance metrics

            summary_data.append(summary_row)

        # Convert the summary data into a pandas DataFrame for better organization
        summary_df = pd.DataFrame(summary_data, columns=data_labels)
        final_df = pd.concat([algorithm_names_df, summary_df], axis=1)
        # Print the completed summary of the TRACE analysis
        print("  -> TRACE has completed. Footprint analysis results:")
        print(" ")
        print(final_df)

        # Return the results as a TraceOut dataclass instance
        return TraceOutputs(
            space=space,
            good=good,
            best=best,
            hard=hard,
            summary=final_df,
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

        labels = self.run_dbscan(y_bin, unique_rows)[0]
        flag = False
        polygon_body: Polygon = Polygon()
        for i in range(0, np.max(labels) + 1):
            polydata = unique_rows[labels == i]

            aux = self.fit_poly(polydata, y_bin)
            if aux:
                if not flag:
                    polygon_body = aux
                    flag = True
                else:
                    polygon_body = polygon_body.union(aux)

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

        base_polygon = base.polygon
        test_polygon = test.polygon

        max_tries = 3
        num_tries = 1
        contradiction = base_polygon.intersection(test_polygon)

        while not contradiction.is_empty and num_tries <= max_tries:
            num_elements = np.sum(
                [contradiction.contains(point) for point in MultiPoint(self.z).geoms],
            )
            num_good_elements_base = np.sum(
                [
                    contradiction.contains(point)
                    for point in MultiPoint(self.z[y_base]).geoms
                ],
            )
            num_good_elements_test = np.sum(
                [
                    contradiction.contains(point)
                    for point in MultiPoint(self.z[y_test]).geoms
                ],
            )

            purity_base = num_good_elements_base / num_elements
            purity_test = num_good_elements_test / num_elements

            if purity_base > purity_test:
                c_area = contradiction.area / test_polygon.area
                print(
                    f"        -> {round(100 * c_area, 1)}% of the test footprint "
                    "is contradictory.",
                )
                test_polygon = test_polygon.difference(contradiction)
                if num_tries < max_tries:
                    test_polygon = self.tight(test_polygon, y_test)
            elif purity_test > purity_base:
                c_area = contradiction.area / base_polygon.area
                print(
                    f"        -> {round(100 * c_area, 1)}% of the base footprint "
                    "is contradictory.",
                )
                base_polygon = base_polygon.difference(contradiction)
                if num_tries < max_tries:
                    base_polygon = self.tight(base_polygon, y_base)
            else:
                print(
                    "        -> Purity of the contradicting areas is equal for both "
                    "footprints.",
                )
                print("        -> Ignoring the contradicting area.")
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
    ) -> Polygon | None:
        """Refine an existing polygon by removing slivers and improving its shape.

        Parameters:
        ----------
        polygon : Polygon | MultiPolygon
            The polygon or multipolygon to be refined.
        y_bin : NDArray[np.bool_]
            Binary array indicating which data points belong to the polygon.

        Returns:
        -------
        Polygon | None:
            The refined polygon, or None if the refinement fails.
        """
        if polygon is None:
            return None

        splits = (
            [item for item in polygon.geoms]
            if isinstance(polygon, MultiPolygon)
            else [polygon]
        )
        n_polygons = len(splits)
        refined_polygons = []

        for i in range(n_polygons):
            criteria = np.logical_and(splits[i].contains(MultiPoint(self.z)), y_bin)
            polydata = self.z[criteria]

            if polydata.shape[0] < POLYGON_MIN_POINT_REQUIREMENT:
                continue

            temp_polygon = Polygon(polydata)

            boundary = temp_polygon.boundary
            filtered_polydata = polydata[boundary]
            aux = self.fit_poly(filtered_polydata, y_bin)

            if aux:
                refined_polygons.append(aux)

        if len(refined_polygons) > 0:
            return unary_union(refined_polygons)
        return None

    def fit_poly(
        self,
        polydata: NDArray[np.double],
        y_bin: NDArray[np.bool_],
    ) -> Polygon | None:
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
        Polygon | None:
            The fitted polygon, or None if the fitting fails.
        """
        if polydata.shape[0] < POLYGON_MIN_POINT_REQUIREMENT:
            return None

        polygon = alphashape.alphashape(polydata, 2.15).simplify(
            0.05,
        )

        if not np.all(y_bin):
            if polygon.is_empty:
                return None
            tri = triangulate(polygon)
            for piece in tri:
                elements = np.sum(
                    [piece.contains(point) for point in MultiPoint(self.z).geoms],
                )
                good_elements = np.sum(
                    [
                        piece.contains(point)
                        for point in MultiPoint(self.z[y_bin]).geoms
                    ],
                )
                if (good_elements / elements) < self.opts.purity:
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
        density = int(footprint.density) if footprint.density is not None else 0
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

    @staticmethod
    def throw() -> Footprint:
        """Generate a footprint with default values, indicating insufficient data.

        Returns:
        -------
        Footprint:
            An instance of Footprint with default values.
        """
        print("        -> There are not enough instances to calculate a footprint.")
        print("        -> The subset of instances used is too small.")
        return Footprint(None, 0, 0, 0, 0, 0)

    @staticmethod
    def run_dbscan(
            y_bin: NDArray[np.bool_],
            data: NDArray[np.double],
    ) -> NDArray[np.int_] | tuple:
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
        nn = max(min(np.ceil(np.sum(y_bin) / 20), 50), 3)
        # Compute Eps
        eps = TraceStage.epsilon(data, nn)
        return TraceStage.dbscan(data, nn, eps)

    @staticmethod
    def epsilon(x: NDArray[np.double], k: int):
        """
        Analytical way of estimating neighborhood radius for DBSCAN.

        Parameters:
        x - data matrix (m, n); m-objects, n-variables
        k - number of objects in a neighborhood of an object
            (minimal number of objects considered as a cluster)

        Returns:
        Eps - Estimated neighborhood radius
        """
        m, n = x.shape
        ranges = np.max(x, axis=0) - np.min(x, axis=0)
        prod_ranges = np.prod(ranges)
        numerator = prod_ranges * k * gamma(0.5 * n + 1)
        denominator = m * np.sqrt(np.pi ** n)
        eps = (numerator / denominator) ** (1.0 / n)
        return eps

    @staticmethod
    def dist(i, x):
        """
        Calculates the Euclidean distances between the i-th object and all objects in x

        Parameters:
        i - an object (1, n)
        x - data matrix (m, n); m-objects, n-variables

        Returns:
        D - Euclidean distance (m,)
        """
        m, n = x.shape
        if n == 1:
            d = np.abs(x - i).flatten()
        else:
            d = np.sqrt(np.sum((x - i) ** 2, axis=1))
        return d

    @staticmethod
    def dbscan(x: NDArray[np.double], k: int, eps: float = None):
        """
        Density-Based Spatial Clustering of Applications with Noise (DBSCAN) algorithm.

        Parameters:
        x   - data matrix (m, n); m-objects, n-variables
        k   - minimum number of points to form a cluster
        Eps - neighborhood radius; if None, it will be estimated using the epsilon
        function

        Returns:
        class_ - Cluster assignments for each point (-1 for noise)
        type   - Type of each point (1: core, 0: border, -1: noise)
        """
        m, n = x.shape
        if eps is None:
            eps = TraceStage.epsilon(x, k)
        # Augment x with indices
        x_with_index = np.hstack((np.arange(m).reshape(m, 1), x))
        type_ = np.zeros(m)  # 1: core, 0: border, -1: noise
        no = 1  # Cluster label
        touched = np.zeros(m)  # 0: not processed, 1: processed
        classes = np.zeros(m)  # Cluster assignment
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
        return classes, type_

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
        start_time = time.time()
        print(f"    -> Good performance footprint for '{self.algo_labels[i]}'")
        good_performance = self.build(self.y_bin[:, i])

        print(f"    -> Best performance footprint for '{self.algo_labels[i]}'")
        bool_array: NDArray[np.bool_] = np.array(
            [int(v) == i for v in self.p],
            dtype=np.bool_,
        )
        best_performance = self.build(bool_array)

        elapsed_time = time.time() - start_time
        print(
            f"    -> Algorithm '{self.algo_labels[i]}' completed. "
            f"Elapsed time: {elapsed_time:.2f}s",
        )

        return i, good_performance, best_performance

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
        # Determine the number of workers available for parallel processing
        good: list[Footprint] = [self.throw() for _ in range(n_algos)]
        best: list[Footprint] = [self.throw() for _ in range(n_algos)]
        worker_count = min(self.parallel_opts.n_cores, multiprocessing.cpu_count())
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = [
                executor.submit(self.process_algorithm, i) for i in range(n_algos)
            ]
            for future in as_completed(futures):
                i: int
                good_performance: Footprint
                best_performance: Footprint
                i, good_performance, best_performance = future.result()
                good[i] = good_performance
                best[i] = best_performance

        return good, best
