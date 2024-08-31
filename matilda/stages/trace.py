import math
import multiprocessing
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from numpy.typing import NDArray
from scipy.special import gamma
from shapely.geometry import MultiPoint, MultiPolygon, Polygon
from shapely.ops import triangulate, unary_union
from sklearn.cluster import DBSCAN, HDBSCAN  # this is here due to clients request. I didn't test it yet. do not delete
import alphashape
from matilda.data.options import TraceOptions
from matilda.data.model import Footprint, TraceOut, TraceDataChanged
import pandas as pd


class Trace:
    """A class to manage the TRACE analysis process.

    Attributes:
    ----------
    z : NDArray[np.double]
        The space of instances.
    y_bin : NDArray[np.bool_]
        Binary indicators of performance.
    p : NDArray[np.double]
        Performance metrics for algorithms.
    beta : NDArray[np.bool_]
        Specific beta threshold for footprint calculation.
    algo_labels : List[str]
        Labels for each algorithm.
    opts : TraceOptions
        Configuration options for TRACE and its subroutines.
    """

    z: NDArray[np.double]
    y_bin: NDArray[np.bool_]
    p: NDArray[np.integer]
    beta: NDArray[np.bool_]
    algo_labels: list[str]
    opts: TraceOptions

    def __init__(
        self,
        z: NDArray[np.double],
        y_bin: NDArray[np.bool_],
        p: NDArray[np.integer],
        beta: NDArray[np.bool_],
        algo_labels: list[str],
        opts: TraceOptions,
    ) -> None:
        """Initialize the Trace analysis.

        Parameters:
        ----------
        z : NDArray[np.double]
            The space of instances.
        y_bin : NDArray[np.bool_]
            Binary indicators of performance.
        p : NDArray[np.double]
            Performance metrics for algorithms.
        beta : NDArray[np.bool_]
            Specific beta threshold for footprint calculation.
        algo_labels : List[str]
            Labels for each algorithm.
        opts : TraceOptions
            Configuration options for TRACE and its subroutines.
        """
        self.z = z
        self.y_bin = y_bin
        self.p = p
        self.beta = beta
        self.algo_labels = algo_labels
        self.opts = opts

    def run(self) -> tuple[TraceDataChanged, TraceOut]:
        """Perform the TRACE footprint analysis.

        Returns:
        -------
        TraceDataChanged:
            Should be Empty
        TraceOut:
            An instance of TraceOut containing the analysis results,
             including the calculated footprints and summary statistics.
        """
        # Determine the number of algorithms being analyzed
        n_algos = self.y_bin.shape[1]

        # Create a boolean array where all values are True (used to calculate the space footprint)
        true_array: NDArray[np.bool_] = np.array([True for _ in self.y_bin], dtype=np.bool_)

        # Calculate the space footprint (area and density)
        print("  -> TRACE is calculating the space area and density.")
        space = self.build(true_array)  # Build the footprint for the entire space
        print(f'    -> Space area: {space.area} | Space density: {space.density}')

        # Prepare to calculate footprints for each algorithm's good and best performance regions
        print("-------------------------------------------------------------------------")
        print("  -> TRACE is calculating the algorithm footprints.")

        # Determine the number of workers available for parallel processing
        n_workers = self.get_num_workers()

        # Calculate the good and best performance footprints for all algorithms in parallel
        good, best = self.parallel_processing(n_workers, n_algos)

        # Detect and resolve contradictions between the best performance footprints of each algorithm
        print("-------------------------------------------------------------------------")
        print("  -> TRACE is detecting and removing contradictory sections of the footprints.")
        for i in range(n_algos):
            print(f"  -> Base algorithm '{self.algo_labels[i]}'")
            start_base = time.time()  # Track the start time for processing this base algorithm

            for j in range(i + 1, n_algos):
                print(f"      -> TRACE is comparing '{self.algo_labels[i]}' with '{self.algo_labels[j]}'")
                start_test = time.time()  # Track the start time for the comparison

                # Create boolean arrays indicating which points correspond to each algorithm's best performance
                algo_1: NDArray[np.bool_] = np.array([v == i for v in self.p], dtype=np.bool_)
                algo_2: NDArray[np.bool_] = np.array([v == j for v in self.p], dtype=np.bool_)

                # Resolve contradictions between the two compared algorithms' best footprints
                best[i], best[j] = self.contra(best[i], best[j], algo_1, algo_2)

                # Print the elapsed time for the comparison
                elapsed_test = time.time() - start_test
                print(f"      -> Test algorithm '{self.algo_labels[j]}' completed. Elapsed time: {elapsed_test:.2f}s")

            # Print the elapsed time for processing this base algorithm
            elapsed_base = time.time() - start_base
            print(f"  -> Base algorithm '{self.algo_labels[i]}' completed. Elapsed time: {elapsed_base:.2f}s")

        # Calculate the footprint for the beta threshold, which is a stricter performance threshold
        print("-------------------------------------------------------------------------")
        print("  -> TRACE is calculating the beta-footprint.")
        hard = self.build(~self.beta)  # Build the footprint for instances not meeting the beta threshold

        # Prepare the summary table for all algorithms, which includes various performance metrics
        print("-------------------------------------------------------------------------")
        print("  -> TRACE is preparing the summary table.")
        summary_data = [
            [
                "Algorithm",
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
            ],
        ]

        # Populate the summary table with metrics for each algorithm's good and best footprints
        for i, label in enumerate(self.algo_labels):
            summary_row = [label]
            summary_row += self.summary(good[i], space.area, space.density)  # Add good performance metrics
            summary_row += self.summary(best[i], space.area, space.density)  # Add best performance metrics
            summary_data.append(summary_row)

        # Convert the summary data into a pandas DataFrame for better organization and readability
        summary_df = pd.DataFrame(summary_data[1:], columns=summary_data[0])

        # Print the completed summary of the TRACE analysis
        print("  -> TRACE has completed. Footprint analysis results:")
        print(" ")
        print(summary_df)

        # Return the results as a TraceOut dataclass instance
        return (TraceDataChanged(), TraceOut(
            space=space,
            good=good,
            best=best,
            hard=hard,
            summary=summary_df
        ))

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
        if unique_rows.shape[0] < 3:
            footprint = self.throw()
            return footprint

        labels = self.run_dbscan(y_bin, unique_rows)
        flag = False
        polygon_body = None
        for i in range(0, np.max(labels) + 1):
            polydata = unique_rows[labels == i]

            # hull = ConvexHull(polydata)
            # polydata = polydata[hull.vertices, :]
            aux = self.fit_poly(polydata, y_bin)
            if aux:
                if not flag:
                    polygon_body = aux
                    flag = True
                else:
                    polygon_body = polygon_body.union(aux)

        return self.compute_footprint(polygon_body, y_bin, True)


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
                [contradiction.contains(point) for point in MultiPoint(self.z[y_base]).geoms],
            )
            num_good_elements_test = np.sum(
                [contradiction.contains(point) for point in MultiPoint(self.z[y_test]).geoms],
            )

            purity_base = num_good_elements_base / num_elements
            purity_test = num_good_elements_test / num_elements

            if purity_base > purity_test:
                c_area = contradiction.area / test_polygon.area
                print(
                    f"        -> {round(100 * c_area, 1)}% of the test footprint is contradictory.",
                )
                test_polygon = test_polygon.difference(contradiction)
                if num_tries < max_tries:
                    test_polygon = self.tight(test_polygon, y_test)
            elif purity_test > purity_base:
                c_area = contradiction.area / base_polygon.area
                print(
                    f"        -> {round(100 * c_area, 1)}% of the base footprint is contradictory.",
                )
                base_polygon = base_polygon.difference(contradiction)
                if num_tries < max_tries:
                    base_polygon = self.tight(base_polygon, y_base)
            else:
                print(
                    "        -> Purity of the contradicting areas is equal for both footprints.",
                )
                print("        -> Ignoring the contradicting area.")
                break

            if base_polygon.is_empty or test_polygon.is_empty:
                break
            else:
                contradiction = base_polygon.intersection(test_polygon)

            num_tries += 1

        base = self.compute_footprint(base_polygon, y_base)
        test = self.compute_footprint(test_polygon, y_test)

        return base, test

    def tight(
        self, polygon: Polygon | MultiPolygon, y_bin: NDArray[np.bool_],
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

            if polydata.shape[0] < 3:
                continue

            # Create a polygon from these points (may use convex_hull instead if needed but the client used this)
            temp_polygon = Polygon(polydata)

            # Get the boundary of the polygon
            boundary = temp_polygon.boundary
            filtered_polydata = polydata[boundary]
            aux = self.fit_poly(filtered_polydata, y_bin)

            if aux:
                refined_polygons.append(aux)

        if len(refined_polygons) > 0:
            return unary_union(refined_polygons)
        else:
            return None

    def fit_poly(
        self, polydata: NDArray[np.double], y_bin: NDArray[np.bool_],
    ) -> Polygon | None:
        """Fit a polygon to the given data points, ensuring it adheres to purity constraints.

        Parameters:
        ----------
        polydata : NDArray[np.double]
            The data points to fit the polygon to.
        y_bin : NDArray[np.bool_]
            Binary array indicating which data points should be considered for the polygon.

        Returns:
        -------
        Polygon | None:
            The fitted polygon, or None if the fitting fails.
        """
        if polydata.shape[0] < 3:
            return None

        polygon = alphashape.alphashape(polydata, 2.2).simplify(0.05)  # no need for rmslivers function

        if not np.all(y_bin):
            if polygon.is_empty:
                return None
            tri = triangulate(polygon)
            for piece in tri:
                elements = np.sum([piece.contains(point) for point in MultiPoint(self.z).geoms])
                good_elements = np.sum([piece.contains(point) for point in MultiPoint(self.z[y_bin]).geoms])
                if (good_elements / elements) < self.opts.pi:
                    polygon = polygon.difference(piece)

        return polygon

    def summary(
        self, footprint: Footprint, space_area: float, space_density: float,
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
            A list containing summarized metrics such as area, normalized area, density, normalized density, and purity.
        """
        area = footprint.area if footprint.area is not None else 0
        normalised_area = float(area / space_area) \
            if ((space_area is not None) and (space_area != 0)) \
            else float(area)
        density = int(footprint.density) if footprint.density is not None else 0
        normalised_density = float(density / space_density) \
            if ((space_density is not None) and (space_density != 0)) \
            else float(footprint.density)
        purity = float(footprint.purity)

        out = [
            area,
            normalised_area,
            density,
            normalised_density,
            purity
        ]
        out = [element if ((element is not None) and (not np.isnan(element))) else 0 for element in out]
        return out

    def throw(self) -> Footprint:
        """Generate an empty footprint with default values, indicating insufficient data.

        Returns:
        -------
        Footprint:
            An instance of Footprint with default values.
        """
        print("        -> There are not enough instances to calculate a footprint.")
        print("        -> The subset of instances used is too small.")
        return Footprint(None, 0, 0, 0, 0, 0)

    def run_dbscan(self, y_bin, data: NDArray[np.double]) -> NDArray[np.int_]:
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
        nn = max(min(np.ceil((np.sum(y_bin)) / 20), 50), 3)
        m, n = data.shape

        # Compute the range of each feature
        feature_ranges = np.max(data, axis=0) - np.min(data, axis=0)

        # Product of the feature ranges
        product_ranges = np.prod(feature_ranges)

        # Compute the gamma function value
        gamma_val = gamma(0.5 * n + 1)

        # Compute Eps
        eps = ((product_ranges * nn * gamma_val) / (m * math.sqrt(math.pi**n))) ** (
            1 / n
        )

        labels = DBSCAN(eps=eps, min_samples=int(nn), metric='euclidean').fit_predict(data)
        return labels

    def get_num_workers(self):
        """Get the number of available workers for parallel processing.

        Returns:
        -------
        int:
            Number of worker processes available.
        """
        try:
            # Try to get the current pool if it exists
            pool = multiprocessing.get_context().Pool()
            num_workers = pool._processes  # Number of processes in the pool
            pool.close()
            pool.join()
        except AttributeError:
            num_workers = 0  # Pool does not exist
        return num_workers

    def process_algorithm(self, i: int):
        """Process a single algorithm to calculate its good and best performance footprints.

        Parameters:
        ----------
        i : int
            Index of the algorithm to process.

        Returns:
        -------
        Tuple[int, Footprint, Footprint]:
            The index of the algorithm, and its good and best performance footprints.
        """
        start_time = time.time()
        print(f"    -> Good performance footprint for '{self.algo_labels[i]}'")
        good_performance = self.build(self.y_bin[:, i])

        print(f"    -> Best performance footprint for '{self.algo_labels[i]}'")
        bool_array: NDArray[np.bool_] = np.array(
            [v == i for v in self.p], dtype=np.bool_,
        )
        best_performance = self.build(bool_array)

        elapsed_time = time.time() - start_time
        print(
            f"    -> Algorithm '{self.algo_labels[i]}' completed. Elapsed time: {elapsed_time:.2f}s",
        )

        return i, good_performance, best_performance

    def parallel_processing(
        self, nworkers: int, nalgos,
    ) -> tuple[list[Footprint], list[Footprint]]:
        """Perform parallel processing to calculate footprints for multiple algorithms.

        Parameters:
        ----------
        nworkers : int
            Number of worker threads to use.
        nalgos : int
            Number of algorithms to process.

        Returns:
        -------
        Tuple[List[Footprint], List[Footprint]]:
            Lists of good and best performance footprints for each algorithm.
        """
        good: list[Footprint | None] = [None for _ in range(nalgos)]
        best: list[Footprint | None] = [None for _ in range(nalgos)]
        with ThreadPoolExecutor(max_workers=nworkers) as executor:
            futures = [
                executor.submit(self.process_algorithm, i) for i in range(nalgos)
            ]
            for future in as_completed(futures):
                i: int
                good_performance: Footprint
                best_performance: Footprint
                i, good_performance, best_performance = future.result()
                good[i] = good_performance
                best[i] = best_performance

        return good, best

    def compute_footprint(self, polygon: Polygon, y_bin: NDArray[np.bool_], smoothen=False) -> Footprint:
        """Create a Footprint object based on the given polygon.

        Parameters:
        ----------
        polygon : Polygon
            The polygon to create the footprint from.
        y_bin : NDArray[np.bool_]
            Binary array indicating the points corresponding to the footprint.
        smoothen : Bool
            Indicates we if need to smoothen the polygon borders.

        Returns:
        -------
        Footprint:
            The created footprint, or an empty one if the polygon is empty.
        """
        if polygon is None:
            return self.throw()
        if smoothen:
            polygon.buffer(0.01).buffer(-0.01)

        elements = np.sum([polygon.contains(point) for point in MultiPoint(self.z).geoms])
        good_elements = np.sum([polygon.contains(point) for point in MultiPoint(self.z[y_bin]).geoms])
        density = elements / polygon.area
        purity = good_elements / elements

        return Footprint(polygon, polygon.area, elements, good_elements, density, purity)





