from typing import List, Tuple, Dict
import numpy as np
from numpy.typing import NDArray
from sklearn.cluster import DBSCAN
from shapely.geometry import MultiPoint, Polygon, MultiPolygon
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import pandas as pd
from scipy.special import gamma
import math
from scipy.spatial import ConvexHull
from shapely.ops import triangulate, unary_union
import multiprocessing
import time
from matilda.data.options import TraceOptions




class Trace:
    """
    A class to manage the TRACE analysis process.

    Attributes:
    -----------
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
    p: NDArray[np.double]
    beta: NDArray[np.bool_]
    algo_labels: List[str]
    opts: TraceOptions


    def __init__(
            self,
            z: NDArray[np.double],
            y_bin: NDArray[np.bool_],
            p: NDArray[np.double],
            beta: NDArray[np.bool_],
            algo_labels: List[str],
            opts: TraceOptions,
    ) -> None:
        """Initialize the Trace analysis.

                Parameters:
                -----------
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

    def run(self) -> Dict[str, List]:
        """Perform the TRACE footprint analysis.

        Returns:
        --------
        dict:
            A dictionary containing the analysis results, including the calculated footprints and summary statistics.
        """
        ninst, ncol = self.z.shape
        nalgos = self.y_bin.shape[1]
        true_array = np.ones((ninst, ncol), dtype=bool)
        # Calculate the space footprint
        print('  -> TRACE is calculating the space area and density.')
        out = {'space': self.build(true_array)}
        print(f'    -> Space area: {out["space"].area} | Space density: {out["space"].density}')

        # Calculate footprints for good and best performance regions
        print('-------------------------------------------------------------------------')
        print('  -> TRACE is calculating the algorithm footprints.')
        nworkers = self.get_num_workers()
        good, best = self.parallel_processing(nworkers, nalgos)

        out['good'] = good
        out['best'] = best

        # Detect and remove contradictions
        print('-------------------------------------------------------------------------')
        print('  -> TRACE is detecting and removing contradictory sections of the footprints.')
        for i in range(nalgos):
            print(f"  -> Base algorithm '{self.algo_labels[i]}'")
            start_base = time.time()
            for j in range(i + 1, nalgos):
                print(f"      -> TRACE is comparing '{self.algo_labels[i]}' with '{self.algo_labels[j]}'")
                start_test = time.time()
                algo_1: NDArray[np.bool_] = np.array([v == i for v in self.p], dtype=np.bool_)
                algo_2: NDArray[np.bool_] = np.array([v == j for v in self.p], dtype=np.bool_)

                out['best'][i], out['best'][j] = self.contra(
                    out['best'][i], out['best'][j], algo_1, algo_2)

                elapsed_test = time.time() - start_test
                print(f"      -> Test algorithm '{self.algo_labels[j]}' completed. Elapsed time: {elapsed_test:.2f}s")
            elapsed_base = time.time() - start_base
            print(f"  -> Base algorithm '{self.algo_labels[i]}' completed. Elapsed time: {elapsed_base:.2f}s")

        # Calculate the beta footprint
        print('-------------------------------------------------------------------------')
        print('  -> TRACE is calculating the beta-footprint.')
        out['hard'] = self.build(~self.beta)

        # Prepare the summary table
        print('-------------------------------------------------------------------------')
        print('  -> TRACE is preparing the summary table.')
        out['summary'] = [
            ['Algorithm', 'Area_Good', 'Area_Good_Normalised', 'Density_Good', 'Density_Good_Normalised', 'Purity_Good',
             'Area_Best', 'Area_Best_Normalised', 'Density_Best', 'Density_Best_Normalised', 'Purity_Best']]
        for i, label in enumerate(self.algo_labels):
            summary_row = [label]
            summary_row += self.summary(out['good'][i], out['space'].area, out['space'].density)
            summary_row += self.summary(out['best'][i], out['space'].area, out['space'].density)
            out['summary'].append(summary_row)

        print('  -> TRACE has completed. Footprint analysis results:')
        print(' ')
        for row in out['summary']:
            print(row)

        return out

    def build(self, y_bin: NDArray[np.bool_]) -> Footprint:
        """Construct a footprint polygon using DBSCAN clustering.

        Parameters:
        -----------
        y_bin : NDArray[np.bool_]
            Binary indicator vector indicating which data points are of interest.

        Returns:
        --------
        Footprint:
            The constructed footprint with calculated area, density, and purity.
        """

        # Extract rows where Ybin is True
        filtered_z = self.z[y_bin].reshape(self.z.shape)

        # Find unique rows
        unique_rows = np.unique(filtered_z, axis=0)

        # Check the number of unique rows
        if unique_rows.shape[0] < 3:
            footprint = self.throw()

        footprint = Footprint()

        labels = self.run_dbscan(y_bin, unique_rows)

        flag = False
        for i in range(1, np.max(labels) + 1):
            polydata = unique_rows[labels == i]
            polydata = polydata[ConvexHull(polydata).vertices]
            aux = self.fit_poly(polydata,y_bin)
            if aux:
                if not flag:
                    footprint.polygon = aux
                    flag = True
                else:
                    footprint.polygon = footprint.polygon.union(aux)

        if footprint.polygon.is_empty:
            return self.throw()

        footprint.polygon = footprint.polygon.buffer(0.01).buffer(-0.01)
        footprint.area = footprint.polygon.area
        footprint.elements = np.sum([footprint.polygon.contains(point) for point in MultiPoint(self.z)])
        footprint.good_elements = np.sum([footprint.polygon.contains(point) for point in MultiPoint(self.z[y_bin,])])
        footprint.density = footprint.elements / footprint.area
        footprint.purity = footprint.good_elements / footprint.elements

        return footprint

    def contra(self, base: Footprint, test: Footprint, y_base: NDArray[np.bool_] , y_test: NDArray[np.bool_]) \
            -> Tuple[Footprint, Footprint]:
        """Detect and resolve contradictions between two footprint polygons.

                Parameters:
                -----------
                base : Footprint
                    The base footprint polygon.
                test : Footprint
                    The test footprint polygon.
                y_base : NDArray[np.bool_]
                    Binary array indicating the points corresponding to the base footprint.
                y_test : NDArray[np.bool_]
                    Binary array indicating the points corresponding to the test footprint.

                Returns:
                --------
                tuple:
                    Updated base and test footprints after resolving contradictions.
                """

        if base.polygon.is_empty or test.polygon.is_empty:
            return base, test

        max_tries = 3
        num_tries = 1
        contradiction = base.polygon.intersection(test.polygon)

        while not contradiction.is_empty and num_tries <= max_tries:
            num_elements = np.sum([contradiction.contains(point) for point in MultiPoint(self.z)])
            num_good_elements_base = np.sum(
                [contradiction.contains(point) for point in MultiPoint(self.z[y_base])])
            num_good_elements_test = np.sum(
                [contradiction.contains(point) for point in MultiPoint(self.z[y_test])])

            purity_base = num_good_elements_base / num_elements
            purity_test = num_good_elements_test / num_elements

            if purity_base > purity_test:
                c_area = contradiction.area / test.polygon.area
                print(f'        -> {round(100 * c_area, 1)}% of the test footprint is contradictory.')
                test.polygon = test.polygon.difference(contradiction)
                if num_tries < max_tries:
                    test.polygon = self.tight(test.polygon, y_test)
            elif purity_test > purity_base:
                c_area = contradiction.area / base.polygon.area
                print(f'        -> {round(100 * c_area, 1)}% of the base footprint is contradictory.')
                base.polygon = base.polygon.difference(contradiction)
                if num_tries < max_tries:
                    base.polygon = self.tight(base.polygon, y_base)
            else:
                print('        -> Purity of the contradicting areas is equal for both footprints.')
                print('        -> Ignoring the contradicting area.')
                break

            if base.polygon.is_empty or test.polygon.is_empty:
                break
            else:
                contradiction = base.polygon.intersection(test.polygon)

            num_tries += 1

        if base.polygon.is_empty:
            base = self.throw()
        else:
            base.area = base.polygon.area
            base.elements = np.sum([base.polygon.contains(point) for point in MultiPoint(self.z)])
            base.good_elements = np.sum([base.polygon.contains(point) for point in MultiPoint(self.z[y_base])])
            base.density = base.elements / base.area
            base.purity = base.good_elements / base.elements

        if test.polygon.is_empty:
            test = self.throw()
        else:
            test.area = test.polygon.area
            test.elements = np.sum([test.polygon.contains(point) for point in MultiPoint(self.z)])
            test.good_elements = np.sum([test.polygon.contains(point) for point in MultiPoint(self.z[y_test])])
            test.density = test.elements / test.area
            test.purity = test.good_elements / test.elements

        return base, test

    def tight(self, polygon: Polygon | MultiPolygon, y_bin: NDArray[np.bool_]) -> Polygon | None:
        """Refine an existing polygon by removing slivers and improving its shape.

        Parameters:
        -----------
        polygon : Polygon | MultiPolygon
            The polygon or multipolygon to be refined.
        y_bin : NDArray[np.bool_]
            Binary array indicating which data points belong to the polygon.

        Returns:
        --------
        Polygon | None:
            The refined polygon, or None if the refinement fails.
        """
        if not polygon:
            return None

        splits = [item for item in polygon] if isinstance(polygon, MultiPolygon) else [polygon]
        npolygons = len(splits)
        refined_polygons = []

        for i in range(npolygons):
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

    def fit_poly(self, polydata: NDArray[np.double], y_bin: NDArray[np.bool_]) -> Polygon | None:
        """Fit a polygon to the given data points, ensuring it adheres to purity constraints.

        Parameters:
        -----------
        polydata : NDArray[np.double]
            The data points to fit the polygon to.
        y_bin : NDArray[np.bool_]
            Binary array indicating which data points should be considered for the polygon.

        Returns:
        --------
        Polygon | None:
            The fitted polygon, or None if the fitting fails.
        """
        if polydata.shape[0] < 3:
            return None

        polygon = Polygon(polydata).simplify(0.05)  # no need for rmslivers function

        if not np.all(y_bin):
            if polygon.is_empty:
                return None
            tri = triangulate(polygon)
            for piece in tri:
                elements = np.sum(tri.contains(MultiPoint(self.z)))
                good_elements = np.sum(tri.contains(MultiPoint(self.z[y_bin])))
                if self.opts.PI > (good_elements / elements):
                    polygon = polygon.difference(piece)

        return polygon

    def summary(self, footprint: Footprint, space_area: float, space_density: float) -> List[float]:
        """Summarize the footprint metrics.

        Parameters:
        -----------
        footprint : Footprint
            The footprint to summarize.
        space_area : float
            The total area of the space being analyzed.
        space_density : float
            The density of the entire space.

        Returns:
        --------
        list:
            A list containing summarized metrics such as area, normalized area, density, normalized density, and purity.
        """
        out = [
            footprint.area,
            footprint.area / space_area,
            footprint.density,
            footprint.density / space_density,
            footprint.purity
        ]
        out[np.isnan(out)] = 0
        return out

    def throw(self) -> Footprint:
        """Generate an empty footprint with default values, indicating insufficient data.

        Returns:
        --------
        Footprint:
            An instance of Footprint with default values.
        """
        print('        -> There are not enough instances to calculate a footprint.')
        print('        -> The subset of instances used is too small.')
        return Footprint()

    def run_dbscan(self, y_bin, data: NDArray[np.double]) -> NDArray[np.int_]:
        """Perform DBSCAN clustering on the dataset.

        Parameters:
        -----------
        y_bin : NDArray[np.bool_]
            Binary indicator vector to filter the data points.
        data : NDArray[np.double]
            The dataset to cluster.

        Returns:
        --------
        NDArray[np.int_]:
            Array of cluster labels for each data point.
        """
        nn = max(min(np.ceil(np.sum(y_bin) / 20), 50), 3)
        m, n = data.shape

        # Compute the range of each feature
        feature_ranges = np.max(data, axis=0) - np.min(data, axis=0)

        # Product of the feature ranges
        product_ranges = np.prod(feature_ranges)

        # Compute the gamma function value
        gamma_val = gamma(0.5 * n + 1)

        # Compute Eps
        eps = ((product_ranges * nn * gamma_val) / (m * math.sqrt(math.pi ** n))) ** (1 / n)

        labels = DBSCAN(eps=eps, min_samples= int(nn)).fit_predict(data)
        return labels

    def get_num_workers(self):
        """Get the number of available workers for parallel processing.

        Returns:
        --------
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
        -----------
        i : int
            Index of the algorithm to process.

        Returns:
        --------
        Tuple[int, Footprint, Footprint]:
            The index of the algorithm, and its good and best performance footprints.
        """
        start_time = time.time()
        print(f"    -> Good performance footprint for '{self.algo_labels[i]}'")
        good_performance = self.build(self.y_bin[:, i])

        print(f"    -> Best performance footprint for '{self.algo_labels[i]}'")
        bool_array: NDArray[np.bool_] = np.array([v == i for v in self.p], dtype=np.bool_)
        best_performance = self.build(bool_array)

        elapsed_time = time.time() - start_time
        print(f"    -> Algorithm '{self.algo_labels[i]}' completed. Elapsed time: {elapsed_time:.2f}s")

        return i, good_performance, best_performance

    def parallel_processing(self, nworkers: int, nalgos) -> Tuple[List[Footprint], List[Footprint]] :
        """Perform parallel processing to calculate footprints for multiple algorithms.

        Parameters:
        -----------
        nworkers : int
            Number of worker threads to use.
        nalgos : int
            Number of algorithms to process.

        Returns:
        --------
        Tuple[List[Footprint], List[Footprint]]:
            Lists of good and best performance footprints for each algorithm.
        """
        good: List[Footprint] = [Footprint() for _ in range(nalgos)]
        best: List[Footprint] = [Footprint() for _ in range(nalgos)]
        with (ThreadPoolExecutor(max_workers=nworkers) as executor):
            futures = [
                executor.submit(self.process_algorithm, i)
                for i in range(nalgos)
            ]
            for future in as_completed(futures):
                i: int
                good_performance: Footprint
                best_performance: Footprint
                i, good_performance, best_performance = future.result()
                good[i] = good_performance
                best[i] = best_performance

        return good, best



