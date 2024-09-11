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


from __future__ import annotations

import math
import multiprocessing
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import alphashape
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.special import gamma
from shapely.geometry import MultiPoint, MultiPolygon, Polygon
from shapely.ops import triangulate, unary_union
from sklearn.cluster import DBSCAN

from matilda.data.model import Footprint, TraceDataChanged, TraceOut
from matilda.data.options import TraceOptions
from matilda.stages.stage import Stage

POLYGON_MIN_POINT_REQUIREMENT = 3


class TraceStage(Stage):
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
        p: NDArray[np.double],
        beta: NDArray[np.bool_],
        algo_labels: list[str],
    ) -> None:
        """Initialise the Trace stage.

            Args
            ----
        z (NDArray[np.double]): The space of instances
        y_bin (NDArray[np.bool_]): Binary indicators of performance
        p (NDArray[np.double]): Performance metrics for algorithms
        beta (NDArray[np.bool_]): Specific beta threshold for footprint calculation
        algo_labels (list[str]): Labels for each algorithm. Note that the datatype
            is still in deciding

        Returns
        -------
        None
        """
        self.z = z
        self.y_bin = y_bin
        self.p = p
        self.beta = beta
        self.algo_labels = algo_labels

    @staticmethod
    def _inputs() -> list[tuple[str, type]]:
        """Use the method for determining the inputs for trace.

        Args
        ----
            None

        Returns
        -------
            list[tuple[str, type]]
                List of inputs for the stage
        """
        return [
    ("z", NDArray[np.double]),
    ("y_bin", NDArray[np.bool_]),
    ("p", NDArray[np.double]),
    ("beta", NDArray[np.bool_]),
    ("algo_labels", list[str]),
]

    @staticmethod
    def _outputs() -> list[tuple[str, type]]:
        """Use the method for determining the outputs for trace.

        Args
        ----
            None

        Returns
        -------
            list[tuple[str, type]]
                List of outputs for the stage
        """
        return [
    ("space", Footprint),
    ("good", list[Footprint]),
    ("best", list[Footprint]),
    ("hard", Footprint),
    ("summary", pd.DataFrame),
]

    def _run(
        self,
        options: TraceOptions,
    ) -> tuple[Footprint, list[Footprint], list[Footprint], Footprint, pd.DataFrame]:
        """Use the method for running the trace stage as well as surrounding buildIS.

        Args
        ----
            options (TraceOptions): Configuration options for TRACE and its subroutines

        Returns
        -------
            tuple[Footprint, list[Footprint], list[Footprint], Footprint, pd.DataFrame]
                The results of the trace stage
        """
        # All the code including the code in the buildIS should be here
        raise NotImplementedError

    @staticmethod
    def trace(
        z: NDArray[np.double],
        y_bin: NDArray[np.bool_],
        p: NDArray[np.double],
        beta: NDArray[np.bool_],
        algo_labels: list[str],
    ) -> tuple[Footprint, list[Footprint], list[Footprint], Footprint, pd.DataFrame]:
        """Use the method for running the trace stage.

        Args
        ----
            z (NDArray[np.double]): The space of instances
            y_bin (NDArray[np.bool_]): Binary indicators of performance
            p (NDArray[np.double]): Performance metrics for algorithms
            beta (NDArray[np.bool_]): Specific beta threshold for footprint calculation
            algo_labels (list[str]): Labels for each algorithm. Note that the datatype
                is still in deciding

        Returns
        -------
            tuple[Footprint, list[Footprint], list[Footprint], Footprint, pd.DataFrame]
                The results of the trace stage
        """
        # This has code specific to TRACE.m
        raise NotImplementedError
