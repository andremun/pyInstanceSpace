"""TRACE: Calculating the algorithm footprints.

Triangulation with Removal of Areas with Contradicting Evidence (TRACE)
is an algorithm used to estimate the area of good performance of an
algorithm within the space.
For more details, please read the original Matlab code and liveDemo.

"""

# Allows using Trace type inside the Trace class. This is only required in static
# constructors.
from __future__ import annotations

import numpy as np
import time as time
from numpy.typing import NDArray
import pandas as pd
from matilda.data.model import Footprint, PolyShape, TraceOut
from matilda.data.option import TraceOptions
import warnings
class Trace:
    """See file docstring."""

    z: NDArray[np.double]
    y_bin: NDArray[np.bool_]
    p: NDArray[np.double]
    beta: NDArray[np.bool_]
    algo_labels: list[str]
    opts: TraceOptions

    def __init__(
        self,
        z: NDArray[np.double],
        y_bin: NDArray[np.bool_],
        p: NDArray[np.double],
        beta: NDArray[np.bool_],
        algo_labels: list[str],
        opts: TraceOptions,
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
            opts (TraceOptions): Configuration options for TRACE and its subroutines
        """
        self.z = z
        self.y_bin = y_bin
        self.p = p
        self.beta = beta
        self.algo_labels = algo_labels
        self.opts = opts

    @staticmethod
    def run(
        z: NDArray[np.double],
        y_bin: NDArray[np.bool_],
        p: NDArray[np.double],
        beta: NDArray[np.bool_],
        algo_labels: list[str],
        opts: TraceOptions,
    ) -> TraceOut:
        """Estimate the good performance area of algorithms within the space.

        Parameters
        ----------
        z : NDArray[np.double]
            The space of instances
        y_bin : NDArray[np.bool_]
            Binary indicators of performance
        p : NDArray[np.double]
            Performance metrics for algorithms
        beta : NDArray[np.bool_]
            Specific beta threshold for footprint calculation
        algo_labels : list[str]
            Labels for each algorithm. Note that the datatype is still in deciding
        opts : TraceOptions
            Configuration options for TRACE and its subroutines

        Returns
        -------
        TraceOut :
            A structured output containing the results of the TRACE analysis
            including algorithm footprints and performance summaries.
        """
        # TODO: parallel pool
        warnings.filterwarnings("ignore")

        trace = Trace(z, y_bin, p, beta, algo_labels, opts)  # noqa: F841
        print("  -> TRACE is calculating the space area and density.")
        ninst = z.shape[0]
        nalgos = y_bin.shape[1]
        space = trace.build(z, np.ones((ninst, 1), dtype=bool), opts)
        
        good=[None]*len(algo_labels)
        best=[None]*len(algo_labels)
        summary=pd.DataFrame()

        print(f"    -> Space area: {space.area} | Space density: {space.density}")
        
        print("-" * 10)
        print("  -> TRACE is calculating the algorithm footprints.")
    
        
        print("-" * 10)
        print("  -> TRACE is detecting and removing contradictory sections of the footprints.")
    
        for i in range(nalgos):
            print(f"  -> Base algorithm '{algo_labels[i]}'")

            start_base = time.time()
            for j in range(i + 1, nalgos):
                print(f"      -> TRACE is comparing '{algo_labels[i]}' with '{algo_labels[j]}'")

                start_test = time.time()
                good[i], best[j] = trace.contra(best[i], best[j], z, p == i, p == j, opts)
                print(f"      -> Test algorithm '{algo_labels[j]}' completed. Elapsed time: {time.time() - start_test:.2f}s")
        print(f"  -> Base algorithm '{algo_labels[i]}' completed. Elapsed time: {time.time() - start_base:.2f}s")

        print("-" * 10)
        print("  -> TRACE is calculating the beta-footprint.")
        hard = trace.build(z, np.logical_not(beta), opts)

        print("-" * 10)
        print("  -> TRACE is preparing the summary table.")

        summary_columns=['','Area_Good', 'Area_Good_Normalized', 
                        'Density_Good', 'Density_Good_Normalized', 'Purity_Good', 'Area_Best', 
                        'Area_Best_Normalized', 'Density_Best', 'Density_Best_Normalized', 'Purity_Best']
        summary = pd.DataFrame(index=algo_labels, columns=summary_columns)

        for i in range(nalgos):
            summary_good = Trace.summary(good[i], space.area, space.density)
            summary_best = Trace.summary(best[i], space.area, space.density)

            # Concatenate summaries for good and best performance
            row_data =  [algo_labels[i]] + summary_good + summary_best    
            row = pd.DataFrame([row_data],columns=summary_columns)

            row = row.round(3)
            summary = pd.concat([summary, row]).dropna()

        summary.set_index('',inplace=True)
        
        out = TraceOut(space,good,best,hard,summary)
        print("  -> TRACE has completed. Footprint analysis results:")
        return out

    """
    % =========================================================================
    % SUBFUNCTIONS
    % =========================================================================
    """


    def build(
        self,
    ) -> Footprint:
        """Build footprints for good or best performance of algorithms.

        Parameters
        ----------
        z: NDArray[np.double]
            The space of instances.
        y_bin: NDArray[np.bool_]
            Binary indicators of performance.
        opts: TraceOptions
            Configuration options for TRACE.

        Returns
        -------
        Footprint: A footprint structure containing polygons, area, density, and purity.
        """
        # TODO: Rewrite TRACEbuild logic in python
        # raise NotImplementedError
        pass

    def contra(
        self,
        base: Footprint,
        test: Footprint,
        y_base: NDArray[np.bool_],
        y_test: NDArray[np.bool_],
    ) -> tuple[Footprint, Footprint]:
        """Detect and remove contradictory sections between two algorithm footprints.

        Parameters
        ----------
        base, Footprint:
            the footprints of base
        test , Footprint:
            the footprints of test
        z,NDArray[np.double]:
            The space of instances.
        y_base,NDArray[np.bool_]:
            Performance indicators
        y_test,NDArray[np.bool_]:
            Performance indicators
        opts,TraceOptions:
            Configuration options for TRACE.

        Returns
        -------
            tuple[Footprint, Footprint]
            still need to decide the return values type.
        """
        # not sure whether the returned value is tuple or list, needs further decision
        # TODO: Rewrite TRACEcontra logic in python
        # raise NotImplementedError
        pass


    def tight(
        self,
        polygon: PolyShape,
    ) -> PolyShape:
        """Refer the original Matlab function to get more info.

        Parameters
        ----------
        polygon : PolyShape
            The initial polygon shape.
        z : NDArray[np.double]
            The space of instances.
        y_bin : NDArray[np.bool_]
            Not pretty sure the meaning of this parameter
        opts : TraceOptions
            Configuration options for TRACE

        Returns
        -------
        PolyShape
            Not pretty sure the meaning
        """
        # TODO: Rewrite TRACEtight logic in python
        # raise NotImplementedError
        pass


    # note that for polydata, it is  highly probably a 2 dimensional array
    def fitpoly(
        self,
        poly_data: NDArray[np.double],
    ) -> PolyShape:
        """Fits a polygon to the given data points according to TRACE criteria.

        Parameters
        ----------
        poly_data : NDArray[np.double]
            Not pretty sure the meaning.
        z : NDArray[np.double]
            Not pretty sure the meaning.
        y_bin : NDArray[np.bool_]
            Not pretty sure the meaning.
        opts : TraceOptions
            Configuration options for TRACE,

        Returns
        -------
        PolyShape
            Not pretty sure the meaning.
        """
        # TODO: Rewrite TRACEfitpoly logic in python
        # raise NotImplementedError
        pass


    def summary(
        self,
        footprint: Footprint,
        space_area: float,
        space_density: float,
    ) -> list[float]:
        """Generate a summary of a footprint's relative characteristics.

        Parameters
        ----------
        footprint : Footprint
            The footprint for which to generate a summary.
        space_area : double
            Not pretty sure the meaning.
        space_density : double
            Not pretty sure the meaning.

        Returns
        -------
        list[float]
            A list containing summary statistics of the footprint,
            such as its area, normalized area, density, normalized density, and purity.
        """
        # TODO: Rewrite TRACEsummary logic in python
        # raise NotImplementedError
        pass


    def throw(
        self,
    ) -> Footprint:
        """Generate a default 'empty' footprint.

        Returns
        -------
        Footprint
            with polygon set to an empty list and all numerical values set to 0,
            indicating an insufficient data scenario.
        """
        # TODO: Rewrite TRACEthrow logic in python
        # raise NotImplementedError
        pass


    def dbscan(
        self,
        x: NDArray[np.double],
        k: int,
        eps: float,
    ) -> tuple[NDArray[np.intc], NDArray[np.intc]]:
        """Perform DBSCAN clustering on the given data set.

        Parameters
        ----------
        x : NDArray[np.double]
            Data set for clustering,
        k : int
            The minimum number of neighbors within `eps` radius
        eps : double
            The maximum distance between two points for one
            to be considered as in the neighborhood of the other.
            note that parameter:eps could be dropped.

        Returns
        -------
        tuple[NDArray[np.intc], NDArray[np.intc]]
            tuple with arrays: the first indicates the cluster labels for each point,
            and the second array indicates the point types (core, border, outlier).
            not sure whether the returned value is tuple or list, needs further decision
        """
        # TODO: Rewrite dbscan logic in python

        # raise NotImplementedError
        pass


    def epsilon(
        self,
        x: NDArray[np.double],
        k: int,
    ) -> float:
        """Estimates the optimal epsilon value for DBSCAN based on the data.

        Parameters
        ----------
        x : NDArray[np.double]
            The data set used for clustering.
        k : int
            The minimum number of neighbors within `eps`
            radius to consider a point as a core point.

        Returns
        -------
        double
            The estimated optimal epsilon value for the given data set and `k`.
        """
        # TODO: Rewrite epsilon logic in python
        # raise NotImplementedError
        pass


    def dist(
        self,
        i: NDArray[np.double],
        x: NDArray[np.double],
    ) -> NDArray[np.double]:
        """Calculate the Euclidean distance between a point and multiple other points.

        Parameters
        ----------
        i : NDArray[np.double]
            an object (1,n).
        x : NDArray[np.double]
            data matrix (m,n); m-objects, n-variables.

        Returns
        -------
        NDArray[np.double]
            Euclidean distance (m,1).
        """
        # TODO: Rewrite dist logic in python
        #raise NotImplementedError
        pass