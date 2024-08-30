"""
TRACE: Calculating the algorithm footprints.

Triangulation with Removal of Areas with Contradicting Evidence (TRACE)
is an algorithm used to estimate the area of good performance of an
algorithm within the space.
For more details, please read the original Matlab code and liveDemo.

"""

# Allows using Trace type inside the Trace class. This is only required in static
# constructors.
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from matilda.data.model import Footprint, PolyShape, TraceDataChanged, TraceOut
from matilda.data.options import TraceOptions


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
        """
        Initialise the Trace stage.

        Args:
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
    ) -> tuple[TraceDataChanged, TraceOut]:
        """
        Estimate the good performance area of algorithms within the space.

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
        trace = Trace(z, y_bin, p, beta, algo_labels, opts)  # noqa: F841
        # TODO: Rewrite TRACE logic in python
        raise NotImplementedError

    """
    % =========================================================================
    % SUBFUNCTIONS
    % =========================================================================
    """

    def build(
        self,
    ) -> Footprint:
        """
        Build footprints for good or best performance of algorithms.

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
        raise NotImplementedError

    def contra(
        self,
        base: Footprint,
        test: Footprint,
        y_base: NDArray[np.bool_],
        y_test: NDArray[np.bool_],
    ) -> tuple[Footprint, Footprint]:
        """
        Detect and remove contradictory sections between two algorithm footprints.

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
        raise NotImplementedError

    def tight(
        self,
        polygon: PolyShape,
    ) -> PolyShape:
        """
        Refer the original Matlab function to get more info.

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
        raise NotImplementedError

    # note that for polydata, it is  highly probably a 2 dimensional array
    def fitpoly(
        self,
        poly_data: NDArray[np.double],
    ) -> PolyShape:
        """
        Fits a polygon to the given data points according to TRACE criteria.

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
        raise NotImplementedError

    def summary(
        self,
        footprint: Footprint,
        space_area: float,
        space_density: float,
    ) -> list[float]:
        """
        Generate a summary of a footprint's relative characteristics.

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
        raise NotImplementedError

    def throw(
        self,
    ) -> Footprint:
        """
        Generate a default 'empty' footprint.

        Returns
        -------
        Footprint
            with polygon set to an empty list and all numerical values set to 0,
            indicating an insufficient data scenario.

        """
        # TODO: Rewrite TRACEthrow logic in python
        raise NotImplementedError

    def dbscan(
        self,
        x: NDArray[np.double],
        k: int,
        eps: float,
    ) -> tuple[NDArray[np.intc], NDArray[np.intc]]:
        """
        Perform DBSCAN clustering on the given data set.

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

        raise NotImplementedError

    def epsilon(
        self,
        x: NDArray[np.double],
        k: int,
    ) -> float:
        """
        Estimates the optimal epsilon value for DBSCAN based on the data.

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
        raise NotImplementedError

    def dist(
        self,
        i: NDArray[np.double],
        x: NDArray[np.double],
    ) -> NDArray[np.double]:
        """
        Calculate the Euclidean distance between a point and multiple other points.

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
        raise NotImplementedError
