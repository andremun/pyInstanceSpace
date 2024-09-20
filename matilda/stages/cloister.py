"""CLOISTER Stage Module for Correlation-Based Boundary Estimation.

This module implements the CLOISTER stage, which estimates boundaries in a dataset
by calculating feature correlations and using convex hull construction to define
the boundary. It is designed to analyze the relationship between features and
algorithmic performance, using correlation matrices to determine which features
are significant.

The CLOISTER stage has several key steps:
1. Calculate Pearson correlation coefficients between features.
2. Filter correlations based on statistical significance (using a p-value threshold).
3. Generate boundary estimates using minimum and maximum feature values.
4. Apply convex hull construction to define the boundary of the feature space.

This module is structured around the `CloisterStage` class, which encapsulates
the entire boundary estimation process. Utility methods are provided to calculate
correlation matrices, generate binary representations for boundary selection, and
compute convex hulls.

Dependencies:
- numpy
- scipy
- loguru

Classes
-------
CloisterStage :
    The primary class that implements the CLOISTER stage, providing methods to
    estimate boundaries using correlation and convex hulls.

Functions
---------
cloister(x, a, options):
    The main function to estimate boundaries for a dataset, using a feature matrix `x`
    and projection matrix `a`.
"""

import numpy as np
from loguru import logger
from numpy.typing import NDArray
from scipy.spatial import ConvexHull, QhullError
from scipy.stats import pearsonr

from matilda.data.options import CloisterOptions
from matilda.stages.stage import Stage


class CloisterStage(Stage):
    """CloisterStage class for Correlation-Based Boundary Estimation.

    The `CloisterStage` class implements the core functionality of the CLOISTER stage,
    which estimates boundaries in a dataset by analyzing the correlation between
    features.

    The class provides methods to compute Pearson correlation coefficients, filter
    insignificant correlations, and generate convex hulls to create boundary estimates.

    Attributes
    ----------
    x : NDArray[np.double]
        The feature matrix (instances x features) provided to the CLOISTER stage.
    a : NDArray[np.double]
        The projection matrix provided to the CLOISTER stage.

    Methods
    -------
    __init__(x, a)
        Initializes the `CloisterStage` with the provided feature matrix `x` and
        projection matrix `a`.

    _run(options)
        Executes the CLOISTER stage to estimate boundaries based on the configuration
        options.

    cloister(x, a, options)
        Main method that estimates boundaries by analyzing correlations between features
        and applying convex hull construction.

    _inputs()
        Defines the input parameters required for the CLOISTER stage, which include
        the feature matrix `x` and the projection matrix `a`.

    _outputs()
        Defines the output parameters returned by the CLOISTER stage, which include
        the estimated boundary points (`z_edge`) and the correlation-based boundary
        points (`z_ecorr`).

    _compute_correlation(x, options)
        Computes the Pearson correlation matrix for the feature matrix `x`, and filters
        correlations based on statistical significance using the provided p-value
        threshold.

    _generate_boundaries(x, rho, options)
        Generates boundary points for the feature matrix `x` based on the computed
        correlation matrix `rho` and configuration options.

    _compute_convex_hull(points)
        Computes the convex hull for a given set of points to estimate the boundary
        of the dataset.

    _decimal_to_binary_matrix(nfeats)
        Generates a binary matrix representing all possible boundary combinations for
        a given number of features.
    """

    def __init__(self, x: NDArray[np.double], a: NDArray[np.double]) -> None:
        """Initialize CloisterStage with feature matrix and projection matrix.

        Parameters
        ----------
        x : NDArray[np.double]
            Feature matrix (instances x features) to process.
        a : NDArray[np.double]
            A projection matrix computed from Pilot.
        """
        self.x = x
        self.a = a

    @staticmethod
    def _inputs() -> list[tuple[str, type]]:
        """Define the inputs for the CloisterStage.

        Returns
        -------
        list[tuple[str, type]]
            A list of tuples representing the names and types of input parameters.
        """
        return [("x", NDArray[np.double]), ("a", NDArray[np.double])]

    @staticmethod
    def _outputs() -> list[tuple[str, type]]:
        """Define the outputs for the CloisterStage.

        Returns
        -------
        list[tuple[str, type]]
            A list of tuples representing the names and types of output parameters.
        """
        return [("z_edge", NDArray[np.double]), ("z_ecorr", NDArray[np.double])]

    def _run(
        self,
        options: CloisterOptions,
    ) -> tuple[NDArray[np.double], NDArray[np.double]]:
        """Execute the CLOISTER stage to estimate boundaries.

        Parameters
        ----------
        options : CloisterOptions
            Configuration options for the CLOISTER stage.

        Returns
        -------
        tuple[NDArray[np.double], NDArray[np.double]]
            The estimated boundary points (z_edge) and correlated boundary points
            (z_ecorr).
        """
        return CloisterStage.cloister(self.x, self.a, options)

    @staticmethod
    def cloister(
        x: NDArray[np.double],
        a: NDArray[np.double],
        options: CloisterOptions,
    ) -> tuple[NDArray[np.double], NDArray[np.double]]:
        """Estimate a boundary for the space using correlation.

        Parameters
        ----------
        x : NDArray[np.double]
            Feature matrix (instances x features) to process.
        a : NDArray[np.double]
            Projection matrix computed from Pilot.
        options : CloisterOptions
            Configuration options for CLOISTER.

        Returns
        -------
        tuple[NDArray[np.double], NDArray[np.double]]
            A tuple containing z_edge (estimated boundary points) and z_ecorr
            (correlated boundary points).
        """
        logger.info(
            "  -> CLOISTER is using correlation to estimate a boundary for the space.",
        )

        rho = CloisterStage._compute_correlation(x, options)
        x_edge, remove = CloisterStage._generate_boundaries(x, rho, options)
        z_edge = CloisterStage._compute_convex_hull(np.dot(x_edge, a.T))
        z_ecorr = CloisterStage._compute_convex_hull(np.dot(x_edge[~remove, :], a.T))

        if z_ecorr.size == 0:
            logger.info("  -> The acceptable correlation threshold was too strict.")
            logger.info("  -> The features are weakely correlated.")
            logger.info("  -> Please consider increasing it.")
            z_ecorr = z_edge

        logger.info("-----------------------------------------------------------------")
        logger.info("  -> CLOISTER has completed.")

        return (z_edge, z_ecorr)

    @staticmethod
    def _compute_correlation(
        x: NDArray[np.double],
        options: CloisterOptions,
    ) -> NDArray[np.double]:
        """Calculate the Pearson correlation coefficient for the dataset.

        Parameters
        ----------
        x : NDArray[np.double]
            The feature matrix (instances x features).
        options : CloisterOptions
            Configuration options for CLOISTER, including p-value threshold.

        Returns
        -------
        NDArray[np.double]
            A matrix of Pearson correlation coefficients between each pair of features.
        """
        nfeats = x.shape[1]

        rho = np.zeros((nfeats, nfeats))
        pval = np.zeros((nfeats, nfeats))

        for i in range(nfeats):
            for j in range(nfeats):
                if i != j:
                    rho[i, j], pval[i, j] = pearsonr(x[:, i], x[:, j])
                else:
                    rho[i, j] = 0
                    pval[i, j] = 1

        # Create a boolean mask where calculated pval exceeds specified p-value
        # threshold from the option.
        insignificant_pvals = pval > options.p_val

        # Set the correlation coefficients to zero where correlations are not
        # statistically significant
        rho[insignificant_pvals] = 0

        return rho

    @staticmethod
    def _decimal_to_binary_matrix(nfeats: int) -> NDArray[np.intc]:
        """Generate a binary matrix representation of decimal numbers.

        Parameters
        ----------
        nfeats : int
            Number of features (columns) in the dataset.

        Returns
        -------
        NDArray[np.intc]
            A matrix where each row represents a binary number as an array of bits.
        """
        decimals = np.arange(2**nfeats)
        binary_strings = [np.binary_repr(dec, width=nfeats) for dec in decimals]
        binary_matrix = np.array(
            [[int(bit) for bit in string] for string in binary_strings],
        )
        return binary_matrix[:, ::-1]

    @staticmethod
    def _compute_convex_hull(points: NDArray[np.double]) -> NDArray[np.double]:
        """Calculate the convex hull of a set of points.

        Parameters
        ----------
        points : NDArray[np.double]
            A 2D array of points (instances x features).

        Returns
        -------
        NDArray[np.double]
            The vertices of the convex hull or an empty array if an error occurs.
        """
        try:
            hull = ConvexHull(points)
            return points[hull.vertices, :]
        except QhullError as qe:
            logger.info("QhullError: Encountered geometrical degeneracy:", str(qe))
            return np.array([])
        except ValueError as ve:
            logger.info("ValueError: Imcompatible value encountered:", str(ve))
            return np.array([])

    @staticmethod
    def _generate_boundaries(
        x: NDArray[np.double],
        rho: NDArray[np.double],
        options: CloisterOptions,
    ) -> tuple[NDArray[np.double], NDArray[np.bool_]]:
        """Generate boundaries based on the correlation matrix and configuration option.

        Parameters
        ----------
        x : NDArray[np.double]
            Feature matrix (instances x features).
        rho : NDArray[np.double]
            Correlation matrix computed using Pearson correlation.
        options : CloisterOptions
            Configuration options for CLOISTER.

        Returns
        -------
        tuple[NDArray[np.double], NDArray[np.bool_]]
            A tuple containing the boundary coordinates (x_edge) and a boolean array
            indicating which boundaries should be removed.
        """
        # if no feature selection. then make a note in the boundary construction
        # that it won't work, because nfeats is so large that decimal to binary matrix
        # conversion wont be able to make a matrix.
        nfeats = x.shape[1]

        idx = CloisterStage._decimal_to_binary_matrix(nfeats)
        ncomb = idx.shape[0]

        x_bnds = np.array([np.min(x, axis=0), np.max(x, axis=0)])
        x_edge = np.zeros((ncomb, nfeats))
        remove = np.zeros(ncomb, dtype=bool)

        for i in range(ncomb):
            # Convert the binary indices to flat indices for the boundary selection
            ind = np.ravel_multi_index(
                (idx[i], np.arange(nfeats)),
                (2, nfeats),
                order="F",
            )
            # Select the boundary points corresponding to the flat indices
            x_edge[i, :] = x_bnds.T.flatten()[ind]
            for j in range(nfeats):
                for k in range(j + 1, nfeats):
                    # Check for valid points give the correlation trend
                    if (
                        rho[j, k] > options.c_thres
                        and np.sign(x_edge[i, j]) != np.sign(x_edge[i, k])
                    ) or (
                        rho[j, k] < -options.c_thres
                        and np.sign(x_edge[i, j]) == np.sign(x_edge[i, k])
                    ):
                        remove[i] = True
                    if remove[i]:
                        break
                if remove[i]:
                    break

        return (x_edge, remove)
