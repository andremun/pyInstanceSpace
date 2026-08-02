# SPDX-License-Identifier: LicenseRef-PolyForm-Noncommercial-1.0.0
# Copyright (c) 2024-2026 Mario Andrés Muñoz
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

from typing import NamedTuple

import numpy as np
from loguru import logger
from numpy.typing import NDArray
from scipy.spatial import ConvexHull, QhullError
from scipy.stats import pearsonr

from instancespace.data.options import CloisterOptions
from instancespace.stages.stage import Stage


class CloisterInput(NamedTuple):
    """Inputs for the Cloister stage.

    Attributes
    ----------
    x : NDArray[np.double]
        The feature matrix (instances x features) provided to the CLOISTER stage.
    a : NDArray[np.double]
        The projection matrix provided to the CLOISTER stage.
    cloister_options : Cloister Options
        Options for running Cloister.
    """

    x: NDArray[np.double]
    a: NDArray[np.double]
    cloister_options: CloisterOptions


class CloisterOutput(NamedTuple):
    """Outputs from the Cloister stage.

    Attributes
    ----------
    z_edge : NDArray[np.double]
        Estimated boundary points.
    z_ecorr : NDArray[np.double]
        Correlated boundary points.
    """

    z_edge: NDArray[np.double]
    z_ecorr: NDArray[np.double]


class CloisterStage(Stage[CloisterInput, CloisterOutput]):
    """CloisterStage class for Correlation-Based Boundary Estimation.

    The `CloisterStage` class implements the core functionality of the CLOISTER stage,
    which estimates boundaries in a dataset by analyzing the correlation between
    features.

    The class provides methods to compute Pearson correlation coefficients, filter
    insignificant correlations, and generate convex hulls to create boundary estimates.

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

    @staticmethod
    def _inputs() -> type[CloisterInput]:
        return CloisterInput

    @staticmethod
    def _outputs() -> type[CloisterOutput]:
        return CloisterOutput

    @staticmethod
    def _run(
        inputs: CloisterInput,
    ) -> CloisterOutput:
        """Execute the CLOISTER stage to estimate boundaries.

        Parameters
        ----------
        inputs : CloisterInput
            Inputs for the cloister stage.

        Returns
        -------
        CloisterOutput
            Output of the Cloister stage.
        """
        return CloisterStage.cloister(inputs.x, inputs.a, inputs.cloister_options)

    @staticmethod
    def cloister(
        x: NDArray[np.double],
        a: NDArray[np.double],
        options: CloisterOptions,
    ) -> CloisterOutput:
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
        The output of the Cloister stage.
        """
        logger.info(
            "[CLOISTER]   -> CLOISTER is using correlation to estimate a boundary"
            " for the space.",
        )

        hull_dims = None if options.hull_dims == "all" else options.hull_dims

        nfeats = x.shape[1]
        if nfeats > options.max_features:
            # Corner enumeration below is 2**nfeats - intractable past a
            # point. Matches MATLAB's opts.maxFeatures fallback: skip
            # enumeration entirely and use a plain convex hull of the
            # projected instances as the boundary instead.
            logger.warning(
                f"[CLOISTER]   -> CLOISTER skipped: {nfeats} features exceeds "
                f"limit of {options.max_features}. Using convex hull as boundary.",
            )
            z_all = CloisterStage._compute_convex_hull(np.dot(x, a.T), hull_dims)
            logger.info("[CLOISTER] " + "-" * 65)
            logger.info("[CLOISTER]   -> CLOISTER has completed.")
            return CloisterOutput(z_all, z_all)

        rho = CloisterStage._compute_correlation(x, options)
        x_edge, remove = CloisterStage._generate_boundaries(x, rho, options)
        z_edge = CloisterStage._compute_convex_hull(np.dot(x_edge, a.T), hull_dims)

        if z_edge.size == 0:
            # Unlike a too-strict correlation threshold (below), an empty
            # z_edge means the boundary polygon itself couldn't be built at
            # all (degenerate points, NaN propagation, etc.) - MATLAB lets
            # this fail loudly rather than silently returning an empty
            # boundary, so this is logged as an error, not folded into the
            # "threshold too strict" message below.
            logger.error(
                "[CLOISTER]   -> Could not construct a boundary polygon from "
                "the feature bounds - check for degenerate or NaN-heavy "
                "input data.",
            )
            z_ecorr = z_edge
        else:
            z_ecorr = CloisterStage._compute_convex_hull(
                np.dot(x_edge[~remove, :], a.T),
                hull_dims,
            )
            if z_ecorr.size == 0:
                logger.info(
                    "[CLOISTER]   -> The acceptable correlation threshold was too"
                    " strict.",
                )
                logger.info("[CLOISTER]   -> The features are weakly correlated.")
                logger.info("[CLOISTER]   -> Please consider increasing it.")
                z_ecorr = z_edge

        logger.info("[CLOISTER] " + "-" * 65)
        logger.info("[CLOISTER]   -> CLOISTER has completed.")

        return CloisterOutput(z_edge, z_ecorr)

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
        min_valid_pairs = 2

        rho = np.zeros((nfeats, nfeats))
        pval = np.zeros((nfeats, nfeats))

        # A feature column can carry sparse NaNs here (matching MATLAB's own
        # documented "sparse NaNs reach CLOISTER" design) - pearsonr on raw
        # columns silently returns (nan, nan) for a NaN-containing pair
        # instead of computing over the valid overlap, which then flows
        # through unfiltered (nan > p_val is False, so insignificant_pvals
        # never catches it). Mask each pair's rows to the ones where both
        # columns are valid instead.
        for i in range(nfeats):
            for j in range(nfeats):
                if i != j:
                    valid = ~(np.isnan(x[:, i]) | np.isnan(x[:, j]))
                    if np.sum(valid) < min_valid_pairs:
                        rho[i, j] = 0.0
                        pval[i, j] = 1.0
                    else:
                        rho[i, j], pval[i, j] = pearsonr(x[valid, i], x[valid, j])
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
    def _compute_convex_hull(
        points: NDArray[np.double],
        hull_dims: int | None = None,
    ) -> NDArray[np.double]:
        """Calculate the convex hull of a set of points.

        Parameters
        ----------
        points : NDArray[np.double]
            A 2D array of points (instances x features).
        hull_dims : int | None
            Restrict the hull geometry to the first `hull_dims` columns of
            `points` (matching MATLAB's `core/CLOISTER.m`, which always
            builds a 2D hull on the first two projected columns). `None`
            (this port's own default) uses every column, letting
            `scipy.spatial.ConvexHull` build the hull in its native
            dimensionality. #299 audit finding, issue 5. Either way, the
            returned vertices keep every column of `points` - only the hull
            geometry itself is computed on the restricted view.

        Returns
        -------
        NDArray[np.double]
            The vertices of the convex hull or an empty array if an error occurs.
        """
        hull_points = points if hull_dims is None else points[:, :hull_dims]
        try:
            hull = ConvexHull(hull_points)
            return points[hull.vertices, :]
        except QhullError as qe:
            logger.warning(
                f"[CLOISTER] QhullError: Encountered geometrical degeneracy: {qe}",
            )
            return np.array([])
        except ValueError as ve:
            logger.warning(
                f"[CLOISTER] ValueError: Incompatible value encountered: {ve}",
            )
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
        # Caller (cloister()) already guards nfeats > options.max_features
        # before reaching here, so 2**nfeats corner enumeration below stays
        # tractable.
        nfeats = x.shape[1]

        idx = CloisterStage._decimal_to_binary_matrix(nfeats)
        ncomb = idx.shape[0]

        # nanmin/nanmax (not min/max): a feature column can carry sparse
        # NaNs (matching MATLAB's documented "sparse NaNs reach CLOISTER"
        # design, via its own omitnan bounds) - plain min/max would return
        # NaN for that column and propagate through x_edge into
        # ConvexHull, which errors on NaN input.
        x_bnds = np.array([np.nanmin(x, axis=0), np.nanmax(x, axis=0)])
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
