"""
CLOISTER is using correlation to estimate a boundary for the space.

The function uses the correlation between the features and the performance of the
algorithms to estimate a boundary for the space. It constructs edges based on the
correlation between the features. The function then uses these edges to construct
a convex hull, providing a boundary estimate for the dataset.
"""

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import ConvexHull, QhullError
from scipy.stats import pearsonr

from matilda.data.model import BoundaryResult, CloisterOut
from matilda.data.option import CloisterOptions


class Cloister:
    """
    Estimate a boundary for the space using correlation.

    Attributes
    ----------
    x : NDArray[np.double]
        2D array representing feature data
    a : NDArray[np.double]
        2D array representing projection matrix calculated from Pilot.
    options : CloisterOptions
        Configuration options for boundary calculations.
    nfeats : int
        Number of features in the dataset 'x'.

    Methods
    -------
        compute_correlation():
            Calcupate correlation on data x using pearson's correlation
        decimal_to_binary_matrix():
            Convert decimal from 0 to nfeat into binary matrix
        calculate_convex_hull():
            Calculate the convex hull for a set of points in a multidimensional space.
        generate_boundaries():
            Generate boundaries based on the correlation matrix and specified options.
        run():
            Run cloister analysis and returns the result.

    """

    def __init__(
        self: "Cloister",
        x: NDArray[np.double],
        a: NDArray[np.double],
        options: CloisterOptions,
    ) -> None:
        """
        Initialize the Cloister object with datasets and options.

        Arguments:
        ---------
            x : NDArray[np.double]
                2D array representing feature data.
            a : NDArray[np.double]
                2D array representing projection matrix calculated from Pilot.
            options : CloisterOptions
                Configuration options for boundary calculation.

        """
        self.x = x
        self.a = a
        self.options = options
        self.nfeats = x.shape[1]

    def compute_correlation(self: "Cloister") -> NDArray[np.double]:
        """
        Calculate the Pearson correlation coefficient for the dataset.

        Returns
        -------
            NDArray[np.double]:
                A matrix of Pearson correlation coefficients between each pair of
                features.

        """
        rho = np.zeros((self.nfeats, self.nfeats))
        pval = np.zeros((self.nfeats, self.nfeats))

        for i in range(self.nfeats):
            for j in range(self.nfeats):
                if i != j:
                    rho[i, j], pval[i, j] = pearsonr(self.x[:, i], self.x[:, j])
                else:
                    rho[i, j] = 0
                    pval[i, j] = 1

        rho[pval > self.options.p_val] = 0

        return rho

    def decimal_to_binary_matrix(self: "Cloister") -> NDArray[np.intc]:
        """
        Generate a matrix converting decimal numbers to binary representation.

        Returns
        -------
            NDArray[np.intc]:
                A matrix where each row represents a binary number as an array of bits.

        """
        decimals = np.arange(2**self.nfeats)
        binary_strings = [np.binary_repr(dec, width=self.nfeats) for dec in decimals]
        binary_matrix = np.array(
            [[int(bit) for bit in string] for string in binary_strings],
        )
        return binary_matrix[:, ::-1]

    def calculate_convex_hull(
        self: "Cloister",
        points: NDArray[np.double],
    ) -> NDArray[np.double]:
        """
        Calculate the convex hull of a set of points.

        Arguments:
        ---------
            points : NDArray[np.double]
                A 2D array of points.

        Returns:
        -------
            NDArray[np.double]:
                The vertices of the convex hull, or an empty array if an error occurs.

        """
        try:
            hull = ConvexHull(points)
            return points[hull.vertices, :]
        except QhullError as qe:
            print("QhullError: Encountered geometrical degeneracy:", str(qe))
            return np.array([])
        except ValueError as ve:
            print("ValueError: Imcompatible value encountered:", str(ve))
            return np.array([])

    def generate_boundaries(
        self: "Cloister",
        rho: NDArray[np.double],
    ) -> BoundaryResult:
        """
        Generate boundaries based on the correlation matrix and options.

        Arguments:
        ---------
            rho : NDArray[np.double]
                Correlation matrix.

        Returns:
        -------
            BoundaryResult:
                Contains the coordinates of boundary edges and indicators for which
                should be removed.

        """
        x_bnds = np.array([np.min(self.x, axis=0), np.max(self.x, axis=0)])
        idx = self.decimal_to_binary_matrix()
        ncomb = idx.shape[0]
        x_edge = np.zeros((ncomb, self.nfeats))
        remove = np.zeros(ncomb, dtype=bool)

        for i in range(ncomb):
            ind = np.ravel_multi_index(
                (idx[i], np.arange(self.nfeats)),
                (2, self.nfeats),
                order="F",
            )
            x_edge[i, :] = x_bnds.T.flatten()[ind]
            for j in range(self.nfeats):
                for k in range(self.nfeats):
                    if (
                        (rho[j, k] > self.options.c_thres
                        and np.sign(x_edge[i, j]) != np.sign(x_edge[i, k]))
                        or (rho[j, k] < -self.options.c_thres
                        and np.sign(x_edge[i, j]) == np.sign(x_edge[i, k]))
                    ):
                        remove[i] = True
                    if remove[i]:
                        break
                if remove[i]:
                    break

        return BoundaryResult(x_edge=x_edge, remove=remove)

    def run(self: "Cloister") -> CloisterOut:
        """
        Run boundary estimation analysis on the dataset.

        Steps:
        1. Compute the correlation matrix for the dataset.
        2. Use the correlation matrix to establish preliminary boundaries.
        3. Transform these points using a specified matrix and calculate their
           convex hull to determine the boundary 'z_edge'.
        4. Refine this boundary by filtering out points based on predefined criteria
           related to their correlation, creating a more selective boundary 'z_ecorr'.

        Returns
        -------
        CloisterOut:
            An instance containing two attributes:
            - `z_edge` : Numpy array representing the complete boundary
            - `z_ecorr`: Numpy array representing the refined boundary after filtering
                         out less relevant points.

        """
        print(
            "  -> CLOISTER is using correlation to estimate a boundary for the space.",
        )
        rho = self.compute_correlation()
        x_edge, remove = self.generate_boundaries(rho)
        z_edge = self.calculate_convex_hull(np.dot(x_edge, self.a.T))
        z_ecorr = self.calculate_convex_hull(np.dot(x_edge[~remove, :], self.a.T))

        if z_ecorr.size == 0:
            print("  -> The acceptable correlation threshold was too strict.")
            print("  -> The features are weakely correlated.")
            print("  -> Please consider increasing it.")
            z_ecorr = z_edge

        print("-----------------------------------------------------------------------")
        print("  -> CLOISTER has completed.")

        return CloisterOut(z_edge=z_edge, z_ecorr=z_ecorr)
