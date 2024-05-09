"""CLOISTER is using correlation to estimate a boundary for the space.

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
from matilda.data.option import CloisterOptions, MissingOptionsError, Options


class Cloister:
    """See file docstring."""

    x: NDArray[np.double]
    a: NDArray[np.double]
    opts: CloisterOptions
    nfeats: int

    def __init__(
        self,
        x: NDArray[np.double],
        a: NDArray[np.double],
        opts: Options,
    ) -> None:
        """Initialize the Cloister stage.

        Args
        ----
            x (NDArray): The feature matrix (instances x features) to process.
            a (NDArray): A projection matrix computed from Pilot
            opts (CloisterOptions): Configuration options for CLOISTER
        """
        self.x = x
        self.a = a
        self.opts = opts.cloister
        self.nfeats = x.shape[1]

    @staticmethod
    def run(
        x: NDArray[np.double],
        a: NDArray[np.double],
        opts: Options,
    ) -> CloisterOut:
        """Estimate a boundary for the space using correlation.

        Args
        ----
            x (NDArray[np.double]): The feature matrix (instances x features) to process
            a (NDArray[np.double]): A projection matrix computed from Pilot
            opts (CloisterOptions): Configuration options for CLOISTER

        Returns
        -------
            cloister_out: A structure containing z_edge and z_ecorr
        """
        print(
            "  -> CLOISTER is using correlation to estimate a boundary for the space.",
        )

        cloister = Cloister(x, a, opts)
        rho = cloister.compute_correlation()
        x_edge, remove = cloister.generate_boundaries(rho)
        z_edge = cloister.compute_convex_hull(np.dot(x_edge, cloister.a.T))
        z_ecorr = cloister.compute_convex_hull(np.dot(x_edge[~remove, :], cloister.a.T))

        if z_ecorr.size == 0:
            print("  -> The acceptable correlation threshold was too strict.")
            print("  -> The features are weakely correlated.")
            print("  -> Please consider increasing it.")
            z_ecorr = z_edge

        print("-----------------------------------------------------------------------")
        print("  -> CLOISTER has completed.")

        return CloisterOut(z_edge=z_edge, z_ecorr=z_ecorr)


    """
    % =========================================================================
    % SUBFUNCTIONS
    % =========================================================================
    """


    def compute_correlation(self) -> NDArray[np.double]:
        """Calculate the Pearson correlation coefficient for the dataset.

        Returns
        -------
            NDArray[np.double]: A matrix of Pearson correlation coefficients between
            each pair of features.
        """
        if self.opts.p_val is None:
            print("CLOISTER cannot be ran if CloisterOptions.p_val isn't set.")
            raise MissingOptionsError

        rho = np.zeros((self.nfeats, self.nfeats))
        pval = np.zeros((self.nfeats, self.nfeats))

        for i in range(self.nfeats):
            for j in range(self.nfeats):
                if i != j:
                    rho[i, j], pval[i, j] = pearsonr(self.x[:, i], self.x[:, j])
                else:
                    rho[i, j] = 0
                    pval[i, j] = 1

        rho[pval > self.opts.p_val] = 0

        return rho

    def decimal_to_binary_matrix(self) -> NDArray[np.intc]:
        """Generate a matrix converting decimal numbers to binary representation.

        Returns
        -------
            NDArray[np.intc]: A matrix where each row represents a binary number as an
            array of bits.
        """
        decimals = np.arange(2**self.nfeats)
        binary_strings = [np.binary_repr(dec, width=self.nfeats) for dec in decimals]
        binary_matrix = np.array(
            [[int(bit) for bit in string] for string in binary_strings],
        )
        return binary_matrix[:, ::-1]

    def compute_convex_hull(self, points: NDArray[np.double]) -> NDArray[np.double]:
        """Calculate the convex hull of a set of points.

        Args
        ----
            points (NDArray[np.double]): A 2d array of points

        Returns
        -------
            NDArray[np.double]: The vertices of the convex hull, or an empty array if
            an error occurs.
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

    def generate_boundaries(self, rho: NDArray[np.double]) -> BoundaryResult:
        """Generate boundaries based on the correlation matrix and options.

        Args
        ----
            rho (NDArray[np.double]): Correlation matrix.

        Returns
        -------
            BoundaryResult: Contains the coordinates of boundary edges and indicators
            for which should be removed.
        """
        if self.opts.c_thres is None:
            print("CLOISTER cannot be ran if CloisterOptions.c_thres isn't set.")
            raise MissingOptionsError

        # if no feature selection. then make a note in the boundary construction
        # that it won't work, because nfeats is so large that decimal to binary matrix
        # conversion wont be able to make a matrix.

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
                for k in range(j + 1, self.nfeats):
                    # Check for valid points give the correlation trend
                    if (
                        (rho[j, k] > self.opts.c_thres
                        and np.sign(x_edge[i, j]) != np.sign(x_edge[i, k]))
                        or (rho[j, k] < -self.opts.c_thres
                        and np.sign(x_edge[i, j]) == np.sign(x_edge[i, k]))
                    ):
                        remove[i] = True
                    if remove[i]:
                        break
                if remove[i]:
                    break

        return BoundaryResult(x_edge=x_edge, remove=remove)
