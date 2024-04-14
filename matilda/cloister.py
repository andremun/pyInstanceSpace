"""
CLOISTER is using correlation to estimate a boundary for the space.

The function uses the correlation between the features and the performance of the
algorithms to estimate a boundary for the space. It constructs edges based on the
correlation between the features. The function then uses these edges to construct
a convex hull, providing a boundary estimate for the dataset.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.spatial import ConvexHull, QhullError
from scipy.stats import pearsonr

from matilda.data.model import BoundaryResult, CloisterOut
from matilda.data.option import CloisterOptions

filepath = "matilda/trials/timetable/" # TODO: remove after completion

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
        Initialize the Cloister object with dataset, auxiliary data, and options.

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
        Calculate the Pearson correlation coefficient matrix for the dataset.

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

        np.savetxt(filepath + "rho_python.csv", rho, delimiter=",") #TODO: remove later
        np.savetxt(filepath + "pval_python.csv", pval, delimiter=",") #TODO: remove

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

        np.savetxt(filepath + "Xbnd_python.csv", x_bnds, delimiter=",") #TODO: remove
        np.savetxt(filepath + "pre_idx_python.csv", idx, delimiter=",") #TODO: remove

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

        np.savetxt(filepath + "post_Xedge_python.csv", x_edge, delimiter=",") #TODO: r
        np.savetxt(filepath + "post_remove_python.csv", remove, delimiter=",") #TODO: r

        return BoundaryResult(x_edge=x_edge, remove=remove)

    def run(self: "Cloister") -> CloisterOut:
        """Run analysis to estimate the boundary of given dataset."""
        raise NotImplementedError



# TODO: Remove below functions after completion
def load_csv_to_numpy(filename: str) -> NDArray[np.double]:
    """Load csv to numpy."""
    df = pd.read_csv(filepath + filename, header=None)
    return df.to_numpy(dtype=np.double)


def load_json(filepath: str) -> dict:
    """Load JSON file."""
    path = Path(filepath)
    with path.open("r") as file:
        return json.load(file)


if __name__ == "__main__":
    x = load_csv_to_numpy("input_X.csv")
    y = load_csv_to_numpy("input_A.csv")
    option = CloisterOptions(p_val=0.05, c_thres=0.7)
    cloister = Cloister(x, y, option)
    rho = cloister.compute_correlation()
    x_edge, remove = cloister.generate_boundaries(rho)
