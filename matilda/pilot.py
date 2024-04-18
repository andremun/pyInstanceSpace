"""
PILOT: Obtaining a two-dimensional projection.

Projecting Instances with Linearly Observable Trends (PILOT)
is a dimensionality reduction algorithm which aims to facilitate
the identification of relationships between instances and
algorithms by unveiling linear trends in the data, increasing
from one edge of the space to the opposite.

"""

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import minimize
from scipy.linalg import eig

from matilda.data.model import PilotOut
from matilda.data.option import PilotOptions


def pilot(
    x: NDArray[np.double],
    y: NDArray[np.double],
    feat_labels: list[str],
    opts: PilotOptions,
) -> PilotOut:
    """
    Produce the final subset of features using a two-dimensional projection
    to uncover linear trends in the data.

    Args:
    x : NDArray[np.float64] -- input feature matrix.
    y : NDArray[np.float64] -- additional data or target features to be included in the projection.
    feat_labels : list[str] -- labels for the features in x.
    opts : PilotOptions -- options for running the PILOT algorithm, specifying whether to
                           use an analytic or numerical approach and the number of tries
                           for the numerical solution.

    Returns:
    PilotOut -- A custom data class instance containing the output of the PILOT algorithm.
    """

    n = x.shape[1]
    x_bar = np.hstack([x, y])
    m = x_bar.shape[1]
    hd = pdist(x.T)

    if opts.pilot.analytic:
        print("Solving analytically...")
        x_bar_t = x_bar.T
        covariance_matrix = x_bar_t @ x_bar_t.T
        eigenvalues, eigenvectors = eig(covariance_matrix)
        indicies = np.argsort(-np.abs(eigenvalues))
        v = eigenvectors[:, indicies[:2]]

        out_b = v[:n, :]
        out_c = v[n:, :].T
        x_r = np.linalg.pinv(x @ x.T)
        out_a = v.T @ x_bar_t @ x_r
        out_z = out_a @ x
        x_hat = np.vstack([out_b @ out_z, out_c.T @ out_z])

        error = np.sum((x_bar - x_hat)**2)
        r2 = np.diag(np.corrcoef(x_bar, x_hat, rowvar=False)[:m, m:])**2
    
    else:
        print("Solving numerically...")
        if hasattr(opts.pilot, 'alpha') and opts.pilot.alpha is not None and opts.pilot.alpha.shape == (2 * m + 2 * n,):
            alpha = opts.pilot.alpha
        else:
            if hasattr(opts.pilot, 'x0') and opts.pilot.x0 is not None:
                x0 = opts.pilot.x0
            else:
                np.random.seed(0)
                x0 = 2 * np.random(2 * m + 2 * n, opts.pilot.ntries) - 1
            
            results = [minimize(lambda a: error_function(a, x_bar, n, m), x0[:, i], method='BFGS') for i in range(x0.shape[1])]
            alphas = np.array([res.x for res in results])
            errors = np.array([res.fun for res in results])

            best_index = np.argmin(errors)
            alpha = alphas[:, best_index]
        
        out_a = alpha[:2 * n].reshape(2, n)
        out_z = x @ out_a.T
        b = alpha[2 * n:].reshape(m, 2)
        x_hat = out_z @ b.T

        out_b = b[:n, :]
        out_c = b[n:, :].T
        error = np.sum((x_bar - x_hat)**2)
        r2 = np.diag(np.corrcoef(x_bar, x_hat, rowvar=False)[:m, m:]) ** 2

    
    return PilotOut(A=out_a, B=out_b, C=out_c, Z=out_z, error=error, R2=r2)

def error_function(alpha: NDArray[np.float64], x_bar: NDArray[np.float64], n: int, m: int) -> float:
    """
    Error function used for numerical optimization in the PILOT algorithm.
    
    Args:
    alpha : NDArray[np.float64] -- Flattened parameter vector containing both A (2*n size)
                                   and B (m*2 size) matrices.
    x_bar : NDArray[np.float64] -- Combined matrix of X and Y.
    n : int -- Number of original features.
    m : int -- Total number of features including appended Y.
    
    Returns:
    float -- The mean squared error between x_bar and its low-dimensional approximation.
    """

    a = alpha[:2 * n].reshape(2, n)
    b = alpha[2 * n:].reshape(m, 2)
    x_bar_approx = b @ a @ x_bar[:, :n].T
    mse = np.nanmean((x_bar.T - x_bar_approx.T) ** 2)
    return mse