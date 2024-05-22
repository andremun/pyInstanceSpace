"""PILOT: Obtaining a two-dimensional projection.

Projecting Instances with Linearly Observable Trends (PILOT)
is a dimensionality reduction algorithm which aims to facilitate
the identification of relationships between instances and
algorithms by unveiling linear trends in the data, increasing
from one edge of the space to the opposite.

"""

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.spatial.distance import pdist
from scipy.optimize import fmin_l_bfgs_b, minimize
from scipy.linalg import eig
from numpy.random import default_rng
from typing import List

from matilda.data.model import PilotDataChanged, PilotOut
from matilda.data.option import PilotOptions


class Pilot:
    """See file docstring."""

    def __init__(self):
        pass

    def error_function(self, alpha: NDArray[np.float64], x_bar: NDArray[np.float64], n: int, m: int) -> float:
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
        A = alpha[:2 * n].reshape(2, n)
        B = alpha[2 * n:].reshape(m, 2)
        
        # Compute the approximation of x_bar
        x_bar_approx = x_bar[:, :n].T
        X_bar_approx = (B @ A @ x_bar_approx).T
        
        # Calculate the mean squared error
        mse = np.nanmean(np.nanmean((x_bar - X_bar_approx) ** 2, axis=1), axis=0)
        
        return mse

    @staticmethod
    def run(
        x: NDArray[np.double],
        y: NDArray[np.double],
        feat_labels: list[str],
        opts: PilotOptions,
    ) -> tuple[PilotDataChanged, PilotOut]:
        """Produce the final subset of features.

        opts.pilot.analytic determines whether the analytic (set as TRUE) or the
        numerical (set as FALSE) solution to be adopted.

        opts.pilot.n_tries number of iterations that the numerical solution is attempted.
        """

        n = x.shape[1]
        x_bar = np.hstack([x, y])
        m = x_bar.shape[1]
        hd = pdist(x.T)

        if opts.analytic:
            print("Solving analytically...")
            x_bar_t = x_bar.T
            covariance_matrix = x_bar_t @ x_bar_t.T
            eigenvalues, eigenvectors = eig(covariance_matrix)
            indices = np.argsort(-np.abs(eigenvalues))
            v = eigenvectors[:, indices[:2]]

            out_b = v[:n, :]
            out_c = v[n:, :].T
            x_r = np.linalg.pinv(x @ x.T)
            out_a = v.T @ x_bar_t @ x_r
            out_z = out_a @ x

            # Correct dimensions for x_hat computation
            x_hat = out_z.T @ np.vstack([out_b, out_c])
            x_hat = x_hat.T


            error = np.sum((x_bar - x_hat)**2)
            r2 = np.diag(np.corrcoef(x_bar, x_hat, rowvar=False)[:m, m:])**2
        
        else:
            print("Solving numerically...")
            if hasattr(opts, 'alpha') and opts.alpha is not None and opts.alpha.shape == (2 * m + 2 * n,):
                alpha = opts.alpha
            else:
                if hasattr(opts, 'x0') and opts.x0 is not None:
                    x0 = opts.x0
                else:
                    np.random.seed(0)
                    x0 = 2 * np.random.rand(2 * m + 2 * n, opts.n_tries) - 1
                
                results = [minimize(lambda a: Pilot.error_function(Pilot, a, x_bar, n, m), x0[:, i], method='BFGS') for i in range(x0.shape[1])]
                alphas = np.array([res.x for res in results])
                eoptim = np.array([res.fun for res in results])
                perf = np.array([np.corrcoef(hd, pdist(x @ res.x[:2 * n].reshape(2, n)))[0, 1] for res in results])

                best_index = np.argmax(perf)
                alpha = alphas[:, best_index]
            
            out_a = alpha[:2 * n].reshape(2, n)
            out_z = x @ out_a.T
            b = alpha[2 * n:].reshape(m, 2)

            x_hat = out_z.T @ np.vstack([out_b, out_c])
            x_hat = x_hat.T


            out_b = b[:n, :]
            out_c = b[n:, :].T
            error = np.sum((x_bar - x_hat)**2)
            r2 = (np.diag(np.corrcoef(x_bar, x_hat, rowvar=False)[:m, m:]) ** 2).astype(np.float64)

        # Ensure r2 is of floating type
        if r2.dtype != np.float64:
            r2 = r2.astype(np.float64)

        summary = pd.DataFrame(
            data=np.vstack([["Z_1", "Z_2"], np.round(out_a, 4)]),
            columns=["Feature"] + feat_labels
        )

        pout = PilotOut(X0=x0, alpha=alpha, eoptim=eoptim, perf=perf, a=out_a, z=out_z, c=out_c, b=out_b, error=error, r2=r2, summary=summary)
        pda = PilotDataChanged()

        return [pout, pda]

    