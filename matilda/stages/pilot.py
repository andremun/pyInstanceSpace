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
import scipy.optimize as optim
from scipy.stats import pearsonr
from scipy.linalg import eig
from numpy.random import default_rng
from typing import List

from matilda.data.model import PilotDataChanged, PilotOut
from matilda.data.option import PilotOptions


class Pilot:
    """See file docstring."""

    def __init__(self):
        pass

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
        x_bar = np.concatenate((x, y), axis=1)
        m = x_bar.shape[1]
        hd = pdist(x).T
        print(hd)

        if opts.analytic:
            print("Solving analytically...")
            x_bar = x_bar.T
            x = x.T
            covariance_matrix = x_bar @ x_bar.T
            eigenvalues, eigenvectors = eig(covariance_matrix)
            indices = np.argsort(-np.abs(eigenvalues))
            indices = indices[::-1]
            v = eigenvectors[:, indices[:2]]

            out_b = v[:n, :]
            out_c = v[n:, :].T
            x_r = np.linalg.pinv(x @ x.T)
            out_a = v.T @ x_bar @ x_r
            out_z = out_a @ x

            # Correct dimensions for x_hat computation
            x_hat = out_z.T @ np.vstack([out_b, out_c])
            x_hat = x_hat.T


            error = np.sum((x_bar - x_hat)**2)
            r2 = np.diag(np.corrcoef(x_bar, x_hat, rowvar=False)[:m, m:])**2

        # Numerical solution
        else:
            if (hasattr(opts, "alpha") and opts.alpha is not None
                and opts.alpha.shape == (2 * m + 2 * n, 1)):
                print(" -> PILOT is using a pre-calculated solution.")
                alpha = opts.alpha
            else:
                if hasattr(opts, "x0") and opts.x0 is not None:
                    print("  -> PILOT is using a user defined starting points"
                        " for BFGS.")
                    x0 = opts.x0
                else:
                    print("  -> PILOT is using random starting points for BFGS.")
                    rng = np.random.default_rng(0)
                    x0 = 2 * rng.random((2 * m + 2 * n, opts.n_tries)) - 1

                alpha = np.zeros((2 * m + 2 * n, opts.n_tries))
                eoptim = np.zeros(opts.n_tries)
                perf = np.zeros(opts.n_tries)

                print("-------------------------------------------------------------------------")
                print("  -> PILOT is solving numerically the projection problem.")
                print("  -> This may take a while. Trials will not be"
                      "run sequentially.")
                print("-------------------------------------------------------------------------")

                for i in range(opts.n_tries):
                    initial_guess = x0[:, i]
                    result = optim.fmin_bfgs(Pilot.error_function, initial_guess,
                                            args=(x_bar, n, m), full_output=True,
                                            disp=False)
                    print(len(result))
                    (xopts, fopts, _, _, _, _, _) = result
                    alpha[:, i] = xopts
                    eoptim[i] = fopts

                    print(Pilot.error_function(xopts,x_bar,n,m))


                    aux = alpha[:, i]
                    A = aux[0:2*n].reshape(2, n)
                    Z = np.dot(x, A.T)

                    perf[i], _ = pearsonr(hd, pdist(Z))
                    idx = np.argmax(perf)
                    print(f"Pilot has completed trial {i + 1}")

            out_a = alpha[:2 * n, idx].reshape(2, n)
            out_z = x @ out_a.T
            b = alpha[2 * n:, idx].reshape(m, 2)
            x_hat = out_z @ b.T
            out_c = b[n+1:m, :].T
            out_b = b[:n, :]
            error = np.sum((x_bar - x_hat)**2)
            r2 = np.diag(np.corrcoef(x_bar, x_hat) ** 2)

        if r2.dtype != np.float64:
            r2 = r2.astype(np.float64)

        data = np.round(out_a, 4)
        row_labels = ['Z_{1}','Z_{2}']
        summary = pd.DataFrame(index=[None] + row_labels, columns=[None] + feat_labels)

        summary.iloc[1:, 0] = row_labels
        summary.iloc[0, 1:] = feat_labels
        summary.iloc[1:, 1:] = data

        pout = PilotOut(X0=x0, alpha=alpha, eoptim=[eoptim], perf=[perf], a=out_a, z=out_z, c=out_c, b=out_b, error=error, r2=r2, summary=summary)
        pda = PilotDataChanged()

        return [pout, pda]
