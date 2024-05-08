"""PILOT: Obtaining a two-dimensional projection.

Projecting Instances with Linearly Observable Trends (PILOT)
is a dimensionality reduction algorithm which aims to facilitate
the identification of relationships between instances and
algorithms by unveiling linear trends in the data, increasing
from one edge of the space to the opposite.

"""

import numpy as np
from numpy.typing import NDArray
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import fmin_l_bfgs_b
from scipy.linalg import eig
from numpy.random import default_rng
from typing import List

from matilda.data.model import PilotOut
from matilda.data.option import PilotOptions


class Pilot:
    """See file docstring."""

    def __init__(self, analytic: bool = False, ntries: int = 1):
        self.analytic = analytic
        self.ntries = ntries

    @staticmethod
    def error_function(self, alpha, Xbar, n, m):
        A = alpha[:2*n].reshape(2, n)
        B = alpha[2*n:2*n+2*m].reshape(m, 2)
        return np.nanmean((Xbar - (B @ A @ Xbar[:n, :].T).T) ** 2)

    def run(self, x: np.ndarray, y: np.ndarray, feat_labels: List[str], opts: PilotOptions) -> PilotOut:
        X = x.T
        Y = y.T
        Xbar = np.vstack([X, Y])
        n = X.shape[1]
        m = Xbar.shape[0]

        if opts.analytic:
            XbarT = Xbar.T
            V, D = eig(Xbar @ XbarT)
            idx = np.abs(D).argsort()[::-1]
            V = V[:, idx[:2]]
            out_B = V[:n, :]
            out_C = V[n:m, :].T
            Xr = np.linalg.pinv(X @ X.T)
            out_A = V.T @ Xbar @ Xr
            out_Z = out_A @ X
            Xhat = np.vstack([out_B @ out_Z, out_C.T @ out_Z])
            error = np.sum((Xbar - Xhat) ** 2)
            out_R2 = np.diag(np.corrcoef(Xbar, Xhat, rowvar=False)) ** 2
            summary = [[''] + feat_labels, ['Z1', 'Z2']] + [list(np.round(out_A, 4))]
        else:
            if opts.alpha is not None:
                alpha = opts.alpha
            else:
                rng = default_rng()
                X0 = opts.X0 if opts.X0 is not None else rng.uniform(-1, 1, size=(2*m+2*n, opts.ntries))
                alpha = np.zeros((2*m+2*n, opts.ntries))
                eoptim = np.zeros(opts.ntries)
                perf = np.zeros(opts.ntries)

                for i in range(opts.ntries):
                    res = fmin_l_bfgs_b(self.error_function, X0[:, i], args=(Xbar, n, m), approx_grad=True)
                    alpha[:, i] = res[0]
                    eoptim[i] = res[1]
                    A = alpha[:2*n, i].reshape(2, n)
                    Z = X @ A.T
                    perf[i] = np.corrcoef(pdist(Z.T), pdist(X.T))[0, 1]

                idx = np.argmax(perf)
                out_A = alpha[:2*n, idx].reshape(2, n)
                out_Z = X @ out_A.T
                out_B = alpha[2*n:2*n+2*m, idx].reshape(m, 2)
                out_C = out_B[n:m, :].T
                Xhat = out_Z @ out_B.T
                error = np.sum((Xbar - Xhat) ** 2)
                out_R2 = np.diag(np.corrcoef(Xbar, Xhat, rowvar=False)) ** 2
                summary = [[''] + feat_labels, ['Z1', 'Z2']] + [list(np.round(out_A, 4))]

        return PilotOut(A=out_A, B=out_B, C=out_C, Z=out_Z, Xhat=Xhat, error=error, R2=out_R2, summary=summary)