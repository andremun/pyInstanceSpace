"""PILOT: Obtaining a two-dimensional projection.

Projecting Instances with Linearly Observable Trends (PILOT)
is a dimensionality reduction algorithm which aims to facilitate
the identification of relationships between instances and
algorithms by unveiling linear trends in the data, increasing
from one edge of the space to the opposite.

"""

import time
from typing import Any, NamedTuple

import numpy as np
import pandas as pd
import scipy.linalg as la
import scipy.optimize as optim
from numpy.typing import NDArray
from scipy.spatial.distance import pdist
from scipy.stats import pearsonr

from matilda.data.options import PilotOptions
from matilda.stages.stage import Stage


class PilotInput(NamedTuple):
    """Inputs for the Pilot stage.

    Attributes
    ----------
    x : NDArray[np.double]
        The feature matrix (instances x features) to process.
    y : NDArray[np.double]
        The data points for the selected feature.
    feat_labels : list[str]
        List feature names.
    options: PilotOptions
        The options enabled for the Pilot Class
    """

    x: NDArray[np.double]
    y: NDArray[np.double]
    feat_labels: list[str]
    pilot_options: PilotOptions


class PilotOutput(NamedTuple):
    """Outputs for the Pilot stage.

    Attributes
    ----------
    X0 : NDArray[np.double] | None
        TODO: This
    alpha : NDArray[np.double] | None
        TODO: This
    eoptim : NDArray[np.double] | None
        TODO: This
    perf : NDArray[np.double] | None
        TODO: This
    a : NDArray[np.double]
        TODO: This
    z : NDArray[np.double]
        TODO: This
    c : NDArray[np.double]
        TODO: This
    b : NDArray[np.double]
        TODO: This
    error : NDArray[np.double]
        TODO: This
    r2 : NDArray[np.double]
        TODO: This
    summary : pd.DataFrame
        TODO: This
    """

    X0: NDArray[np.double] | None
    alpha: NDArray[np.double] | None
    eoptim: NDArray[np.double] | None
    perf: NDArray[np.double] | None
    a: NDArray[np.double]
    z: NDArray[np.double]
    c: NDArray[np.double]
    b: NDArray[np.double]
    error: NDArray[np.double]
    r2: NDArray[np.double]
    summary: pd.DataFrame


class PilotStage(Stage[PilotInput, PilotOutput]):
    """Class for PILOT stage."""

    def __init__(
        self,
        x: NDArray[np.double],
        y: NDArray[np.double],
        feat_labels: list[str],
    ) -> None:
        """Initialize the Pilot stage.

        The Initialize functon is used to create a Pilot class.

        Args
        ----
            x (NDArray[np.double]): The feature matrix (instances x features) to
                process.
            y (NDArray[np.double]): The data points for the selected feature
            feat_labels (list[str]): List feature names

        Returns
        -------
            None
        """
        self.x = x
        self.y = y
        self.feat_labels = feat_labels

    @staticmethod
    def _inputs() -> type[PilotInput]:
        return PilotInput

    @staticmethod
    def _outputs() -> type[PilotOutput]:
        return PilotOutput

    @staticmethod
    def _run(inputs: PilotInput) -> PilotOutput:
        """Implement all the code in and around this class in buildIS.

        Args
        -------
        options : PilotOptions
            The options enabled for the Pilot Class

        Return
        -------
        X0
            NDArray[np.double] | None  # not sure about the dimensions
        alpha
            NDArray[np.double] | None
        eoptim
            NDArray[np.double] | None
        perf
            NDArray[np.double] | None
        a
            NDArray[np.double]
        z
            NDArray[np.double]
        c
            NDArray[np.double]
        b
            NDArray[np.double]
        error
            NDArray[np.double]  # or just the double
        r2
            NDArray[np.double]
        summary
            pd.DataFrame

        """
        start = time.perf_counter()
        output = PilotStage.pilot(
            inputs.x,
            inputs.y,
            inputs.feat_labels,
            inputs.pilot_options,
        )
        print(f"PILOT took {time.perf_counter() - start} seconds to run.")
        return output

    @staticmethod
    def _pilot_print(
        a: Any,  # noqa: ANN401
        _do_output: bool,
    ) -> None:
        if _do_output:
            print(a)

    @staticmethod
    def pilot(
        x: NDArray[np.double],
        y: NDArray[np.double],
        feat_labels: list[str],
        options: PilotOptions,
        _do_output: bool = True,
    ) -> PilotOutput:
        """Run the PILOT dimensionality reduction algorithm.

        Args
        -------
        x : NDArray[double]
            The feature matrix (instances x features) to process.
        y: NDArray[double]
            The data points for the selected feature.
        feat_labels :  list[str]
            List feature names.
        options : PilotOptions
            The options enabled for the Pilot Class.

        Return
        -------
        PilotOutput
            Outputs from the Pilot stage.

        """
        n = x.shape[1]
        x_bar = np.concatenate((x, y), axis=1)
        m = x_bar.shape[1]
        hd = pdist(x).T

        # Following parameters are not generated in the matlab code
        # when solving analytically
        x0 = None
        alpha = None
        eoptim = None
        perf = None

        # Analytical solution
        if options.analytic:
            out_a, out_z, out_c, out_b, error, r2 = PilotStage.analytic_solve(
                x,
                x_bar,
                n,
                m,
            )

        # Numerical solution
        else:
            if options.alpha is not None and options.alpha.shape == (2 * m + 2 * n, 1):
                PilotStage._pilot_print(
                    " -> PILOT is using a pre-calculated solution.",
                    _do_output,
                )
                alpha = options.alpha
            else:
                if options.x0 is not None:
                    PilotStage._pilot_print(
                        "  -> PILOT is using a user defined starting points"
                        " for BFGS.",
                        _do_output,
                    )
                    x0 = options.x0
                else:
                    PilotStage._pilot_print(
                        "  -> PILOT is using random starting points for BFGS.",
                        _do_output,
                    )
                    rng = np.random.default_rng(seed=0)
                    x0 = 2 * rng.random((2 * m + 2 * n, options.n_tries)) - 1

                alpha = np.zeros((2 * m + 2 * n, options.n_tries))
                eoptim = np.zeros(options.n_tries)
                perf = np.zeros(options.n_tries)

                idx, alpha, eoptim, perf = PilotStage.numerical_solve(
                    x,
                    hd,
                    x0,
                    x_bar,
                    n,
                    m,
                    alpha,
                    eoptim,
                    perf,
                    options,
                    _do_output,
                )

            out_a = alpha[: 2 * n, idx].reshape(2, n)
            out_z = x @ out_a.T
            b = alpha[2 * n :, idx].reshape(m, 2)
            x_hat = out_z @ b.T
            out_c = b[n + 1 : m, :].T
            out_b = b[:n, :]
            error = np.sum((x_bar - x_hat) ** 2)
            r2 = np.diag(np.corrcoef(x_bar, x_hat) ** 2).astype(np.double)

        if options.analytic:
            summary = pd.DataFrame(out_a)
            summary.rename(
                columns={
                    summary.columns[num]: feat_labels[num]
                    for num in range(len(feat_labels))
                },
                inplace=True,
            )
        else:
            summary = pd.DataFrame(out_a, columns=feat_labels)

        row_labels = ["Z_{1}", "Z_{2}"]
        rldf = pd.DataFrame(row_labels)
        summary = rldf.join(summary)

        if alpha is not None and x0 is not None:
            alpha = alpha.astype(np.float16)
            x_init: NDArray[np.double] = x0
            pout = PilotOutput(
                x_init,
                alpha,
                eoptim,
                perf,
                out_a,
                out_z,
                out_c,
                out_b,
                error,
                r2,
                summary,
            )
        else:
            pout = PilotOutput(
                x0,
                alpha,
                eoptim,
                perf,
                out_a,
                out_z,
                out_c,
                out_b,
                error,
                r2,
                summary,
            )

        PilotStage._pilot_print(
            "-------------------------------------------------------------------------",
            _do_output,
        )
        PilotStage._pilot_print(
            "  -> PILOT has completed. The projection matrix A is:",
            _do_output,
        )
        PilotStage._pilot_print(out_a, _do_output)

        return pout

    @staticmethod
    def analytic_solve(
        x: NDArray[np.double],
        x_bar: NDArray[np.double],
        n: int,
        m: int,
        _do_output: bool = True,
    ) -> tuple[
        NDArray[np.double],
        NDArray[np.double],
        NDArray[np.double],
        NDArray[np.double],
        NDArray[np.double],
        NDArray[np.float16],
    ]:
        """Solve the projection problem analytically.

        Args:
        -------
        x : NDArray[np.double]
            The feature matrix (instances x features) to process.
        x_bar : NDArray[np.double]
            Combined matrix of X and Y.
        n : int
            Number of original features.
        m : int
            Total number of features including appended Y.

        Returns:
        -------
        NDArray[np.double]
            Matrix A.
        NDArray[np.double]
            Matrix B.
        NDArray[np.double]
            Matrix C.
        NDArray[np.double]
            Matrix Z.
        NDArray[np.double]
            The mean squared error between x_bar and its
            low-dimensional approximation.
        NDArray[np.float16]
            The coefficient of determination between x_bar
            and its low-dimensional approximation.
        """
        PilotStage._pilot_print(
            "-------------------------------------------------------------------------",
            _do_output,
        )
        PilotStage._pilot_print(
            "  -> PILOT is solving analytically the projection problem.",
            _do_output,
        )
        PilotStage._pilot_print(
            "-------------------------------------------------------------------------",
            _do_output,
        )
        x_bar = x_bar.T

        x = x.T

        covariance_matrix = np.dot(x_bar, x_bar.T)

        d, v = la.eig(covariance_matrix)

        indices = np.argsort(np.abs(d))
        indices = indices[::-1]
        v = -1 * v[:, indices[:2]]

        out_b = v[:n, :]

        out_c = v[n:m, :].T

        x_transpose = x.T
        xx_transpose = np.dot(x, x.T)
        xx_transpose_inverse = np.linalg.inv(xx_transpose)

        x_r = np.dot(x_transpose, xx_transpose_inverse)

        out_a = v.T @ x_bar @ x_r
        out_z = out_a @ x

        # Correct dimensions for x_hat computation
        bz = np.dot(out_b, out_z)
        cz = np.dot(out_c.T, out_z)
        x_hat = np.vstack((bz, cz))

        out_z = out_z.T

        error = np.sum((x_bar - x_hat) ** 2)
        r2 = np.diag(np.corrcoef(x_bar.T, x_hat.T, rowvar=False)[:m, m:]) ** 2

        a: NDArray[np.double] = out_a
        z: NDArray[np.double] = out_z
        c: NDArray[np.double] = out_c
        b: NDArray[np.double] = out_b
        err: NDArray[np.double] = error
        corref: NDArray[np.float16] = r2.astype(np.float16)

        return (a, z, c, b, err, corref)

    @staticmethod
    def numerical_solve(
        x: NDArray[np.double],
        hd: NDArray[np.double],
        x0: NDArray[np.double],
        x_bar: NDArray[np.double],
        n: int,
        m: int,
        alpha: NDArray[np.double],
        eoptim: NDArray[np.double],
        perf: NDArray[np.double],
        opts: PilotOptions,
        _do_output: bool = True,
    ) -> tuple[int, NDArray[np.double], NDArray[np.double], NDArray[np.double]]:
        """Solve the projection problem numerically.

        Args:
        -------
        x : NDArray[np.double]
            The feature matrix (instances x features)
            to process.
        x0 : NDArray[np.double]
            Initial guess for the solution.
        x_bar : NDArray[np.double]
            Combined matrix of X and Y.
        n : int
            Number of original features.
        m : int
            Total number of features including appended Y.
        alpha : NDArray[np.double]
            Flattened parameter vector containing
            both A (2*n size) and B (m*2 size) matrices.
        eoptim : NDArray[np.double]
            Optimized error function.
        perf : NDArray[np.double]
            Optimized performance matrix.
        opts : PilotOptions
            Configuration options for PILOT.

        Returns:
        -------
        NDArray[np.double]
            Flattened parameter vector containing
            both A (2*n size) and B (m*2 size) matrices.
        NDArray[np.double]
            Optimized error function.
        NDArray[np.double]
            Optimized performance matrix.
        int
            The index for the most optimal array indices
        """
        PilotStage._pilot_print(
            "-------------------------------------------------------------------------",
            _do_output,
        )
        PilotStage._pilot_print(
            "  -> PILOT is solving numerically the projection problem.",
            _do_output,
        )
        PilotStage._pilot_print(
            "  -> This may take a while. Trials will not be run sequentially.",
            _do_output,
        )
        PilotStage._pilot_print(
            "-------------------------------------------------------------------------",
            _do_output,
        )

        for i in range(opts.n_tries):
            initial_guess = x0[:, i]
            result = optim.fmin_bfgs(
                PilotStage.error_function,
                initial_guess,
                args=(x_bar, n, m),
                full_output=True,
                disp=False,
            )

            (xopts, fopts, _, _, _, _, _) = result
            alpha[:, i] = xopts
            eoptim[i] = fopts

            aux = alpha[:, i].astype(np.float64)
            a = aux[0 : 2 * n].reshape(2, n)
            z = np.dot(x, a.T)

            perf[i], _ = pearsonr(hd, pdist(z))
            idx = np.argmax(perf).astype(int)
            PilotStage._pilot_print(f"Pilot has completed trial {i + 1}", _do_output)

            al: NDArray[np.float16] = alpha.astype(np.float16)
            ept: NDArray[np.double] = eoptim
            prf: NDArray[np.double] = perf

        return idx, al, ept, prf

    @staticmethod
    def error_function(
        alpha: NDArray[np.float64],
        x_bar: NDArray[np.float64],
        n: int,
        m: int,
    ) -> float:
        """Error function used for numerical optimization in the PILOT algorithm.

        Args:
        -------
        alpha : NDArray[np.float64]
            Flattened parameter vector containing
            both A (2*n size) and B (m*2 size) matrices.
        x_bar : NDArray[np.float64]
            Combined matrix of X and Y.
        n : int
            Number of original features.
        m : int
            Total number of features including appended Y.

        Returns:
        -------
        float
            The mean squared error between x_bar and its
            low-dimensional approximation.
        """
        a = alpha[: 2 * n].reshape(2, n)
        b = alpha[2 * n :].reshape(m, 2)

        # Compute the approximation of x_bar
        x_bar_approx = x_bar[:, :n].T
        x_bar_approx = (b @ a @ x_bar_approx).T

        return float(
            np.nanmean(np.nanmean((x_bar - x_bar_approx) ** 2, axis=1), axis=0),
        )
