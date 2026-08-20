# SPDX-License-Identifier: LicenseRef-PolyForm-Noncommercial-1.0.0
# Copyright (c) 2024-2026 Mario Andrés Muñoz
"""PILOT: Obtaining a two- or three-dimensional projection.

Projecting Instances with Linearly Observable Trends (PILOT)
is a dimensionality reduction algorithm which aims to facilitate
the identification of relationships between instances and
algorithms by unveiling linear trends in the data, increasing
from one edge of the space to the opposite.

"""

import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, NamedTuple

import numpy as np
import pandas as pd
import scipy.linalg as la
import scipy.optimize as optim
from loguru import logger
from numpy.typing import NDArray
from scipy.spatial.distance import pdist
from scipy.stats import mode, pearsonr

from instancespace.data.options import (
    PILOT_3D_DIMS,
    GeneralOptions,
    ParallelOptions,
    PilotOptions,
)
from instancespace.stages.pilot_viewpoint import (
    PilotViewpointResult,
    pilot_viewpoint,
)
from instancespace.stages.pilot_viewpoint import (
    _default_starts as _matlab_default_starts,
)
from instancespace.stages.stage import Stage


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
    parallel_options : ParallelOptions
        The parallel options, specifying whether to run in parallel and the
        number of cores - used to parallelise the numerical solver's `ntries`
        restart loop.
    general_options : GeneralOptions
        General options (e.g. the RNG seed), not specific to any one stage.
    y_bin : NDArray[np.bool_]
        Binary matrix indicating good algorithm performance per instance,
        used only when `pilot_options.adjust_rotation` is set.
    """

    x: NDArray[np.double]
    y: NDArray[np.double]
    feat_labels: list[str]
    pilot_options: PilotOptions
    parallel_options: ParallelOptions
    general_options: GeneralOptions
    y_bin: NDArray[np.bool_]


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
    pilot_summary: pd.DataFrame
    viewpoint: PilotViewpointResult | None = None


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
        output = PilotStage.pilot(
            inputs.x,
            inputs.y,
            inputs.feat_labels,
            inputs.pilot_options,
            general_options=inputs.general_options,
            y_bin=inputs.y_bin,
            parallel_options=inputs.parallel_options,
        )
        if inputs.pilot_options.dims != PILOT_3D_DIMS:
            return output

        viewpoint = pilot_viewpoint(
            output.z,
            inputs.y,
            view_groups=inputs.pilot_options.view_groups,
            n_tries=inputs.pilot_options.n_tries,
            x0=inputs.pilot_options.x0,
            parallel_options=inputs.parallel_options,
        )
        return output._replace(viewpoint=viewpoint)

    @staticmethod
    def _pilot_print(
        a: Any,  # noqa: ANN401
        _do_output: bool,
    ) -> None:
        if _do_output:
            logger.info(f"[PILOT] {a}")

    @staticmethod
    def _pilot_print_detail(
        a: Any,  # noqa: ANN401
        _do_output: bool,
        general_options: GeneralOptions,
    ) -> None:
        if _do_output and general_options.verbose:
            logger.debug(f"[PILOT] {a}")

    @staticmethod
    def _validate_numerical_option_shapes(
        options: PilotOptions,
        expected_rows: int,
    ) -> None:
        """Validate the highest-precedence solver matrix for this dataset.

        MATLAB gives a valid ``precalcAlpha`` precedence over ``X0``. Unlike
        MATLAB, Python rejects an explicitly supplied but wrongly shaped
        ``precalcAlpha`` instead of silently falling through to ``X0``.
        """
        if options.precalc_alpha is not None:
            expected_shape = (expected_rows, 1)
            if options.precalc_alpha.shape != expected_shape:
                msg = (
                    "opts.pilot.precalcAlpha must have shape "
                    f"{expected_shape}. Got {options.precalc_alpha.shape}."
                )
                raise ValueError(msg)
            return
        if options.x0 is not None and options.x0.shape[0] != expected_rows:
            msg = (
                f"opts.pilot.x0 must have {expected_rows} rows. "
                f"Got {options.x0.shape}."
            )
            raise ValueError(msg)

    @staticmethod
    def _default_numerical_starts(
        parameter_count: int,
        n_tries: int,
    ) -> NDArray[np.double]:
        """Return MATLAB `rng('default')` starts without touching global state.

        PILOT and PILOTviewpoint share the same MT19937 seed-5489, legacy
        53-bit conversion, and column-major matrix-fill contract.
        """
        return _matlab_default_starts(parameter_count, n_tries)

    @staticmethod
    def _pack_solution(
        a: NDArray[np.double],
        b: NDArray[np.double],
    ) -> NDArray[np.double]:
        """Pack PILOT matrices using MATLAB's column-major vector layout."""
        return np.concatenate(
            (
                np.asarray(a, dtype=np.double).reshape(-1, order="F"),
                np.asarray(b, dtype=np.double).reshape(-1, order="F"),
            ),
        )

    @staticmethod
    def _unpack_solution(
        theta: NDArray[np.double],
        dims: int,
        n: int,
        m: int,
    ) -> tuple[NDArray[np.double], NDArray[np.double]]:
        """Unpack MATLAB's ``[A(:); B(:)]`` numerical solution vector."""
        flat_theta = np.asarray(theta, dtype=np.double).reshape(-1)
        expected_size = dims * (n + m)
        if flat_theta.size != expected_size:
            msg = (
                "PILOT solution vector must contain "
                f"{expected_size} values. Got {flat_theta.size}."
            )
            raise ValueError(msg)
        split = dims * n
        a = flat_theta[:split].reshape((dims, n), order="F")
        b = flat_theta[split:].reshape((m, dims), order="F")
        return a, b

    @staticmethod
    def pilot(
        x: NDArray[np.double],
        y: NDArray[np.double],
        feat_labels: list[str],
        options: PilotOptions,
        general_options: GeneralOptions,
        y_bin: NDArray[np.bool_] | None = None,
        _do_output: bool = True,
        parallel_options: ParallelOptions | None = None,
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
        y_bin : NDArray[np.bool_] | None
            Binary matrix (instances x algorithms) indicating good algorithm
            performance. Required only when `options.adjust_rotation` is set.
        parallel_options : ParallelOptions | None
            Whether (and how much) to parallelise the numerical solver's
            `ntries` restart loop. `None` (the default for direct callers
            that don't pass one) runs the restarts sequentially, same as
            `flag=False`.

        Return
        -------
        PilotOutput
            Outputs from the Pilot stage.

        """
        n = x.shape[1]
        x_bar = np.concatenate((x, y), axis=1)
        m = x_bar.shape[1]
        dims = options.dims
        hd = pdist(x).T

        # Following parameters are not generated in the matlab code
        # when solving analytically or via PLS
        x0 = None
        alpha = None
        eoptim = None
        perf = None

        # Partial Least Squares (MATLAB's method='pls', F2/#262) - an
        # alternative to the standard analytic/numeric solvers below, not a
        # variant of either. Checked first since it ignores `analytic`
        # entirely, matching MATLAB's own opts.method dispatch order
        # (core/PILOT.m).
        if options.method == "pls":
            out_a, out_z, out_c, out_b, error, r2 = PilotStage.pls_solve(
                x,
                y,
                x_bar,
                m,
                dims=dims,
                _do_output=_do_output,
            )

        # Analytical solution
        elif options.analytic and np.linalg.matrix_rank(x) == n:
            out_a, out_z, out_c, out_b, error, r2 = PilotStage.analytic_solve(
                x,
                x_bar,
                n,
                m,
                options.cost_weight,
                dims=dims,
                _do_output=_do_output,
            )

        # Numerical solution
        else:
            if options.analytic:
                logger.warning(
                    "Feature matrix rank-deficient; falling back to numerical "
                    "solution.",
                )
            expected_rows = dims * (m + n)
            PilotStage._validate_numerical_option_shapes(options, expected_rows)
            if options.precalc_alpha is not None:
                PilotStage._pilot_print(
                    " -> PILOT is using a pre-calculated solution.",
                    _do_output,
                )
                alpha = options.precalc_alpha
                idx = 0
            else:
                if options.x0 is not None:
                    PilotStage._pilot_print(
                        "  -> PILOT is using a user defined starting points"
                        " for BFGS.",
                        _do_output,
                    )
                    x0 = options.x0
                    n_tries = x0.shape[1]
                else:
                    PilotStage._pilot_print(
                        "  -> PILOT is using random starting points for BFGS.",
                        _do_output,
                    )
                    x0 = PilotStage._default_numerical_starts(
                        expected_rows,
                        options.n_tries,
                    )
                    n_tries = options.n_tries

                alpha = np.zeros((expected_rows, n_tries))
                eoptim = np.zeros(n_tries)
                perf = np.zeros(n_tries)

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
                    general_options,
                    _do_output,
                    parallel_options,
                )

            out_a, b = PilotStage._unpack_solution(alpha[:, idx], dims, n, m)
            out_z = x @ out_a.T
            x_hat = out_z @ b.T
            # MATLAB's 1-based ``B(n+1:m, :)`` starts at the first
            # performance row.  In Python's 0-based indexing that is
            # ``b[n:m]``; ``n + 1`` silently dropped the first algorithm.
            out_c = b[n:m, :].T
            out_b = b[:n, :]
            error = np.sum((x_bar - x_hat) ** 2)
            # rowvar=False + the [:m, m:] block mirrors analytic_solve()'s (correct)
            # R^2 computation: per-column correlation between x_bar and x_hat, not
            # the row-wise (instance-to-instance) correlation corrcoef's default
            # would otherwise compute.
            r2 = (np.diag(np.corrcoef(x_bar, x_hat, rowvar=False)[:m, m:]) ** 2).astype(
                np.double,
            )

        if options.adjust_rotation:
            out_z, out_a = PilotStage._maybe_adjust_rotation(
                out_z,
                out_a,
                y_bin,
                _do_output,
            )

        summary = pd.DataFrame(np.round(out_a, 4), columns=feat_labels)
        row_labels = [f"Z_{{{index}}}" for index in range(1, dims + 1)]
        rldf = pd.DataFrame(row_labels)
        summary = rldf.join(summary)

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
    def _maybe_adjust_rotation(
        out_z: NDArray[np.double],
        out_a: NDArray[np.double],
        y_bin: NDArray[np.bool_] | None,
        _do_output: bool,
    ) -> tuple[NDArray[np.double], NDArray[np.double]]:
        """Apply `adjust_rotation()` when there are poorly-solved instances.

        Split out of `pilot()` purely to keep that method's branch count
        down; see `adjust_rotation()` for what the rotation itself does.
        """
        if y_bin is None:
            msg = "PILOT cannot adjust rotation without y_bin."
            raise ValueError(msg)
        bad_instances = PilotStage._bad_instances(y_bin)
        if not bad_instances.any():
            PilotStage._pilot_print(
                "  -> PILOT could not adjust the IS rotation: there are no "
                "poorly-solved instances.",
                _do_output,
            )
            return out_z, out_a
        PilotStage._pilot_print(
            "  -> PILOT is adjusting the IS rotation so poorly-solved "
            "instances face a consistent direction.",
            _do_output,
        )
        out_z, rot = PilotStage.adjust_rotation(out_z, bad_instances)
        out_a = rot @ out_a
        return out_z, out_a

    @staticmethod
    def _bad_instances(y_bin: NDArray[np.bool_]) -> NDArray[np.bool_]:
        """Flag instances where most algorithms perform poorly.

        Per-instance majority vote across `y_bin` (instances x algorithms):
        an instance counts as "bad" when the mode of its good/bad flags
        across algorithms is `False`, i.e. most algorithms are not good for
        it.
        """
        majority_good, _ = mode(y_bin.astype(int), axis=1, keepdims=True)
        bad_instances: NDArray[np.bool_] = majority_good[:, 0] == 0
        return bad_instances

    @staticmethod
    def adjust_rotation(
        z: NDArray[np.double],
        bad_instances: NDArray[np.bool_],
        theta: float = 135.0,
    ) -> tuple[NDArray[np.double], NDArray[np.double]]:
        """Rotate the 2D projection so poorly-solved instances face `theta`.

        Ported from PyISpace's `pilot.adjust_rotation()`
        (gitlab.com/ita-ml/pyispace). Rotation preserves all pairwise
        distances in `z`, so it only changes the space's visual orientation
        - error, R2, and footprint areas are unaffected. The centroid of the
        instances flagged by `bad_instances` is rotated to sit at `theta`
        degrees (135 = upper-left quadrant, matching PyISpace's default), so
        that similar datasets come out consistently oriented across runs.

        Args
        ----
        z : NDArray[np.double]
            The 2D projection (instances x 2) to rotate.
        bad_instances : NDArray[np.bool_]
            Boolean mask selecting the instances whose centroid should be
            placed at `theta`.
        theta : float
            Target angle, in degrees, for the bad-instance centroid.

        Returns
        -------
        NDArray[np.double]
            The rotated projection, same shape as `z`.
        NDArray[np.double]
            The 2x2 rotation matrix applied.
        """
        centroid_bad = np.mean(z[bad_instances], axis=0)[::-1]
        theta_rad = np.radians(theta) - np.arctan2(*centroid_bad)
        rot = np.array(
            (
                (np.cos(theta_rad), -np.sin(theta_rad)),
                (np.sin(theta_rad), np.cos(theta_rad)),
            ),
        )
        z_rot = np.dot(rot, z.T)
        return z_rot.T, rot

    @staticmethod
    def analytic_solve(
        x: NDArray[np.double],
        x_bar: NDArray[np.double],
        n: int,
        m: int,
        cost_weight: float = 1.0,
        _do_output: bool = True,
        dims: int = 2,
    ) -> tuple[
        NDArray[np.double],
        NDArray[np.double],
        NDArray[np.double],
        NDArray[np.double],
        NDArray[np.double],
        NDArray[np.double],
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
        cost_weight : float
            Scalar performance-reconstruction weight (MATLAB's costWeight).
            Scales the performance block relative to the feature block
            before the eigendecomposition; 1.0 weights both equally and
            reproduces the pre-cost_weight behaviour exactly.
        dims : int
            Output projection dimensionality, either 2 or 3.

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
        NDArray[np.double]
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

        # Scale the performance block (rows n:m, since x_bar is m x instances
        # here) by sqrt(cost_weight) before the eigendecomposition, matching
        # MATLAB's Xbarw(:,n+1:m) = sqrt(costWeight) * Xbarw(:,n+1:m) - a
        # separate weighted copy, not a mutation of x_bar itself. MATLAB
        # uses this weighted matrix for both the eigendecomposition and A;
        # the unweighted matrix remains necessary for error/R2.
        x_bar_weighted = x_bar.copy()
        x_bar_weighted[n:m, :] = np.sqrt(cost_weight) * x_bar_weighted[n:m, :]

        covariance_matrix = np.dot(x_bar_weighted, x_bar_weighted.T)

        d, v = la.eig(covariance_matrix)

        indices = np.argsort(np.abs(d))
        indices = indices[::-1]
        v = -1 * v[:, indices[:dims]]

        out_b = v[:n, :]

        # Undo the weighting so C is expressed in the original Y units,
        # matching MATLAB's out.C = V(n+1:m,:)./sqrt(costWeight).
        out_c = v[n:m, :].T / np.sqrt(cost_weight)

        x_transpose = x.T
        xx_transpose = np.dot(x, x.T)
        # pinv (rather than inv) keeps this well-defined for rank-deficient X,
        # matching inv's result whenever X is full rank.
        xx_transpose_inverse = np.linalg.pinv(xx_transpose)

        x_r = np.dot(x_transpose, xx_transpose_inverse)

        out_a = v.T @ x_bar_weighted @ x_r
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
        corref: NDArray[np.double] = r2.astype(np.double)

        return (a, z, c, b, err, corref)

    @staticmethod
    def pls_solve(
        x: NDArray[np.double],
        y: NDArray[np.double],
        x_bar: NDArray[np.double],
        m: int,
        dims: int = 2,
        _do_output: bool = True,
    ) -> tuple[
        NDArray[np.double],
        NDArray[np.double],
        NDArray[np.double],
        NDArray[np.double],
        NDArray[np.double],
        NDArray[np.double],
    ]:
        """Solve the projection problem via Partial Least Squares.

        MATLAB's `method='pls'` alternative (core/PILOT.m, F2/#262) to the
        analytic/numeric eigen-decomposition solvers above - a genuinely
        different algorithm, not a variant of either. `dims` matches
        MATLAB's own `plsregress(X, Y, d)` call.

        Implements the SIMPLS routine used by MATLAB R2026a's `plsregress`.
        Both inputs are mean-centred but not variance-scaled, matching the
        default MATLAB call in `PILOT.m`.

        Args
        ----
        x : NDArray[np.double]
            The feature matrix (instances x features) to process.
        y : NDArray[np.double]
            The performance matrix (instances x algorithms).
        x_bar : NDArray[np.double]
            Combined matrix of X and Y.
        m : int
            Total number of features including appended Y.
        dims : int
            Output projection dimensionality, either 2 or 3.

        Returns
        -------
        NDArray[np.double]
            Matrix A (dims x n), the projection matrix:
            Z = (X - mean(X)) @ A.T.
        NDArray[np.double]
            Matrix Z (instances x dims), the projected coordinates.
        NDArray[np.double]
            Matrix C (dims x q), the performance-block reconstruction
            matrix.
        NDArray[np.double]
            Matrix B (n x dims), the feature-block reconstruction matrix.
        NDArray[np.double]
            The sum of squared reconstruction error between x_bar and its
            low-dimensional approximation.
        NDArray[np.double]
            The per-column coefficient of determination between x_bar and
            its low-dimensional approximation.
        """
        PilotStage._pilot_print(
            "-------------------------------------------------------------------------",
            _do_output,
        )
        PilotStage._pilot_print(
            "  -> PILOT is using partial least squares (opts.pilot.method='pls').",
            _do_output,
        )
        PilotStage._pilot_print(
            "-------------------------------------------------------------------------",
            _do_output,
        )

        max_components = min(x.shape[0] - 1, x.shape[1])
        if dims <= 0 or dims > max_components:
            msg = (
                "PILOT PLS components must be between 1 and "
                f"{max_components}. Got {dims}."
            )
            raise ValueError(msg)

        x_mean = x.mean(axis=0)
        y_mean = y.mean(axis=0)
        x_centered = x - x_mean
        y_centered = y - y_mean

        out_b, y_loadings, out_z, weights = PilotStage._simpls(
            x_centered,
            y_centered,
            dims,
        )
        out_a = weights.T
        out_c = y_loadings.T

        b_c_stack = np.vstack([out_b, out_c.T])
        x_hat = out_z @ b_c_stack.T + np.concatenate([x_mean, y_mean])
        error = np.sum((x_bar - x_hat) ** 2)
        r2 = np.diag(np.corrcoef(x_bar, x_hat, rowvar=False)[:m, m:]) ** 2

        a: NDArray[np.double] = out_a
        z: NDArray[np.double] = out_z
        c: NDArray[np.double] = out_c
        b: NDArray[np.double] = out_b
        err: NDArray[np.double] = error
        corref: NDArray[np.double] = r2.astype(np.double)

        return (a, z, c, b, err, corref)

    @staticmethod
    def _simpls(
        x_centered: NDArray[np.double],
        y_centered: NDArray[np.double],
        n_components: int,
    ) -> tuple[
        NDArray[np.double],
        NDArray[np.double],
        NDArray[np.double],
        NDArray[np.double],
    ]:
        """Port MATLAB R2026a `plsregress`'s private SIMPLS routine.

        MATLAB permits a degenerate component norm to cascade into non-finite
        outputs. Python deliberately raises a named ``ValueError`` instead so
        an unusable projection fails at its source with an actionable message.
        """
        n_instances, n_features = x_centered.shape
        n_targets = y_centered.shape[1]
        x_loadings = np.zeros((n_features, n_components), dtype=np.double)
        y_loadings = np.zeros((n_targets, n_components), dtype=np.double)
        x_scores = np.zeros((n_instances, n_components), dtype=np.double)
        weights = np.zeros((n_features, n_components), dtype=np.double)

        # Orthonormal basis for the span of successive X loadings. MATLAB
        # uses it to deflate X0'Y0 without explicitly deflating X0 or Y0.
        loading_basis = np.zeros((n_features, n_components), dtype=np.double)
        covariance = x_centered.T @ y_centered

        for component in range(n_components):
            left_vectors, singular_values, right_vectors_t = np.linalg.svd(
                covariance,
                full_matrices=False,
            )
            x_weight = left_vectors[:, 0]
            y_weight = right_vectors_t[0, :]
            singular_value = singular_values[0]

            x_score = x_centered @ x_weight
            score_norm = np.linalg.norm(x_score)
            if score_norm == 0 or not np.isfinite(score_norm):
                msg = (
                    f"PILOT SIMPLS component {component + 1} has a zero or "
                    "non-finite X-score norm."
                )
                raise ValueError(msg)
            x_score = x_score / score_norm
            x_loading = x_centered.T @ x_score
            y_loading = singular_value * y_weight / score_norm

            x_loadings[:, component] = x_loading
            y_loadings[:, component] = y_loading
            x_scores[:, component] = x_score
            weights[:, component] = x_weight / score_norm

            # MATLAB repeats modified Gram-Schmidt twice for numerical
            # stability before applying both deflation projections.
            basis_vector = x_loading.copy()
            for _ in range(2):
                for previous in range(component):
                    previous_basis = loading_basis[:, previous]
                    basis_vector -= (previous_basis @ basis_vector) * previous_basis
            loading_norm = np.linalg.norm(basis_vector)
            if loading_norm == 0 or not np.isfinite(loading_norm):
                msg = (
                    f"PILOT SIMPLS component {component + 1} has a zero or "
                    "non-finite orthogonalized X-loading norm."
                )
                raise ValueError(msg)
            basis_vector /= loading_norm
            loading_basis[:, component] = basis_vector

            covariance -= np.outer(basis_vector, basis_vector @ covariance)
            current_basis = loading_basis[:, : component + 1]
            covariance -= current_basis @ (current_basis.T @ covariance)

        # MATLAB orthogonalizes Y scores only when that optional output is
        # requested. PILOT consumes X scores/loadings and stats.W, so no Y
        # scores are formed here.
        return x_loadings, y_loadings, x_scores, weights

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
        general_options: GeneralOptions,
        _do_output: bool = True,
        parallel_options: ParallelOptions | None = None,
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
            both A (dims*n size) and B (m*dims size) matrices.
        eoptim : NDArray[np.double]
            Optimized error function.
        perf : NDArray[np.double]
            Optimized performance matrix.
        opts : PilotOptions
            Configuration options for PILOT.
        parallel_options : ParallelOptions | None
            Whether (and how much) to parallelise the `ntries` restarts
            across OS processes (matching MATLAB's `parfor`). `None` or
            `flag=False` runs the restarts sequentially. Ignored - falls
            back to sequential - when already running inside another
            process pool's worker (see `multiprocessing.parent_process()`
            check below): SIFTED's GA fitness function calls into `pilot()`
            from inside its own `ProcessPoolExecutor` workers, and opening a
            second, nested pool there would reintroduce MATLAB's
            nested-parfor-inside-GA bug. In practice this never triggers
            today, since SIFTED always calls `pilot()` with `analytic=True`
            (bypassing this numerical branch entirely) - this check guards
            against that invariant ever changing silently.

        Returns:
        -------
        NDArray[np.double]
            Flattened parameter vector containing
            both A (dims*n size) and B (m*dims size) matrices.
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

        n_tries = x0.shape[1]
        use_pool = (
            parallel_options is not None
            and parallel_options.flag
            and n_tries > 1
            and multiprocessing.parent_process() is None
        )

        if use_pool:
            # Narrows for mypy; use_pool being True already implies this.
            assert parallel_options is not None
            n_workers = max(
                1,
                min(parallel_options.n_cores, n_tries, os.cpu_count() or 1),
            )
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                futures = {
                    executor.submit(
                        PilotStage._solve_one_trial,
                        x0[:, i],
                        x,
                        hd,
                        x_bar,
                        n,
                        m,
                        opts.cost_weight,
                        opts.dims,
                    ): i
                    for i in range(n_tries)
                }
                for future in as_completed(futures):
                    i = futures[future]
                    xopts, fopts, perf_i = future.result()
                    alpha[:, i] = xopts
                    eoptim[i] = fopts
                    perf[i] = perf_i
                    PilotStage._pilot_print_detail(
                        f"Pilot has completed trial {i + 1}",
                        _do_output,
                        general_options,
                    )
        else:
            for i in range(n_tries):
                xopts, fopts, perf_i = PilotStage._solve_one_trial(
                    x0[:, i],
                    x,
                    hd,
                    x_bar,
                    n,
                    m,
                    opts.cost_weight,
                    opts.dims,
                )
                alpha[:, i] = xopts
                eoptim[i] = fopts
                perf[i] = perf_i
                PilotStage._pilot_print_detail(
                    f"Pilot has completed trial {i + 1}",
                    _do_output,
                    general_options,
                )

        idx = int(np.argmax(perf))
        return idx, alpha, eoptim, perf

    @staticmethod
    def _solve_one_trial(
        initial_guess: NDArray[np.double],
        x: NDArray[np.double],
        hd: NDArray[np.double],
        x_bar: NDArray[np.double],
        n: int,
        m: int,
        cost_weight: float,
        dims: int,
    ) -> tuple[NDArray[np.double], float, float]:
        """Run one BFGS restart of the numerical PILOT solver.

        Split out of `numerical_solve()` so a single restart is a
        self-contained, picklable unit of work that can be submitted to a
        `ProcessPoolExecutor` - each restart only depends on its own
        starting point, not on any other restart's progress, so which one
        "wins" (highest `perf`) doesn't depend on run order.
        """
        result = optim.fmin_bfgs(
            PilotStage.error_function,
            initial_guess,
            args=(x_bar, n, m, cost_weight, dims),
            full_output=True,
            disp=False,
        )
        xopts, fopts, _, _, _, _, _ = result
        a, _ = PilotStage._unpack_solution(xopts, dims, n, m)
        z = np.dot(x, a.T)
        perf_i, _ = pearsonr(hd, pdist(z))
        return xopts, float(fopts), float(perf_i)

    @staticmethod
    def error_function(
        alpha: NDArray[np.float64],
        x_bar: NDArray[np.float64],
        n: int,
        m: int,
        cost_weight: float = 1.0,
        d: int = 2,
    ) -> float:
        """Error function used for numerical optimization in the PILOT algorithm.

        Args:
        -------
        alpha : NDArray[np.float64]
            Flattened parameter vector containing
            both A (d*n size) and B (m*d size) matrices.
        x_bar : NDArray[np.float64]
            Combined matrix of X and Y.
        n : int
            Number of original features.
        m : int
            Total number of features including appended Y.
        cost_weight : float
            Scalar weight applied to the performance columns' squared error
            (MATLAB's `pilotErrorFcn` costWeight). 1.0 weights every column
            equally, reproducing the pre-cost_weight behaviour exactly.
        d : int
            Output projection dimensionality, either 2 or 3.

        Returns:
        -------
        float
            The mean squared error between x_bar and its
            low-dimensional approximation.
        """
        a, b = PilotStage._unpack_solution(alpha, d, n, m)

        # Compute the approximation of x_bar
        x_bar_approx = x_bar[:, :n].T
        x_bar_approx = (b @ a @ x_bar_approx).T

        sq_err = (x_bar - x_bar_approx) ** 2
        weights = np.ones(m, dtype=np.double)
        weights[n:m] = cost_weight

        column_losses = np.nanmean(sq_err * weights, axis=0)
        return float(np.nanmean(column_losses))
