# SPDX-License-Identifier: LicenseRef-PolyForm-Noncommercial-1.0.0
# Copyright (c) 2024-2026 Mario Andrés Muñoz
"""Optimal 2D camera views of a three-dimensional PILOT projection."""

import multiprocessing
import os
from collections.abc import Sequence
from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import NamedTuple, cast

import numpy as np
import scipy.optimize as optim
from numpy.typing import NDArray
from scipy.spatial.distance import pdist

from instancespace.data.options import ParallelOptions

_PROJECTION_DIMS = 3
_VIEW_DIMS = 2
_ARRAY_DIMS = 2
_MIN_TRIALS = 1
_MIN_CORRELATION_VALUES = 2
_ORTHOGONALITY_WEIGHT = 0.2
_DEFAULT_N_TRIES = 10
_MAX_ITERATIONS = 30_000
_FUNCTION_TOLERANCE = 1e-20
_DEFAULT_SEED = 42

FloatArray = NDArray[np.float64]
ViewGroups = Sequence[Sequence[int]]


@dataclass(frozen=True, slots=True)
class PilotViewpointResult:
    """Immutable optimized viewpoints for zero-based algorithm groups.

    Attributes
    ----------
    groups : tuple[tuple[int, ...], ...]
        Resolved zero-based algorithm indices, one tuple per viewpoint.
    a : tuple[NDArray[np.float64], ...]
        Read-only 2-by-3 view matrix for each group.
    azimuth : tuple[float, ...]
        Camera azimuth angles in radians.
    elevation : tuple[float, ...]
        Camera elevation angles in radians.
    """

    groups: tuple[tuple[int, ...], ...]
    a: tuple[FloatArray, ...]
    azimuth: tuple[float, ...]
    elevation: tuple[float, ...]

    def __post_init__(self) -> None:
        """Copy and freeze every stored view matrix."""
        if not (
            len(self.groups) == len(self.a) == len(self.azimuth) == len(self.elevation)
        ):
            msg = "Viewpoint result fields must contain one value per group."
            raise ValueError(msg)

        frozen_views: list[FloatArray] = []
        for view in self.a:
            matrix = np.array(view, dtype=np.float64, copy=True)
            if matrix.shape != (_VIEW_DIMS, _PROJECTION_DIMS):
                msg = "Every PILOT viewpoint matrix must have shape (2, 3)."
                raise ValueError(msg)
            matrix.setflags(write=False)
            frozen_views.append(matrix)
        object.__setattr__(self, "a", tuple(frozen_views))


class _TrialResult(NamedTuple):
    """Normalized view and topology score from one BFGS restart."""

    a: FloatArray
    score: float


class _FunctionChangeStopper:
    """Stop SciPy BFGS using MATLAB's relative objective-change rule."""

    def __init__(self, initial_value: float, tolerance: float) -> None:
        self._previous_value = initial_value
        self._tolerance = tolerance

    def __call__(self, intermediate_result: optim.OptimizeResult) -> None:
        """Raise ``StopIteration`` when the objective change is small enough."""
        current_value = float(intermediate_result.fun)
        previous_value = self._previous_value
        self._previous_value = current_value
        if not np.isfinite(previous_value) or not np.isfinite(current_value):
            return

        relative_threshold = self._tolerance * (1.0 + abs(previous_value))
        if abs(previous_value - current_value) < relative_threshold:
            raise StopIteration


def pilot_viewpoint(
    z: NDArray[np.double],
    y: NDArray[np.double],
    *,
    view_groups: ViewGroups | None = None,
    n_tries: int = _DEFAULT_N_TRIES,
    x0: NDArray[np.double] | None = None,
    parallel_options: ParallelOptions | None = None,
    seed: int | None = _DEFAULT_SEED,
) -> PilotViewpointResult:
    """Find optimal 2D camera viewpoints for a 3D PILOT projection.

    This is the R2026a ``PILOTviewpoint`` numerical engine. Each algorithm
    group jointly fits a 2-by-3 view matrix and a group-specific linear
    performance reconstruction. Restart selection maximizes preservation of
    pairwise distances in ``z``.

    Parameters
    ----------
    z : NDArray[np.double]
        Three-dimensional PILOT coordinates, shape ``(instances, 3)``.
    y : NDArray[np.double]
        Algorithm performance matrix, shape ``(instances, algorithms)``.
    view_groups : Sequence[Sequence[int]] | None
        Zero-based algorithm groups. ``None`` or an empty outer sequence uses
        one group containing every algorithm.
    n_tries : int
        Number of deterministic BFGS starts when ``x0`` is not accepted.
    x0 : NDArray[np.double] | None
        Optional start matrix. For a group of size ``g``, it is accepted only
        at shape ``(6 + 2*g, trials)`` with at least one trial. Its column count
        then replaces ``n_tries`` for that group. Other shapes fall back to
        deterministic starts.
    parallel_options : ParallelOptions | None
        Optional process-level parallelism across independent restarts.
    seed : int | None
        Stage RNG seed for generated starts. MATLAB's default is 42.

    Returns
    -------
    PilotViewpointResult
        View matrices and camera angles for the resolved groups.
    """
    z_array, y_array = _validate_data(z, y)
    groups = _resolve_groups(view_groups, y_array.shape[1])
    high_dimensional_distances = pdist(z_array)

    views: list[FloatArray] = []
    azimuths: list[float] = []
    elevations: list[float] = []

    for group in groups:
        group_y = y_array[:, group]
        parameter_count = _VIEW_DIMS * _PROJECTION_DIMS + _VIEW_DIMS * len(group)
        starts = _resolve_starts(x0, parameter_count, n_tries, seed)
        trials = _run_trials(
            starts,
            z_array,
            group_y,
            high_dimensional_distances,
            parallel_options,
        )
        best = _select_best_trial(trials)
        azimuth, elevation = _view_angles(best.a)
        views.append(best.a)
        azimuths.append(azimuth)
        elevations.append(elevation)

    return PilotViewpointResult(
        groups=groups,
        a=tuple(views),
        azimuth=tuple(azimuths),
        elevation=tuple(elevations),
    )


def _validate_data(
    z: NDArray[np.double],
    y: NDArray[np.double],
) -> tuple[FloatArray, FloatArray]:
    """Validate the projection/performance boundary and return float arrays."""
    z_array = np.asarray(z, dtype=np.float64)
    y_array = np.asarray(y, dtype=np.float64)
    if z_array.ndim != _ARRAY_DIMS:
        msg = "Z must be a two-dimensional array."
        raise ValueError(msg)
    if y_array.ndim != _ARRAY_DIMS:
        msg = "Y must be a two-dimensional array."
        raise ValueError(msg)
    if z_array.shape[0] != y_array.shape[0]:
        msg = (
            "Z and Y must have the same number of rows "
            f"(got {z_array.shape[0]} and {y_array.shape[0]})."
        )
        raise ValueError(msg)
    if z_array.shape[1] != _PROJECTION_DIMS:
        msg = (
            "Z must be an (instances, 3) PILOT projection "
            f"(got {z_array.shape[1]} columns)."
        )
        raise ValueError(msg)
    if y_array.shape[1] == 0:
        msg = "Y must contain at least one algorithm column."
        raise ValueError(msg)
    return z_array, y_array


def _resolve_groups(
    view_groups: ViewGroups | None,
    n_algorithms: int,
) -> tuple[tuple[int, ...], ...]:
    """Resolve and validate zero-based algorithm groups."""
    if view_groups is None or len(view_groups) == 0:
        return (tuple(range(n_algorithms)),)

    groups: list[tuple[int, ...]] = []
    for group_number, raw_group in enumerate(view_groups):
        if isinstance(raw_group, str | bytes):
            msg = f"View group {group_number} must be a sequence of indices."
            raise ValueError(msg)
        group = tuple(raw_group)
        if not group:
            msg = f"View group {group_number} must not be empty."
            raise ValueError(msg)

        validated: list[int] = []
        for raw_index in group:
            if isinstance(raw_index, bool | np.bool_) or not isinstance(
                raw_index,
                int | np.integer,
            ):
                msg = (
                    f"View group {group_number} contains a non-integer "
                    f"algorithm index: {raw_index!r}."
                )
                raise ValueError(msg)
            index = int(raw_index)
            if index < 0 or index >= n_algorithms:
                msg = (
                    f"View group {group_number} index {index} is outside the "
                    f"zero-based range 0..{n_algorithms - 1}."
                )
                raise ValueError(msg)
            validated.append(index)
        groups.append(tuple(validated))
    return tuple(groups)


def _validate_n_tries(n_tries: int) -> None:
    """Require a positive integer restart count."""
    if isinstance(n_tries, bool) or not isinstance(n_tries, int | np.integer):
        msg = "n_tries must be a positive integer."
        raise ValueError(msg)
    if n_tries < _MIN_TRIALS:
        msg = "n_tries must be at least 1."
        raise ValueError(msg)


def _resolve_starts(
    x0: NDArray[np.double] | None,
    parameter_count: int,
    n_tries: int,
    seed: int | None = _DEFAULT_SEED,
) -> FloatArray:
    """Accept an exact MATLAB-shaped X0 or create deterministic starts."""
    if x0 is not None:
        starts = np.asarray(x0)
        if (
            starts.ndim == _ARRAY_DIMS
            and starts.shape[0] == parameter_count
            and starts.shape[1] >= _MIN_TRIALS
            and np.issubdtype(starts.dtype, np.number)
            and not np.issubdtype(starts.dtype, np.complexfloating)
        ):
            return starts.astype(np.float64, copy=True)
    _validate_n_tries(n_tries)
    return _default_starts(parameter_count, n_tries, seed)


def _default_starts(
    parameter_count: int,
    n_tries: int,
    seed: int | None = _DEFAULT_SEED,
) -> FloatArray:
    """Return MATLAB-twister starts without changing global RNG state."""
    rng = np.random.RandomState(seed)
    values = rng.random_sample(parameter_count * n_tries)
    matlab_matrix = values.reshape((parameter_count, n_tries), order="F")
    return np.asarray(2.0 * matlab_matrix - 1.0, dtype=np.float64)


def _viewpoint_objective(
    theta: NDArray[np.double],
    z: FloatArray,
    y: FloatArray,
    orthogonality_weight: float = _ORTHOGONALITY_WEIGHT,
) -> float:
    """Return reconstruction MSE plus normalized-row orthogonality penalty."""
    group_size = y.shape[1]
    view_parameter_count = _VIEW_DIMS * _PROJECTION_DIMS
    view = np.asarray(theta[:view_parameter_count], dtype=np.float64).reshape(
        _VIEW_DIMS,
        _PROJECTION_DIMS,
        order="F",
    )
    reconstruction = np.asarray(
        theta[view_parameter_count:],
        dtype=np.float64,
    ).reshape(group_size, _VIEW_DIMS, order="F")
    predicted = (reconstruction @ view @ z.T).T

    with np.errstate(invalid="ignore"):
        column_mse = np.nanmean(np.square(y - predicted), axis=0)
        reconstruction_error = np.nanmean(column_mse)

    unit_view = _normalise_view_rows(view)
    penalty = orthogonality_weight * abs(float(np.dot(unit_view[0], unit_view[1])))
    return float(reconstruction_error + penalty)


def _normalise_view_rows(view: FloatArray) -> FloatArray:
    """Normalize both viewing-plane directions independently."""
    norms = np.linalg.norm(view, axis=1)
    safe_norms = np.maximum(norms, np.finfo(np.float64).eps)
    return np.asarray(view / safe_norms[:, None], dtype=np.float64)


def _solve_one_trial(
    initial_guess: FloatArray,
    z: FloatArray,
    y: FloatArray,
    high_dimensional_distances: FloatArray,
) -> _TrialResult:
    """Optimize, normalize, and score one independent BFGS restart."""
    initial_value = _viewpoint_objective(initial_guess, z, y)
    function_change_stopper = _FunctionChangeStopper(
        initial_value,
        _FUNCTION_TOLERANCE,
    )
    result = optim.minimize(
        _viewpoint_objective,
        initial_guess,
        args=(z, y),
        method="BFGS",
        callback=function_change_stopper,
        options={"disp": False, "maxiter": _MAX_ITERATIONS},
    )
    theta = np.asarray(result.x, dtype=np.float64)
    view = theta[: _VIEW_DIMS * _PROJECTION_DIMS].reshape(
        _VIEW_DIMS,
        _PROJECTION_DIMS,
        order="F",
    )
    normalized_view = _normalise_view_rows(view)
    projected_distances = pdist(z @ normalized_view.T)
    score = _distance_correlation(
        high_dimensional_distances,
        projected_distances,
    )
    return _TrialResult(normalized_view, score)


def _distance_correlation(first: FloatArray, second: FloatArray) -> float:
    """Return Pearson correlation, or NaN when either distance vector is constant."""
    if first.size != second.size or first.size < _MIN_CORRELATION_VALUES:
        return float("nan")
    first_centered = first - np.mean(first)
    second_centered = second - np.mean(second)
    denominator = float(
        np.sqrt(np.dot(first_centered, first_centered))
        * np.sqrt(np.dot(second_centered, second_centered)),
    )
    if denominator == 0.0 or not np.isfinite(denominator):
        return float("nan")
    return float(np.dot(first_centered, second_centered) / denominator)


def _run_trials(
    starts: FloatArray,
    z: FloatArray,
    y: FloatArray,
    high_dimensional_distances: FloatArray,
    parallel_options: ParallelOptions | None,
) -> tuple[_TrialResult, ...]:
    """Run BFGS restarts sequentially or in a safe process pool."""
    n_trials = starts.shape[1]
    use_pool = (
        parallel_options is not None
        and parallel_options.flag
        and n_trials > 1
        and multiprocessing.parent_process() is None
    )
    if not use_pool:
        return tuple(
            _solve_one_trial(starts[:, trial], z, y, high_dimensional_distances)
            for trial in range(n_trials)
        )

    if parallel_options is None:  # pragma: no cover - narrowed by use_pool
        msg = "Parallel options are required when process execution is enabled."
        raise RuntimeError(msg)
    worker_count = max(
        1,
        min(parallel_options.n_cores, n_trials, os.cpu_count() or 1),
    )
    completed: list[_TrialResult | None] = [None] * n_trials
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        futures: dict[Future[_TrialResult], int] = {
            executor.submit(
                _solve_one_trial,
                starts[:, trial],
                z,
                y,
                high_dimensional_distances,
            ): trial
            for trial in range(n_trials)
        }
        for future in as_completed(futures):
            completed[futures[future]] = future.result()
    return tuple(cast(_TrialResult, trial) for trial in completed)


def _select_best_trial(trials: tuple[_TrialResult, ...]) -> _TrialResult:
    """Select the highest topology score, using the first trial if all are NaN."""
    scores = np.asarray([trial.score for trial in trials], dtype=np.float64)
    comparable_scores = np.where(np.isnan(scores), -np.inf, scores)
    return trials[int(np.argmax(comparable_scores))]


def _view_angles(view: FloatArray) -> tuple[float, float]:
    """Convert the viewing-plane normal to MATLAB cart2sph angles in radians."""
    direction = np.cross(view[0], view[1])
    azimuth = float(np.arctan2(direction[1], direction[0]))
    horizontal = float(np.hypot(direction[0], direction[1]))
    elevation = float(np.arctan2(direction[2], horizontal))
    return azimuth, elevation
