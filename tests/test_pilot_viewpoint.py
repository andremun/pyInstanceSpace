"""Contract tests for the isolated MATLAB PILOTviewpoint engine."""

# ruff: noqa: PLR2004, SLF001

import multiprocessing
import os
from collections.abc import Callable
from concurrent.futures import Future
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
import scipy.optimize as optim
from numpy.typing import NDArray
from scipy.spatial.distance import pdist

import instancespace.stages.pilot_viewpoint as viewpoint
from instancespace.data.options import ParallelOptions

FloatArray = NDArray[np.float64]


@pytest.fixture()
def sample_data() -> tuple[FloatArray, FloatArray]:
    """Return a nondegenerate 3D projection and three performance columns."""
    z = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.2],
            [0.0, 1.0, 0.4],
            [0.3, 0.2, 1.0],
            [1.0, 1.0, 0.7],
        ],
        dtype=np.float64,
    )
    y = np.column_stack((z[:, 0], z[:, 1], z[:, 0] - z[:, 2]))
    return z, np.asarray(y, dtype=np.float64)


def _pack_parameters(view: FloatArray, reconstruction: FloatArray) -> FloatArray:
    """Pack A and C using MATLAB's column-major reshape convention."""
    return np.concatenate(
        (
            view.reshape(-1, order="F"),
            reconstruction.reshape(-1, order="F"),
        ),
    )


def _fixed_trial_runner(
    captured_starts: list[FloatArray],
) -> Any:
    """Return a trial-runner stub that records starts and emits a fixed view."""

    def run(
        starts: FloatArray,
        z: FloatArray,
        y: FloatArray,
        high_dimensional_distances: FloatArray,
        parallel_options: ParallelOptions | None,
    ) -> tuple[viewpoint._TrialResult, ...]:
        del z, y, high_dimensional_distances, parallel_options
        captured_starts.append(starts.copy())
        view = np.array(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=np.float64,
        )
        return (viewpoint._TrialResult(view, 1.0),)

    return run


def test_objective_is_zero_for_exact_orthogonal_reconstruction(
    sample_data: tuple[FloatArray, FloatArray],
) -> None:
    """Match MATLAB's column-major A/C reconstruction objective."""
    z, _ = sample_data
    view = np.array(
        [[1.0, 2.0, 3.0], [2.0, -1.0, 0.0]],
        dtype=np.float64,
    )
    reconstruction = np.array(
        [[0.5, 1.25], [-2.0, 0.75]],
        dtype=np.float64,
    )
    y = (reconstruction @ view @ z.T).T
    theta = _pack_parameters(view, reconstruction)

    objective = viewpoint._viewpoint_objective(theta, z, y)

    assert objective == pytest.approx(0.0, abs=1e-15)


def test_objective_penalty_uses_unit_view_rows() -> None:
    """The 0.2 penalty is scale-invariant and reaches 0.2 for parallel rows."""
    z = np.zeros((3, 3), dtype=np.float64)
    y = np.zeros((3, 1), dtype=np.float64)
    reconstruction = np.zeros((1, 2), dtype=np.float64)
    unit_view = np.array(
        [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        dtype=np.float64,
    )
    scaled_view = np.array(
        [[10.0, 0.0, 0.0], [0.01, 0.0, 0.0]],
        dtype=np.float64,
    )

    unit_cost = viewpoint._viewpoint_objective(
        _pack_parameters(unit_view, reconstruction),
        z,
        y,
    )
    scaled_cost = viewpoint._viewpoint_objective(
        _pack_parameters(scaled_view, reconstruction),
        z,
        y,
    )

    assert unit_cost == pytest.approx(0.2)
    assert scaled_cost == pytest.approx(unit_cost)


def test_objective_matches_matlab_nested_nanmean() -> None:
    """Average instance errors per algorithm before averaging algorithms."""
    z = np.zeros((3, 3), dtype=np.float64)
    y = np.array(
        [[1.0, np.nan], [3.0, 10.0], [np.nan, np.nan]],
        dtype=np.float64,
    )
    view = np.array(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=np.float64,
    )
    reconstruction = np.zeros((2, 2), dtype=np.float64)

    objective = viewpoint._viewpoint_objective(
        _pack_parameters(view, reconstruction),
        z,
        y,
    )

    assert objective == pytest.approx(52.5)


@pytest.mark.parametrize(
    ("z", "y", "message"),
    [
        (np.zeros(3), np.zeros((1, 1)), "Z must be a two-dimensional"),
        (np.zeros((1, 3)), np.zeros(1), "Y must be a two-dimensional"),
        (np.zeros((2, 3)), np.zeros((3, 1)), "same number of rows"),
        (np.zeros((2, 2)), np.zeros((2, 1)), "got 2 columns"),
        (np.zeros((2, 3)), np.zeros((2, 0)), "at least one algorithm"),
    ],
)
def test_data_shape_validation(
    z: FloatArray,
    y: FloatArray,
    message: str,
) -> None:
    """Reject arrays that cannot satisfy the 3D MATLAB engine contract."""
    with pytest.raises(ValueError, match=message):
        viewpoint.pilot_viewpoint(z, y, n_tries=1)


def test_empty_groups_default_to_all_algorithms_and_result_is_immutable(
    sample_data: tuple[FloatArray, FloatArray],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Resolve a global zero-based group and freeze the returned matrices."""
    z, y = sample_data
    captured: list[FloatArray] = []
    monkeypatch.setattr(viewpoint, "_run_trials", _fixed_trial_runner(captured))

    result = viewpoint.pilot_viewpoint(z, y, view_groups=[], n_tries=2)

    assert result.groups == ((0, 1, 2),)
    assert len(result.a) == 1
    assert result.a[0].shape == (2, 3)
    assert not result.a[0].flags.writeable
    with pytest.raises(ValueError, match="read-only"):
        result.a[0][0, 0] = 3.0
    with pytest.raises(FrozenInstanceError):
        setattr(result, "groups", ((0,),))


def test_custom_zero_based_groups_produce_one_view_each(
    sample_data: tuple[FloatArray, FloatArray],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Preserve custom group order without MATLAB's one-based indexing."""
    z, y = sample_data
    captured: list[FloatArray] = []
    monkeypatch.setattr(viewpoint, "_run_trials", _fixed_trial_runner(captured))

    result = viewpoint.pilot_viewpoint(
        z,
        y,
        view_groups=[[2, 0], [1]],
        n_tries=1,
    )

    assert result.groups == ((2, 0), (1,))
    assert len(result.a) == len(result.groups)
    assert len(result.azimuth) == len(result.groups)
    assert len(result.elevation) == len(result.groups)


@pytest.mark.parametrize(
    "groups",
    [
        [[]],
        [[-1]],
        [[3]],
        [[1.0]],
        [[True]],
    ],
)
def test_group_validation_rejects_empty_noninteger_and_out_of_range_indices(
    sample_data: tuple[FloatArray, FloatArray],
    groups: list[list[int]],
) -> None:
    """Validate groups where Python data enters the numerical engine."""
    z, y = sample_data
    with pytest.raises(ValueError, match="View group"):
        viewpoint.pilot_viewpoint(z, y, view_groups=groups, n_tries=1)


@pytest.mark.parametrize("n_tries", [0, -1, True])
def test_restart_count_must_be_a_positive_integer(
    sample_data: tuple[FloatArray, FloatArray],
    n_tries: int,
) -> None:
    """Reject restart counts that cannot create a default start matrix."""
    z, y = sample_data
    with pytest.raises(ValueError, match="n_tries"):
        viewpoint.pilot_viewpoint(z, y, n_tries=n_tries)


def test_exact_x0_shape_replaces_configured_restart_count(
    sample_data: tuple[FloatArray, FloatArray],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Use every supplied X0 column when its group-dependent row count matches."""
    z, y = sample_data
    captured: list[FloatArray] = []
    monkeypatch.setattr(viewpoint, "_run_trials", _fixed_trial_runner(captured))
    x0 = np.arange(30, dtype=np.float64).reshape(10, 3)

    viewpoint.pilot_viewpoint(
        z,
        y,
        view_groups=[[0, 1]],
        n_tries=7,
        x0=x0,
    )

    assert len(captured) == 1
    np.testing.assert_array_equal(captured[0], x0)


def test_valid_x0_ignores_an_invalid_default_restart_count(
    sample_data: tuple[FloatArray, FloatArray],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Let valid X0 columns define the trial count, as MATLAB does."""
    z, y = sample_data
    captured: list[FloatArray] = []
    monkeypatch.setattr(viewpoint, "_run_trials", _fixed_trial_runner(captured))
    x0 = np.arange(20, dtype=np.float64).reshape(10, 2)

    viewpoint.pilot_viewpoint(
        z,
        y,
        view_groups=[[0, 1]],
        n_tries=0,
        x0=x0,
    )

    np.testing.assert_array_equal(captured[0], x0)


def test_x0_is_reused_only_for_groups_with_matching_size(
    sample_data: tuple[FloatArray, FloatArray],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fall back independently when another group's parameter count differs."""
    z, y = sample_data
    captured: list[FloatArray] = []
    monkeypatch.setattr(viewpoint, "_run_trials", _fixed_trial_runner(captured))
    x0 = np.arange(16, dtype=np.float64).reshape(8, 2)

    viewpoint.pilot_viewpoint(
        z,
        y,
        view_groups=[[0], [1, 2]],
        n_tries=3,
        x0=x0,
    )

    np.testing.assert_array_equal(captured[0], x0)
    np.testing.assert_array_equal(captured[1], viewpoint._default_starts(10, 3))


def test_invalid_x0_shape_uses_deterministic_default_starts(
    sample_data: tuple[FloatArray, FloatArray],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ignore malformed X0 deterministically, matching MATLAB's fallback."""
    z, y = sample_data
    first_capture: list[FloatArray] = []
    second_capture: list[FloatArray] = []
    wrong_rows = np.ones((9, 4), dtype=np.float64)

    monkeypatch.setattr(
        viewpoint,
        "_run_trials",
        _fixed_trial_runner(first_capture),
    )
    viewpoint.pilot_viewpoint(
        z,
        y,
        view_groups=[[0, 1]],
        n_tries=2,
        x0=wrong_rows,
    )
    monkeypatch.setattr(
        viewpoint,
        "_run_trials",
        _fixed_trial_runner(second_capture),
    )
    viewpoint.pilot_viewpoint(
        z,
        y,
        view_groups=[[0, 1]],
        n_tries=2,
        x0=wrong_rows,
    )

    assert first_capture[0].shape == (10, 2)
    np.testing.assert_array_equal(first_capture[0], second_capture[0])


def test_default_starts_match_matlab_seeded_twister_sequence() -> None:
    """Preserve MATLAB's seed-42 values and column-major matrix fill."""
    starts = viewpoint._default_starts(4, 2)
    expected_unscaled = np.array(
        [
            [0.3745401188473625, 0.15601864044243652],
            [0.9507143064099162, 0.15599452033620265],
            [0.7319939418114051, 0.05808361216819946],
            [0.5986584841970366, 0.8661761457749352],
        ],
        dtype=np.float64,
    )

    np.testing.assert_allclose(
        starts,
        2.0 * expected_unscaled - 1.0,
        rtol=0.0,
        atol=1e-15,
    )


def test_solver_uses_bfgs_and_normalizes_each_view_row(
    sample_data: tuple[FloatArray, FloatArray],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Lock the MATLAB quasi-Newton configuration and post-trial normalization."""
    z, _ = sample_data
    y = z[:, :1]
    view = np.array(
        [[2.0, 0.0, 0.0], [0.0, 3.0, 0.0]],
        dtype=np.float64,
    )
    start = _pack_parameters(view, np.zeros((1, 2), dtype=np.float64))
    captured_method: list[str] = []
    captured_callbacks: list[Callable[[optim.OptimizeResult], None]] = []
    captured_options: list[dict[str, object]] = []

    def fake_minimize(
        function: Callable[..., float],
        initial_guess: FloatArray,
        *,
        args: tuple[FloatArray, FloatArray],
        method: str,
        callback: Callable[[optim.OptimizeResult], None],
        options: dict[str, object],
    ) -> SimpleNamespace:
        del function, initial_guess, args
        captured_method.append(method)
        captured_callbacks.append(callback)
        captured_options.append(options)
        return SimpleNamespace(x=start.copy())

    monkeypatch.setattr(optim, "minimize", fake_minimize)

    trial = viewpoint._solve_one_trial(start, z, y, pdist(z))

    assert captured_method == ["BFGS"]
    assert len(captured_callbacks) == 1
    assert captured_options[0]["maxiter"] == 30_000
    np.testing.assert_allclose(np.linalg.norm(trial.a, axis=1), np.ones(2))


def test_function_tolerance_uses_relative_objective_change() -> None:
    """Map MATLAB FunctionTolerance instead of SciPy's gradient tolerance."""
    stopper = viewpoint._FunctionChangeStopper(2.0, 1e-4)

    stopper(optim.OptimizeResult(fun=1.0))
    with pytest.raises(StopIteration):
        stopper(optim.OptimizeResult(fun=1.00001))


def test_known_exact_start_is_deterministic_end_to_end(
    sample_data: tuple[FloatArray, FloatArray],
) -> None:
    """Run the real BFGS path twice from an exact reconstructing viewpoint."""
    z, _ = sample_data
    view = np.array(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=np.float64,
    )
    reconstruction = np.eye(2, dtype=np.float64)
    y = (reconstruction @ view @ z.T).T
    x0 = _pack_parameters(view, reconstruction)[:, None]

    first = viewpoint.pilot_viewpoint(z, y, x0=x0)
    second = viewpoint.pilot_viewpoint(z, y, x0=x0)

    np.testing.assert_array_equal(first.a[0], second.a[0])
    np.testing.assert_allclose(np.linalg.norm(first.a[0], axis=1), np.ones(2))
    assert first.azimuth == pytest.approx((0.0,))
    assert first.elevation == pytest.approx((np.pi / 2,))


@pytest.mark.parametrize(
    ("view", "expected"),
    [
        (
            np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
            (0.0, np.pi / 2),
        ),
        (
            np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
            (0.0, 0.0),
        ),
    ],
)
def test_view_angles_match_matlab_cart2sph(
    view: FloatArray,
    expected: tuple[float, float],
) -> None:
    """Return azimuth and elevation in radians from cross(row1, row2)."""
    azimuth, elevation = viewpoint._view_angles(view)

    assert azimuth == pytest.approx(expected[0])
    assert elevation == pytest.approx(expected[1])


def test_trial_selection_uses_highest_topology_correlation() -> None:
    """Choose by distance preservation and keep MATLAB's first-tie behavior."""
    first_view = np.eye(2, 3, dtype=np.float64)
    second_view = np.fliplr(first_view)
    first = viewpoint._TrialResult(first_view, 0.4)
    second = viewpoint._TrialResult(second_view, 0.9)

    assert viewpoint._select_best_trial((first, second)) is second
    assert (
        viewpoint._select_best_trial(
            (
                viewpoint._TrialResult(first_view, float("nan")),
                second,
            ),
        )
        is second
    )
    assert (
        viewpoint._select_best_trial(
            (
                viewpoint._TrialResult(first_view, float("nan")),
                viewpoint._TrialResult(second_view, float("nan")),
            ),
        ).a
        is first_view
    )


def test_parallel_restarts_are_suppressed_inside_a_worker(
    sample_data: tuple[FloatArray, FloatArray],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Avoid nested process pools, matching the existing PILOT safety pattern."""
    z, y = sample_data
    starts = viewpoint._default_starts(8, 2)
    mock_pool = MagicMock()
    monkeypatch.setattr(multiprocessing, "parent_process", lambda: object())
    monkeypatch.setattr(viewpoint, "ProcessPoolExecutor", mock_pool)

    def fake_solve(
        initial: FloatArray,
        _z: FloatArray,
        _y: FloatArray,
        _distances: FloatArray,
    ) -> viewpoint._TrialResult:
        return viewpoint._TrialResult(
            np.eye(2, 3, dtype=np.float64),
            float(initial[0]),
        )

    monkeypatch.setattr(
        viewpoint,
        "_solve_one_trial",
        fake_solve,
    )

    trials = viewpoint._run_trials(
        starts,
        z,
        y[:, :1],
        pdist(z),
        ParallelOptions(flag=True, n_cores=2),
    )

    assert len(trials) == 2
    mock_pool.assert_not_called()


def test_enabled_parallel_restarts_preserve_order_and_cap_workers(
    sample_data: tuple[FloatArray, FloatArray],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise the process-pool path without launching child processes."""
    z, y = sample_data
    starts = viewpoint._default_starts(8, 3)
    mock_pool = MagicMock()
    executor = mock_pool.return_value.__enter__.return_value
    submitted: list[FloatArray] = []

    def fake_solve(
        initial: FloatArray,
        _z: FloatArray,
        _y: FloatArray,
        _distances: FloatArray,
    ) -> viewpoint._TrialResult:
        return viewpoint._TrialResult(
            np.eye(2, 3, dtype=np.float64),
            float(initial[0]),
        )

    def immediate_submit(
        function: Callable[
            [FloatArray, FloatArray, FloatArray, FloatArray],
            viewpoint._TrialResult,
        ],
        initial: FloatArray,
        z_argument: FloatArray,
        y_argument: FloatArray,
        distances: FloatArray,
    ) -> Future[viewpoint._TrialResult]:
        submitted.append(initial.copy())
        future: Future[viewpoint._TrialResult] = Future()
        future.set_result(
            function(initial, z_argument, y_argument, distances),
        )
        return future

    executor.submit.side_effect = immediate_submit
    monkeypatch.setattr(multiprocessing, "parent_process", lambda: None)
    monkeypatch.setattr(os, "cpu_count", lambda: 2)
    monkeypatch.setattr(viewpoint, "ProcessPoolExecutor", mock_pool)
    monkeypatch.setattr(viewpoint, "_solve_one_trial", fake_solve)

    trials = viewpoint._run_trials(
        starts,
        z,
        y[:, :1],
        pdist(z),
        ParallelOptions(flag=True, n_cores=8),
    )

    mock_pool.assert_called_once_with(max_workers=2)
    assert len(submitted) == 3
    np.testing.assert_array_equal(np.column_stack(submitted), starts)
    np.testing.assert_array_equal(
        [trial.score for trial in trials],
        starts[0, :],
    )


def test_matlab_source_invariants_when_reference_repo_is_available() -> None:
    """Probe the local gold-standard source without making CI depend on it."""
    matlab_source = (
        Path(__file__).resolve().parents[2]
        / "InstanceSpace"
        / "core"
        / "PILOTviewpoint.m"
    )
    if not matlab_source.exists():
        pytest.skip("MATLAB reference repository is not available")

    source = matlab_source.read_text(encoding="utf-8")
    assert "LAMBDA = 0.2" in source
    assert "size(opts.X0,1)==2*n+2*n2" in source
    assert "rng(opts.seed, 'twister')" in source
    assert "'FunctionTolerance',1e-20" in source
    assert "perf(i) = corr(Hd, pdist(Z*A')')" in source
    assert "viewdir = cross(A(1,:), A(2,:))" in source
