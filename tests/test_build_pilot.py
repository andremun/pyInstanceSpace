"""Test module for Pilot class to verify its functionality.

The file contains multiple unit tests to ensure that the `Pilot` class corretly
perform its tasks. The basic mechanism of the test is to compare its output against
output from MATLAB and check if the outputs are the same or reasonable similar. The
tests also include some boundary test where appropriate to test the boundary of the
statement within the methods to ensure they are implemented appropriately.

Tests include:
- Correct import for the data.
- Correct output dimensionality
- Analytic option is correctly detected
- Error handling from convex hull calculation
"""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy.io import loadmat
from scipy.spatial.distance import pdist

from instancespace.data.options import GeneralOptions, ParallelOptions, PilotOptions
from instancespace.stages.pilot import PilotInput, PilotStage
from instancespace.stages.pilot_viewpoint import PilotViewpointResult

script_dir = Path(__file__).parent


class SampleData:
    """Data class for testing the Pilot stage for analytic purposes.

    This class contains the data used for testing the Pilot stage for analytic purposes.
    """

    def __init__(self) -> None:
        """Initialize the sample data for the Pilot stage."""
        fp_sampledata = script_dir / "test_data/pilot/input/test_analytic.mat"
        data = loadmat(fp_sampledata)
        self.x_sample = data["X"]
        self.y_sample = data["Y"]
        feat_labels_sample = data["featlabels"][0]
        self.feat_labels_sample = [str(label[0]) for label in feat_labels_sample]


class SampleDataNum:
    """Data class for testing the Pilot stage for numerical purposes.

    This class contains the data used for testing the Pilot stage for
    numerical purposes.
    """

    def __init__(self) -> None:
        """Initialize the sample data for the Pilot stage."""
        fp_sampledata = script_dir / "test_data/pilot/input/test_numerical.mat"
        data = loadmat(fp_sampledata)
        self.x_sample = data["X_test"]
        self.y_sample = data["Y_test"]
        feat_labels = data["featlabels"][0]
        self.feat_labels_sample = [str(label[0]) for label in feat_labels]
        analytic = bool(data["optsPilot"][0, 0]["analytic"][0, 0])
        n_tries = int(data["optsPilot"][0, 0]["ntries"][0, 0])
        self.opts_sample = PilotOptions(None, None, analytic, n_tries)


class MatlabResults:
    """Data class for verifying the output of the Pilot analytical method.

    This class contains the data used for verifying the output of the
    analytical Pilot stage.
    """

    def __init__(self) -> None:
        """Initialize the sample data for the Pilot stage."""
        fp_outdata = script_dir / "test_data/pilot/output/matlab_results_ana.mat"
        self.data = loadmat(fp_outdata)


class MatlabResultsNum:
    """Data class for verifying the output of the Pilot numerical method.

    This class contains the data used for verifying the output of the
    numerical Pilot stage.
    """

    def __init__(self) -> None:
        """Initialize the sample data for the Pilot stage."""
        fp_outdata = script_dir / "test_data/pilot/output/matlab_results_num.mat"
        self.data = loadmat(fp_outdata)


def _matlab_simpls_reference(
    x: NDArray[np.double],
    y: NDArray[np.double],
    dims: int,
) -> tuple[
    NDArray[np.double],
    NDArray[np.double],
    NDArray[np.double],
    NDArray[np.double],
]:
    """Evaluate MATLAB R2026a `plsregress.m` lines 142-203 directly."""
    x_centered = x - x.mean(axis=0)
    y_centered = y - y.mean(axis=0)
    x_loadings = np.zeros((x.shape[1], dims), dtype=np.double)
    y_loadings = np.zeros((y.shape[1], dims), dtype=np.double)
    x_scores = np.zeros((x.shape[0], dims), dtype=np.double)
    weights = np.zeros((x.shape[1], dims), dtype=np.double)
    basis = np.zeros((x.shape[1], dims), dtype=np.double)
    covariance = x_centered.T @ y_centered

    for component in range(dims):
        left, singular_values, right_t = np.linalg.svd(
            covariance,
            full_matrices=False,
        )
        x_weight = left[:, 0]
        y_weight = right_t[0, :]
        score = x_centered @ x_weight
        score_norm = np.linalg.norm(score)
        score /= score_norm
        x_loading = x_centered.T @ score

        x_loadings[:, component] = x_loading
        y_loadings[:, component] = singular_values[0] * y_weight / score_norm
        x_scores[:, component] = score
        weights[:, component] = x_weight / score_norm

        basis_vector = x_loading.copy()
        for _ in range(2):
            for previous in range(component):
                previous_basis = basis[:, previous]
                basis_vector -= (previous_basis @ basis_vector) * previous_basis
        basis_vector /= np.linalg.norm(basis_vector)
        basis[:, component] = basis_vector
        covariance -= np.outer(basis_vector, basis_vector @ covariance)
        current_basis = basis[:, : component + 1]
        covariance -= current_basis @ (current_basis.T @ covariance)

    return x_loadings, y_loadings, x_scores, weights


def test_run_analytic() -> None:
    """Test the run function for the Pilot stage for analytic purposes."""
    sd = SampleData()
    mtr = MatlabResults()

    x_sample = sd.x_sample
    y_sample = sd.y_sample
    feat_labels_sample = sd.feat_labels_sample
    opts = PilotOptions(None, None, True, 5)
    pilot = PilotStage(x_sample, y_sample, feat_labels_sample)
    result = pilot.pilot(
        x_sample,
        y_sample,
        feat_labels_sample,
        opts,
        GeneralOptions.default(),
    )

    a = result[4]
    b = result[7]
    c = result[6]
    z = result[5]
    error = result[8]

    np.testing.assert_almost_equal(abs(a), abs(mtr.data["A"]), decimal=6)
    np.testing.assert_almost_equal(abs(b), abs(mtr.data["B"]), decimal=6)
    np.testing.assert_almost_equal(abs(c), abs(mtr.data["C"]), decimal=6)
    np.testing.assert_almost_equal(abs(z), abs(mtr.data["Z"]), decimal=6)
    np.testing.assert_almost_equal(abs(error), abs(mtr.data["error"]), decimal=6)


def test_run_numerical() -> None:
    """Test the run function for the Pilot stage for numerical purposes."""
    sd = SampleDataNum()
    mtr = MatlabResultsNum()

    x_sample = sd.x_sample
    y_sample = sd.y_sample
    feat_labels_sample = sd.feat_labels_sample
    opts_sample = sd.opts_sample
    opts = PilotOptions(None, None, opts_sample.analytic, opts_sample.n_tries)
    pilot = PilotStage(x_sample, y_sample, feat_labels_sample)
    result = pilot.pilot(
        x_sample,
        y_sample,
        feat_labels_sample,
        opts,
        GeneralOptions.default(),
    )
    eoptim = result[2]
    perf = result[3]

    if eoptim is not None and perf is not None:
        np.testing.assert_almost_equal(eoptim, mtr.data["eoptim"][0], decimal=6)
        np.testing.assert_almost_equal(perf, mtr.data["perf"][0], decimal=1)


def test_numerical_c_keeps_every_algorithm_reconstruction_column() -> None:
    """Translate MATLAB's one-based B(n+1:m, :) slice without dropping Y[:, 0]."""
    x = np.array(
        [
            [0.1, 0.4],
            [0.2, 0.8],
            [0.7, 0.3],
            [0.9, 0.6],
        ],
    )
    y = np.array(
        [
            [0.2, 0.5, 0.8],
            [0.4, 0.3, 0.7],
            [0.6, 0.9, 0.1],
            [0.8, 0.2, 0.4],
        ],
    )
    n_features = x.shape[1]
    total_columns = n_features + y.shape[1]
    alpha = np.arange(1, 2 * total_columns + 2 * n_features + 1, dtype=float)
    alpha = (alpha / 10.0).reshape(-1, 1)

    output = PilotStage.pilot(
        x,
        y,
        ["f0", "f1"],
        PilotOptions.default(analytic=False, precalc_alpha=alpha),
        GeneralOptions.default(verbose=False),
        _do_output=False,
    )

    _, full_reconstruction = PilotStage._unpack_solution(
        alpha[:, 0],
        2,
        n_features,
        total_columns,
    )
    np.testing.assert_array_equal(
        output.c,
        full_reconstruction[n_features:total_columns].T,
    )
    assert output.c.shape == (2, y.shape[1])

    reconstructed = output.z @ np.vstack((output.b, output.c.T)).T
    expected_error = np.sum((np.column_stack((x, y)) - reconstructed) ** 2)
    assert output.error == pytest.approx(expected_error)


def test_pilot_default_starts_inherit_general_seed() -> None:
    """MATLAB threads general.seed into PILOT when no local seed is supplied."""
    rng = np.random.default_rng(42)
    x = rng.random((30, 4))
    y = rng.random((30, 2))
    feat_labels = ["f0", "f1", "f2", "f3"]
    opts = PilotOptions(None, None, False, 3)
    n = x.shape[1]
    m = n + y.shape[1]
    expected_rows = opts.dims * (n + m)

    with patch.object(
        PilotStage,
        "_solve_one_trial",
        return_value=(np.ones(expected_rows), 0.0, 1.0),
    ):
        result_a = PilotStage.pilot(
            x,
            y,
            feat_labels,
            opts,
            GeneralOptions(verbose=False, seed=0),
            _do_output=False,
        )
        result_b = PilotStage.pilot(
            x,
            y,
            feat_labels,
            opts,
            GeneralOptions(verbose=False, seed=1),
            _do_output=False,
        )

    assert result_a.X0 is not None
    assert result_b.X0 is not None
    assert not np.array_equal(result_a.X0, result_b.X0)
    np.testing.assert_array_equal(
        result_a.X0,
        PilotStage._default_numerical_starts(
            expected_rows,
            opts.n_tries,
            0,
        ),
    )


def test_pilot_stage_seed_overrides_general_seed() -> None:
    """An explicit pilot.seed wins over the inherited general seed."""
    rng = np.random.default_rng(42)
    x = rng.random((30, 4))
    y = rng.random((30, 2))
    opts = PilotOptions(None, None, False, 3, seed=7)
    expected_rows = opts.dims * (x.shape[1] + x.shape[1] + y.shape[1])

    with patch.object(
        PilotStage,
        "_solve_one_trial",
        return_value=(np.ones(expected_rows), 0.0, 1.0),
    ):
        result_a = PilotStage.pilot(
            x,
            y,
            ["f0", "f1", "f2", "f3"],
            opts,
            GeneralOptions(verbose=False, seed=0),
            _do_output=False,
        )
        result_b = PilotStage.pilot(
            x,
            y,
            ["f0", "f1", "f2", "f3"],
            opts,
            GeneralOptions(verbose=False, seed=1),
            _do_output=False,
        )

    assert result_a.X0 is not None
    assert result_b.X0 is not None
    np.testing.assert_array_equal(result_a.X0, result_b.X0)
    np.testing.assert_array_equal(
        result_a.X0,
        PilotStage._default_numerical_starts(expected_rows, opts.n_tries, 7),
    )


def test_adjust_rotation_preserves_pairwise_distances() -> None:
    """R1: `adjust_rotation()` only rotates Z - pairwise distances are unchanged.

    Ported from PyISpace's `pilot.adjust_rotation()`
    (gitlab.com/ita-ml/pyispace). Rotation is a rigid transform, so the
    pairwise-distance matrix of Z must be identical before and after.
    """
    rng = np.random.default_rng(7)
    z = rng.random((25, 2))
    bad_instances = np.zeros(25, dtype=bool)
    bad_instances[[2, 5, 9, 14]] = True

    z_rot, rot = PilotStage.adjust_rotation(z, bad_instances)

    np.testing.assert_allclose(pdist(z), pdist(z_rot), atol=1e-10)
    # A rotation matrix is orthonormal: R @ R.T == I.
    np.testing.assert_allclose(rot @ rot.T, np.eye(2), atol=1e-10)


def test_adjust_rotation_places_bad_centroid_at_theta() -> None:
    """The bad-instance centroid lands at the requested angle after rotation."""
    rng = np.random.default_rng(11)
    z = rng.random((20, 2)) - 0.5
    bad_instances = np.zeros(20, dtype=bool)
    bad_instances[[1, 4, 7]] = True

    z_rot, _ = PilotStage.adjust_rotation(z, bad_instances, theta=135.0)

    centroid = np.mean(z_rot[bad_instances], axis=0)
    angle = np.degrees(np.arctan2(centroid[1], centroid[0]))
    np.testing.assert_allclose(angle, 135.0, atol=1e-6)


def test_pilot_adjust_rotation_matches_unrotated_up_to_rotation() -> None:
    """`adjust_rotation=True` rotates Z but leaves pairwise distances unchanged.

    Roadmap R1 checkpoint: confirms Z's pairwise distances are unchanged
    before/after the rotation adjustment.
    """
    rng = np.random.default_rng(3)
    x = rng.random((30, 4))
    y = rng.random((30, 2))
    feat_labels = ["f0", "f1", "f2", "f3"]
    y_bin = np.zeros((30, 2), dtype=bool)
    y_bin[::3, 0] = True

    opts_plain = PilotOptions(None, None, False, 3, adjust_rotation=False)
    opts_rotated = PilotOptions(None, None, False, 3, adjust_rotation=True)
    general_options = GeneralOptions(verbose=False, seed=0)

    plain = PilotStage.pilot(
        x,
        y,
        feat_labels,
        opts_plain,
        general_options,
        y_bin=y_bin,
        _do_output=False,
    )
    rotated = PilotStage.pilot(
        x,
        y,
        feat_labels,
        opts_rotated,
        general_options,
        y_bin=y_bin,
        _do_output=False,
    )

    np.testing.assert_allclose(pdist(plain.z), pdist(rotated.z), atol=1e-8)
    assert not np.allclose(plain.z, rotated.z)


def test_pilot_adjust_rotation_reproducible_across_runs() -> None:
    """Two independent runs on the same data rotate to the same orientation."""
    rng = np.random.default_rng(3)
    x = rng.random((30, 4))
    y = rng.random((30, 2))
    feat_labels = ["f0", "f1", "f2", "f3"]
    y_bin = np.zeros((30, 2), dtype=bool)
    y_bin[::3, 0] = True
    opts = PilotOptions(None, None, False, 3, adjust_rotation=True)
    general_options = GeneralOptions(verbose=False, seed=0)

    run_a = PilotStage.pilot(
        x,
        y,
        feat_labels,
        opts,
        general_options,
        y_bin=y_bin,
        _do_output=False,
    )
    run_b = PilotStage.pilot(
        x,
        y,
        feat_labels,
        opts,
        general_options,
        y_bin=y_bin,
        _do_output=False,
    )

    np.testing.assert_array_equal(run_a.z, run_b.z)


def test_pilot_adjust_rotation_requires_y_bin() -> None:
    """`adjust_rotation=True` without `y_bin` fails loudly, not silently."""
    rng = np.random.default_rng(3)
    x = rng.random((20, 3))
    y = rng.random((20, 2))
    feat_labels = ["f0", "f1", "f2"]
    opts = PilotOptions(None, None, False, 2, adjust_rotation=True)

    with pytest.raises(ValueError, match="y_bin"):
        PilotStage.pilot(
            x,
            y,
            feat_labels,
            opts,
            GeneralOptions(verbose=False, seed=0),
            _do_output=False,
        )


def test_pilot_adjust_rotation_no_bad_instances_leaves_z_unchanged() -> None:
    """No poorly-solved instances: rotation is skipped, Z is returned as-is."""
    rng = np.random.default_rng(3)
    x = rng.random((20, 3))
    y = rng.random((20, 2))
    feat_labels = ["f0", "f1", "f2"]
    y_bin_all_good = np.ones((20, 2), dtype=bool)
    opts_plain = PilotOptions(None, None, False, 2, adjust_rotation=False)
    opts_rotated = PilotOptions(None, None, False, 2, adjust_rotation=True)
    general_options = GeneralOptions(verbose=False, seed=0)

    plain = PilotStage.pilot(
        x,
        y,
        feat_labels,
        opts_plain,
        general_options,
        y_bin=y_bin_all_good,
        _do_output=False,
    )
    rotated = PilotStage.pilot(
        x,
        y,
        feat_labels,
        opts_rotated,
        general_options,
        y_bin=y_bin_all_good,
        _do_output=False,
    )

    np.testing.assert_array_equal(plain.z, rotated.z)


def test_pilot_precalculated_alpha_does_not_crash() -> None:
    """A precomputed `alpha` solution must not hit an undefined `idx`.

    Regression test: when `options.alpha` is provided with the expected
    `(2*m + 2*n, 1)` shape, PILOT skipped setting `idx`, so the later
    `alpha[:, idx]` indexing raised `UnboundLocalError`.
    """
    rng = np.random.default_rng(0)
    x = rng.random((15, 3))
    y = rng.random((15, 2))
    feat_labels = ["f0", "f1", "f2"]
    n = x.shape[1]
    m = n + y.shape[1]
    alpha = rng.random((2 * m + 2 * n, 1))
    opts = PilotOptions(None, alpha, False, 1)

    result = PilotStage.pilot(
        x,
        y,
        feat_labels,
        opts,
        GeneralOptions(verbose=False, seed=0),
        _do_output=False,
    )

    assert result.a.shape == (2, n)
    assert result.z.shape == (15, 2)


def test_valid_precalculated_alpha_takes_precedence_over_x0() -> None:
    """Match MATLAB's precalcAlpha-first numerical dispatch."""
    rng = np.random.default_rng(73)
    x = rng.random((12, 3))
    y = rng.random((12, 2))
    n = x.shape[1]
    m = n + y.shape[1]
    expected_a = rng.random((2, n))
    expected_full_b = rng.random((m, 2))
    precalc_alpha = PilotStage._pack_solution(
        expected_a,
        expected_full_b,
    ).reshape(-1, 1)
    # This X0 is contextually invalid for the dataset. A valid precalculated
    # solution has higher precedence, so MATLAB never consults it.
    ignored_x0 = np.ones((1, 1), dtype=np.double)

    with patch.object(
        PilotStage,
        "_solve_one_trial",
        side_effect=AssertionError("X0 must be ignored when precalcAlpha is valid"),
    ):
        result = PilotStage.pilot(
            x,
            y,
            ["f0", "f1", "f2"],
            PilotOptions.default(
                analytic=False,
                x0=ignored_x0,
                precalc_alpha=precalc_alpha,
            ),
            GeneralOptions.default(verbose=False),
            _do_output=False,
        )

    assert result.X0 is None
    assert result.alpha is not None
    np.testing.assert_array_equal(result.alpha, precalc_alpha)
    np.testing.assert_array_equal(result.a, expected_a)
    np.testing.assert_array_equal(result.b, expected_full_b[:n, :])
    np.testing.assert_array_equal(result.c, expected_full_b[n:m, :].T)


@pytest.mark.parametrize(
    ("options", "message"),
    [
        (
            PilotOptions(None, np.ones((15, 1)), False, 1),
            "precalcAlpha must have shape",
        ),
        (
            PilotOptions(np.ones((15, 1)), None, False, 2),
            "x0 must have 16 rows",
        ),
        (
            PilotOptions(
                np.ones((16, 1)),
                np.ones((15, 1)),
                False,
                1,
            ),
            "precalcAlpha must have shape",
        ),
    ],
)
def test_pilot_rejects_contextually_wrong_solver_matrix_shapes(
    options: PilotOptions,
    message: str,
) -> None:
    """PILOT checks option matrix dimensions after feature counts are known."""
    x = np.ones((5, 3))
    y = np.ones((5, 2))

    with pytest.raises(ValueError, match=message):
        PilotStage.pilot(
            x,
            y,
            ["one", "two", "three"],
            options,
            GeneralOptions(verbose=False, seed=0),
            _do_output=False,
        )


def test_pilot_x0_columns_define_the_numerical_restart_count() -> None:
    """A supplied X0 matrix overrides the configured random restart count."""
    x = np.arange(15, dtype=np.double).reshape(5, 3)
    y = np.arange(10, dtype=np.double).reshape(5, 2)
    expected_rows = 16
    x0 = np.ones((expected_rows, 3))
    options = PilotOptions(x0, None, False, 1)

    with patch.object(
        PilotStage,
        "_solve_one_trial",
        return_value=(np.ones(expected_rows), 0.0, 1.0),
    ) as solve_trial:
        result = PilotStage.pilot(
            x,
            y,
            ["one", "two", "three"],
            options,
            GeneralOptions(verbose=False, seed=0),
            _do_output=False,
        )

    assert solve_trial.call_count == x0.shape[1]
    assert result.X0 is not None
    assert result.X0.shape == x0.shape
    assert result.eoptim is not None
    assert result.eoptim.shape == (x0.shape[1],)


def test_pilot_analytic_rank_deficiency_falls_back_to_numerical() -> None:
    """Match MATLAB's warning and numerical fallback for rank-deficient X."""
    rng = np.random.default_rng(1)
    x_base = rng.random((20, 2))
    x = np.column_stack([x_base, x_base[:, 0]])
    y = rng.random((20, 2))
    feat_labels = ["f0", "f1", "f2"]
    n = x.shape[1]
    m = n + y.shape[1]
    expected_a = rng.random((2, n))
    expected_full_b = rng.random((m, 2))
    precalc_alpha = PilotStage._pack_solution(
        expected_a,
        expected_full_b,
    ).reshape(-1, 1)
    opts = PilotOptions.default(
        analytic=True,
        n_tries=1,
        precalc_alpha=precalc_alpha,
    )

    with patch("instancespace.stages.pilot.logger.warning") as warning:
        result = PilotStage.pilot(
            x,
            y,
            feat_labels,
            opts,
            GeneralOptions(verbose=False, seed=0),
            _do_output=False,
        )

    warning.assert_called_once_with(
        "Feature matrix rank-deficient; falling back to numerical solution.",
    )
    np.testing.assert_array_equal(result.a, expected_a)
    np.testing.assert_array_equal(result.z, x @ expected_a.T)


def test_pilot_numerical_r2_has_one_value_per_column() -> None:
    """Numerical-branch R^2 must be per-column, matching the analytic branch.

    Regression test: without `rowvar=False`, `corrcoef` treated each
    *instance* (row) as a variable instead of each of the `m` (features +
    algorithms) columns, producing a wrongly-shaped, wrongly-computed R^2.
    """
    rng = np.random.default_rng(2)
    x = rng.random((25, 3))
    y = rng.random((25, 2))
    feat_labels = ["f0", "f1", "f2"]
    opts = PilotOptions(None, None, False, 1)

    result = PilotStage.pilot(
        x,
        y,
        feat_labels,
        opts,
        GeneralOptions(verbose=False, seed=0),
        _do_output=False,
    )

    tolerance = 1e-8
    m = x.shape[1] + y.shape[1]
    assert result.r2.shape == (m,)
    assert np.all(result.r2 >= -tolerance)
    assert np.all(result.r2 <= 1 + tolerance)


def test_pilot_numerical_solve_keeps_full_precision() -> None:
    """`numerical_solve()` must return `alpha` at float64, not truncated to float16.

    Regression test: the solver previously downcast the whole `alpha` matrix
    to `float16` before returning it, discarding precision that then fed
    into the projection matrices A/B/C/Z computed from it and into the raw
    solver result exposed in `PilotOutput.alpha`.
    """
    rng = np.random.default_rng(3)
    x = rng.random((15, 3))
    y = rng.random((15, 2))
    n = x.shape[1]
    m = n + y.shape[1]
    x_bar = np.concatenate((x, y), axis=1)
    hd = pdist(x).T
    x0 = 2 * rng.random((2 * m + 2 * n, 2)) - 1
    alpha = np.zeros((2 * m + 2 * n, 2))
    eoptim = np.zeros(2)
    perf = np.zeros(2)
    opts = PilotOptions(None, None, False, 2)

    _idx, out_alpha, _eoptim, _perf = PilotStage.numerical_solve(
        x,
        hd,
        x0,
        x_bar,
        n,
        m,
        alpha,
        eoptim,
        perf,
        opts,
        GeneralOptions(verbose=False, seed=0),
        _do_output=False,
    )

    assert out_alpha.dtype == np.float64

    result = PilotStage.pilot(
        x,
        y,
        ["f0", "f1", "f2"],
        PilotOptions.default(analytic=False, precalc_alpha=out_alpha[:, :1]),
        GeneralOptions.default(verbose=False),
        _do_output=False,
    )
    assert result.alpha is not None
    assert result.alpha.dtype == np.float64


def test_pilot_options_precalc_alpha_is_keyword_settable() -> None:
    """`precalc_alpha` (renamed from `alpha`, #301 issue 1) is a distinct field.

    Regression test: previously `PilotOptions.alpha` was overloaded to mean
    both a precomputed solution vector and (nowhere, since it didn't exist)
    a scalar performance-reconstruction weight. They're now separate fields.
    """
    rng = np.random.default_rng(29)
    vector = rng.random((10, 1))
    opts = PilotOptions.default(precalc_alpha=vector)

    assert opts.precalc_alpha is vector
    assert opts.cost_weight == 1.0


def test_analytic_solve_default_cost_weight_is_unweighted() -> None:
    """`cost_weight` defaults to 1.0, reproducing the pre-`cost_weight` output.

    This is what makes adding `cost_weight` an additive change rather than a
    behaviour change for any existing caller: weighting the performance
    block by `sqrt(1.0)` before the eigendecomposition, then dividing by
    `sqrt(1.0)` afterwards, is a no-op.
    """
    rng = np.random.default_rng(13)
    x = rng.random((20, 3))
    y = rng.random((20, 2))
    n = x.shape[1]
    x_bar = np.concatenate((x, y), axis=1)
    m = x_bar.shape[1]

    a_default, z_default, c_default, b_default, _err_d, _r2_d = (
        PilotStage.analytic_solve(x, x_bar, n, m)
    )
    a_explicit, z_explicit, c_explicit, b_explicit, _err_e, _r2_e = (
        PilotStage.analytic_solve(x, x_bar, n, m, cost_weight=1.0)
    )

    np.testing.assert_array_equal(a_default, a_explicit)
    np.testing.assert_array_equal(z_default, z_explicit)
    np.testing.assert_array_equal(c_default, c_explicit)
    np.testing.assert_array_equal(b_default, b_explicit)


def test_analytic_solve_cost_weight_changes_projection() -> None:
    """A non-default `cost_weight` must change `C` (#301 issues 1/3).

    Regression test: previously MATLAB's `costWeight` had no Python
    equivalent at all, so emphasising the performance block relative to the
    feature block was impossible.
    """
    rng = np.random.default_rng(17)
    x = rng.random((20, 3))
    y = rng.random((20, 2))
    n = x.shape[1]
    x_bar = np.concatenate((x, y), axis=1)
    m = x_bar.shape[1]

    _a1, _z1, c1, _b1, _err1, _r2_1 = PilotStage.analytic_solve(
        x,
        x_bar,
        n,
        m,
        cost_weight=1.0,
    )
    _a2, _z2, c2, _b2, _err2, _r2_2 = PilotStage.analytic_solve(
        x,
        x_bar,
        n,
        m,
        cost_weight=4.0,
    )

    assert not np.allclose(c1, c2)


def test_analytic_solve_weighted_a_matches_matlab_formula() -> None:
    """Use weighted Xbar when deriving A, matching MATLAB PILOT.m."""
    rng = np.random.default_rng(47)
    x = rng.random((24, 4))
    y = rng.random((24, 3))
    n = x.shape[1]
    x_bar = np.column_stack((x, y))
    m = x_bar.shape[1]
    cost_weight = 9.0

    a, _z, c, b, _error, _r2 = PilotStage.analytic_solve(
        x,
        x_bar,
        n,
        m,
        cost_weight=cost_weight,
        _do_output=False,
    )

    weighted_x_bar = x_bar.copy()
    weighted_x_bar[:, n:m] *= np.sqrt(cost_weight)
    weighted_loadings = np.vstack((b, c.T * np.sqrt(cost_weight)))
    x_transpose = x.T
    x_right_inverse = x_transpose.T @ np.linalg.pinv(
        x_transpose @ x_transpose.T,
    )
    expected_a = weighted_loadings.T @ weighted_x_bar.T @ x_right_inverse
    old_unweighted_a = weighted_loadings.T @ x_bar.T @ x_right_inverse

    np.testing.assert_allclose(a, expected_a)
    assert not np.allclose(a, old_unweighted_a)


def test_analytic_solve_preserves_sixth_positional_output_argument() -> None:
    """The historical sixth positional argument remains `_do_output`."""
    rng = np.random.default_rng(53)
    x = rng.random((20, 4))
    y = rng.random((20, 2))
    n = x.shape[1]
    x_bar = np.column_stack((x, y))
    m = x_bar.shape[1]

    positional = PilotStage.analytic_solve(x, x_bar, n, m, 1.0, False)
    keyword = PilotStage.analytic_solve(
        x,
        x_bar,
        n,
        m,
        cost_weight=1.0,
        _do_output=False,
        dims=2,
    )

    assert positional[0].shape == (2, n)
    for positional_value, keyword_value in zip(positional, keyword, strict=True):
        np.testing.assert_array_equal(positional_value, keyword_value)


def test_error_function_default_cost_weight_matches_unweighted() -> None:
    """`error_function`'s `cost_weight` defaults to 1.0, an exact no-op."""
    rng = np.random.default_rng(19)
    x = rng.random((15, 3))
    y = rng.random((15, 2))
    n = x.shape[1]
    x_bar = np.concatenate((x, y), axis=1)
    m = x_bar.shape[1]
    alpha = rng.random(2 * (n + m))

    err_default = PilotStage.error_function(alpha, x_bar, n, m)
    err_explicit = PilotStage.error_function(alpha, x_bar, n, m, cost_weight=1.0, d=2)

    assert err_default == err_explicit


def test_error_function_cost_weight_reweights_performance_columns() -> None:
    """A non-default `cost_weight` must change the reported error (#301 issue 7)."""
    rng = np.random.default_rng(23)
    x = rng.random((15, 3))
    y = rng.random((15, 2))
    n = x.shape[1]
    x_bar = np.concatenate((x, y), axis=1)
    m = x_bar.shape[1]
    alpha = rng.random(2 * (n + m))

    err_unweighted = PilotStage.error_function(alpha, x_bar, n, m, cost_weight=1.0)
    err_weighted = PilotStage.error_function(alpha, x_bar, n, m, cost_weight=10.0)

    assert err_unweighted != err_weighted


def test_numerical_solution_vector_uses_matlab_column_major_order() -> None:
    """PILOT packs ``[A(:); B(:)]`` exactly as MATLAB does."""
    theta = np.arange(1, 15, dtype=np.double)
    expected_a = np.array([[1.0, 3.0, 5.0], [2.0, 4.0, 6.0]])
    expected_b = np.array(
        [
            [7.0, 11.0],
            [8.0, 12.0],
            [9.0, 13.0],
            [10.0, 14.0],
        ],
    )

    a, b = PilotStage._unpack_solution(
        theta,
        dims=2,
        n=3,
        m=4,
    )

    np.testing.assert_array_equal(a, expected_a)
    np.testing.assert_array_equal(b, expected_b)
    np.testing.assert_array_equal(
        PilotStage._pack_solution(a, b),
        theta,
    )


def test_default_numerical_starts_match_r2026a_without_global_rng_mutation() -> None:
    """Pin MATLAB's seed-42 values, 53-bit conversion, and matrix fill."""
    global_state_before = np.random.get_state()

    starts = PilotStage._default_numerical_starts(4, 2)

    global_state_after = np.random.get_state()
    expected_unscaled = np.array(
        [
            [0.3745401188473625, 0.15601864044243652],
            [0.9507143064099162, 0.15599452033620265],
            [0.7319939418114051, 0.05808361216819946],
            [0.5986584841970366, 0.8661761457749352],
        ],
        dtype=np.double,
    )
    np.testing.assert_allclose(
        starts,
        2.0 * expected_unscaled - 1.0,
        atol=1e-15,
        rtol=0,
    )
    np.testing.assert_equal(global_state_after, global_state_before)


def test_error_function_matches_matlab_columnwise_nanmean_order() -> None:
    """Average instances within each column before averaging the columns."""
    x_bar = np.array(
        [
            [1.0, 1.0, np.nan, 2.0],
            [2.0, np.nan, 10.0, np.nan],
            [3.0, 3.0, 20.0, 4.0],
        ],
    )
    n = 1
    m = x_bar.shape[1]
    theta = np.zeros(2 * (n + m), dtype=np.double)

    result = PilotStage.error_function(
        theta,
        x_bar,
        n,
        m,
        cost_weight=2.0,
        d=2,
    )

    expected = np.mean(
        [
            np.mean([1.0, 4.0, 9.0]),
            2.0 * np.mean([1.0, 9.0]),
            2.0 * np.mean([100.0, 400.0]),
            2.0 * np.mean([4.0, 16.0]),
        ],
    )
    assert result == pytest.approx(expected)


def test_pilot_dims_two_preserves_the_default_analytic_result() -> None:
    """Making the historical two-dimensional default explicit is a no-op."""
    rng = np.random.default_rng(31)
    x = rng.random((24, 5))
    y = rng.random((24, 3))
    labels = [f"f{index}" for index in range(x.shape[1])]

    implicit = PilotStage.pilot(
        x,
        y,
        labels,
        PilotOptions.default(analytic=True),
        GeneralOptions.default(verbose=False),
        _do_output=False,
    )
    explicit = PilotStage.pilot(
        x,
        y,
        labels,
        PilotOptions.default(analytic=True, dims=2),
        GeneralOptions.default(verbose=False),
        _do_output=False,
    )

    np.testing.assert_array_equal(implicit.a, explicit.a)
    np.testing.assert_array_equal(implicit.z, explicit.z)
    np.testing.assert_array_equal(implicit.b, explicit.b)
    np.testing.assert_array_equal(implicit.c, explicit.c)
    np.testing.assert_array_equal(implicit.r2, explicit.r2)
    assert implicit.pilot_summary.equals(explicit.pilot_summary)
    assert implicit.viewpoint is None
    assert explicit.viewpoint is None


def test_pilot_stage_run_skips_viewpoint_for_two_dimensions() -> None:
    """The outer stage leaves historical 2D output unchanged."""
    rng = np.random.default_rng(33)
    x = rng.random((20, 4))
    y = rng.random((20, 2))
    options = PilotOptions.default(analytic=True, dims=2)
    inputs = PilotInput(
        x=x,
        y=y,
        feat_labels=[f"f{index}" for index in range(x.shape[1])],
        pilot_options=options,
        parallel_options=ParallelOptions(flag=False, n_cores=1),
        general_options=GeneralOptions.default(verbose=False),
        y_bin=np.zeros_like(y, dtype=np.bool_),
    )

    with patch("instancespace.stages.pilot.pilot_viewpoint") as calculate:
        result = PilotStage._run(inputs)

    assert result.viewpoint is None
    calculate.assert_not_called()


def test_pilot_analytic_supports_three_dimensions() -> None:
    """The standard analytic branch exposes consistent 3D matrices."""
    rng = np.random.default_rng(37)
    x = rng.random((30, 5))
    y = rng.random((30, 3))
    labels = [f"f{index}" for index in range(x.shape[1])]
    x_bar = np.column_stack((x, y))

    result = PilotStage.pilot(
        x,
        y,
        labels,
        PilotOptions.default(analytic=True, dims=3),
        GeneralOptions.default(verbose=False),
        _do_output=False,
    )

    assert result.a.shape == (3, x.shape[1])
    assert result.z.shape == (x.shape[0], 3)
    assert result.b.shape == (x.shape[1], 3)
    assert result.c.shape == (3, y.shape[1])
    assert result.r2.dtype == np.float64
    np.testing.assert_allclose(result.z, x @ result.a.T)
    reconstruction = result.z @ np.vstack((result.b, result.c.T)).T
    assert result.error == pytest.approx(np.sum((x_bar - reconstruction) ** 2))
    assert result.pilot_summary.shape == (3, x.shape[1] + 1)
    assert result.pilot_summary.iloc[:, 0].tolist() == ["Z_{1}", "Z_{2}", "Z_{3}"]
    np.testing.assert_array_equal(
        result.pilot_summary.iloc[:, 1:].to_numpy(),
        np.round(result.a, 4),
    )
    assert result.viewpoint is None


def test_pilot_stage_run_attaches_viewpoint_for_three_dimensions() -> None:
    """Compute a 3D viewpoint once at the outer stage boundary."""
    rng = np.random.default_rng(39)
    x = rng.random((20, 4))
    y = rng.random((20, 2))
    x0 = np.linspace(-1.0, 1.0, 16, dtype=np.double).reshape(8, 2)
    options = PilotOptions.default(
        analytic=True,
        dims=3,
        n_tries=7,
        x0=x0,
        view_groups=((1,), (0,)),
    )
    parallel_options = ParallelOptions(flag=False, n_cores=2)
    inputs = PilotInput(
        x=x,
        y=y,
        feat_labels=[f"f{index}" for index in range(x.shape[1])],
        pilot_options=options,
        parallel_options=parallel_options,
        general_options=GeneralOptions.default(verbose=False),
        y_bin=np.zeros_like(y, dtype=np.bool_),
    )
    expected = PilotViewpointResult(
        groups=((1,), (0,)),
        a=(
            np.eye(2, 3, dtype=np.float64),
            np.array(
                [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                dtype=np.float64,
            ),
        ),
        azimuth=(0.0, 0.0),
        elevation=(np.pi / 2, 0.0),
    )

    with patch(
        "instancespace.stages.pilot.pilot_viewpoint",
        return_value=expected,
    ) as calculate:
        result = PilotStage._run(inputs)

    assert result.viewpoint is expected
    calculate.assert_called_once()
    call_args = calculate.call_args
    np.testing.assert_array_equal(call_args.args[0], result.z)
    np.testing.assert_array_equal(call_args.args[1], y)
    assert call_args.kwargs == {
        "view_groups": options.view_groups,
        "n_tries": options.n_tries,
        "x0": options.x0,
        "parallel_options": parallel_options,
        "seed": inputs.general_options.seed,
    }


def test_pilot_numerical_supports_three_dimensions() -> None:
    """The standard numerical caller and output preserve a 3D solution."""
    rng = np.random.default_rng(41)
    x = rng.random((18, 4))
    y = rng.random((18, 2))
    labels = [f"f{index}" for index in range(x.shape[1])]
    n = x.shape[1]
    m = n + y.shape[1]
    dims = 3
    expected_a = rng.random((3, n))
    expected_full_b = rng.random((m, 3))
    theta = PilotStage._pack_solution(
        expected_a,
        expected_full_b,
    ).reshape(-1, 1)

    with patch.object(
        PilotStage,
        "_solve_one_trial",
        return_value=(theta[:, 0], 0.0, 1.0),
    ) as solve_trial:
        result = PilotStage.pilot(
            x,
            y,
            labels,
            PilotOptions.default(analytic=False, dims=dims, x0=theta),
            GeneralOptions.default(verbose=False),
            _do_output=False,
        )

    assert solve_trial.call_args.args[-1] == dims
    assert result.alpha is not None
    assert result.alpha.dtype == np.float64
    assert result.r2.dtype == np.float64
    np.testing.assert_array_equal(result.a, expected_a)
    np.testing.assert_array_equal(result.z, x @ expected_a.T)
    np.testing.assert_array_equal(result.b, expected_full_b[:n, :])
    np.testing.assert_array_equal(result.c, expected_full_b[n:m, :].T)
    reconstruction = result.z @ expected_full_b.T
    expected_error = np.sum((np.column_stack((x, y)) - reconstruction) ** 2)
    assert result.error == pytest.approx(expected_error)


def test_pilot_numerical_three_dimensional_bfgs_smoke() -> None:
    """Run one real fixed-start 3D BFGS solve without a numeric pseudo-oracle."""
    rng = np.random.default_rng(71)
    x = rng.normal(size=(10, 3))
    y = rng.normal(size=(10, 1))
    n = x.shape[1]
    m = n + y.shape[1]
    dims = 3
    x0 = np.linspace(-0.4, 0.4, dims * (n + m), dtype=np.double).reshape(-1, 1)

    result = PilotStage.pilot(
        x,
        y,
        ["f0", "f1", "f2"],
        PilotOptions.default(analytic=False, dims=dims, n_tries=1, x0=x0),
        GeneralOptions.default(verbose=False),
        _do_output=False,
        parallel_options=ParallelOptions(flag=False, n_cores=1),
    )

    assert result.alpha is not None
    assert result.eoptim is not None
    assert result.perf is not None
    assert result.a.shape == (dims, n)
    assert result.z.shape == (x.shape[0], dims)
    assert result.b.shape == (n, dims)
    assert result.c.shape == (dims, y.shape[1])
    assert np.all(np.isfinite(result.alpha))
    assert np.all(np.isfinite(result.eoptim))
    assert np.all(np.isfinite(result.perf))
    assert np.all(np.isfinite(result.r2))
    np.testing.assert_allclose(result.z, x @ result.a.T)
    reconstruction = result.z @ np.vstack((result.b, result.c.T)).T
    expected_error = np.sum((np.column_stack((x, y)) - reconstruction) ** 2)
    assert float(result.error) == pytest.approx(expected_error, abs=1e-10)


def test_numerical_solve_parallel_matches_sequential() -> None:
    """Parallelising the `ntries` restart loop must not change the result.

    Regression test for F2's ntries-parallelism work (roadmap #301 issues
    overlap, docs/python_implementation_pathways.md F2): each restart is
    independent (different starting point, same cost function), so which
    one "wins" (highest `perf`) must be identical whether the restarts run
    sequentially or across a process pool.
    """
    rng = np.random.default_rng(7)
    x = rng.random((20, 3))
    y = rng.random((20, 2))
    n = x.shape[1]
    x_bar = np.concatenate((x, y), axis=1)
    m = x_bar.shape[1]
    hd = pdist(x).T
    n_tries = 3
    x0 = 2 * rng.random((2 * m + 2 * n, n_tries)) - 1
    opts = PilotOptions(None, None, False, n_tries)
    general_options = GeneralOptions(verbose=False, seed=0)

    idx_seq, alpha_seq, eoptim_seq, perf_seq = PilotStage.numerical_solve(
        x,
        hd,
        x0.copy(),
        x_bar,
        n,
        m,
        np.zeros((2 * m + 2 * n, n_tries)),
        np.zeros(n_tries),
        np.zeros(n_tries),
        opts,
        general_options,
        _do_output=False,
        parallel_options=ParallelOptions(flag=False, n_cores=2),
    )
    idx_par, alpha_par, eoptim_par, perf_par = PilotStage.numerical_solve(
        x,
        hd,
        x0.copy(),
        x_bar,
        n,
        m,
        np.zeros((2 * m + 2 * n, n_tries)),
        np.zeros(n_tries),
        np.zeros(n_tries),
        opts,
        general_options,
        _do_output=False,
        parallel_options=ParallelOptions(flag=True, n_cores=2),
    )

    assert idx_seq == idx_par
    np.testing.assert_array_equal(alpha_seq, alpha_par)
    np.testing.assert_array_equal(eoptim_seq, eoptim_par)
    np.testing.assert_array_equal(perf_seq, perf_par)


def test_numerical_solve_skips_pool_when_already_in_worker_process() -> None:
    """Guard against reintroducing MATLAB's nested-parfor-inside-GA bug.

    If PILOT's own `ntries` pool ever opened while already running inside
    another process pool's worker (e.g. SIFTED's GA fitness function calling
    `pilot()` from inside `pygad`'s own `ProcessPoolExecutor` workers), it
    would reintroduce that bug (roadmap F2 pathway, #301). This never
    triggers in practice today - SIFTED always calls `pilot()` with
    `analytic=True`, which bypasses this numerical branch entirely - but
    `multiprocessing.parent_process()` is checked as a cheap circuit
    breaker regardless, in case that invariant ever changes silently.
    """
    rng = np.random.default_rng(11)
    x = rng.random((15, 2))
    y = rng.random((15, 1))
    n = x.shape[1]
    x_bar = np.concatenate((x, y), axis=1)
    m = x_bar.shape[1]
    hd = pdist(x).T
    n_tries = 2
    x0 = 2 * rng.random((2 * m + 2 * n, n_tries)) - 1
    opts = PilotOptions(None, None, False, n_tries)
    general_options = GeneralOptions(verbose=False, seed=0)

    with (
        patch(
            "instancespace.stages.pilot.multiprocessing.parent_process",
            return_value=object(),
        ),
        patch("instancespace.stages.pilot.ProcessPoolExecutor") as mock_pool,
    ):
        PilotStage.numerical_solve(
            x,
            hd,
            x0,
            x_bar,
            n,
            m,
            np.zeros((2 * m + 2 * n, n_tries)),
            np.zeros(n_tries),
            np.zeros(n_tries),
            opts,
            general_options,
            _do_output=False,
            parallel_options=ParallelOptions(flag=True, n_cores=2),
        )

    mock_pool.assert_not_called()


def test_pls_solve_produces_correct_shapes() -> None:
    """`pls_solve` must return A/Z/C/B at the expected `dims`-parametrised shapes."""
    rng = np.random.default_rng(0)
    x = rng.random((30, 5))
    y = rng.random((30, 3))
    x_bar = np.concatenate([x, y], axis=1)
    m = x_bar.shape[1]

    a, z, c, b, _err, r2 = PilotStage.pls_solve(x, y, x_bar, m, _do_output=False)

    assert a.shape == (2, 5)
    assert z.shape == (30, 2)
    assert c.shape == (2, 3)
    assert b.shape == (5, 2)
    assert r2.shape == (m,)
    assert np.all(np.isfinite(z))


def test_pls_solve_a_reprojects_new_instances_correctly() -> None:
    """SIMPLS weights satisfy Z = (X - mean) @ A.T."""
    rng = np.random.default_rng(1)
    x = rng.random((30, 6))
    y = rng.random((30, 4))
    x_bar = np.concatenate([x, y], axis=1)
    m = x_bar.shape[1]

    a, z, _c, _b, _err, _r2 = PilotStage.pls_solve(
        x,
        y,
        x_bar,
        m,
        dims=3,
        _do_output=False,
    )

    z_reprojected = (x - x.mean(axis=0)) @ a.T
    np.testing.assert_allclose(z, z_reprojected, atol=1e-8)


@pytest.mark.parametrize("dims", [2, 3])
def test_pls_solve_matches_matlab_simpls_formula_up_to_component_sign(
    dims: int,
) -> None:
    """Match the R2026a SIMPLS source for deterministic 2D and 3D fits."""
    rng = np.random.default_rng(61)
    x = rng.normal(size=(32, 6))
    coefficients = rng.normal(size=(6, 4))
    y = x @ coefficients + 0.1 * rng.normal(size=(32, 4))
    x_bar = np.column_stack((x, y))
    m = x_bar.shape[1]
    expected_b, expected_y_loadings, expected_z, expected_weights = (
        _matlab_simpls_reference(x, y, dims)
    )

    a, z, c, b, error, _r2 = PilotStage.pls_solve(
        x,
        y,
        x_bar,
        m,
        dims=dims,
        _do_output=False,
    )

    # SVD component signs are arbitrary across LAPACK implementations. Align
    # each component before comparing factors; do not assert raw sign identity.
    sign_alignment = np.sign(np.sum(a * expected_weights.T, axis=1))
    sign_alignment[sign_alignment == 0] = 1
    np.testing.assert_allclose(
        a * sign_alignment[:, None],
        expected_weights.T,
        atol=2e-13,
        rtol=0,
    )
    np.testing.assert_allclose(
        z * sign_alignment[None, :],
        expected_z,
        atol=2e-13,
        rtol=0,
    )
    np.testing.assert_allclose(
        b * sign_alignment[None, :],
        expected_b,
        atol=2e-13,
        rtol=0,
    )
    np.testing.assert_allclose(
        c * sign_alignment[:, None],
        expected_y_loadings.T,
        atol=2e-13,
        rtol=0,
    )

    np.testing.assert_allclose(
        z,
        (x - x.mean(axis=0)) @ a.T,
        atol=2e-13,
        rtol=0,
    )
    reconstruction = z @ np.vstack((b, c.T)).T + np.concatenate(
        (x.mean(axis=0), y.mean(axis=0)),
    )
    assert float(error) == pytest.approx(
        np.sum((x_bar - reconstruction) ** 2),
        abs=2e-12,
    )


def test_pls_solve_does_not_call_sklearn_plsregression() -> None:
    """Guard the MATLAB-gold SIMPLS port against a sklearn backend regression."""
    rng = np.random.default_rng(67)
    x = rng.normal(size=(20, 5))
    y = rng.normal(size=(20, 3))
    x_bar = np.column_stack((x, y))

    with patch(
        "sklearn.cross_decomposition.PLSRegression.fit",
        side_effect=AssertionError("PILOT must not call sklearn PLSRegression"),
    ):
        result = PilotStage.pls_solve(
            x,
            y,
            x_bar,
            x_bar.shape[1],
            dims=3,
            _do_output=False,
        )

    assert result[0].shape == (3, x.shape[1])


def test_simpls_rejects_a_degenerate_component_with_named_error() -> None:
    """Report an actionable error instead of propagating non-finite factors."""
    x_centered = np.zeros((8, 4), dtype=np.double)
    y_centered = np.zeros((8, 3), dtype=np.double)

    with pytest.raises(
        ValueError,
        match=r"PILOT SIMPLS component 1.*X-score norm",
    ):
        PilotStage._simpls(
            x_centered,
            y_centered,
            n_components=2,
        )


def test_pls_solve_rejects_more_than_matlab_maximum_components() -> None:
    """Match `plsregress`'s `min(n_instances - 1, n_features)` limit."""
    x = np.arange(15, dtype=np.double).reshape(3, 5)
    y = np.arange(9, dtype=np.double).reshape(3, 3)
    x_bar = np.column_stack((x, y))

    with pytest.raises(ValueError, match="between 1 and 2"):
        PilotStage.pls_solve(
            x,
            y,
            x_bar,
            x_bar.shape[1],
            dims=3,
            _do_output=False,
        )


def test_pls_solve_is_dims_generic_with_no_code_changes() -> None:
    """`pls_solve` must work at `dims=3` (#262)."""
    rng = np.random.default_rng(2)
    x = rng.random((25, 6))
    y = rng.random((25, 4))
    x_bar = np.concatenate([x, y], axis=1)
    m = x_bar.shape[1]

    a, z, c, b, _err, r2 = PilotStage.pls_solve(
        x,
        y,
        x_bar,
        m,
        dims=3,
        _do_output=False,
    )

    tolerance = 1e-6
    assert a.shape == (3, 6)
    assert z.shape == (25, 3)
    assert c.shape == (3, 4)
    assert b.shape == (6, 3)
    assert np.all(np.isfinite(z))
    assert np.all(r2 >= -tolerance)
    assert np.all(r2 <= 1 + tolerance)


def test_pilot_method_pls_dispatches_correctly() -> None:
    """`PilotStage.pilot()` with `method='pls'` must use the PLS solver.

    Regression test: `method='pls'` must be checked before `analytic`, and
    must produce sane output regardless of `analytic`'s value (PLS ignores
    it entirely, matching MATLAB's opts.method dispatch order).
    """
    rng = np.random.default_rng(3)
    x = rng.random((30, 4))
    y = rng.random((30, 2))
    feat_labels = ["f0", "f1", "f2", "f3"]

    opts = PilotOptions.default(method="pls", analytic=True)
    result = PilotStage.pilot(
        x,
        y,
        feat_labels,
        opts,
        GeneralOptions.default(),
        _do_output=False,
    )

    assert result.a.shape == (2, 4)
    assert result.z.shape == (30, 2)
    assert result.alpha is None
    assert result.X0 is None
    assert np.all(np.isfinite(result.z))


def test_pilot_method_pls_passes_three_dimensions_through_public_dispatch() -> None:
    """The public PLS branch passes `PilotOptions.dims` to its solver."""
    rng = np.random.default_rng(43)
    x = rng.random((30, 6))
    y = rng.random((30, 4))
    feat_labels = [f"f{index}" for index in range(x.shape[1])]

    result = PilotStage.pilot(
        x,
        y,
        feat_labels,
        PilotOptions.default(method="pls", analytic=True, dims=3),
        GeneralOptions.default(verbose=False),
        _do_output=False,
    )

    assert result.a.shape == (3, x.shape[1])
    assert result.z.shape == (x.shape[0], 3)
    assert result.b.shape == (x.shape[1], 3)
    assert result.c.shape == (3, y.shape[1])
    assert result.r2.dtype == np.float64
    np.testing.assert_allclose(result.z, (x - x.mean(axis=0)) @ result.a.T)
    assert result.pilot_summary.iloc[:, 0].tolist() == ["Z_{1}", "Z_{2}", "Z_{3}"]


def test_pilot_options_default_method_is_standard() -> None:
    """`PilotOptions.default()`'s `method` must default to `"standard"`.

    Additive-at-default check: existing callers not passing `method`
    explicitly must see no change in behaviour.
    """
    opts = PilotOptions.default()
    assert opts.method == "standard"
