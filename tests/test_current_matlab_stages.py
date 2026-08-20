"""Numerical readers for the verified current MATLAB stage bundle."""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from types import SimpleNamespace
from typing import NamedTuple, cast

import numpy as np
import pandas as pd
import pytest
from numpy.typing import NDArray
from sklearn.neighbors import KNeighborsClassifier

from instancespace.data.options import (
    CloisterOptions,
    GeneralOptions,
    ParallelOptions,
    PilotOptions,
    PrelimOptions,
    PythiaOptions,
    SelvarsOptions,
    SiftedOptions,
)
from instancespace.instance_space import InstanceSpace
from instancespace.stages.cloister import CloisterStage
from instancespace.stages.pilot import PilotStage
from instancespace.stages.prelim import PrelimStage
from instancespace.stages.pythia import PythiaOutput, PythiaStage
from instancespace.stages.sifted import SiftedStage

_BUNDLE = Path(__file__).parent / "fixtures" / "matlab" / "current"
_BUILD = _BUNDLE / "build_data"
_EXPLORE = _BUNDLE / "explore_data"
_KNN_METRICS = ("euclidean", "cityblock", "cosine", "correlation")
_GOOD_PROBABILITY_CUTOFF = 0.5
_QDA_GOOD_COUNT = 4


def _json_object(path: Path) -> dict[str, object]:
    """Read a JSON object while keeping strict typing at the file boundary."""
    value: object = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return cast(dict[str, object], value)


_RESOLVED_DOCUMENT = _json_object(
    _BUNDLE / "resolved_options" / "trace3_default.json",
)
_RESOLVED = cast(dict[str, object], _RESOLVED_DOCUMENT["options"])


def _option_group(name: str) -> dict[str, object]:
    """Return one resolved MATLAB option namespace."""
    return cast(dict[str, object], _RESOLVED[name])


def _boolean(group: str, name: str) -> bool:
    """Read a resolved boolean option."""
    value = _option_group(group)[name]
    assert isinstance(value, bool)
    return value


def _integer(group: str, name: str) -> int:
    """Read a resolved integer option."""
    value = _option_group(group)[name]
    assert isinstance(value, int)
    assert not isinstance(value, bool)
    return value


def _number(group: str, name: str) -> float:
    """Read a resolved numeric option."""
    value = _option_group(group)[name]
    assert isinstance(value, int | float)
    assert not isinstance(value, bool)
    return float(value)


def _string(group: str, name: str) -> str:
    """Read a resolved string option."""
    value = _option_group(group)[name]
    assert isinstance(value, str)
    return value


def _matrix(path: Path) -> NDArray[np.double]:
    """Read a numeric bundle CSV, removing its optional row-label column."""
    frame = pd.read_csv(path)
    if str(frame.columns[0]) == "Row":
        frame = frame.iloc[:, 1:]
    matrix: NDArray[np.double] = frame.to_numpy(dtype=np.double)
    return matrix


def _vector(path: Path) -> NDArray[np.double]:
    """Read a one-column numeric bundle CSV as a vector."""
    return _matrix(path).reshape(-1)


def _labels(path: Path) -> list[str]:
    """Read a one-column text-label bundle CSV."""
    frame = pd.read_csv(path)
    return frame.iloc[:, 0].astype(str).tolist()


def _row_labels(path: Path) -> pd.Series:  # type: ignore[type-arg]
    """Read MATLAB's explicit row labels from a stage input."""
    frame = pd.read_csv(path)
    assert str(frame.columns[0]) == "Row"
    return frame.iloc[:, 0].astype(str)


def _general_options() -> GeneralOptions:
    """Construct general options from the resolved MATLAB artifact."""
    return GeneralOptions(
        verbose=_boolean("general", "verbose"),
        seed=_integer("general", "seed"),
    )


def _parallel_options() -> ParallelOptions:
    """Construct parallel options from the resolved MATLAB artifact."""
    return ParallelOptions(
        flag=_boolean("general", "parallel"),
        n_cores=_integer("general", "ncores"),
    )


def _selvars_options() -> SelvarsOptions:
    """Construct selection options from the resolved MATLAB artifact."""
    return SelvarsOptions.default(
        small_scale_flag=_boolean("selvars", "smallscaleflag"),
        small_scale=_number("selvars", "smallscale"),
        file_idx_flag=_boolean("selvars", "fileidxflag"),
        file_idx=_string("selvars", "fileidx"),
        selvars_type=_string("selvars", "type"),
        min_distance=_number("selvars", "mindistance"),
        density_flag=_boolean("selvars", "densityflag"),
    )


def _prelim_options() -> PrelimOptions:
    """Construct PRELIM options from the resolved MATLAB artifact."""
    return PrelimOptions.default(
        max_perf=_boolean("perf", "MaxPerf"),
        abs_perf=_boolean("perf", "AbsPerf"),
        epsilon=_number("perf", "epsilon"),
        beta_threshold=_number("perf", "betaThreshold"),
        bound=_boolean("bound", "flag"),
        norm=_boolean("norm", "flag"),
        iqr_multiplier=_number("prelim", "iqrMultiplier"),
        preproc=_boolean("auto", "preproc"),
        nan_threshold=_number("prelim", "nanThreshold"),
    )


def _sifted_options() -> SiftedOptions:
    """Construct active SIFTED options from the resolved MATLAB artifact."""
    return SiftedOptions.default(
        flag=_boolean("sifted", "flag"),
        rho=_number("sifted", "rho"),
        pval=_number("sifted", "pval"),
        k=_integer("sifted", "K"),
        max_iter=_integer("sifted", "MaxIter"),
        replicates=_integer("sifted", "Replicates"),
    )


def _assert_same_closed_cycle(
    actual: NDArray[np.double],
    expected_closed: NDArray[np.double],
) -> None:
    """Compare hull vertices modulo start, direction, and MATLAB's closure row."""
    np.testing.assert_allclose(
        expected_closed[0],
        expected_closed[-1],
        atol=2e-13,
        rtol=0,
    )
    expected = expected_closed[:-1]
    assert actual.shape == expected.shape

    candidates = [
        np.roll(oriented, shift, axis=0)
        for oriented in (actual, actual[::-1])
        for shift in range(actual.shape[0])
    ]
    best = min(
        candidates,
        key=lambda candidate: float(np.max(np.abs(candidate - expected))),
    )
    np.testing.assert_allclose(best, expected, atol=2e-13, rtol=0)


def _assert_knn_probability_semantics(
    actual: NDArray[np.double],
    expected: NDArray[np.double],
    predictions: NDArray[np.bool_],
    params: NDArray[np.double],
) -> None:
    """Compare exact KNN probabilities except correlation-distance tie choices."""
    assert actual.shape == expected.shape == predictions.shape
    assert np.all((actual >= 0) & (actual <= 1))
    assert np.all((expected >= 0) & (expected <= 1))

    for index, (neighbors_raw, metric_raw) in enumerate(params):
        neighbors = int(neighbors_raw)
        metric_index = int(metric_raw) - 1
        np.testing.assert_allclose(
            actual[:, index] * neighbors,
            np.rint(actual[:, index] * neighbors),
            atol=2e-13,
            rtol=0,
        )
        np.testing.assert_allclose(
            expected[:, index] * neighbors,
            np.rint(expected[:, index] * neighbors),
            atol=2e-13,
            rtol=0,
        )
        np.testing.assert_array_equal(
            actual[:, index] < _GOOD_PROBABILITY_CUTOFF,
            predictions[:, index],
        )
        np.testing.assert_array_equal(
            expected[:, index] < _GOOD_PROBABILITY_CUTOFF,
            predictions[:, index],
        )
        if _KNN_METRICS[metric_index] != "correlation":
            np.testing.assert_allclose(
                actual[:, index],
                expected[:, index],
                atol=2e-14,
                rtol=0,
            )


class _CurrentPythiaRun(NamedTuple):
    """One shared current-bundle PYTHIA training result."""

    training_z: NDArray[np.double]
    params: NDArray[np.double]
    output: PythiaOutput
    sklearn_warnings: tuple[str, ...]


@pytest.fixture(scope="module")
def current_pythia_run() -> _CurrentPythiaRun:
    """Train current-bundle KNN models once using exported MATLAB-unit params."""
    root = _BUILD / "pythia" / "trace3_default"
    inputs = root / "inputs"
    outputs = root / "outputs"
    hyperparameters = pd.read_csv(outputs / "hyperparameters.csv")
    params: NDArray[np.double] = hyperparameters[["param1", "param2"]].to_numpy(
        dtype=np.double,
    )
    training_z = _matrix(inputs / "z.csv")
    options = PythiaOptions.default(
        cv_folds=_integer("pythia", "kFold"),
        is_poly_krnl=_boolean("pythia", "ispolykrnl"),
        use_weights=_boolean("pythia", "useweights"),
        classifier=_string("pythia", "classifier"),
        tuning="none",
        n_tuning_iter=_integer("pythia", "nTuningIter"),
        skip=_boolean("pythia", "skip"),
        params=params,
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        output = PythiaStage.pythia(
            training_z,
            _matrix(inputs / "y_raw.csv"),
            _matrix(inputs / "y_bin.csv").astype(np.bool_),
            _vector(inputs / "y_best.csv"),
            _labels(inputs / "algorithm_labels.csv"),
            options,
            _parallel_options(),
            _general_options(),
        )

    return _CurrentPythiaRun(
        training_z,
        params,
        output,
        tuple(str(item.message) for item in caught),
    )


class _ExplorePythiaHarness:
    """Minimal model owner accepted by the production explore method."""

    def __init__(self, run: _CurrentPythiaRun) -> None:
        self._model = SimpleNamespace(
            pilot=SimpleNamespace(z=run.training_z),
            pythia=SimpleNamespace(
                svm=run.output.svm,
                precision=run.output.precision,
            ),
        )

    def _require_model(self) -> SimpleNamespace:
        """Return the stage fields consumed by `_explore_pythia`."""
        return self._model


def test_current_bundle_is_verified_r2026a_source() -> None:
    """Pin these numerical oracles to the reviewed clean MATLAB export."""
    manifest = _json_object(_BUNDLE / "manifest.json")
    matlab = cast(dict[str, object], manifest["matlab"])
    generator = cast(dict[str, object], manifest["generator"])

    assert manifest["schema_version"] == "pyinstancespace.matlab-fixtures/v1"
    assert manifest["trust"] == "matlab-verified"
    assert matlab["repo_commit"] == "34c01293fef99b4eabd53323c393cb184cc95a8e"
    assert matlab["repo_dirty"] is False
    assert matlab["release"] == "R2026a"
    assert matlab["platform"] == "MACA64"
    assert generator["repo_commit"] == "cf3cde0da5a3067300bd94a48d4d09ff5cf20b0c"
    assert generator["repo_dirty"] is False
    assert _RESOLVED_DOCUMENT["schema_version"] == (
        "pyinstancespace.resolved-options/v1"
    )
    assert _RESOLVED_DOCUMENT["name"] == "trace3_default"


def test_current_matlab_prelim_numerical_output() -> None:
    """Reproduce PRELIM transforms and every exported fit parameter."""
    root = _BUILD / "prelim" / "default"
    inputs = root / "inputs"
    outputs = root / "outputs"
    x_raw = _matrix(inputs / "x_raw.csv")
    y_raw = _matrix(inputs / "y_raw.csv")

    (
        x,
        y,
        y_bin,
        y_best,
        p,
        num_good_algos,
        beta,
        med_val,
        iq_range,
        hi_bound,
        lo_bound,
        min_x,
        lambda_x,
        mu_x,
        sigma_x,
        min_y,
        lambda_y,
        sigma_y,
        mu_y,
    ) = PrelimStage.prelim(
        x_raw.copy(),
        y_raw.copy(),
        x_raw.copy(),
        y_raw.copy(),
        None,
        _row_labels(inputs / "x_raw.csv"),
        _prelim_options(),
        _selvars_options(),
        _general_options(),
    )

    np.testing.assert_allclose(
        x,
        _matrix(inputs / "x_processed.csv"),
        atol=5e-14,
        rtol=0,
    )
    np.testing.assert_allclose(
        y,
        _matrix(inputs / "y_processed.csv"),
        atol=5e-14,
        rtol=0,
    )
    np.testing.assert_array_equal(
        y_bin,
        _matrix(outputs / "prelim_ybin.csv").astype(bool),
    )

    feature_params = pd.read_csv(outputs / "prelim_feature_params.csv")
    for actual, column in (
        (min_x, "min_x"),
        (lambda_x, "lambda_x"),
        (mu_x, "mu_x"),
        (sigma_x, "sigma_x"),
        (med_val, "medval"),
        (iq_range, "iqrange"),
        (hi_bound, "hi_bound"),
        (lo_bound, "lo_bound"),
    ):
        np.testing.assert_allclose(
            actual,
            feature_params[column].to_numpy(dtype=np.double),
            atol=1e-12,
            rtol=0,
        )

    algorithm_params = pd.read_csv(outputs / "prelim_algo_params.csv")
    for actual, column in (
        (lambda_y, "lambda_y"),
        (mu_y, "mu_y"),
        (sigma_y, "sigma_y"),
    ):
        np.testing.assert_allclose(
            actual,
            algorithm_params[column].to_numpy(dtype=np.double),
            atol=1e-12,
            rtol=0,
        )
    assert min_y == pytest.approx(_vector(outputs / "prelim_scalars.csv")[0], abs=1e-15)

    instances = pd.read_csv(outputs / "prelim_instance_outputs.csv")
    np.testing.assert_allclose(
        y_best,
        instances["y_best"].to_numpy(dtype=np.double),
        atol=1e-14,
        rtol=0,
    )
    np.testing.assert_array_equal(
        num_good_algos,
        instances["num_good_algos"].to_numpy(dtype=np.double),
    )
    np.testing.assert_array_equal(
        beta,
        instances["beta"].to_numpy(dtype=np.bool_),
    )

    # MATLAB randomizes exact best-algorithm ties; both answers must still point
    # to a genuine raw-performance minimum, and unique minima must agree exactly.
    finite_y = np.where(np.isnan(y_raw), np.inf, y_raw)
    raw_best = np.min(finite_y, axis=1)
    tied_best = finite_y == raw_best[:, None]
    matlab_p = instances["p_best_algo"].to_numpy(dtype=np.int_)
    rows = np.arange(y_raw.shape[0])
    assert np.all(tied_best[rows, p - 1])
    assert np.all(tied_best[rows, matlab_p - 1])
    unique_best = np.sum(tied_best, axis=1) == 1
    np.testing.assert_array_equal(p[unique_best], matlab_p[unique_best])


def test_current_matlab_sifted_numerical_output() -> None:
    """Reproduce SIFTED correlations and its exact 1-based feature selection."""
    root = _BUILD / "sifted" / "default"
    inputs = root / "inputs"
    outputs = root / "outputs"
    x = _matrix(inputs / "x.csv")
    y = _matrix(inputs / "y.csv")
    y_bin = _matrix(inputs / "y_bin.csv").astype(np.bool_)
    num_good_algos = np.sum(y_bin, axis=1, dtype=np.double)

    output = SiftedStage.sifted(
        x=x,
        y=y,
        y_bin=y_bin,
        x_raw=x.copy(),
        y_raw=y.copy(),
        beta=num_good_algos >= _number("perf", "betaThreshold") * y.shape[1],
        num_good_algos=num_good_algos,
        y_best=np.nanmin(y, axis=1),
        p=np.nanargmin(y, axis=1).astype(np.int_) + 1,
        inst_labels=_row_labels(inputs / "x.csv"),
        s=None,
        feat_labels=_labels(inputs / "feature_labels.csv"),
        opts=_sifted_options(),
        opts_selvars=_selvars_options(),
        data_dense=None,
        parallel_options=_parallel_options(),
        general_options=_general_options(),
    )

    assert output.rho is not None
    assert output.pval is not None
    np.testing.assert_allclose(
        output.rho,
        _matrix(outputs / "correlation_rho.csv"),
        atol=1e-12,
        rtol=0,
    )
    np.testing.assert_allclose(
        output.pval,
        _matrix(outputs / "correlation_pval.csv"),
        atol=1e-12,
        rtol=0,
    )

    selected = _vector(outputs / "selected_indices.csv").astype(np.intc) - 1
    ranked = (
        pd.read_csv(outputs / "sifted_indices.csv")["original_index"].to_numpy(
            dtype=np.intc,
        )
        - 1
    )
    np.testing.assert_array_equal(output.selvars, selected)
    np.testing.assert_array_equal(output.selvars, ranked)
    np.testing.assert_array_equal(output.x, x[:, selected])


def test_current_matlab_pilot_precalculated_solution_oracle() -> None:
    """Decode MATLAB's selected alpha and reproduce A/B/C exactly.

    This isolates the column-major ``[A(:); B(:)]`` contract from optimizer
    stopping differences. MATLAB selected the alpha column with maximum `perf`,
    so feeding that same column through Python must reproduce every factor exactly
    and all derived reconstruction quantities to floating-point precision.
    """
    root = _BUILD / "pilot" / "default"
    inputs = root / "inputs"
    outputs = root / "outputs"
    x = _matrix(inputs / "x.csv")
    y = _matrix(inputs / "y.csv")
    matlab_alpha = _matrix(outputs / "pilot_alpha.csv")
    selected_index = int(np.argmax(_vector(outputs / "pilot_perf.csv")))
    selected_alpha = matlab_alpha[:, [selected_index]]

    output = PilotStage.pilot(
        x,
        y,
        _labels(inputs / "feature_labels.csv"),
        PilotOptions.default(
            analytic=False,
            n_tries=_integer("pilot", "ntries"),
            precalc_alpha=selected_alpha,
            cost_weight=_number("pilot", "alpha"),
            method=_string("pilot", "method"),
            dims=_integer("pilot", "dims"),
        ),
        _general_options(),
        _do_output=False,
    )

    assert output.alpha is not None
    np.testing.assert_array_equal(output.alpha, selected_alpha)
    np.testing.assert_array_equal(output.a, _matrix(outputs / "pilot_a_raw.csv"))
    np.testing.assert_array_equal(output.b, _matrix(outputs / "pilot_b.csv"))
    np.testing.assert_array_equal(output.c, _matrix(outputs / "pilot_c.csv"))
    np.testing.assert_array_equal(output.z, x @ output.a.T)
    np.testing.assert_allclose(
        output.z,
        _matrix(outputs / "pilot_z.csv"),
        atol=1e-14,
        rtol=0,
    )
    assert float(output.error) == pytest.approx(
        _vector(outputs / "pilot_error.csv")[0],
        abs=3e-12,
    )
    np.testing.assert_allclose(
        output.r2,
        _vector(outputs / "pilot_r2.csv"),
        atol=2e-15,
        rtol=0,
    )


def test_current_matlab_pilot_numerical_optimizer_quality() -> None:
    """Match PILOT objectives and the optimizer-invariant projection subspace.

    MATLAB ``fminunc`` and SciPy BFGS choose different coordinates and different
    near-equal restarts on the flat factorization manifold.  The exported X0 makes
    both deterministic. Trial objectives remain close, while reconstruction error,
    R2, and the two-dimensional column space use narrow bounds calibrated to the
    two solvers' documented stopping-point differences. Raw A/Z coordinates are not
    treated as unique; their exact MATLAB-order decoding is tested separately above.
    """
    root = _BUILD / "pilot" / "default"
    inputs = root / "inputs"
    outputs = root / "outputs"
    x = _matrix(inputs / "x.csv")
    y = _matrix(inputs / "y.csv")
    x0 = _matrix(outputs / "pilot_x0.csv")
    output = PilotStage.pilot(
        x,
        y,
        _labels(inputs / "feature_labels.csv"),
        PilotOptions.default(
            analytic=_boolean("pilot", "analytic"),
            n_tries=_integer("pilot", "ntries"),
            x0=x0,
            cost_weight=_number("pilot", "alpha"),
            method=_string("pilot", "method"),
            dims=_integer("pilot", "dims"),
        ),
        _general_options(),
        _do_output=False,
        parallel_options=_parallel_options(),
    )

    assert output.X0 is not None
    assert output.eoptim is not None
    np.testing.assert_array_equal(output.X0, x0)
    np.testing.assert_allclose(
        output.eoptim,
        _vector(outputs / "pilot_eoptim.csv"),
        atol=2e-7,
        rtol=0,
    )
    # On a 2668.7 objective, the fixed-X0 solver difference is 2.492e-6;
    # 5e-6 covers that stopping-point delta without obscuring reconstruction bugs.
    assert float(output.error) == pytest.approx(
        _vector(outputs / "pilot_error.csv")[0],
        abs=5e-6,
    )
    # The maximum observed R2 delta is 5.449e-5 on the same fitted solution.
    np.testing.assert_allclose(
        output.r2,
        _vector(outputs / "pilot_r2.csv"),
        atol=6e-5,
        rtol=0,
    )

    np.testing.assert_array_equal(output.z, x @ output.a.T)
    matlab_a = _matrix(outputs / "pilot_a_raw.csv")
    matlab_z = _matrix(outputs / "pilot_z.csv")
    np.testing.assert_allclose(matlab_z, x @ matlab_a.T, atol=2e-14, rtol=0)
    python_basis = np.linalg.qr(output.z, mode="reduced")[0]
    matlab_basis = np.linalg.qr(matlab_z, mode="reduced")[0]
    subspace_cosines = np.linalg.svd(
        python_basis.T @ matlab_basis,
        compute_uv=False,
    )
    # The less-aligned direction has 1-cos(theta)=6.469e-9.
    np.testing.assert_allclose(subspace_cosines, np.ones(2), atol=1e-8, rtol=0)

    # C must include every performance column.  Together B/C reconstruct the
    # exact objective that PILOT reports, catching a former n+1 0-based slice.
    assert output.c.shape == (2, y.shape[1])
    reconstruction = output.z @ np.vstack((output.b, output.c.T)).T
    reconstructed_error = np.sum((np.column_stack((x, y)) - reconstruction) ** 2)
    assert float(output.error) == pytest.approx(reconstructed_error, abs=1e-10)


def test_current_matlab_cloister_numerical_output() -> None:
    """Reproduce both CLOISTER hull cycles to floating-point precision."""
    root = _BUILD / "cloister" / "default"
    inputs = root / "inputs"
    outputs = root / "outputs"
    output = CloisterStage.cloister(
        _matrix(inputs / "x.csv"),
        _matrix(inputs / "projection_a.csv"),
        CloisterOptions.default(
            p_val=_number("cloister", "pval"),
            c_thres=_number("cloister", "corrThreshold"),
            max_features=_integer("cloister", "maxFeatures"),
        ),
    )

    # MATLAB repeats the first point as a closing row; scipy returns each hull
    # vertex once. Starting vertex and traversal direction are also arbitrary.
    _assert_same_closed_cycle(output.z_edge, _matrix(outputs / "z_edge.csv"))
    _assert_same_closed_cycle(output.z_ecorr, _matrix(outputs / "z_ecorr.csv"))


def test_current_matlab_pythia_build_numerical_output(
    current_pythia_run: _CurrentPythiaRun,
) -> None:
    """Bridge exported KNN params and exactly reproduce full-fit predictions.

    Correlation distance in two dimensions creates many equidistant neighbours;
    MATLAB and sklearn can choose different tied neighbours, so those two posterior
    columns are compared by their exact KNN probability lattice and class decision.
    Non-correlation posteriors and every final prediction remain numerical oracles.
    Cross-validation fold assignment is implementation-specific and is not treated as
    a pseudo-oracle for final-model parity.
    """
    root = _BUILD / "pythia" / "trace3_default"
    inputs = root / "inputs"
    outputs = root / "outputs"
    output = current_pythia_run.output
    expected_y_hat = _matrix(outputs / "yhat.csv").astype(np.bool_)

    np.testing.assert_allclose(
        output.mu,
        _vector(outputs / "normalization_mu.csv"),
        atol=1e-14,
        rtol=0,
    )
    np.testing.assert_allclose(
        output.sigma,
        _vector(outputs / "normalization_sigma.csv"),
        atol=1e-14,
        rtol=0,
    )
    np.testing.assert_array_equal(output.y_hat, expected_y_hat)
    np.testing.assert_array_equal(output.box_consnt, current_pythia_run.params[:, 0])
    np.testing.assert_array_equal(output.k_scale, current_pythia_run.params[:, 1])

    labels = _labels(inputs / "algorithm_labels.csv")
    hyperparameters = pd.read_csv(outputs / "hyperparameters.csv")
    assert hyperparameters["algo"].astype(str).tolist() == labels
    for estimator, (neighbors_raw, metric_raw) in zip(
        output.svm,
        current_pythia_run.params,
        strict=True,
    ):
        assert isinstance(estimator, KNeighborsClassifier)
        assert estimator.n_neighbors == int(neighbors_raw)
        assert estimator.metric == _KNN_METRICS[int(metric_raw) - 1]

    _assert_knn_probability_semantics(
        output.pr0_hat,
        _matrix(outputs / "pr0hat.csv"),
        expected_y_hat,
        current_pythia_run.params,
    )

    matlab_selection = _vector(outputs / "selection0.csv").astype(np.int_)
    np.testing.assert_array_equal(output.selection0 == -1, matlab_selection == 0)
    qda_index = labels.index("QDA")
    assert (
        np.count_nonzero(_matrix(inputs / "y_bin.csv")[:, qda_index]) == _QDA_GOOD_COUNT
    )
    assert any(
        "least populated class" in message
        for message in current_pythia_run.sklearn_warnings
    )


def test_current_matlab_pythia_explore_numerical_output(
    current_pythia_run: _CurrentPythiaRun,
) -> None:
    """Run production explore inference and exactly reproduce MATLAB decisions."""
    root = _EXPLORE / "pythia" / "trace3_default"
    z = _matrix(root / "inputs" / "z.csv")
    harness = cast(InstanceSpace, _ExplorePythiaHarness(current_pythia_run))

    y_hat, pr0_hat, selection0 = InstanceSpace._explore_pythia(  # noqa: SLF001
        harness,
        z,
    )
    expected_y_hat = _matrix(root / "outputs" / "predictions.csv").astype(np.bool_)
    np.testing.assert_array_equal(y_hat, expected_y_hat)
    _assert_knn_probability_semantics(
        pr0_hat,
        _matrix(root / "outputs" / "probabilities.csv"),
        expected_y_hat,
        current_pythia_run.params,
    )

    assert np.all((selection0 >= -1) & (selection0 < y_hat.shape[1]))
    selected = selection0 >= 0
    rows = np.arange(y_hat.shape[0])[selected]
    assert np.all(y_hat[rows, selection0[selected]])
