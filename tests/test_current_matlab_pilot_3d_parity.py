"""PILOT 2D/3D parity readers for the verified MATLAB reference bundle."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest
from numpy.typing import NDArray
from scipy.spatial.distance import pdist

from instancespace.data.options import GeneralOptions, ParallelOptions, PilotOptions
from instancespace.instance_space import InstanceSpace
from instancespace.model import Model
from instancespace.stages.pilot import PilotInput, PilotOutput, PilotStage
from instancespace.stages.pilot_viewpoint import PilotViewpointResult, pilot_viewpoint
from tools.fixture_provenance import validate_bundle

_CURRENT = Path(__file__).parent / "fixtures" / "matlab" / "current"
_BUNDLE = Path(os.environ.get("PYIS_MATLAB_REFERENCE_BUNDLE", str(_CURRENT)))
_ALLOW_DIAGNOSTIC = os.environ.get("PYIS_ALLOW_DIAGNOSTIC_FIXTURES") == "1"

_PROFILE_V2 = "pyinstancespace.reference-export/v2"
_VERIFIED_TRUST = "matlab-verified"
_DIAGNOSTIC_TRUST = "matlab-diagnostic"
_VARIANTS = (
    "pilot_standard_analytic_3d",
    "pilot_standard_numerical_3d_x0",
    "pilot_standard_numerical_3d_precalc",
    "pilot_pls_2d",
    "pilot_pls_3d_grouped",
)
_DIRECT_VARIANTS = (
    "pilot_standard_analytic_3d",
    "pilot_pls_2d",
    "pilot_pls_3d_grouped",
)
_VIEW_VARIANTS = (
    "pilot_standard_analytic_3d",
    "pilot_standard_numerical_3d_x0",
    "pilot_pls_3d_grouped",
)
_MIN_TOPOLOGY_SCORE = 0.60
_ORTHOGONALITY_WEIGHT = 0.2
# Retain the audited cross-solver envelopes for analytic and numerical X0
# viewpoints; v0.9.1's refreshed PLS gap is 0.056%. These limits compare the
# optimized scientific quantity instead of raw coordinates; PLS identity is
# additionally pinned by its projection plane.
_MAX_RELATIVE_VIEW_OBJECTIVE_GAP = {
    "pilot_standard_analytic_3d": 0.20,
    "pilot_standard_numerical_3d_x0": 0.035,
    "pilot_pls_3d_grouped": 0.001,
}
_MAX_VIEW_TOPOLOGY_DROP: dict[str, float | None] = {
    # This oracle deliberately uses ntries=1. In MATLAB, topology is only
    # used to choose among completed restarts, so it is not a selection
    # constraint in this one-start case. SciPy and fminunc may follow
    # different valid quasi-Newton trajectories; require the shared absolute
    # scientific floor instead of a platform-calibrated delta from MATLAB.
    "pilot_standard_analytic_3d": None,
    "pilot_standard_numerical_3d_x0": 0.015,
    "pilot_pls_3d_grouped": 0.002,
}
# On the numerical-X0 oracle, only 22 of these 256 deterministic random planes
# pass the objective/topology contract. Keep at least 90% discrimination.
_RANDOM_VIEW_TRIALS = 256
_MAX_RANDOM_VIEW_PASS_FRACTION = 0.10
# The integrated Python PILOT->viewpoint path reaches a locally equivalent PLS
# plane with 1-cos(theta)=5.34e-4; a 1e-3 cap remains far below the 0.19 gap
# produced by replacing either MATLAB plane with the raw XY coordinate plane.
_PLS_PLANE_COSINE_TOLERANCE = 1e-3


def _has_v2_bundle() -> bool:
    manifest_path = _BUNDLE / "manifest.json"
    if not manifest_path.is_file():
        return False
    document: object = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(document, dict) or document.get("profile") != _PROFILE_V2:
        return False
    report = validate_bundle(_BUNDLE, allow_diagnostic=_ALLOW_DIAGNOSTIC)
    return report.trust == _VERIFIED_TRUST or (
        _ALLOW_DIAGNOSTIC and report.trust == _DIAGNOSTIC_TRUST
    )


pytestmark = pytest.mark.skipif(
    not _has_v2_bundle(),
    reason="the installed MATLAB reference bundle does not yet contain PILOT v2",
)


@dataclass(frozen=True)
class _PilotCase:
    """One exported case and its Python PILOT replay."""

    variant: str
    x: NDArray[np.double]
    y: NDArray[np.double]
    output: PilotOutput
    expected_a: NDArray[np.double]
    expected_b: NDArray[np.double]
    expected_c: NDArray[np.double]
    expected_z: NDArray[np.double]
    expected_error: float
    expected_r2: NDArray[np.double]


def _frame(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, float_precision="round_trip")


def _matrix(path: Path) -> NDArray[np.double]:
    frame = _frame(path)
    if str(frame.columns[0]) == "Row":
        frame = frame.iloc[:, 1:]
    return np.asarray(frame.to_numpy(dtype=np.double), dtype=np.double)


def _vector(path: Path) -> NDArray[np.double]:
    return _matrix(path).reshape(-1)


def _labels(path: Path) -> list[str]:
    return _frame(path).iloc[:, 0].astype(str).tolist()


def _resolved_options(variant: str) -> dict[str, Any]:
    document: object = json.loads(
        (_BUNDLE / "resolved_options" / f"{variant}.json").read_text(
            encoding="utf-8",
        ),
    )
    assert isinstance(document, dict)
    options = document["options"]
    assert isinstance(options, dict)
    return cast(dict[str, Any], options)


def _pilot_options(variant: str) -> PilotOptions:
    options = _resolved_options(variant)
    pilot = cast(dict[str, Any], options["pilot"])
    build_inputs = _BUNDLE / "build_data" / "pilot" / variant / "inputs"
    x0 = (
        _matrix(build_inputs / "x0.csv")
        if (build_inputs / "x0.csv").is_file()
        else None
    )
    precalc = (
        _matrix(build_inputs / "precalc_alpha.csv")
        if (build_inputs / "precalc_alpha.csv").is_file()
        else None
    )
    matlab_groups = cast(list[list[int]], pilot["viewGroups"])
    groups = tuple(tuple(index - 1 for index in group) for group in matlab_groups)
    return PilotOptions.default(
        analytic=cast(bool, pilot["analytic"]),
        n_tries=cast(int, pilot["ntries"]),
        x0=x0,
        precalc_alpha=precalc,
        cost_weight=float(pilot["alpha"]),
        method=cast(str, pilot["method"]),
        dims=cast(int, pilot["dims"]),
        view_groups=groups,
    )


@cache
def _case(variant: str) -> _PilotCase:
    root = _BUNDLE / "build_data" / "pilot" / variant
    inputs = root / "inputs"
    outputs = root / "outputs"
    x = _matrix(inputs / "x.csv")
    y = _matrix(inputs / "y.csv")
    output = PilotStage._run(
        PilotInput(
            x=x,
            y=y,
            feat_labels=_labels(inputs / "feature_labels.csv"),
            pilot_options=_pilot_options(variant),
            parallel_options=ParallelOptions.default(flag=False, n_cores=1),
            general_options=GeneralOptions.default(verbose=False, seed=42),
            y_bin=np.zeros_like(y, dtype=np.bool_),
        ),
    )
    return _PilotCase(
        variant=variant,
        x=x,
        y=y,
        output=output,
        expected_a=_matrix(outputs / "pilot_a_raw.csv"),
        expected_b=_matrix(outputs / "pilot_b.csv"),
        expected_c=_matrix(outputs / "pilot_c.csv"),
        expected_z=_matrix(outputs / "pilot_z.csv"),
        expected_error=float(_vector(outputs / "pilot_error.csv")[0]),
        expected_r2=_vector(outputs / "pilot_r2.csv"),
    )


def _component_signs(
    actual: NDArray[np.double],
    expected: NDArray[np.double],
) -> NDArray[np.double]:
    dots = np.sum(actual * expected, axis=1)
    signs = np.sign(dots)
    signs[signs == 0] = 1
    return np.asarray(signs, dtype=np.double)


def _assert_component_parity(case: _PilotCase) -> None:
    signs = _component_signs(case.output.a, case.expected_a)
    np.testing.assert_allclose(
        case.output.a * signs[:, None],
        case.expected_a,
        atol=1e-11,
        rtol=0,
    )
    np.testing.assert_allclose(
        case.output.b * signs[None, :],
        case.expected_b,
        atol=1e-11,
        rtol=0,
    )
    np.testing.assert_allclose(
        case.output.c * signs[:, None],
        case.expected_c,
        atol=1e-11,
        rtol=0,
    )
    np.testing.assert_allclose(
        case.output.z * signs[None, :],
        case.expected_z,
        atol=2e-11,
        rtol=0,
    )
    assert float(case.output.error) == pytest.approx(case.expected_error, abs=1e-9)
    np.testing.assert_allclose(
        case.output.r2,
        case.expected_r2,
        atol=1e-11,
        rtol=0,
    )


@pytest.mark.parametrize("variant", _DIRECT_VARIANTS)
def test_current_matlab_pilot_analytic_and_simpls_components(variant: str) -> None:
    """Match identifiable analytic/SIMPLS factors after component sign alignment."""
    case = _case(variant)
    _assert_component_parity(case)
    assert case.output.alpha is None
    assert case.output.X0 is None
    assert case.output.eoptim is None
    assert case.output.perf is None


def test_current_matlab_pilot_pls_2d_is_prefix_of_pls_3d() -> None:
    """Keep R2026a SIMPLS 2D unchanged while extending it to three components."""
    two = _case("pilot_pls_2d")
    three = _case("pilot_pls_3d_grouped")
    signs = _component_signs(two.output.a, three.output.a[:2])
    np.testing.assert_allclose(
        two.output.a * signs[:, None],
        three.output.a[:2],
        atol=1e-12,
        rtol=0,
    )
    np.testing.assert_allclose(
        two.output.z * signs[None, :],
        three.output.z[:, :2],
        atol=1e-12,
        rtol=0,
    )


def test_current_matlab_pilot_3d_precalculated_solution() -> None:
    """Decode MATLAB's selected 3D column-major solution without optimization."""
    case = _case("pilot_standard_numerical_3d_precalc")
    output = case.output
    assert output.alpha is not None
    assert output.alpha.shape == (84, 1)
    np.testing.assert_allclose(output.a, case.expected_a, atol=2e-15, rtol=0)
    np.testing.assert_allclose(output.b, case.expected_b, atol=2e-15, rtol=0)
    np.testing.assert_allclose(output.c, case.expected_c, atol=2e-15, rtol=0)
    np.testing.assert_allclose(output.z, case.expected_z, atol=2e-14, rtol=0)
    assert float(output.error) == pytest.approx(case.expected_error, abs=1e-9)
    np.testing.assert_allclose(output.r2, case.expected_r2, atol=1e-11, rtol=0)
    assert output.X0 is None
    assert output.eoptim is None
    assert output.perf is None


def test_current_matlab_pilot_3d_x0_solver_quality() -> None:
    """Match fixed-X0 objectives and the optimizer-invariant 3D subspace."""
    case = _case("pilot_standard_numerical_3d_x0")
    output = case.output
    outputs = _BUNDLE / "build_data" / "pilot" / case.variant / "outputs"
    assert output.X0 is not None
    assert output.eoptim is not None
    assert output.perf is not None
    np.testing.assert_array_equal(output.X0, _matrix(outputs / "pilot_x0.csv"))
    assert output.X0.shape == (84, 3)
    np.testing.assert_allclose(
        output.eoptim,
        _vector(outputs / "pilot_eoptim.csv"),
        # R2026a fminunc and SciPy BFGS stop on the same shallow minimum;
        # the observed three-start objective delta is at most 2.390e-5.
        atol=5e-5,
        rtol=0,
    )
    matlab_perf = _vector(outputs / "pilot_perf.csv")
    # Trial coordinates and near-equal restart ordering are not unique. Compare
    # the best achieved scientific score, rather than requiring both optimizers
    # to attach it to the same start column.
    assert float(np.max(output.perf)) == pytest.approx(
        float(np.max(matlab_perf)),
        abs=1e-2,
    )
    assert np.all(output.perf >= _MIN_TOPOLOGY_SCORE)
    assert float(output.error) == pytest.approx(case.expected_error, rel=1e-4)
    # The factorization manifold permits different per-column reconstructions
    # at nearly identical total error; the observed maximum R2 delta is 0.01680.
    np.testing.assert_allclose(output.r2, case.expected_r2, atol=0.025, rtol=0)

    python_basis = np.linalg.qr(output.z, mode="reduced")[0]
    matlab_basis = np.linalg.qr(case.expected_z, mode="reduced")[0]
    subspace_cosines = np.linalg.svd(
        python_basis.T @ matlab_basis,
        compute_uv=False,
    )
    # The least-aligned direction has 1-cos(theta)=3.5433e-4.
    np.testing.assert_allclose(subspace_cosines, np.ones(3), atol=7.5e-4, rtol=0)


@pytest.mark.parametrize("variant", _VARIANTS)
def test_current_matlab_pilot_explore_projection_is_dimension_generic(
    variant: str,
) -> None:
    """Replay MATLAB's public uncentred explore projection for 2D and 3D."""
    root = _BUNDLE / "explore_data" / "pilot" / variant
    x_frame = _frame(root / "inputs" / "x.csv")
    expected_frame = _frame(root / "outputs" / "pilot_z.csv")
    x = np.asarray(x_frame.iloc[:, 1:].to_numpy(dtype=np.double), dtype=np.double)
    a = _matrix(root / "inputs" / "projection_a.csv")

    space = InstanceSpace.__new__(InstanceSpace)
    space._model = cast(
        Model,
        SimpleNamespace(pilot=SimpleNamespace(a=a)),
    )
    actual = space._explore_pilot(x)

    np.testing.assert_allclose(
        actual,
        expected_frame.iloc[:, 1:].to_numpy(dtype=np.double),
        atol=2e-13,
        rtol=0,
    )
    assert (
        x_frame.iloc[:, 0].astype(str).tolist()
        == expected_frame.iloc[:, 0].astype(str).tolist()
    )
    assert list(expected_frame.columns[1:]) == [
        f"z_{index}" for index in range(1, a.shape[0] + 1)
    ]


def _exported_viewpoint(
    variant: str,
) -> tuple[tuple[tuple[int, ...], ...], tuple[NDArray[np.double], ...]]:
    outputs = _BUNDLE / "build_data" / "pilot" / variant / "outputs"
    group_frame = _frame(outputs / "viewpoint_groups.csv")
    groups = tuple(
        tuple(group["algorithm_index"].astype(int).to_numpy() - 1)
        for _, group in group_frame.groupby("group", sort=True)
    )
    a_frame = _frame(outputs / "viewpoint_a.csv")
    views = tuple(
        np.asarray(group.iloc[:, 2:].to_numpy(dtype=np.double), dtype=np.double)
        for _, group in a_frame.groupby("group", sort=True)
    )
    return groups, views


def _topology_score(z: NDArray[np.double], view: NDArray[np.double]) -> float:
    return float(np.corrcoef(pdist(z), pdist(z @ view.T))[0, 1])


def _view_quality(
    z: NDArray[np.double],
    y: NDArray[np.double],
    group: tuple[int, ...],
    view: NDArray[np.double],
) -> tuple[float, float]:
    """Return MATLAB-equivalent refitted objective and topology preservation."""
    projected = z @ view.T
    coefficients, *_ = np.linalg.lstsq(projected, y[:, group], rcond=None)
    reconstruction = projected @ coefficients
    unit_view = view / np.linalg.norm(view, axis=1, keepdims=True)
    penalty = _ORTHOGONALITY_WEIGHT * abs(float(unit_view[0] @ unit_view[1]))
    column_mse = np.nanmean(np.square(y[:, group] - reconstruction), axis=0)
    objective = float(np.nanmean(column_mse) + penalty)
    return objective, _topology_score(z, view)


def _view_meets_quality_contract(
    variant: str,
    expected: tuple[float, float],
    actual: tuple[float, float],
) -> bool:
    expected_objective, expected_topology = expected
    actual_objective, actual_topology = actual
    objective_limit = expected_objective * (
        1.0 + _MAX_RELATIVE_VIEW_OBJECTIVE_GAP[variant]
    )
    maximum_topology_drop = _MAX_VIEW_TOPOLOGY_DROP[variant]
    topology_floor = (
        _MIN_TOPOLOGY_SCORE
        if maximum_topology_drop is None
        else max(
            _MIN_TOPOLOGY_SCORE,
            expected_topology - maximum_topology_drop,
        )
    )
    return actual_objective <= objective_limit and actual_topology >= topology_floor


def _unit_normal(view: NDArray[np.double]) -> NDArray[np.double]:
    normal = np.cross(view[0], view[1])
    return np.asarray(normal / np.linalg.norm(normal), dtype=np.double)


@cache
def _python_viewpoint(variant: str) -> PilotViewpointResult:
    case = _case(variant)
    assert case.output.viewpoint is not None
    return case.output.viewpoint


@pytest.mark.parametrize("variant", _VIEW_VARIANTS)
def test_current_matlab_pilot_3d_viewpoint_scientific_parity(variant: str) -> None:
    """Compare viewpoint objective and topology without raw-BFGS coordinates."""
    case = _case(variant)
    expected_groups, expected_views = _exported_viewpoint(variant)
    actual = _python_viewpoint(variant)
    assert actual.groups == expected_groups

    for group, expected_view, actual_view, azimuth, elevation in zip(
        expected_groups,
        expected_views,
        actual.a,
        actual.azimuth,
        actual.elevation,
        strict=True,
    ):
        np.testing.assert_allclose(
            np.linalg.norm(actual_view, axis=1),
            np.ones(2),
            atol=2e-14,
            rtol=0,
        )
        component_signs = _component_signs(case.output.a, case.expected_a)
        aligned_actual_view = actual_view * component_signs[None, :]
        expected_quality = _view_quality(case.expected_z, case.y, group, expected_view)
        actual_quality = _view_quality(
            case.expected_z,
            case.y,
            group,
            aligned_actual_view,
        )
        assert _view_meets_quality_contract(
            variant,
            expected_quality,
            actual_quality,
        ), (
            f"{variant} viewpoint quality {actual_quality} does not preserve "
            f"MATLAB's objective/topology {expected_quality}"
        )
        if variant == "pilot_pls_3d_grouped":
            plane_cosine = abs(
                float(
                    _unit_normal(expected_view) @ _unit_normal(aligned_actual_view),
                ),
            )
            assert plane_cosine == pytest.approx(
                1.0,
                abs=_PLS_PLANE_COSINE_TOLERANCE,
            )

        actual_normal = _unit_normal(actual_view)
        actual_azimuth = float(np.arctan2(actual_normal[1], actual_normal[0]))
        actual_elevation = float(
            np.arctan2(
                actual_normal[2],
                np.hypot(actual_normal[0], actual_normal[1]),
            ),
        )
        assert azimuth == pytest.approx(actual_azimuth, abs=2e-14)
        assert elevation == pytest.approx(actual_elevation, abs=2e-14)


def test_current_matlab_numerical_view_quality_rejects_random_planes() -> None:
    """The optimizer-invariant contract rejects at least 90% of random views."""
    variant = "pilot_standard_numerical_3d_x0"
    case = _case(variant)
    groups, expected_views = _exported_viewpoint(variant)
    group = groups[0]
    expected = _view_quality(case.expected_z, case.y, group, expected_views[0])
    rng = np.random.default_rng(20260820)
    passing = 0
    for _ in range(_RANDOM_VIEW_TRIALS):
        candidate = rng.normal(size=(2, 3))
        candidate /= np.linalg.norm(candidate, axis=1, keepdims=True)
        quality = _view_quality(case.expected_z, case.y, group, candidate)
        passing += _view_meets_quality_contract(variant, expected, quality)

    assert passing / _RANDOM_VIEW_TRIALS <= _MAX_RANDOM_VIEW_PASS_FRACTION


def test_single_start_analytic_view_uses_absolute_topology_floor() -> None:
    """Do not turn one optimizer's trajectory into a cross-platform oracle."""
    variant = "pilot_standard_analytic_3d"
    expected = (0.88, 0.90)
    assert _view_meets_quality_contract(variant, expected, (0.88, 0.65))
    assert not _view_meets_quality_contract(variant, expected, (0.88, 0.59))


@pytest.mark.parametrize(
    ("variant", "candidate"),
    [
        (
            "pilot_standard_analytic_3d",
            np.asarray([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.double),
        ),
        (
            "pilot_standard_numerical_3d_x0",
            np.asarray([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.double),
        ),
    ],
)
def test_current_matlab_view_quality_rejects_deterministic_corruption(
    variant: str,
    candidate: NDArray[np.double],
) -> None:
    """Each solver-invariant contract rejects a fixed incoherent plane."""
    case = _case(variant)
    groups, expected_views = _exported_viewpoint(variant)
    expected = _view_quality(case.expected_z, case.y, groups[0], expected_views[0])
    corrupted = _view_quality(case.expected_z, case.y, groups[0], candidate)
    assert not _view_meets_quality_contract(variant, expected, corrupted)


def test_current_matlab_pls_plane_rejects_coherent_xy_substitution() -> None:
    """PLS' identifiable plane rejects an otherwise high-quality XY substitute."""
    variant = "pilot_pls_3d_grouped"
    case = _case(variant)
    groups, expected_views = _exported_viewpoint(variant)
    xy_view = np.asarray(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=np.double,
    )
    xy_normal = _unit_normal(xy_view)

    for group, expected_view in zip(groups, expected_views, strict=True):
        expected_quality = _view_quality(
            case.expected_z,
            case.y,
            group,
            expected_view,
        )
        xy_quality = _view_quality(case.expected_z, case.y, group, xy_view)
        assert xy_quality[0] <= expected_quality[0] * 1.01
        assert xy_quality[1] >= expected_quality[1]
        plane_cosine = abs(float(_unit_normal(expected_view) @ xy_normal))
        assert not np.isclose(
            plane_cosine,
            1.0,
            atol=_PLS_PLANE_COSINE_TOLERANCE,
            rtol=0,
        )


def test_current_matlab_pilot_replay_viewpoint_fallback_is_stable() -> None:
    """Pin MATLAB/Python fallback when solver X0 has the wrong viewpoint shape."""
    x0_case = _case("pilot_standard_numerical_3d_x0")
    x0 = pilot_viewpoint(
        x0_case.expected_z,
        x0_case.y,
        n_tries=1,
        x0=_pilot_options("pilot_standard_numerical_3d_x0").x0,
        parallel_options=ParallelOptions.default(flag=False, n_cores=1),
    )
    precalc = pilot_viewpoint(
        _case("pilot_standard_numerical_3d_precalc").expected_z,
        _case("pilot_standard_numerical_3d_precalc").y,
        n_tries=1,
        parallel_options=ParallelOptions.default(flag=False, n_cores=1),
    )
    assert x0.groups == precalc.groups
    for x0_view, precalc_view in zip(x0.a, precalc.a, strict=True):
        np.testing.assert_array_equal(x0_view, precalc_view)
    np.testing.assert_array_equal(x0.azimuth, precalc.azimuth)
    np.testing.assert_array_equal(x0.elevation, precalc.elevation)
