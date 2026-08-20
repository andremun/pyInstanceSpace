"""Native 3D TRACE parity reader for the verified MATLAB reference bundle."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from numpy.typing import NDArray

from instancespace.data.model import Footprint, TraceOut
from instancespace.data.options import (
    GeneralOptions,
    ParallelOptions,
    PythiaOptions,
    TraceOptions,
)
from instancespace.stages.trace import TraceInputs, TraceOutputs, TraceStage
from instancespace.utils.alpha_shape import AlphaShape3D, TetrahedralMesh
from tools.fixture_provenance import validate_bundle

_CURRENT = Path(__file__).parent / "fixtures" / "matlab" / "current"
_BUNDLE = Path(os.environ.get("PYIS_MATLAB_REFERENCE_BUNDLE", str(_CURRENT)))
_ALLOW_DIAGNOSTIC = os.environ.get("PYIS_ALLOW_DIAGNOSTIC_FIXTURES") == "1"
_PROFILE_V2 = "pyinstancespace.reference-export/v2"
_VERIFIED_TRUST = "matlab-verified"
_DIAGNOSTIC_TRUST = "matlab-diagnostic"
_VARIANT = "pilot_standard_analytic_3d"
_SCALAR_TOLERANCE = 2e-10
_THREE_DIMENSIONS = 3


def _trace3d_bundle_available(
    bundle: Path,
    *,
    allow_diagnostic: bool,
) -> bool:
    manifest_path = bundle / "manifest.json"
    if not manifest_path.is_file():
        return False
    document: object = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(document, dict) or document.get("profile") != _PROFILE_V2:
        return False
    report = validate_bundle(bundle, allow_diagnostic=allow_diagnostic)
    return report.trust == _VERIFIED_TRUST or (
        allow_diagnostic and report.trust == _DIAGNOSTIC_TRUST
    )


def _has_trace3d_bundle() -> bool:
    return _trace3d_bundle_available(
        _BUNDLE,
        allow_diagnostic=_ALLOW_DIAGNOSTIC,
    )


pytestmark = pytest.mark.skipif(
    not _has_trace3d_bundle(),
    reason="the MATLAB reference bundle does not yet contain native 3D TRACE",
)


@dataclass(frozen=True)
class _ExpectedMesh:
    vertices: NDArray[np.double]
    tetrahedra: NDArray[np.int_]
    faces: NDArray[np.int_]
    spectrum: NDArray[np.double]


@dataclass(frozen=True)
class _Trace3DCase:
    labels: tuple[str, ...]
    build_z: NDArray[np.double]
    built: TraceOut
    metrics: pd.DataFrame
    meshes: dict[tuple[str, str], _ExpectedMesh]
    explore_z: NDArray[np.double]
    expected_membership: pd.DataFrame
    actual_membership: pd.DataFrame
    rescored: TraceOut


def _indexed_frame(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, index_col=0, float_precision="round_trip")


def _double_matrix(path: Path) -> NDArray[np.double]:
    return np.asarray(_indexed_frame(path).to_numpy(dtype=np.double), dtype=np.double)


def _bool_matrix(path: Path) -> NDArray[np.bool_]:
    return np.asarray(_indexed_frame(path).to_numpy(dtype=np.bool_), dtype=np.bool_)


def _int_vector(path: Path) -> NDArray[np.int_]:
    return np.asarray(
        _indexed_frame(path).to_numpy(dtype=np.int_).reshape(-1),
        dtype=np.int_,
    )


def _labels(path: Path) -> tuple[str, ...]:
    return tuple(
        pd.read_csv(path, float_precision="round_trip")["algorithm_name"]
        .astype(str)
        .tolist(),
    )


def _connectivity(path: Path, width: int) -> NDArray[np.int_]:
    frame = pd.read_csv(path, float_precision="round_trip")
    if frame.empty:
        return np.empty((0, width), dtype=np.int_)
    return np.asarray(frame.iloc[:, 1:].to_numpy(dtype=np.int_) - 1, dtype=np.int_)


def _expected_mesh(output_root: Path, prefix: str) -> _ExpectedMesh:
    vertices_frame = pd.read_csv(
        output_root / f"{prefix}_vertices.csv",
        float_precision="round_trip",
    )
    vertices = (
        np.asarray(
            vertices_frame.iloc[:, 1:].to_numpy(dtype=np.double),
            dtype=np.double,
        )
        if not vertices_frame.empty
        else np.empty((0, 3), dtype=np.double)
    )
    spectrum_frame = pd.read_csv(
        output_root / f"{prefix}_alpha_spectrum.csv",
        float_precision="round_trip",
    )
    spectrum = (
        np.asarray(spectrum_frame["alpha"].to_numpy(dtype=np.double), dtype=np.double)
        if not spectrum_frame.empty
        else np.empty(0, dtype=np.double)
    )
    return _ExpectedMesh(
        vertices,
        _connectivity(output_root / f"{prefix}_tetrahedra.csv", 4),
        _connectivity(output_root / f"{prefix}_boundary_faces.csv", 3),
        spectrum,
    )


def _trace_out(outputs: TraceOutputs) -> TraceOut:
    return TraceOut(
        outputs.space,
        outputs.good,
        outputs.best,
        outputs.hard,
        outputs.trace_summary,
    )


@cache
def _case() -> _Trace3DCase:
    build_root = _BUNDLE / "build_data" / "trace" / _VARIANT
    explore_root = _BUNDLE / "explore_data" / "trace" / _VARIANT
    build_inputs = build_root / "inputs"
    labels = _labels(build_inputs / "algorithm_labels.csv")
    build_z = _double_matrix(build_inputs / "z.csv")
    build_y_bin = _bool_matrix(build_inputs / "y_bin.csv")
    build_y_hat = _bool_matrix(build_inputs / "y_hat.csv")
    build_p = _int_vector(build_inputs / "p.csv")
    build_beta = _bool_matrix(build_inputs / "beta.csv").reshape(-1)
    outputs = TraceStage._run(  # noqa: SLF001
        TraceInputs(
            z=build_z,
            selection0=np.zeros(build_z.shape[0], dtype=np.int_),
            p=build_p,
            beta=build_beta,
            algo_labels=list(labels),
            y_hat=build_y_hat,
            y_bin=build_y_bin,
            trace_options=TraceOptions.default(
                method="trace3",
                purity=0.6,
                contra=False,
                min_instances=4,
                min_area_frac=0.01,
            ),
            parallel_options=ParallelOptions.default(flag=False, n_cores=1),
            general_options=GeneralOptions.default(verbose=False, seed=42),
            pythia_options=PythiaOptions.default(skip=True),
        ),
    )
    built = _trace_out(outputs)
    output_root = build_root / "outputs"
    meshes = {
        (kind, label): _expected_mesh(output_root, f"{kind}_{label}")
        for kind in ("good", "best")
        for label in labels
    }
    meshes[("hard", "")] = _expected_mesh(output_root, "hard")

    explore_inputs = explore_root / "inputs"
    explore_z_frame = _indexed_frame(explore_inputs / "z.csv")
    explore_z = np.asarray(explore_z_frame.to_numpy(dtype=np.double), dtype=np.double)
    explore_y_bin = _bool_matrix(explore_inputs / "y_bin.csv")
    explore_p = _int_vector(explore_inputs / "p.csv")
    explore_beta = _bool_matrix(explore_inputs / "beta.csv").reshape(-1)
    expected_membership = _indexed_frame(
        explore_root / "outputs" / "membership.csv",
    ).astype(bool)
    actual_membership = pd.DataFrame(
        np.column_stack(
            [
                *(_covers(footprint, explore_z) for footprint in built.good),
                *(_covers(footprint, explore_z) for footprint in built.best),
            ],
        ),
        index=explore_z_frame.index,
        columns=expected_membership.columns,
    )
    rescored = TraceStage.rescore(
        built,
        explore_z,
        explore_y_bin,
        explore_p,
        explore_beta,
        list(labels),
    )
    return _Trace3DCase(
        labels,
        build_z,
        built,
        pd.read_csv(
            output_root / "raw_metrics.csv",
            keep_default_na=False,
            float_precision="round_trip",
        ),
        meshes,
        explore_z,
        expected_membership,
        actual_membership,
        rescored,
    )


def _covers(footprint: Footprint, points: NDArray[np.double]) -> NDArray[np.bool_]:
    if not isinstance(footprint.polygon, TetrahedralMesh):
        return np.zeros(points.shape[0], dtype=np.bool_)
    return footprint.polygon.covers(points)


def _footprint(case: _Trace3DCase, kind: str, label: str) -> Footprint:
    if kind == "space":
        return case.built.space
    if kind == "hard":
        return case.built.hard
    index = case.labels.index(label)
    return case.built.good[index] if kind == "good" else case.built.best[index]


def _coordinate_key(point: NDArray[np.double]) -> tuple[float, ...]:
    return tuple(float(value) for value in point)


def _canonical_cells(
    vertices: NDArray[np.double],
    connectivity: NDArray[np.int_],
) -> set[tuple[tuple[float, ...], ...]]:
    return {
        tuple(sorted(_coordinate_key(vertices[index]) for index in simplex))
        for simplex in connectivity
    }


def _metric_number(row: pd.Series[Any], column: str) -> float:
    return float(row[column])


def _assert_footprint_metrics(
    footprint: Footprint,
    row: pd.Series[Any],
    *,
    compare_good_elements: bool = True,
) -> None:
    assert footprint.measure == pytest.approx(
        _metric_number(row, "measure"),
        abs=_SCALAR_TOLERANCE,
    )
    assert footprint.elements == int(float(row["elements"]))
    if compare_good_elements:
        assert footprint.good_elements == int(float(row["good_elements"]))
    assert footprint.density == pytest.approx(
        _metric_number(row, "density"),
        abs=_SCALAR_TOLERANCE,
    )
    assert footprint.purity == pytest.approx(
        _metric_number(row, "purity"),
        abs=_SCALAR_TOLERANCE,
    )


def test_current_matlab_trace3_3d_build_metrics_and_topology() -> None:
    """Match every 3D footprint through coordinate-canonical tetrahedra and faces."""
    case = _case()
    for _, raw in case.metrics.iterrows():
        kind = str(raw["kind"])
        label = str(raw["algorithm"])
        footprint = _footprint(case, kind, label)
        if kind == "space":
            # MATLAB leaves the whole-space goodElements field unset; Python keeps
            # the useful in-memory count.  The remaining serialized metrics agree.
            _assert_footprint_metrics(
                footprint,
                raw,
                compare_good_elements=False,
            )
            assert footprint.dimension == _THREE_DIMENSIONS
            assert footprint.polygon is None
            continue

        _assert_footprint_metrics(footprint, raw)
        expected = case.meshes[(kind, label)]
        empty = bool(int(float(raw["empty"])))
        if empty:
            assert footprint.polygon is None
            assert expected.vertices.shape == (0, 3)
            assert expected.tetrahedra.shape == (0, 4)
            assert expected.faces.shape == (0, 3)
            assert expected.spectrum.size == 0
            continue

        assert isinstance(footprint.polygon, TetrahedralMesh)
        actual = footprint.polygon
        assert {_coordinate_key(point) for point in actual.vertices} == {
            _coordinate_key(point) for point in expected.vertices
        }
        assert _canonical_cells(actual.vertices, actual.tetrahedra) == _canonical_cells(
            expected.vertices,
            expected.tetrahedra,
        )
        assert _canonical_cells(
            actual.vertices,
            actual.boundary_faces,
        ) == _canonical_cells(
            expected.vertices,
            expected.faces,
        )
        assert actual.alpha == pytest.approx(
            _metric_number(raw, "alpha_radius"),
            abs=_SCALAR_TOLERANCE,
        )
        assert actual.region_threshold == pytest.approx(
            _metric_number(raw, "region_threshold"),
            abs=_SCALAR_TOLERANCE,
        )
        assert actual.region_count == int(float(raw["region_count"]))
        assert actual.tetrahedra.shape[0] == int(float(raw["tetrahedron_count"]))
        assert actual.boundary_faces.shape[0] == int(float(raw["boundary_face_count"]))
        assert expected.spectrum.size == int(float(raw["alpha_spectrum_count"]))
        assert actual.volume == pytest.approx(
            _metric_number(raw, "volume"),
            abs=_SCALAR_TOLERANCE,
        )
        assert actual.surface_area == pytest.approx(
            _metric_number(raw, "surface_area"),
            abs=_SCALAR_TOLERANCE,
        )


def test_current_matlab_trace3_3d_alpha_spectra() -> None:
    """Match MATLAB's full descending Delaunay circumsphere spectra."""
    case = _case()
    for (kind, label), expected in case.meshes.items():
        raw = case.metrics[
            (case.metrics["kind"] == kind) & (case.metrics["algorithm"] == label)
        ].iloc[0]
        if bool(int(float(raw["empty"]))):
            continue
        if kind == "good":
            column = case.labels.index(label)
            support = _bool_matrix(
                _BUNDLE / "build_data" / "trace" / _VARIANT / "inputs" / "y_bin.csv",
            )[:, column]
        elif kind == "best":
            support = (
                _int_vector(
                    _BUNDLE / "build_data" / "trace" / _VARIANT / "inputs" / "p.csv",
                )
                == case.labels.index(label) + 1
            )
        else:
            beta = _bool_matrix(
                _BUNDLE / "build_data" / "trace" / _VARIANT / "inputs" / "beta.csv",
            ).reshape(-1)
            support = ~beta
        shape = AlphaShape3D.from_points(case.build_z[support])
        assert shape is not None
        np.testing.assert_allclose(
            shape.spectrum,
            expected.spectrum,
            atol=_SCALAR_TOLERANCE,
            rtol=2e-12,
        )
        assert shape.critical_radius >= float(np.min(shape.spectrum))


def test_current_matlab_trace3_3d_explore_membership_and_rescore() -> None:
    """Use inclusive tetrahedral membership and retain geometry while rescoring."""
    case = _case()
    pd.testing.assert_frame_equal(
        case.actual_membership,
        case.expected_membership,
        check_dtype=True,
    )
    expected_summary = _indexed_frame(
        _BUNDLE / "explore_data" / "trace" / _VARIANT / "outputs" / "eval_summary.csv",
    )
    actual_summary = case.rescored.summary.set_index("Algorithm")
    np.testing.assert_allclose(
        actual_summary.to_numpy(dtype=np.double),
        expected_summary.to_numpy(dtype=np.double),
        atol=5e-4,
        rtol=0,
    )
    for trained, rescored in zip(
        [*case.built.good, *case.built.best, case.built.hard],
        [*case.rescored.good, *case.rescored.best, case.rescored.hard],
        strict=True,
    ):
        assert rescored.polygon is trained.polygon
        assert rescored.measure == trained.measure
