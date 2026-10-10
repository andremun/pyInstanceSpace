# SPDX-License-Identifier: LicenseRef-PolyForm-Noncommercial-1.0.0
# Copyright (c) 2024-2026 Mario Andrés Muñoz
# ruff: noqa: SLF001
"""Controlled local MATLAB references for boundary dimension and topology."""

import hashlib
import json
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.path import Path as PlotPath
from numpy.typing import NDArray
from scipy.spatial import ConvexHull
from shapely.geometry import MultiPolygon, Polygon

from instancespace import _serialisers as serialisers
from instancespace.data.model import CloisterOut, pointwise_covers
from instancespace.data.options import (
    CloisterOptions,
    GeneralOptions,
    ParallelOptions,
    TraceOptions,
)
from instancespace.stages.cloister import CloisterStage
from instancespace.stages.trace import TraceStage
from instancespace.utils.alpha_shape import AlphaShape2D

ROOT = Path(__file__).parent / "fixtures/matlab/geometry"
MANIFEST_SHA = "016be05cf4ef6be05f815e759b3306d6ab9e5ee36272f5a1beab672cccd77eff"


@pytest.fixture(scope="module", autouse=True)
def _verified_geometry_reference() -> None:
    """Authenticate the exact locally generated bundle and its exporter."""
    manifest_bytes = (ROOT / "manifest.json").read_bytes()
    assert hashlib.sha256(manifest_bytes).hexdigest() == MANIFEST_SHA
    manifest = json.loads(manifest_bytes)
    assert manifest["schema_version"] == "pyinstancespace.geometry-reference/v1"
    assert manifest["matlab_commit"] == "929acfd889e7a17c7ee40c4004ad1bc18639b0e6"
    generator = ROOT.parents[3] / manifest["generator"]
    assert (
        hashlib.sha256(generator.read_bytes()).hexdigest()
        == manifest["generator_sha256"]
    )
    expected = {entry["path"] for entry in manifest["files"]} | {"manifest.json"}
    assert {p.name for p in ROOT.glob("*.json")} == expected
    for entry in manifest["files"]:
        assert (
            hashlib.sha256((ROOT / entry["path"]).read_bytes()).hexdigest()
            == entry["sha256"]
        )


def _case(name: str) -> dict[str, Any]:
    return json.loads((ROOT / f"{name}.json").read_text())  # type: ignore[no-any-return]


def _surface(vertices: NDArray[np.double], faces: NDArray[np.int_]) -> float:
    triangles = vertices[faces]
    normals = np.cross(
        triangles[:, 1] - triangles[:, 0],
        triangles[:, 2] - triangles[:, 0],
    )
    return float(np.linalg.norm(normals, axis=1).sum() / 2)


def _sorted_rows(values: NDArray[np.double]) -> NDArray[np.double]:
    return np.asarray(values[np.lexsort(values.T[::-1])], dtype=np.double)


@pytest.mark.parametrize("name", ["solid", "coplanar", "fallback"])
def test_cloister_geometry_and_face_export(name: str, tmp_path: Path) -> None:
    case = _case(f"cloister_{name}")
    x, a = np.array(case["x"], float), np.array(case["a"], float)
    result = CloisterStage.cloister(
        x,
        a,
        CloisterOptions.default(max_features=case["options"]["maxFeatures"]),
    )
    # Keep existing two-array unpacking while adding derived face accessors.
    edge, corr = result
    for actual, reference, faces, ref_faces in (
        (edge, case["vertices"], result.z_edge_faces, case["faces"]),
        (
            corr,
            case["correlated_vertices"],
            result.z_ecorr_faces,
            case["correlated_faces"],
        ),
    ):
        reference_array = np.array(reference, float)
        np.testing.assert_allclose(
            _sorted_rows(actual),
            _sorted_rows(reference_array),
            atol=1e-12,
        )
        assert faces.ndim == 2
        assert faces.shape[1] == 3
        assert np.all((faces >= 0) & (faces < len(actual)))
        assert _surface(actual, faces) == pytest.approx(
            _surface(reference_array, np.array(ref_faces, int) - 1),
            abs=1e-12,
        )
        # Containment in the hull's intrinsic dimension covers planar bounds.
        centered = actual - actual.mean(axis=0)
        rank = np.linalg.matrix_rank(centered)
        _, _, basis = np.linalg.svd(centered, full_matrices=False)
        hull = ConvexHull(centered @ basis[:rank].T)
        projected = (x @ a.T - actual.mean(axis=0)) @ basis[:rank].T
        assert (
            np.max(projected @ hull.equations[:, :-1].T + hull.equations[:, -1]) < 1e-12
        )

    model = CloisterOut(edge, corr)
    joblib.dump(model, tmp_path / "boundary.joblib")
    restored = joblib.load(tmp_path / "boundary.joblib")
    np.testing.assert_array_equal(restored.z_edge_faces, model.z_edge_faces)
    serialisers._write_cloister_faces(restored, tmp_path)
    manifest = json.loads((tmp_path / "bounds_mesh_manifest.json").read_text())
    assert manifest["schema_version"] == "pyinstancespace.cloister-mesh/v1"
    assert manifest["index_base"] == 0
    np.testing.assert_array_equal(
        pd.read_csv(tmp_path / "bounds_faces.csv"),
        model.z_edge_faces,
    )
    # A subsequent 2D export cannot leave obsolete 3D face files behind.
    serialisers._write_cloister_faces(CloisterOut(edge[:, :2], corr[:, :2]), tmp_path)
    assert not (tmp_path / "bounds_faces.csv").exists()
    assert not (tmp_path / "bounds_mesh_manifest.json").exists()


def test_cloister_degenerate_3d_matches_matlab_rejection() -> None:
    case = _case("cloister_collinear")
    assert case["error"] == "ISA:CLOISTER:degenerateBoundary"
    with pytest.raises(ValueError, match="degenerate"):
        CloisterStage.cloister(
            np.array(case["x"], float),
            np.array(case["a"], float),
            CloisterOptions.default(),
        )


@pytest.mark.parametrize("name", ["hole", "components"])
@pytest.mark.parametrize("predictions", [False, True])
def test_trace_topology_membership_and_roundtrip(
    name: str,
    predictions: bool,
    tmp_path: Path,
) -> None:
    case = _case(f"trace_{name}")
    query = np.array(case["queries"], float)
    alpha = AlphaShape2D.from_points(np.array(case["support"], float))
    assert alpha is not None
    fixed = alpha.geometry(case["alpha"])
    assert fixed is not None
    assert fixed.area == pytest.approx(case["fixed_area"], abs=1e-12)
    np.testing.assert_array_equal(
        pointwise_covers(fixed, query),
        case["fixed_membership"],
    )
    z, labels = np.array(case["z"], float), np.array(case["labels"], bool)
    stage = TraceStage(
        z=z,
        y_bin=labels[:, None],
        p=np.ones(len(z), dtype=np.int_),
        beta=labels,
        algo_labels=["algorithm"],
        trace_opts=TraceOptions.default(
            method="trace3",
            purity=case["options"]["PI"],
            min_area_frac=0,
        ),
        parallel_opts=ParallelOptions.default(),
        general_opts=GeneralOptions.default(),
        y_hat=np.ones((len(z), 1), dtype=np.bool_) if predictions else None,
    )
    footprint = stage._trace3().good[0]
    polygon = footprint.polygon
    assert isinstance(polygon, Polygon | MultiPolygon)
    assert footprint.area == pytest.approx(case["trace_area"], abs=1e-12)
    assert footprint.elements == case["trace_elements"]
    assert footprint.good_elements == case["trace_good_elements"]
    assert footprint.purity == pytest.approx(case["trace_purity"])
    np.testing.assert_array_equal(
        pointwise_covers(polygon, query),
        case["trace_membership"],
    )
    parts = list(polygon.geoms) if isinstance(polygon, MultiPolygon) else [polygon]
    assert len(parts) == case["trace_regions"]
    boundary = np.array(case["trace_boundary"], float)
    cycles = int(np.isnan(boundary).all(axis=1).sum()) + 1
    assert sum(len(part.interiors) for part in parts) == cycles - len(parts)
    frame = serialisers._footprint_boundary_frame(polygon)
    frame.to_csv(tmp_path / "boundary.csv", index=False)
    loaded = pd.read_csv(tmp_path / "boundary.csv")
    restored = []
    for _, part in loaded.groupby("Part"):
        rings = {
            ring: rows.sort_values("Vertex")[["z_1", "z_2"]].to_numpy()
            for ring, rows in part.groupby("Ring")
        }
        restored.append(Polygon(rings.pop("exterior"), list(rings.values())))
    roundtrip = MultiPolygon(restored)
    assert roundtrip.symmetric_difference(polygon).area < 1e-12
    np.testing.assert_array_equal(
        pointwise_covers(roundtrip, query),
        case["trace_membership"],
    )
    fig, ax = plt.subplots()
    try:
        serialisers._draw_footprint(ax, footprint, (0.0, 0.0, 1.0, 1.0), 0.3)
        assert (
            sum(
                np.count_nonzero(patch.get_path().codes == PlotPath.MOVETO)
                for patch in ax.patches
            )
            == cycles
        )
    finally:
        plt.close(fig)
