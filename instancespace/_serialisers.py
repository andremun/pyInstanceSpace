# SPDX-License-Identifier: LicenseRef-PolyForm-Noncommercial-1.0.0
# Copyright (c) 2024-2026 Mario Andrés Muñoz
from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from matplotlib.patches import PathPatch
from matplotlib.path import Path as MatplotlibPath
from mpl_toolkits.mplot3d import Axes3D  # type: ignore[import-untyped]
from numpy.typing import NDArray
from scipy.io import savemat
from shapely import MultiPolygon, Polygon
from shapely.geometry.polygon import orient  # type: ignore[import-untyped]

from instancespace.data.model import (
    CloisterOut,
    Data,
    FeatSel,
    Footprint,
    PilotOut,
    PythiaOut,
    SiftedOut,
    TraceOut,
)
from instancespace.data.options import InstanceSpaceOptions
from instancespace.plotting import (
    _apply_view_angle,
    _draw_tetrahedral_mesh,
    _projection_dimensions,
    _resolve_view_angle,
    _scatter_projection,
    _ViewAngle,
)
from instancespace.utils.alpha_shape import TetrahedralMesh

_FOOTPRINT_COLUMNS = ["Row", "Part", "Ring", "Vertex", "z_1", "z_2"]
_TRACE_MESH_SCHEMA = "pyinstancespace.trace-mesh/v1"
_TRACE_MESH_MANIFEST = "footprint_meshes.json"
_TRACE_MESH_VERTEX_COLUMNS = ["vertex", "z_1", "z_2", "z_3"]
_TRACE_MESH_TETRAHEDRON_COLUMNS = [
    "tetrahedron",
    "v_1",
    "v_2",
    "v_3",
    "v_4",
]
_TRACE_MESH_FACE_COLUMNS = ["face", "v_1", "v_2", "v_3"]
_PROJECTION_ARRAY_DIMENSIONS = 2
_FOOTPRINT_DIMENSIONS = 2
_THREE_DIMENSIONS = 3
_SUPPORTED_PROJECTION_DIMENSIONS = {2, 3}
_INVALID_STEM_CHARACTERS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_MAX_STEM_LENGTH = 80
_WINDOWS_RESERVED_STEMS = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}


class SerializationError(RuntimeError):
    """Report a failed output write with its operation and target path."""


def _portable_stems(labels: list[str], fallback: str) -> list[str]:
    """Return deterministic, unique filename stems for user labels."""
    stems: list[str] = []
    used: set[str] = set()

    for index, label in enumerate(labels, start=1):
        stem = unicodedata.normalize("NFKC", str(label))
        stem = _INVALID_STEM_CHARACTERS.sub("_", stem).strip(" .")
        if not stem or stem in {".", ".."}:
            stem = f"{fallback}_{index}"
        if stem.split(".", maxsplit=1)[0].upper() in _WINDOWS_RESERVED_STEMS:
            stem = f"_{stem}"

        stem = stem[:_MAX_STEM_LENGTH].rstrip(" .") or f"{fallback}_{index}"
        candidate = stem
        suffix_number = 2
        while candidate.casefold() in used:
            suffix = f"_{suffix_number}"
            candidate = f"{stem[: _MAX_STEM_LENGTH - len(suffix)]}{suffix}"
            suffix_number += 1

        used.add(candidate.casefold())
        stems.append(candidate)

    return stems


def _write_dataframe_to_csv(
    frame: pd.DataFrame,
    filename: Path,
    *,
    index: bool = False,
    index_label: str | None = None,
) -> None:
    """Write one CSV file and add the target path to write errors."""
    try:
        frame.to_csv(filename, index=index, index_label=index_label)
    except Exception as exc:
        raise SerializationError(f"Could not write CSV file '{filename}'.") from exc


def _footprint_boundary_frame(polygon: Polygon | MultiPolygon) -> pd.DataFrame:
    """Convert a polygon geometry to the lossless footprint CSV v2 schema."""
    records: list[dict[str, int | float | str]] = []
    parts = polygon.geoms if isinstance(polygon, MultiPolygon) else [polygon]

    for part_number, part in enumerate(parts, start=1):
        rings = [("exterior", part.exterior)]
        rings.extend(
            (f"hole_{hole_number}", ring)
            for hole_number, ring in enumerate(part.interiors, start=1)
        )
        for ring_name, ring in rings:
            coordinates = np.asarray(ring.coords, dtype=np.double)
            if coordinates.shape[0] > 1 and np.array_equal(
                coordinates[0],
                coordinates[-1],
            ):
                coordinates = coordinates[:-1]
            for vertex_number, (z_1, z_2) in enumerate(coordinates, start=1):
                records.append(
                    {
                        "Row": len(records) + 1,
                        "Part": part_number,
                        "Ring": ring_name,
                        "Vertex": vertex_number,
                        "z_1": float(z_1),
                        "z_2": float(z_2),
                    },
                )

    return pd.DataFrame.from_records(records, columns=_FOOTPRINT_COLUMNS)


def _trace_mesh_frames(
    mesh: TetrahedralMesh | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build the three one-based tables for one 3D TRACE footprint."""
    vertices = np.empty((0, 3), dtype=np.double) if mesh is None else mesh.vertices
    tetrahedra = (
        np.empty((0, 4), dtype=np.int_) if mesh is None else mesh.tetrahedra + 1
    )
    boundary_faces = (
        np.empty((0, 3), dtype=np.int_) if mesh is None else mesh.boundary_faces + 1
    )

    vertex_frame = pd.DataFrame(vertices, columns=_TRACE_MESH_VERTEX_COLUMNS[1:])
    vertex_frame.insert(
        0,
        _TRACE_MESH_VERTEX_COLUMNS[0],
        np.arange(1, len(vertex_frame) + 1, dtype=np.int_),
    )
    tetrahedron_frame = pd.DataFrame(
        tetrahedra,
        columns=_TRACE_MESH_TETRAHEDRON_COLUMNS[1:],
    )
    tetrahedron_frame.insert(
        0,
        _TRACE_MESH_TETRAHEDRON_COLUMNS[0],
        np.arange(1, len(tetrahedron_frame) + 1, dtype=np.int_),
    )
    face_frame = pd.DataFrame(
        boundary_faces,
        columns=_TRACE_MESH_FACE_COLUMNS[1:],
    )
    face_frame.insert(
        0,
        _TRACE_MESH_FACE_COLUMNS[0],
        np.arange(1, len(face_frame) + 1, dtype=np.int_),
    )
    return vertex_frame, tetrahedron_frame, face_frame


def _mesh_metric(value: float, name: str) -> float:
    """Return one standards-compliant finite JSON mesh metric."""
    metric = float(value)
    if not np.isfinite(metric):
        raise ValueError(f"3D TRACE mesh {name} must be finite, got {metric!r}.")
    return metric


def _write_trace_mesh_footprint(
    output_directory: Path,
    footprint: Footprint,
    *,
    kind: str,
    stem: str,
    algorithm: str | None,
    algorithm_index: int | None,
) -> dict[str, Any]:
    """Write one footprint's mesh tables and return its manifest record."""
    if footprint.dimension != _THREE_DIMENSIONS:
        raise ValueError(
            "A 3D TRACE mesh export requires three-dimensional Footprint metadata.",
        )
    geometry = footprint.polygon
    if geometry is not None and not isinstance(geometry, TetrahedralMesh):
        raise ValueError(
            "A 3D TRACE mesh export cannot serialize Shapely 2D geometry.",
        )
    mesh = geometry
    footprint_volume = _mesh_metric(footprint.area, "footprint volume")
    elements = int(footprint.elements)
    good_elements = int(footprint.good_elements)
    density = _mesh_metric(footprint.density, "density")
    purity = _mesh_metric(footprint.purity, "purity")
    if (
        elements < 0
        or good_elements < 0
        or good_elements > elements
        or elements != footprint.elements
        or good_elements != footprint.good_elements
        or density < 0
        or not 0 <= purity <= 1
    ):
        raise ValueError("3D TRACE footprint statistics are inconsistent.")
    if mesh is None and any(
        value != 0
        for value in (
            footprint_volume,
            elements,
            good_elements,
            density,
            purity,
        )
    ):
        raise ValueError("An empty 3D TRACE footprint must have zero statistics.")
    if mesh is not None and not np.isclose(
        footprint_volume,
        mesh.volume,
        rtol=1e-12,
        atol=1e-12,
    ):
        raise ValueError("Footprint volume does not match its TetrahedralMesh.")
    if mesh is not None:
        expected_density = elements / mesh.volume if mesh.volume > 0 else 0.0
        expected_purity = good_elements / elements if elements > 0 else 0.0
        if not np.isclose(
            density,
            expected_density,
            rtol=1e-12,
            atol=1e-12,
        ) or not np.isclose(
            purity,
            expected_purity,
            rtol=1e-12,
            atol=1e-12,
        ):
            raise ValueError(
                "Footprint density or purity does not match its mesh statistics.",
            )

    if mesh is None:
        alpha: float | None = None
        region_threshold: float | None = None
        region_count = 0
        volume = 0.0
        surface_area = 0.0
        empty = True
    else:
        alpha = _mesh_metric(mesh.alpha, "alpha")
        region_threshold = _mesh_metric(mesh.region_threshold, "region threshold")
        region_count = int(mesh.region_count)
        volume = _mesh_metric(mesh.volume, "volume")
        surface_area = _mesh_metric(mesh.surface_area, "surface area")
        if (
            mesh.is_empty
            or alpha <= 0
            or region_threshold < 0
            or region_count <= 0
            or volume <= 0
            or surface_area <= 0
        ):
            raise ValueError(
                "A present TetrahedralMesh must be nonempty with positive geometry.",
            )
        empty = False

    base = f"footprint_{stem}"
    files = {
        "vertices": f"{base}_vertices.csv",
        "tetrahedra": f"{base}_tetrahedra.csv",
        "boundary_faces": f"{base}_boundary_faces.csv",
    }
    frames = _trace_mesh_frames(mesh)
    for filename, frame in zip(files.values(), frames, strict=True):
        _write_dataframe_to_csv(frame, output_directory / filename)

    return {
        "kind": kind,
        "algorithm": algorithm,
        "algorithm_index": algorithm_index,
        "empty": empty,
        "mesh_present": mesh is not None,
        "files": files,
        "alpha": alpha,
        "region_threshold": region_threshold,
        "region_count": region_count,
        "volume": volume,
        "surface_area": surface_area,
        "elements": elements,
        "good_elements": good_elements,
        "density": density,
        "purity": purity,
    }


def _write_trace_mesh_bundle(
    output_directory: Path,
    algorithm_labels: list[str],
    trace_out: TraceOut,
) -> None:
    """Write the additive versioned 3D TRACE mesh interchange."""
    if len(trace_out.good) != len(algorithm_labels) or len(trace_out.best) != len(
        algorithm_labels,
    ):
        raise ValueError("TRACE footprint counts must match the algorithm labels.")

    records: list[dict[str, Any]] = []
    stems = _portable_stems(algorithm_labels, "algorithm")
    for index, (algorithm, stem) in enumerate(
        zip(algorithm_labels, stems, strict=True),
        start=1,
    ):
        records.append(
            _write_trace_mesh_footprint(
                output_directory,
                trace_out.good[index - 1],
                kind="good",
                stem=f"{stem}_good",
                algorithm=algorithm,
                algorithm_index=index,
            ),
        )
        records.append(
            _write_trace_mesh_footprint(
                output_directory,
                trace_out.best[index - 1],
                kind="best",
                stem=f"{stem}_best",
                algorithm=algorithm,
                algorithm_index=index,
            ),
        )
    records.append(
        _write_trace_mesh_footprint(
            output_directory,
            trace_out.hard,
            kind="hard",
            stem="hard",
            algorithm=None,
            algorithm_index=None,
        ),
    )

    manifest = {
        "schema_version": _TRACE_MESH_SCHEMA,
        "coordinate_dimension": _THREE_DIMENSIONS,
        "algorithm_index_base": 1,
        "mesh_index_base": 1,
        "footprints": records,
    }
    target = output_directory / _TRACE_MESH_MANIFEST
    try:
        target.write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    except Exception as exc:
        raise SerializationError(f"Could not write JSON file '{target}'.") from exc


def _projection_column_names(z: NDArray[Any]) -> pd.Series[str]:
    """Return MATLAB-compatible coordinate names for a 2D or 3D projection."""
    projection = np.asarray(z)
    if (
        projection.ndim != _PROJECTION_ARRAY_DIMENSIONS
        or projection.shape[1] not in _SUPPORTED_PROJECTION_DIMENSIONS
    ):
        raise ValueError("pilot_out.z must be a two-dimensional 2D or 3D projection.")
    return pd.Series(
        [f"z_{dimension}" for dimension in range(1, projection.shape[1] + 1)],
    )


def save_instance_space_to_csv(
    output_directory: Path,
    data: Data,
    sifted_out: SiftedOut,
    trace_out: TraceOut,
    pilot_out: PilotOut,
    cloister_out: CloisterOut,
    pythia_out: PythiaOut,
) -> None:
    if not output_directory.is_dir():
        raise ValueError("output_directory must be an existing directory.")

    projection_columns = _projection_column_names(pilot_out.z)
    projection_dimensions = len(projection_columns)
    num_algorithms = data.y.shape[1]
    algorithm_stems = _portable_stems(data.algo_labels, "algorithm")

    if projection_dimensions == _FOOTPRINT_DIMENSIONS:
        for i in range(num_algorithms):
            best = trace_out.best[i]
            if (
                best is not None
                and best.polygon is not None
                and not best.polygon.is_empty
            ):
                _write_dataframe_to_csv(
                    _footprint_boundary_frame(best.polygon),
                    output_directory / f"footprint_{algorithm_stems[i]}_best.csv",
                )

            good = trace_out.good[i]
            if (
                good is not None
                and good.polygon is not None
                and not good.polygon.is_empty
            ):
                _write_dataframe_to_csv(
                    _footprint_boundary_frame(good.polygon),
                    output_directory / f"footprint_{algorithm_stems[i]}_good.csv",
                )
    else:
        _write_trace_mesh_bundle(output_directory, data.algo_labels, trace_out)

    _write_array_to_csv(
        pilot_out.z,
        projection_columns,
        data.inst_labels,
        output_directory / "coordinates.csv",
    )

    if cloister_out is not None:
        _write_array_to_csv(
            cloister_out.z_edge,
            projection_columns,
            _make_bind_labels(cloister_out.z_edge),
            output_directory / "bounds.csv",
        )
        _write_array_to_csv(
            cloister_out.z_ecorr,
            projection_columns,
            _make_bind_labels(cloister_out.z_ecorr),
            output_directory / "bounds_prunned.csv",
        )

    _write_array_to_csv(
        data.x_raw[:, sifted_out.selvars],
        pd.Series(data.feat_labels),
        data.inst_labels,
        output_directory / "feature_raw.csv",
    )
    _write_array_to_csv(
        data.x,
        pd.Series(data.feat_labels),
        data.inst_labels,
        output_directory / "feature_process.csv",
    )
    _write_array_to_csv(
        data.y_raw,
        pd.Series(data.algo_labels),
        data.inst_labels,
        output_directory / "algorithm_raw.csv",
    )
    _write_array_to_csv(
        data.y,
        pd.Series(data.algo_labels),
        data.inst_labels,
        output_directory / "algorithm_process.csv",
    )
    _write_array_to_csv(
        data.y_bin,
        pd.Series(data.algo_labels),
        data.inst_labels,
        output_directory / "algorithm_bin.csv",
    )
    _write_array_to_csv(
        data.num_good_algos,
        pd.Series(["NumGoodAlgos"]),
        data.inst_labels,
        output_directory / "good_algos.csv",
    )
    _write_array_to_csv(
        data.beta,
        pd.Series(["IsBetaEasy"]),
        data.inst_labels,
        output_directory / "beta_easy.csv",
    )
    _write_array_to_csv(
        data.p,
        pd.Series(["Best_Algorithm"]),
        data.inst_labels,
        output_directory / "portfolio.csv",
    )
    _write_array_to_csv(
        pythia_out.y_hat,
        pd.Series(data.algo_labels),
        data.inst_labels,
        output_directory / "algorithm_svm.csv",
    )
    _write_array_to_csv(
        pythia_out.selection0,
        pd.Series(["Best_Algorithm"]),
        data.inst_labels,
        output_directory / "portfolio_svm.csv",
    )

    trace_summary = trace_out.summary.iloc[:, [0, 2, 4, 5, 7, 9, 10]].copy()
    trace_summary = trace_summary.rename(columns={"Algorithm": "Row"})
    _write_dataframe_to_csv(
        trace_summary,
        output_directory / "footprint_performance.csv",
    )

    if pilot_out.summary is not None:
        pilot_summary = pilot_out.summary.copy().rename(columns={0: "Row"})
        _write_dataframe_to_csv(
            pilot_summary,
            output_directory / "projection_matrix.csv",
        )

    pythia_summary = pythia_out.summary.copy().rename(columns={"Algorithms": "Row"})
    _write_dataframe_to_csv(
        pythia_summary,
        output_directory / "svm_table.csv",
    )


def save_instance_space_for_web(
    output_directory: Path,
    data: Data,
    feat_sel: FeatSel,
) -> None:
    if not output_directory.is_dir():
        raise ValueError("output_directory must be an existing directory.")

    colours = (
        np.array(
            mpl.colormaps["viridis"].resampled(256).__dict__["colors"],
        )[:, :3]
        * 255
    ).astype(np.int_)

    _write_dataframe_to_csv(
        pd.DataFrame(colours, columns=["R", "G", "B"]),
        output_directory / "color_table.csv",
    )

    _write_colour_array_to_csv(
        _colour_scale(data.x_raw[:, feat_sel.idx]),
        pd.Series(data.feat_labels),
        data.inst_labels,
        output_directory / "feature_raw_color.csv",
    )
    _write_colour_array_to_csv(
        _colour_scale(data.y_raw),
        pd.Series(data.algo_labels),
        data.inst_labels,
        output_directory / "algorithm_raw_single_color.csv",
    )
    _write_colour_array_to_csv(
        _colour_scale(data.x),
        pd.Series(data.feat_labels),
        data.inst_labels,
        output_directory / "feature_process_color.csv",
    )
    _write_colour_array_to_csv(
        _colour_scale(data.y),
        pd.Series(data.algo_labels),
        data.inst_labels,
        output_directory / "algorithm_process_single_color.csv",
    )
    _write_colour_array_to_csv(
        _colour_scale_g(data.y_raw),
        pd.Series(data.algo_labels),
        data.inst_labels,
        output_directory / "algorithm_raw_color.csv",
    )
    _write_colour_array_to_csv(
        _colour_scale_g(data.y),
        pd.Series(data.algo_labels),
        data.inst_labels,
        output_directory / "algorithm_process_color.csv",
    )
    _write_colour_array_to_csv(
        _colour_scale_g(data.num_good_algos),
        pd.Series(["NumGoodAlgos"]),
        data.inst_labels,
        output_directory / "good_algos_color.csv",
    )


def save_instance_space_graphs(
    output_directory: Path,
    data: Data,
    options: InstanceSpaceOptions,
    pythia: PythiaOut,
    pilot: PilotOut,
    trace: TraceOut,
) -> None:
    if not output_directory.is_dir():
        raise ValueError("output_directory must be an existing directory.")

    num_feats = data.x.shape[1]
    num_algorithms = data.y.shape[1]
    feature_stems = _portable_stems(data.feat_labels, "feature")
    algorithm_stems = _portable_stems(data.algo_labels, "algorithm")
    _projection_dimensions(pilot.z)
    viewpoint = getattr(pilot, "viewpoint", None)
    global_view = _resolve_view_angle(viewpoint, None)

    x_aux = _minmax_scale(data.x, axis=0)
    y_ind = _minmax_scale(data.y_raw, axis=0)
    y_log = np.full(data.y_raw.shape, np.nan, dtype=np.double)
    log_domain = np.isfinite(data.y_raw) & (data.y_raw > -1)
    y_log[log_domain] = np.log10(data.y_raw[log_domain] + 1)
    y_glb = _minmax_scale(y_log, axis=None)

    # MATLAB scriptpng always labels trained footprints with experimental truth.
    # TRACE's build-time useSim choice does not change the output overlay contract.
    y_foot = data.y_bin
    p_foot = data.p - 1

    for i in range(num_feats):
        filename = f"distribution_feature_{feature_stems[i]}.png"
        _draw_scatter(
            pilot.z,
            x_aux[:, i],
            data.feat_labels[i].replace("_", " "),
            output_directory / filename,
            global_view,
        )

    for i in range(num_algorithms):
        algo_label = data.algo_labels[i]
        algo_stem = algorithm_stems[i]
        algorithm_view = _resolve_view_angle(viewpoint, i)

        filename = f"distribution_performance_global_normalized_{algo_stem}.png"
        _draw_scatter(
            pilot.z,
            y_glb[:, i],
            algo_label.replace("_", " "),
            output_directory / filename,
            algorithm_view,
        )

        filename = f"distribution_performance_individual_normalized_{algo_stem}.png"
        _draw_scatter(
            pilot.z,
            y_ind[:, i],
            algo_label.replace("_", " "),
            output_directory / filename,
            algorithm_view,
        )

        _draw_binary_performance(
            pilot.z,
            data.y_bin[:, i],
            algo_label.replace("_", " "),
            output_directory / f"binary_performance_{algo_stem}.png",
            algorithm_view,
        )

        _draw_binary_performance(
            pilot.z,
            pythia.y_hat[:, i],
            algo_label.replace("_", " "),
            output_directory / f"binary_svm_{algo_stem}.png",
            algorithm_view,
        )

        _draw_good_bad_footprint(
            pilot.z,
            trace.good[i],
            y_foot[:, i],
            algo_label.replace("_", " ") + " Footprint",
            output_directory / f"footprint_{algo_stem}.png",
            algorithm_view,
        )

    _draw_scatter(
        pilot.z,
        data.num_good_algos / num_algorithms,
        "Percentage of good algorithms",
        output_directory / "distribution_number_good_algos.png",
        global_view,
    )

    _draw_portfolio_selections(
        pilot.z,
        data.p - 1,
        np.array(data.algo_labels),
        "Best algorithm",
        output_directory / "distribution_portfolio.png",
        global_view,
    )

    _draw_portfolio_selections(
        pilot.z,
        pythia.selection0,
        np.array(data.algo_labels),
        "Predicted best algorithm",
        output_directory / "distribution_svm_portfolio.png",
        global_view,
    )

    _draw_binary_performance(
        pilot.z,
        data.beta,
        "Beta score",
        output_directory / "distribution_beta_score.png",
        global_view,
    )

    if data.s is not None:
        _draw_sources(
            pilot.z,
            np.array(data.s),
            output_directory / "distribution_sources.png",
            global_view,
        )

    _draw_portfolio_footprint(
        pilot.z,
        trace.best,
        p_foot,
        np.array(data.algo_labels),
        output_directory / "footprint_portfolio.png",
        global_view,
    )


def _write_array_to_csv(
    data: NDArray[Any],
    column_names: pd.Series[str],
    row_names: pd.Series[str],
    filename: Path,
) -> None:
    _write_dataframe_to_csv(
        pd.DataFrame(data, index=row_names, columns=column_names),
        filename,
        index=True,
        index_label="Row",
    )


def _make_bind_labels(
    data: NDArray[Any],
) -> pd.Series[str]:
    return pd.Series([f"bnd_pnt_{i+1}" for i in range(data.shape[0])])


def _write_colour_array_to_csv(
    data: NDArray[np.double],
    column_names: pd.Series[str],
    row_names: pd.Series[str],
    filename: Path,
) -> None:
    """Write integer colours while keeping missing values empty."""
    frame = pd.DataFrame(data, index=row_names, columns=column_names).astype("Int64")
    _write_dataframe_to_csv(frame, filename, index=True, index_label="Row")


def _minmax_scale(
    data: NDArray[Any],
    axis: int | None,
) -> NDArray[np.double]:
    """Scale finite values to zero through one without empty-slice warnings."""
    values = np.asarray(data, dtype=np.double)
    scaled = np.full(values.shape, np.nan, dtype=np.double)

    if axis is None:
        finite = np.isfinite(values)
        if not np.any(finite):
            return scaled
        minimum = float(np.min(values[finite]))
        value_range = float(np.max(values[finite]) - minimum)
        scaled[finite] = 0.0
        if value_range > 0:
            scaled[finite] = (values[finite] - minimum) / value_range
        return scaled

    if axis != 0:
        raise ValueError("axis must be 0 or None.")

    for column in range(values.shape[1]):
        finite = np.isfinite(values[:, column])
        if not np.any(finite):
            continue
        minimum = float(np.min(values[finite, column]))
        value_range = float(np.max(values[finite, column]) - minimum)
        scaled[finite, column] = 0.0
        if value_range > 0:
            scaled[finite, column] = (values[finite, column] - minimum) / value_range
    return scaled


def _colour_scale(
    data: NDArray[Any],
) -> NDArray[np.double]:
    return np.round(255.0 * _minmax_scale(data, axis=0))


def _colour_scale_g(
    data: NDArray[Any],
) -> NDArray[np.double]:
    return np.round(255.0 * _minmax_scale(data, axis=None))


def _save_figure(fig: Figure, output: Path) -> None:
    """Save one figure and add the target path to write errors."""
    try:
        fig.savefig(output)
    except Exception as exc:
        raise SerializationError(f"Could not write image file '{output}'.") from exc


def _add_legend_if_present(ax: Axes) -> None:
    """Add a legend only when the plot has labeled artists."""
    handles, _ = ax.get_legend_handles_labels()
    if handles:
        ax.legend()


def _new_projection_figure(z: NDArray[Any]) -> tuple[Figure, Axes]:
    """Create isolated native axes matching a 2D or 3D projection."""
    if _projection_dimensions(z) == _THREE_DIMENSIONS:
        fig = plt.figure()
        return fig, fig.add_subplot(projection="3d")
    return plt.subplots()


def _configure_projection_axes(
    ax: Axes,
    z: NDArray[Any],
    view_angle: _ViewAngle | None,
) -> None:
    """Keep 2D styling while adding the native third axis and camera."""
    dimensions = _projection_dimensions(z)
    if dimensions == _THREE_DIMENSIONS:
        ax.set_zlabel(r"$z_{3}$")  # type: ignore[attr-defined]
    else:
        ax.set_xlim((-5, 5))
        ax.set_ylim((-5, 5))
    ax.set_xlabel(r"$z_{1}$")
    ax.set_ylabel(r"$z_{2}$")
    _apply_view_angle(ax, z, view_angle or _resolve_view_angle(None, None))


def _draw_sources(
    z: NDArray[Any],
    s: NDArray[np.str_],
    output: Path,
    view_angle: _ViewAngle | None = None,
) -> None:
    source_labels = np.unique(s)
    num_sources = len(source_labels)

    cmap = plt.colormaps["viridis"]
    fig, ax2 = _new_projection_figure(z)
    try:
        ax: Axes = ax2
        fig.suptitle("Sources")

        denominator = max(num_sources - 1, 1)
        for i in reversed(range(num_sources)):
            _scatter_projection(
                ax,
                z[s == source_labels[i]],
                s=8,
                color=cmap(i / denominator),
                label=source_labels[i],
            )

        _configure_projection_axes(ax, z, view_angle)
        _add_legend_if_present(ax)
        _save_figure(fig, output)
    finally:
        plt.close(fig)


def _draw_scatter(
    z: NDArray[Any],
    x: NDArray[Any],
    title_label: str,
    output: Path,
    view_angle: _ViewAngle | None = None,
) -> None:
    values = np.asarray(x, dtype=np.double)
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        lower_bound, upper_bound = 0.0, 1.0
    else:
        lower_bound = float(np.min(finite_values))
        upper_bound = float(np.max(finite_values))
        if lower_bound == upper_bound:
            delta = max(abs(lower_bound) * 0.01, 0.5)
            lower_bound -= delta
            upper_bound += delta

    cmap = plt.colormaps["viridis"].copy()
    cmap.set_bad("#bdbdbd")
    fig, ax2 = _new_projection_figure(z)
    try:
        ax: Axes = ax2
        fig.suptitle(title_label, size=14)
        norm = Normalize(lower_bound, upper_bound)

        finite = np.isfinite(values)
        if np.any(finite):
            _scatter_projection(
                ax,
                z[finite],
                s=8,
                c=values[finite],
                norm=norm,
                cmap=cmap,
            )
        if np.any(~finite):
            _scatter_projection(
                ax,
                z[~finite],
                s=8,
                color="#bdbdbd",
            )
        _configure_projection_axes(ax, z, view_angle)
        fig.colorbar(
            plt.cm.ScalarMappable(
                norm=norm,
                cmap=cmap,
            ),
            ax=ax,
        )
        _save_figure(fig, output)
    finally:
        plt.close(fig)


def _draw_portfolio_selections(
    z: NDArray[Any],
    p: NDArray[Any],
    algorithm_labels: NDArray[np.str_],
    title_label: str,
    output: Path,
    view_angle: _ViewAngle | None = None,
) -> None:
    """Plot a portfolio encoded with zero-based indices and ``-1`` for none."""
    num_algorithms = len(algorithm_labels)
    cmap = plt.colormaps["viridis"]
    fig, ax2 = _new_projection_figure(z)
    try:
        ax: Axes = ax2
        fig.suptitle(title_label)

        denominator = max(num_algorithms, 1)
        for selection in range(-1, num_algorithms):
            selected = p == selection
            if not np.any(selected):
                continue

            _scatter_projection(
                ax,
                z[selected],
                s=8,
                color=cmap((selection + 1) / denominator),
                label=(
                    "None"
                    if selection == -1
                    else algorithm_labels[selection].replace("_", " ")
                ),
            )

        _configure_projection_axes(ax, z, view_angle)
        _add_legend_if_present(ax)
        _save_figure(fig, output)
    finally:
        plt.close(fig)


def _draw_portfolio_footprint(
    z: NDArray[Any],
    best: list[Footprint],
    p: NDArray[Any],
    algorithm_labels: NDArray[np.str_],
    output: Path,
    view_angle: _ViewAngle | None = None,
) -> None:
    """Plot best footprints for a zero-based portfolio with ``-1`` for none."""
    num_algorithms = len(algorithm_labels)

    cmap = plt.colormaps["viridis"]
    fig, ax2 = _new_projection_figure(z)
    try:
        ax: Axes = ax2
        fig.suptitle("Portfolio footprints")

        denominator = max(num_algorithms, 1)
        for selection in range(-1, num_algorithms):
            selected = p == selection
            if not np.any(selected):
                continue

            colour = cmap((selection + 1) / denominator)
            _scatter_projection(
                ax,
                z[selected],
                s=8,
                color=colour,
                label=(
                    "None"
                    if selection == -1
                    else algorithm_labels[selection].replace("_", " ")
                ),
            )

            if selection >= 0:
                _draw_footprint(ax, best[selection], colour, 0.3)

        _configure_projection_axes(ax, z, view_angle)
        _add_legend_if_present(ax)
        _save_figure(fig, output)
    finally:
        plt.close(fig)


def _draw_good_bad_footprint(
    z: NDArray[Any],
    good: Footprint,
    y_bin: NDArray[Any],
    title_label: str,
    output: Path,
    view_angle: _ViewAngle | None = None,
) -> None:
    orange = (1.0, 0.6471, 0.0, 1.0)
    blue = (0.0, 0.0, 1.0, 1.0)

    fig, ax2 = _new_projection_figure(z)
    try:
        ax: Axes = ax2
        fig.suptitle(title_label)
        not_y_bin = y_bin == 0
        good_y_bin = y_bin == 1

        if np.any(not_y_bin):
            _scatter_projection(
                ax,
                z[not_y_bin],
                s=8,
                c=[orange],
                label="BAD",
            )

        if np.any(good_y_bin):
            _scatter_projection(
                ax,
                z[good_y_bin],
                s=8,
                c=[blue],
                label="GOOD",
            )
            _draw_footprint(ax, good, blue, 0.3)

        _configure_projection_axes(ax, z, view_angle)
        _add_legend_if_present(ax)
        _save_figure(fig, output)
    finally:
        plt.close(fig)


def _draw_footprint(
    ax: Axes,
    footprint: Footprint,
    colour: tuple[float, float, float, float],
    alpha: float,
) -> None:
    geometry = footprint.polygon
    axis_is_3d = isinstance(ax, Axes3D)
    expected_dimension = _THREE_DIMENSIONS if axis_is_3d else _FOOTPRINT_DIMENSIONS
    if footprint.dimension != expected_dimension:
        raise ValueError(
            "Footprint metadata and matplotlib axis dimensions do not match.",
        )
    if geometry is None:
        return
    if isinstance(geometry, TetrahedralMesh):
        if not axis_is_3d:
            raise ValueError("A TetrahedralMesh requires a three-dimensional axis.")
        _draw_tetrahedral_mesh(
            ax,
            geometry,
            facecolor=colour,
            edgecolor=colour,
            alpha=alpha,
        )
        return
    if axis_is_3d:
        raise ValueError("Shapely 2D footprint geometry cannot be drawn on a 3D axis.")
    if geometry.is_empty:
        return

    parts = geometry.geoms if isinstance(geometry, MultiPolygon) else [geometry]
    for part in parts:
        oriented = orient(part, sign=1.0)
        vertices: list[tuple[float, float]] = []
        codes: list[int] = []
        for ring in [oriented.exterior, *oriented.interiors]:
            ring_coordinates = list(ring.coords)
            if not ring_coordinates:
                continue
            vertices.extend((float(x), float(y)) for x, y in ring_coordinates)
            codes.extend(
                [
                    int(MatplotlibPath.MOVETO),
                    *([int(MatplotlibPath.LINETO)] * (len(ring_coordinates) - 2)),
                    int(MatplotlibPath.CLOSEPOLY),
                ],
            )

        if vertices:
            path = MatplotlibPath(np.asarray(vertices), np.asarray(codes))
            ax.add_patch(
                PathPatch(
                    path,
                    facecolor=colour,
                    edgecolor=colour,
                    alpha=alpha,
                ),
            )


def _draw_binary_performance(
    z: NDArray[Any],
    y_bin: NDArray[Any],
    title_label: str,
    output: Path,
    view_angle: _ViewAngle | None = None,
) -> None:
    orange = (1.0, 0.6471, 0.0, 1.0)
    blue = (0.0, 0.0, 1.0, 1.0)

    fig, ax2 = _new_projection_figure(z)
    try:
        ax: Axes = ax2
        fig.suptitle(title_label)
        not_y_bin = y_bin == 0
        good_y_bin = y_bin == 1

        if np.any(not_y_bin):
            _scatter_projection(
                ax,
                z[not_y_bin],
                s=8,
                c=[orange],
                label="BAD",
            )

        if np.any(good_y_bin):
            _scatter_projection(
                ax,
                z[good_y_bin],
                s=8,
                c=[blue],
                label="GOOD",
            )

        _configure_projection_axes(ax, z, view_angle)
        _add_legend_if_present(ax)
        _save_figure(fig, output)
    finally:
        plt.close(fig)


def save_instance_space_output_mat(
    output_directory: Path,
    data: Data,
) -> None:
    """Write the algorithm-label-only MAT shim used by the offline dashboard.

    This compatibility file is not full model persistence. ``Model.save`` is
    the canonical round-trip format for complete Python models.
    """
    output = output_directory / "model.mat"
    try:
        savemat(
            output,
            {"data": {"algolabels": np.array(data.algo_labels)}},
        )
    except Exception as exc:
        raise SerializationError(f"Could not write MAT file '{output}'.") from exc
