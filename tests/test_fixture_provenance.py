"""Tests for MATLAB fixture provenance and historical data classification."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
from collections.abc import Iterable
from copy import deepcopy
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

from tools.fixture_provenance import (
    _EXPORTER_SCRIPT,
    _GOLD_MATLAB_COMMIT,
    _REFERENCE_V2_EXPORTER_SHA256,
    _VERIFIED_V2_CONTENT_ROOT_SHA256,
    BUNDLE_SCHEMA,
    DIAGNOSTIC_TRUST,
    REFERENCE_PROFILE,
    REFERENCE_PROFILE_V1,
    RESOLVED_OPTIONS_INDEX_SCHEMA,
    RESOLVED_OPTIONS_SCHEMA,
    VERIFIED_TRUST,
    ProvenanceError,
    _fixed_reference_paths,
    _fixed_reference_paths_v1,
    _geometry_paths,
    _pilot_numerical_trial_metrics,
    _trace3d_mesh_paths,
    install_verified_bundle,
    sha256_file,
    validate_bundle,
    validate_inventory,
)

_COMMIT = "1" * 40
_SCRIPT_HASH = "2" * 64
_MIN_HISTORICAL_FILES = 300
_CURRENT_FIXTURES = Path(__file__).parent / "fixtures" / "matlab" / "current"
_VARIANTS = (
    "trace3_default",
    "trace3_pythia_skip",
    "legacy_svm",
    "pilot_standard_analytic_3d",
    "pilot_standard_numerical_3d_x0",
    "pilot_standard_numerical_3d_precalc",
    "pilot_pls_2d",
    "pilot_pls_3d_grouped",
)
_REFERENCE_ALGORITHM_LABELS = [
    "NB",
    "LDA",
    "QDA",
    "CART",
    "J48",
    "KNN",
    "L_SVM",
    "poly_SVM",
    "RBF_SVM",
    "RandF",
]
_ALGORITHM_LABELS = _REFERENCE_ALGORITHM_LABELS
_FEATURE_LABELS = ["feature_a", "feature_b", "feature_c"]


def _synthetic_x0() -> list[list[float]]:
    return [
        [float(row), float(row) + 0.1, float(row) + 0.2]
        for row in range(len(_synthetic_precalc()))
    ]


def _synthetic_alpha() -> list[list[float]]:
    dims = 3
    projection = np.asarray(_synthetic_pilot_matrix(dims), dtype=np.double)
    b = np.eye(3, dtype=np.double)
    c = np.asarray(
        [
            [0.1 * (index + 1) for index in range(len(_ALGORITHM_LABELS))],
            [0.05 * (index + 1) for index in range(len(_ALGORITHM_LABELS))],
            [0.02 * (index + 1) for index in range(len(_ALGORITHM_LABELS))],
        ],
        dtype=np.double,
    )
    combined = np.vstack((b, c.T))
    tail = combined.reshape(-1, order="F")

    selected = np.concatenate((projection.reshape(-1, order="F"), tail))
    first = np.concatenate(
        (
            np.asarray(
                [[2.0, 0.0, 0.0], [0.0, 0.25, 0.0], [0.0, 0.0, 0.5]],
            ).reshape(-1, order="F"),
            tail,
        ),
    )
    third = np.concatenate(
        (
            np.asarray(
                [[0.25, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 0.5]],
            ).reshape(-1, order="F"),
            tail,
        ),
    )
    return [
        [float(value) for value in row]
        for row in np.column_stack((first, selected, third))
    ]


def _synthetic_precalc() -> list[float]:
    return [row[1] for row in _synthetic_alpha()]


def _effective_options(variant: str) -> dict[str, Any]:
    options: dict[str, Any] = {
        "general": {"seed": 42, "verbose": False, "parallel": False, "ncores": 18},
        "perf": {
            "MaxPerf": False,
            "AbsPerf": False,
            "epsilon": 0.05,
            "betaThreshold": 0.55,
        },
        "prelim": {"iqrMultiplier": 5, "nanThreshold": 0.2},
        "auto": {"preproc": True},
        "bound": {"flag": True},
        "norm": {"flag": True},
        "selvars": {
            "smallscaleflag": False,
            "smallscale": 0.3,
            "fileidxflag": False,
            "fileidx": "",
            "densityflag": False,
            "mindistance": 0.1,
            "type": "Ftr&Good",
        },
        "sifted": {
            "flag": True,
            "rho": 0.1,
            "pval": 0.05,
            "K": 10,
            "MaxIter": 1000,
            "Replicates": 100,
        },
        "pilot": {
            "analytic": False,
            "ntries": 10,
            "dims": 2,
            "method": "standard",
            "alpha": 1.0,
            "viewGroups": [],
            "topoWeight": 0,
            "verbose": False,
        },
        "cloister": {"pval": 0.05, "corrThreshold": 0.7, "maxFeatures": 20},
        "pythia": {
            "flag": True,
            "kFold": 5,
            "tuning": "sobol",
            "nTuningIter": 20,
            "params": [],
            "skip": False,
            "ispolykrnl": False,
            "useweights": False,
            "ensembleMethod": "Bag",
            "verbose": False,
            "seed": 42,
            "classifier": "knn",
        },
        "trace": {
            "method": "trace3",
            "PI": 0.6,
            "minInstances": 4,
            "minAreaFrac": 0.01,
            "contra": False,
        },
        "outputs": {"csv": False, "png": False, "fig": False, "web": False},
    }
    if variant == "trace3_pythia_skip":
        options["pythia"]["skip"] = True
    elif variant == "legacy_svm":
        options["pythia"]["classifier"] = "svm"
        options["trace"].update(method="legacy", PI=0.55, contra=True)
    elif variant.startswith("pilot_"):
        options["pythia"]["skip"] = True
        options["pilot"]["ntries"] = 1
        if variant == "pilot_standard_analytic_3d":
            options["pilot"].update(dims=3, analytic=True)
        elif variant == "pilot_standard_numerical_3d_x0":
            options["pilot"].update(dims=3, analytic=False, X0=_synthetic_x0())
        elif variant == "pilot_standard_numerical_3d_precalc":
            options["pilot"].update(
                dims=3,
                analytic=False,
                precalcAlpha=_synthetic_precalc(),
            )
        elif variant == "pilot_pls_2d":
            options["pilot"].update(method="pls", dims=2, analytic=False)
        elif variant == "pilot_pls_3d_grouped":
            options["pilot"].update(
                method="pls",
                dims=3,
                analytic=True,
                alpha=3.0,
                viewGroups=[[1, 2, 3, 4], [5, 6, 7, 8, 9, 10]],
            )
    return options


def _entry_metadata(relative: str) -> tuple[str, str | None, str]:
    parts = Path(relative).parts
    if parts[0] == "shared_inputs":
        return "shared", None, "reference"
    if parts[0] == "resolved_options":
        return "shared", None, Path(relative).stem
    return parts[0].removesuffix("_data"), parts[1], parts[2]


def _write_csv(
    target: Path,
    header: list[str],
    rows: Iterable[Iterable[object]],
) -> None:
    with target.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)


def _read_csv_for_mutation(path: Path) -> tuple[list[str], list[list[str]]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream)
        return next(reader), list(reader)


def _synthetic_pilot_matrix(dims: int) -> list[list[float]]:
    rows = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    if dims == 3:
        rows.append([0.0, 0.0, 1.0])
    return rows


def _write_pilot_profile_file(
    target: Path,
    relative: str,
    variant: str,
) -> None:
    dims = 2 if variant == "pilot_pls_2d" else 3
    z_header = [f"z_{index}" for index in range(1, dims + 1)]
    is_explore = relative.startswith("explore_data/")
    x = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, -1.0],
    ]
    if is_explore:
        x[0] = [0.1, 0.1, 0.0]
    y = [
        [
            (index + 1) * (0.1 * row[0] + 0.05 * row[1] + 0.02 * row[2])
            for index in range(len(_ALGORITHM_LABELS))
        ]
        for row in x
    ]
    if variant.startswith("pilot_pls_"):
        x = [[row[0] + 0.25, row[1] + 0.5, row[2] + 0.75] for row in x]
        y = [
            [value + 0.4 * (index + 1) for index, value in enumerate(row)] for row in y
        ]
    projection = _synthetic_pilot_matrix(dims)
    projection_x = x
    if variant.startswith("pilot_pls_") and not is_explore:
        means = [sum(row[index] for row in x) / len(x) for index in range(3)]
        projection_x = [
            [value - means[index] for index, value in enumerate(row)] for row in x
        ]
    z = [
        [
            sum(value * projection[dimension][index] for index, value in enumerate(row))
            for dimension in range(dims)
        ]
        for row in projection_x
    ]
    b = [
        [1.0, 0.0, *([0.0] if dims == 3 else [])],
        [0.0, 1.0, *([0.0] if dims == 3 else [])],
        (
            ([0.5, 0.5, 1.0] if dims == 3 else [0.5, 0.5])
            if variant.startswith("pilot_pls_")
            else [0.0, 0.0, 1.0]
        ),
    ]
    c = [
        [0.1 * (index + 1) for index in range(len(_ALGORITHM_LABELS))],
        [0.05 * (index + 1) for index in range(len(_ALGORITHM_LABELS))],
    ]
    if dims == 3:
        c.append([0.02 * (index + 1) for index in range(len(_ALGORITHM_LABELS))])
    x_bar = np.column_stack((x, y))
    factors = np.vstack((b, np.asarray(c, dtype=np.double).T))
    reconstructed = np.asarray(z, dtype=np.double) @ factors.T
    if variant.startswith("pilot_pls_"):
        reconstructed += np.mean(x_bar, axis=0)
    reconstruction_error = float(np.sum(np.square(x_bar - reconstructed)))
    reconstruction_r2 = [
        float(np.corrcoef(x_bar[:, index], reconstructed[:, index])[0, 1] ** 2)
        for index in range(x_bar.shape[1])
    ]
    filename = target.name

    if filename == "stage_context.json":
        is_pls = variant.startswith("pilot_pls_")
        context = {
            "schema_version": "pyinstancespace.pilot-evidence-context/v1",
            "scope": "pilot-stage",
            "upstream_snapshot": "build_data/pilot/default/inputs",
            "sifted_effective_pilot_dims": 2,
            "input_transform": "deterministic-column-shift" if is_pls else "none",
            "feature_shift": [0.25, 0.5, 0.75] if is_pls else [],
            "algorithm_shift": (
                [0.4 * (index + 1) for index in range(len(_ALGORITHM_LABELS))]
                if is_pls
                else []
            ),
            "explore_projection": "InstanceSpace.explore: Z=X*A' (uncentred)",
        }
        target.write_text(json.dumps(context), encoding="utf-8")
    elif filename == "feature_labels.csv":
        _write_csv(target, ["feature_name"], [[label] for label in _FEATURE_LABELS])
    elif filename == "x.csv":
        _write_csv(
            target,
            ["Row", *_FEATURE_LABELS],
            [[f"instance_{index}", *row] for index, row in enumerate(x, start=1)],
        )
    elif filename == "y.csv":
        _write_csv(
            target,
            ["Row", *_ALGORITHM_LABELS],
            [[f"instance_{index}", *row] for index, row in enumerate(y, start=1)],
        )
    elif filename == "projection_a.csv":
        _write_csv(
            target,
            ["Row", *_FEATURE_LABELS],
            [[z_header[index], *row] for index, row in enumerate(projection)],
        )
    elif filename == "pilot_matrix.csv":
        _write_csv(
            target,
            ["Row", *_FEATURE_LABELS],
            [[f"Z_{{{index + 1}}}", *row] for index, row in enumerate(projection)],
        )
    elif filename == "pilot_a_raw.csv":
        _write_csv(target, ["col_1", "col_2", "col_3"], projection)
    elif filename == "pilot_b.csv":
        _write_csv(
            target,
            [f"col_{index}" for index in range(1, dims + 1)],
            b,
        )
    elif filename == "pilot_c.csv":
        _write_csv(
            target,
            [f"col_{index}" for index in range(1, len(_ALGORITHM_LABELS) + 1)],
            c,
        )
    elif filename == "pilot_z.csv":
        if is_explore:
            _write_csv(
                target,
                ["Row", *z_header],
                [[f"instance_{index}", *row] for index, row in enumerate(z, start=1)],
            )
        else:
            _write_csv(target, z_header, z)
    elif filename == "pilot_r2.csv":
        _write_csv(target, ["r2"], [[value] for value in reconstruction_r2])
    elif filename == "pilot_error.csv":
        _write_csv(target, ["error"], [[reconstruction_error]])
    elif filename in {"x0.csv", "pilot_x0.csv"}:
        _write_csv(target, ["col_1", "col_2", "col_3"], _synthetic_x0())
    elif filename == "pilot_alpha.csv":
        if variant == "pilot_standard_numerical_3d_precalc":
            _write_csv(target, ["col_1"], [[value] for value in _synthetic_precalc()])
        else:
            _write_csv(
                target,
                ["col_1", "col_2", "col_3"],
                _synthetic_alpha(),
            )
    elif filename == "precalc_alpha.csv":
        _write_csv(
            target,
            ["precalc_alpha"],
            [[value] for value in _synthetic_precalc()],
        )
    elif filename == "pilot_perf.csv":
        _, perf = _pilot_numerical_trial_metrics(
            _synthetic_alpha(),
            x,
            y,
            dims,
            1.0,
        )
        _write_csv(target, ["perf"], [[value] for value in perf])
    elif filename == "pilot_eoptim.csv":
        eoptim, _ = _pilot_numerical_trial_metrics(
            _synthetic_alpha(),
            x,
            y,
            dims,
            1.0,
        )
        _write_csv(target, ["eoptim"], [[value] for value in eoptim])
    elif filename == "viewpoint_groups.csv":
        groups = (
            [[1, 2, 3, 4], [5, 6, 7, 8, 9, 10]]
            if variant == "pilot_pls_3d_grouped"
            else [list(range(1, len(_ALGORITHM_LABELS) + 1))]
        )
        rows = [
            [
                group_index,
                member_index,
                algorithm_index,
                _ALGORITHM_LABELS[algorithm_index - 1],
            ]
            for group_index, group in enumerate(groups, start=1)
            for member_index, algorithm_index in enumerate(group, start=1)
        ]
        _write_csv(
            target,
            ["group", "member", "algorithm_index", "algorithm"],
            rows,
        )
    elif filename == "viewpoint_a.csv":
        group_count = 2 if variant == "pilot_pls_3d_grouped" else 1
        rows = [
            [
                group,
                dimension,
                float(dimension == 1),
                float(dimension == 2),
                0.0,
            ]
            for group in range(1, group_count + 1)
            for dimension in (1, 2)
        ]
        _write_csv(
            target,
            ["group", "view_dimension", "z_1", "z_2", "z_3"],
            rows,
        )
    elif filename == "viewpoint_angles.csv":
        group_count = 2 if variant == "pilot_pls_3d_grouped" else 1
        _write_csv(
            target,
            ["group", "azimuth", "elevation"],
            [[group, 0.0, 1.5707963267948966] for group in range(1, group_count + 1)],
        )
    else:
        raise AssertionError(f"Unhandled synthetic PILOT artifact: {relative}")


def _synthetic_trace3d_faces() -> list[tuple[int, int, int]]:
    vertices = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
        ],
        dtype=np.double,
    )
    tetrahedra = [(0, 1, 2, 3), (0, 2, 1, 4)]
    counts: dict[tuple[int, int, int], int] = {}
    owners: dict[tuple[int, int, int], tuple[int, int, int, int]] = {}
    for simplex in tetrahedra:
        for omitted in range(4):
            face = cast(
                tuple[int, int, int],
                tuple(sorted(simplex[:omitted] + simplex[omitted + 1 :])),
            )
            counts[face] = counts.get(face, 0) + 1
            owners[face] = simplex
    oriented: list[tuple[int, int, int]] = []
    for face in sorted(item for item, count in counts.items() if count == 1):
        first, second, third = face
        normal = np.cross(
            vertices[second] - vertices[first],
            vertices[third] - vertices[first],
        )
        opposite = next(index for index in owners[face] if index not in face)
        if float(normal @ (vertices[opposite] - vertices[first])) >= 0:
            second, third = third, second
        oriented.append((first, second, third))
    return oriented


def _write_trace3d_profile_file(
    target: Path,
    relative: str,
) -> None:
    points = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, -1.0],
    ]
    if relative.startswith("explore_data/"):
        points[0] = [0.1, 0.1, 0.0]
    row_labels = [f"instance_{index}" for index in range(1, len(points) + 1)]
    filename = target.name
    is_input = "/inputs/" in relative
    if is_input:
        if filename == "z.csv":
            _write_csv(
                target,
                ["Row", "z_1", "z_2", "z_3"],
                [
                    [label, *point]
                    for label, point in zip(row_labels, points, strict=True)
                ],
            )
        elif filename in {"y_bin.csv", "y_hat.csv"}:
            input_rows = []
            for label in row_labels:
                values = [1, *([0] * (len(_ALGORITHM_LABELS) - 1))]
                if filename == "y_hat.csv":
                    values = [0] * len(_ALGORITHM_LABELS)
                input_rows.append([label, *values])
            _write_csv(target, ["Row", *_ALGORITHM_LABELS], input_rows)
        elif filename == "p.csv":
            _write_csv(
                target,
                ["Row", "p_best_algo"],
                [[label, 1] for label in row_labels],
            )
        elif filename == "beta.csv":
            _write_csv(target, ["Row", "beta"], [[label, 1] for label in row_labels])
        elif filename == "algorithm_labels.csv":
            _write_csv(
                target,
                ["algorithm_name"],
                [[label] for label in _ALGORITHM_LABELS],
            )
        else:
            raise AssertionError(f"Unhandled synthetic TRACE3 input: {relative}")
        return

    if filename in {"summary.csv", "eval_summary.csv"}:
        header = [
            "Row",
            "Volume_Good",
            "Volume_Good_Normalized",
            "Density_Good",
            "Density_Good_Normalized",
            "Purity_Good",
            "Volume_Best",
            "Volume_Best_Normalized",
            "Density_Best",
            "Density_Best_Normalized",
            "Purity_Best",
        ]
        populated_metrics = [0.333, 1.0, 15.0, 1.0, 1.0] * 2
        summary_rows = [
            [label, *(populated_metrics if index == 0 else [0.0] * 10)]
            for index, label in enumerate(_ALGORITHM_LABELS)
        ]
        _write_csv(target, header, summary_rows)
        return
    if filename == "membership.csv":
        header = [
            "Row",
            *(f"in_good_{label}" for label in _ALGORITHM_LABELS),
            *(f"in_best_{label}" for label in _ALGORITHM_LABELS),
        ]
        values = [1, *([0] * (len(_ALGORITHM_LABELS) - 1))] * 2
        _write_csv(target, header, [[label, *values] for label in row_labels])
        return
    if filename == "raw_metrics.csv":
        header = [
            "kind",
            "algorithm",
            "measure",
            "measure_label",
            "elements",
            "good_elements",
            "density",
            "purity",
            "alpha_radius",
            "region_threshold",
            "region_count",
            "tetrahedron_count",
            "boundary_face_count",
            "alpha_spectrum_count",
            "volume",
            "surface_area",
            "empty",
        ]
        alpha = np.sqrt(3.0) / 2.0
        surface = 2.0 + np.sqrt(3.0)
        metric_rows: list[list[object]] = []
        for kind in ("good", "best"):
            for index, label in enumerate(_ALGORITHM_LABELS):
                if index == 0:
                    metric_rows.append(
                        [
                            kind,
                            label,
                            1.0 / 3.0,
                            "Volume",
                            5,
                            5,
                            15.0,
                            1.0,
                            alpha,
                            0.0,
                            1,
                            2,
                            6,
                            1,
                            1.0 / 3.0,
                            surface,
                            0,
                        ],
                    )
                else:
                    metric_rows.append(
                        [
                            kind,
                            label,
                            0,
                            "Volume",
                            0,
                            0,
                            0,
                            0,
                            "NaN",
                            "NaN",
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            1,
                        ],
                    )
        metric_rows.append(
            ["hard", "", 0, "Volume", 0, 0, 0, 0, "NaN", "NaN", 0, 0, 0, 0, 0, 0, 1],
        )
        metric_rows.append(
            [
                "space",
                "",
                1.0 / 3.0,
                "Volume",
                5,
                "NaN",
                15.0,
                1,
                "NaN",
                "NaN",
                0,
                0,
                0,
                0,
                1.0 / 3.0,
                "NaN",
                1,
            ],
        )
        _write_csv(target, header, metric_rows)
        return

    prefix = filename
    for suffix in (
        "_vertices.csv",
        "_tetrahedra.csv",
        "_boundary_faces.csv",
        "_alpha_spectrum.csv",
    ):
        if filename.endswith(suffix):
            prefix = filename[: -len(suffix)]
            break
    is_populated = prefix in {"good_NB", "best_NB"}
    if filename.endswith("_vertices.csv"):
        _write_csv(
            target,
            ["vertex", "z_1", "z_2", "z_3"],
            (
                [[index, *point] for index, point in enumerate(points, start=1)]
                if is_populated
                else []
            ),
        )
    elif filename.endswith("_tetrahedra.csv"):
        tetrahedra = [[1, 1, 2, 3, 4], [2, 1, 3, 2, 5]] if is_populated else []
        _write_csv(target, ["tetrahedron", "v_1", "v_2", "v_3", "v_4"], tetrahedra)
    elif filename.endswith("_boundary_faces.csv"):
        face_rows = (
            [
                [index, *(vertex + 1 for vertex in face)]
                for index, face in enumerate(_synthetic_trace3d_faces(), start=1)
            ]
            if is_populated
            else []
        )
        _write_csv(target, ["face", "v_1", "v_2", "v_3"], face_rows)
    elif filename.endswith("_alpha_spectrum.csv"):
        spectrum_rows = [[1, np.sqrt(3.0) / 2.0]] if is_populated else []
        _write_csv(target, ["spectrum_index", "alpha"], spectrum_rows)
    else:
        raise AssertionError(f"Unhandled synthetic TRACE3 artifact: {relative}")


def _write_profile_file(root: Path, relative: str) -> None:
    target = root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    if relative.startswith("resolved_options/"):
        variant = target.stem
        artifact = {
            "schema_version": RESOLVED_OPTIONS_SCHEMA,
            "name": variant,
            "description": f"{variant} reference variant",
            "options": _effective_options(variant),
        }
        target.write_text(json.dumps(artifact), encoding="utf-8")
    elif "/trace/pilot_standard_analytic_3d/" in relative and not relative.endswith(
        "algorithm_labels.csv",
    ):
        _write_trace3d_profile_file(target, relative)
    elif relative.endswith("algorithm_labels.csv"):
        _write_csv(
            target,
            ["algorithm_name"],
            [[label] for label in _ALGORITHM_LABELS],
        )
    elif relative.startswith("build_data/pilot/default/inputs/"):
        _write_pilot_profile_file(
            target,
            relative,
            "pilot_standard_analytic_3d",
        )
    elif (
        "/pilot/" in relative
        and len(Path(relative).parts) >= 3
        and Path(relative).parts[2] in _VARIANTS[3:]
    ):
        _write_pilot_profile_file(target, relative, Path(relative).parts[2])
    elif "/trace/" in relative and (
        target.name.startswith(("good_", "best_")) or target.name == "hard.csv"
    ):
        target.write_text("part,ring,vertex,is_hole,z_1,z_2\n", encoding="utf-8")
    elif relative in {
        "shared_inputs/reference/metadata.csv",
        "shared_inputs/reference/metadata_test.csv",
    }:
        target.write_bytes((_CURRENT_FIXTURES / relative).read_bytes())
    else:
        target.write_text("value\n1\n", encoding="utf-8")


def _manifest_entry(root: Path, relative: str) -> dict[str, Any]:
    target = root / relative
    phase, stage, variant = _entry_metadata(relative)
    media_type = "application/json" if target.suffix == ".json" else "text/csv"
    rows = 0
    columns = 0
    empty = False
    if media_type == "text/csv":
        with target.open("r", encoding="utf-8", newline="") as stream:
            csv_rows = list(csv.reader(stream))
        rows = max(0, len(csv_rows) - 1)
        columns = len(csv_rows[0]) if csv_rows else 0
        empty = rows == 0
    return {
        "path": relative,
        "sha256": sha256_file(target),
        "size_bytes": target.stat().st_size,
        "media_type": media_type,
        "role": relative,
        "phase": phase,
        "stage": stage,
        "variant": variant,
        "empty": empty,
        "rows": rows,
        "columns": columns,
    }


def _write_bundle(
    root: Path,
    *,
    trust: str = DIAGNOSTIC_TRUST,
    release: str = "R2026a",
    matlab_dirty: bool = False,
    profile: str = REFERENCE_PROFILE,
) -> dict[str, Any]:
    evidence_enabled = profile == REFERENCE_PROFILE
    is_verified_v2 = trust == VERIFIED_TRUST and evidence_enabled
    fixed_paths = (
        _fixed_reference_paths() if evidence_enabled else _fixed_reference_paths_v1()
    )
    variants = _VARIANTS if evidence_enabled else _VARIANTS[:3]
    profile_paths = fixed_paths | _geometry_paths(_ALGORITHM_LABELS)
    if evidence_enabled:
        profile_paths |= _trace3d_mesh_paths(_ALGORITHM_LABELS)
    for relative in sorted(profile_paths):
        _write_profile_file(root, relative)
    files = [_manifest_entry(root, relative) for relative in sorted(profile_paths)]
    manifest: dict[str, Any] = {
        "schema_version": BUNDLE_SCHEMA,
        "profile": profile,
        "bundle_id": "study-defaults",
        "trust": trust,
        "generated_at": "2026-08-17T00:00:00Z",
        "dataset": {
            "name": "study",
            "seed": 42,
            "training_input": "shared_inputs/reference/metadata.csv",
            "test_input": "shared_inputs/reference/metadata_test.csv",
        },
        "resolved_options": {
            "schema_version": RESOLVED_OPTIONS_INDEX_SCHEMA,
            "variants": [
                {
                    "name": variant,
                    "description": f"{variant} reference variant",
                    "path": f"resolved_options/{variant}.json",
                }
                for variant in variants
            ],
        },
        "matlab": {
            "repo_commit": _GOLD_MATLAB_COMMIT if is_verified_v2 else _COMMIT,
            "repo_dirty": matlab_dirty,
            "toolkit_version": "0.9.0",
            "release": release,
            "version": "25.1",
            "platform": "maca64",
            "installed_toolboxes": [
                "MATLAB",
                "Statistics and Machine Learning Toolbox",
                "Optimization Toolbox",
                "Global Optimization Toolbox",
                "Financial Toolbox",
            ],
            "required_toolboxes": [
                "MATLAB",
                "Statistics and Machine Learning Toolbox",
                "Optimization Toolbox",
                "Global Optimization Toolbox",
                "Financial Toolbox",
            ],
        },
        "generator": {
            "repo_commit": _COMMIT,
            "repo_dirty": False,
            "script": _EXPORTER_SCRIPT,
            "script_sha256": (
                _REFERENCE_V2_EXPORTER_SHA256 if is_verified_v2 else _SCRIPT_HASH
            ),
        },
        "files": files,
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return manifest


def _copy_verified_bundle(root: Path) -> dict[str, Any]:
    """Copy the installed audited oracle for verified-identity tests."""
    shutil.copytree(_CURRENT_FIXTURES, root, dirs_exist_ok=True)
    return cast(
        dict[str, Any],
        json.loads((root / "manifest.json").read_text(encoding="utf-8")),
    )


def _rewrite_manifest(root: Path, manifest: dict[str, Any]) -> None:
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def _entry_for(manifest: dict[str, Any], relative: str) -> dict[str, Any]:
    return next(entry for entry in manifest["files"] if entry["path"] == relative)


def _refresh_entry(root: Path, entry: dict[str, Any]) -> None:
    refreshed = _manifest_entry(root, entry["path"])
    entry.update(refreshed)


def test_verified_bundle_passes_full_integrity_check(tmp_path: Path) -> None:
    """Accept an intact current-release bundle with complete provenance."""
    _copy_verified_bundle(tmp_path)

    report = validate_bundle(tmp_path)

    assert report.trust == VERIFIED_TRUST
    assert report.matlab_release == "R2026a"
    assert report.file_count == len(
        _fixed_reference_paths()
        | _geometry_paths(_ALGORITHM_LABELS)
        | _trace3d_mesh_paths(_ALGORITHM_LABELS),
    )
    assert report.total_bytes > 0


def test_reference_v2_exporter_identity_matches_checked_in_script() -> None:
    """Force exporter changes to update the pinned v2 identity explicitly."""
    exporter = Path(__file__).parents[1] / _EXPORTER_SCRIPT

    assert sha256_file(exporter) == _REFERENCE_V2_EXPORTER_SHA256


def test_verified_v2_content_root_matches_installed_oracle() -> None:
    """Pin all 423 audited artifact identities, not only source metadata."""
    manifest = cast(
        dict[str, Any],
        json.loads((_CURRENT_FIXTURES / "manifest.json").read_text(encoding="utf-8")),
    )
    digest = hashlib.sha256()
    for entry in sorted(manifest["files"], key=lambda item: item["path"]):
        digest.update(entry["path"].encode("utf-8"))
        digest.update(b"\0")
        digest.update(entry["sha256"].encode("ascii"))
        digest.update(b"\n")

    assert digest.hexdigest() == _VERIFIED_V2_CONTENT_ROOT_SHA256


def test_verified_v2_rejects_coherent_viewpoint_rehash(tmp_path: Path) -> None:
    """A coherent A/angle substitution cannot self-declare verified trust."""
    manifest = _copy_verified_bundle(tmp_path)
    root = "build_data/pilot/pilot_standard_analytic_3d/outputs"
    matrix_relative = f"{root}/viewpoint_a.csv"
    angles_relative = f"{root}/viewpoint_angles.csv"
    matrix = tmp_path / matrix_relative
    angles = tmp_path / angles_relative
    _write_csv(
        matrix,
        ["group", "view_dimension", "z_1", "z_2", "z_3"],
        [[1, 1, 1.0, 0.0, 0.0], [1, 2, 0.0, 1.0, 0.0]],
    )
    _write_csv(
        angles,
        ["group", "azimuth", "elevation"],
        [[1, 0.0, np.pi / 2.0]],
    )
    _refresh_entry(tmp_path, _entry_for(manifest, matrix_relative))
    _refresh_entry(tmp_path, _entry_for(manifest, angles_relative))
    _rewrite_manifest(tmp_path, manifest)

    with pytest.raises(ProvenanceError, match="content root"):
        validate_bundle(tmp_path)


def test_verified_v2_requires_the_gold_matlab_commit(tmp_path: Path) -> None:
    """Reject a clean but self-declared MATLAB source identity."""
    manifest = _copy_verified_bundle(tmp_path)
    manifest["matlab"]["repo_commit"] = "f" * 40
    _rewrite_manifest(tmp_path, manifest)

    with pytest.raises(ProvenanceError, match="gold MATLAB commit"):
        validate_bundle(tmp_path)


def test_verified_v2_requires_the_canonical_dataset_after_rehash(
    tmp_path: Path,
) -> None:
    """Reject a substituted input even when its manifest metadata is refreshed."""
    manifest = _copy_verified_bundle(tmp_path)
    relative = "shared_inputs/reference/metadata.csv"
    target = tmp_path / relative
    contents = target.read_text(encoding="utf-8")
    assert "abalone," in contents
    target.write_text(contents.replace("abalone,", "substitute,", 1), encoding="utf-8")
    _refresh_entry(tmp_path, _entry_for(manifest, relative))
    _rewrite_manifest(tmp_path, manifest)

    with pytest.raises(ProvenanceError, match="canonical dataset"):
        validate_bundle(tmp_path)


def test_verified_v2_requires_the_pinned_exporter_hash(tmp_path: Path) -> None:
    """Reject a syntactically valid self-declared exporter identity."""
    manifest = _copy_verified_bundle(tmp_path)
    manifest["generator"]["script_sha256"] = "f" * 64
    _rewrite_manifest(tmp_path, manifest)

    with pytest.raises(ProvenanceError, match="pinned exporter script hash"):
        validate_bundle(tmp_path)


def test_verified_v2_rejects_rehashed_algorithm_label_substitution(
    tmp_path: Path,
) -> None:
    """Tie every audited algorithm-label artifact to the verified content root."""
    manifest = _copy_verified_bundle(tmp_path)
    for entry in manifest["files"]:
        relative = entry["path"]
        if not relative.endswith("algorithm_labels.csv"):
            continue
        target = tmp_path / relative
        contents = target.read_text(encoding="utf-8")
        assert "NB" in contents
        target.write_text(contents.replace("NB", "altered", 1), encoding="utf-8")
        _refresh_entry(tmp_path, entry)
    _rewrite_manifest(tmp_path, manifest)

    with pytest.raises(ProvenanceError, match="content root"):
        validate_bundle(tmp_path)


def test_reference_profile_declares_the_documented_file_count() -> None:
    """Keep the fixed reference-study profile and documentation synchronized."""
    profile = (
        _fixed_reference_paths()
        | _geometry_paths(_REFERENCE_ALGORITHM_LABELS)
        | _trace3d_mesh_paths(_REFERENCE_ALGORITHM_LABELS)
    )

    assert len(profile) == 423


def test_frozen_v1_profile_remains_valid(tmp_path: Path) -> None:
    """Keep the installed 229-file oracle readable during the v2 migration."""
    _write_bundle(tmp_path, profile=REFERENCE_PROFILE_V1)

    report = validate_bundle(tmp_path, allow_diagnostic=True)

    assert report.file_count == len(
        _fixed_reference_paths_v1() | _geometry_paths(_ALGORITHM_LABELS),
    )


def test_verified_v2_profile_requires_r2026a(tmp_path: Path) -> None:
    """Pin SIMPLS/viewpoint evidence to the audited MATLAB release."""
    manifest = _copy_verified_bundle(tmp_path)
    manifest["matlab"]["release"] = "R2025a"
    _rewrite_manifest(tmp_path, manifest)

    with pytest.raises(ProvenanceError, match="v2 requires MATLAB R2026a"):
        validate_bundle(tmp_path)


@pytest.mark.parametrize("mutation", ["content", "extra", "missing"])
def test_file_set_or_hash_changes_are_rejected(tmp_path: Path, mutation: str) -> None:
    """Reject altered, missing, and untracked fixture content."""
    manifest = _write_bundle(tmp_path)
    target = tmp_path / manifest["files"][0]["path"]
    if mutation == "content":
        target.write_text("changed", encoding="utf-8")
    elif mutation == "extra":
        (tmp_path / "untracked.csv").write_text("x\n1\n", encoding="utf-8")
    else:
        target.unlink()

    with pytest.raises(ProvenanceError):
        validate_bundle(tmp_path, allow_diagnostic=True)


def test_duplicate_casefolded_paths_are_rejected(tmp_path: Path) -> None:
    """Reject archive paths that collide on case-insensitive filesystems."""
    manifest = _write_bundle(tmp_path)
    duplicate = deepcopy(manifest["files"][0])
    duplicate["path"] = duplicate["path"].upper()
    manifest["files"].append(duplicate)
    _rewrite_manifest(tmp_path, manifest)

    with pytest.raises(ProvenanceError, match="case-colliding"):
        validate_bundle(tmp_path, allow_diagnostic=True)


@pytest.mark.parametrize(
    "unsafe",
    ["../escape.csv", "/absolute.csv", "a/../b.csv", "a\\b.csv"],
)
def test_noncanonical_manifest_paths_are_rejected(tmp_path: Path, unsafe: str) -> None:
    """Reject paths that are unsafe or platform-dependent."""
    manifest = _write_bundle(tmp_path)
    manifest["files"][0]["path"] = unsafe
    _rewrite_manifest(tmp_path, manifest)

    with pytest.raises(ProvenanceError, match="Unsafe|noncanonical"):
        validate_bundle(tmp_path, allow_diagnostic=True)


def test_verified_bundle_requires_clean_current_matlab(tmp_path: Path) -> None:
    """Require clean sources and the declared gold MATLAB release."""
    manifest = _copy_verified_bundle(tmp_path)
    manifest["matlab"]["release"] = "R2024a"
    manifest["matlab"]["repo_dirty"] = True
    _rewrite_manifest(tmp_path, manifest)

    with pytest.raises(ProvenanceError, match="clean MATLAB"):
        validate_bundle(tmp_path)


def test_diagnostic_bundle_is_explicitly_opt_in(tmp_path: Path) -> None:
    """Keep older diagnostic exports out of parity assertions by default."""
    _write_bundle(tmp_path, trust=DIAGNOSTIC_TRUST, release="R2024a", matlab_dirty=True)

    with pytest.raises(ProvenanceError, match="not accepted as parity oracles"):
        validate_bundle(tmp_path)
    report = validate_bundle(tmp_path, allow_diagnostic=True)
    assert report.trust == DIAGNOSTIC_TRUST


def test_verified_bundle_installs_atomically_with_layout_intact(tmp_path: Path) -> None:
    """Install only a validated bundle and preserve every relative path."""
    source = tmp_path / "source"
    source.mkdir()
    _copy_verified_bundle(source)
    destination = tmp_path / "fixtures" / "matlab" / "current"

    report = install_verified_bundle(source, destination)

    assert report.root == destination
    assert report.file_count == len(
        _fixed_reference_paths()
        | _geometry_paths(_ALGORITHM_LABELS)
        | _trace3d_mesh_paths(_ALGORITHM_LABELS),
    )
    assert (destination / "manifest.json").is_file()
    assert (
        destination / "shared_inputs" / "reference" / "metadata.csv"
    ).read_bytes() == (
        source / "shared_inputs" / "reference" / "metadata.csv"
    ).read_bytes()
    assert (
        destination
        / "build_data"
        / "trace"
        / "trace3_default"
        / "outputs"
        / "good_NB.csv"
    ).is_file()
    validate_bundle(destination)


def test_install_rejects_diagnostic_or_existing_destination(tmp_path: Path) -> None:
    """Never promote diagnostics or overwrite an existing fixture tree."""
    diagnostic = tmp_path / "diagnostic"
    diagnostic.mkdir()
    _write_bundle(
        diagnostic,
        trust=DIAGNOSTIC_TRUST,
        release="R2024a",
        matlab_dirty=True,
    )
    destination = tmp_path / "current"

    with pytest.raises(ProvenanceError, match="not accepted as parity oracles"):
        install_verified_bundle(diagnostic, destination)

    verified = tmp_path / "verified"
    verified.mkdir()
    _copy_verified_bundle(verified)
    destination.mkdir()
    with pytest.raises(ProvenanceError, match="already exists"):
        install_verified_bundle(verified, destination)


def test_missing_required_toolbox_is_rejected(tmp_path: Path) -> None:
    """Reject an export that lacks a required MATLAB toolbox."""
    manifest = _write_bundle(tmp_path)
    manifest["matlab"]["installed_toolboxes"].remove("Financial Toolbox")
    _rewrite_manifest(tmp_path, manifest)

    with pytest.raises(ProvenanceError, match="Financial Toolbox"):
        validate_bundle(tmp_path, allow_diagnostic=True)


def test_profile_cannot_omit_a_required_toolbox_from_both_lists(
    tmp_path: Path,
) -> None:
    """Require the known exporter dependencies, not only internal list agreement."""
    manifest = _write_bundle(tmp_path)
    manifest["matlab"]["installed_toolboxes"].remove("Financial Toolbox")
    manifest["matlab"]["required_toolboxes"].remove("Financial Toolbox")
    _rewrite_manifest(tmp_path, manifest)

    with pytest.raises(ProvenanceError, match="reference-export dependencies"):
        validate_bundle(tmp_path, allow_diagnostic=True)


def test_csv_shape_and_empty_status_are_verified(tmp_path: Path) -> None:
    """Verify semantic shape metadata in addition to file hashes."""
    manifest = _write_bundle(tmp_path)
    manifest["files"][0]["rows"] = 2
    _rewrite_manifest(tmp_path, manifest)

    with pytest.raises(ProvenanceError, match="row-count"):
        validate_bundle(tmp_path, allow_diagnostic=True)


@pytest.mark.parametrize(
    "relative",
    [
        "shared_inputs/reference/metadata_test.csv",
        "resolved_options/legacy_svm.json",
        "build_data/pilot/default/outputs/pilot_z.csv",
        "build_data/pythia/trace3_default/outputs/raw_metrics.csv",
        "build_data/trace/trace3_default/outputs/raw_metrics.csv",
        "build_data/trace/trace3_default/outputs/good_NB.csv",
        "explore_data/trace/trace3_default/inputs/y_bin.csv",
        "explore_data/pythia/legacy_svm/outputs/predictions.csv",
        "explore_data/trace/trace3_pythia_skip/outputs/membership.csv",
        "build_data/pilot/pilot_standard_analytic_3d/inputs/stage_context.json",
        "build_data/pilot/pilot_standard_numerical_3d_x0/inputs/x0.csv",
        "build_data/pilot/pilot_standard_numerical_3d_precalc/inputs/precalc_alpha.csv",
        "build_data/pilot/pilot_pls_3d_grouped/outputs/viewpoint_a.csv",
        "explore_data/pilot/pilot_pls_3d_grouped/outputs/pilot_z.csv",
        "build_data/trace/pilot_standard_analytic_3d/inputs/z.csv",
        "build_data/trace/pilot_standard_analytic_3d/outputs/good_NB_tetrahedra.csv",
        "build_data/trace/pilot_standard_analytic_3d/outputs/hard_vertices.csv",
        "explore_data/trace/pilot_standard_analytic_3d/outputs/membership.csv",
    ],
)
def test_profile_rejects_deleting_file_and_manifest_entry(
    tmp_path: Path,
    relative: str,
) -> None:
    """Reject self-consistent manifests that omit a required profile artifact."""
    manifest = _write_bundle(tmp_path)
    (tmp_path / relative).unlink()
    manifest["files"] = [
        entry for entry in manifest["files"] if entry["path"] != relative
    ]
    _rewrite_manifest(tmp_path, manifest)

    with pytest.raises(ProvenanceError, match="Reference export profile"):
        validate_bundle(tmp_path, allow_diagnostic=True)


def test_trace3d_reader_validates_a_broken_v2_instead_of_skipping(
    tmp_path: Path,
) -> None:
    """Never turn a malformed declared-v2 oracle into a silent reader skip."""
    from tests.test_current_matlab_trace3_3d_parity import (
        _trace3d_bundle_available,
    )

    manifest = _write_bundle(tmp_path)
    relative = "build_data/trace/pilot_standard_analytic_3d/outputs/raw_metrics.csv"
    (tmp_path / relative).unlink()
    manifest["files"] = [
        entry for entry in manifest["files"] if entry["path"] != relative
    ]
    _rewrite_manifest(tmp_path, manifest)

    with pytest.raises(ProvenanceError, match="Reference export profile"):
        _trace3d_bundle_available(tmp_path, allow_diagnostic=True)


def test_profile_rejects_an_internally_consistent_two_file_bundle(
    tmp_path: Path,
) -> None:
    """Require the exporter profile even when hashes and inventory agree."""
    manifest = _write_bundle(tmp_path)
    retained = {
        "shared_inputs/reference/metadata.csv",
        "shared_inputs/reference/metadata_test.csv",
    }
    for entry in manifest["files"]:
        if entry["path"] not in retained:
            (tmp_path / entry["path"]).unlink()
    manifest["files"] = [
        entry for entry in manifest["files"] if entry["path"] in retained
    ]
    _rewrite_manifest(tmp_path, manifest)

    with pytest.raises(ProvenanceError, match="Reference export profile"):
        validate_bundle(tmp_path, allow_diagnostic=True)


def test_profile_rejects_self_consistent_stale_artifact(tmp_path: Path) -> None:
    """Reject an extra exporter artifact even when it is correctly manifested."""
    manifest = _write_bundle(tmp_path)
    relative = "build_data/pilot/default/outputs/stale.csv"
    _write_profile_file(tmp_path, relative)
    manifest["files"].append(_manifest_entry(tmp_path, relative))
    _rewrite_manifest(tmp_path, manifest)

    with pytest.raises(ProvenanceError, match="file-set mismatch.*stale.csv"):
        validate_bundle(tmp_path, allow_diagnostic=True)


@pytest.mark.parametrize("missing", ["perf", "outputs"])
def test_resolved_options_require_every_top_level_group(
    tmp_path: Path,
    missing: str,
) -> None:
    """Reject partial option artifacts even when their hash is refreshed."""
    manifest = _write_bundle(tmp_path)
    relative = "resolved_options/trace3_default.json"
    target = tmp_path / relative
    artifact = json.loads(target.read_text(encoding="utf-8"))
    artifact["options"].pop(missing)
    target.write_text(json.dumps(artifact), encoding="utf-8")
    _refresh_entry(root=tmp_path, entry=_entry_for(manifest, relative))
    _rewrite_manifest(tmp_path, manifest)

    with pytest.raises(ProvenanceError, match="exact option groups"):
        validate_bundle(tmp_path, allow_diagnostic=True)


def test_resolved_options_require_every_nested_field(tmp_path: Path) -> None:
    """Reject a complete-looking tree with one effective MATLAB field omitted."""
    manifest = _write_bundle(tmp_path)
    relative = "resolved_options/trace3_pythia_skip.json"
    target = tmp_path / relative
    artifact = json.loads(target.read_text(encoding="utf-8"))
    artifact["options"]["pythia"].pop("skip")
    target.write_text(json.dumps(artifact), encoding="utf-8")
    _refresh_entry(root=tmp_path, entry=_entry_for(manifest, relative))
    _rewrite_manifest(tmp_path, manifest)

    with pytest.raises(ProvenanceError, match="pythia.*MATLAB schema"):
        validate_bundle(tmp_path, allow_diagnostic=True)


def test_resolved_options_index_must_link_the_matching_artifact(
    tmp_path: Path,
) -> None:
    """Reject a variant index that points at a different options artifact."""
    manifest = _write_bundle(tmp_path)
    manifest["resolved_options"]["variants"][0][
        "path"
    ] = "resolved_options/legacy_svm.json"
    _rewrite_manifest(tmp_path, manifest)

    with pytest.raises(ProvenanceError, match="Resolved-options path"):
        validate_bundle(tmp_path, allow_diagnostic=True)


def test_resolved_options_index_requires_complete_records(tmp_path: Path) -> None:
    """Reject a variant index that omits its artifact description."""
    manifest = _write_bundle(tmp_path)
    manifest["resolved_options"]["variants"][0].pop("description")
    _rewrite_manifest(tmp_path, manifest)

    with pytest.raises(ProvenanceError, match="invalid structure"):
        validate_bundle(tmp_path, allow_diagnostic=True)


def test_resolved_options_artifact_name_must_match_variant(tmp_path: Path) -> None:
    """Reject an artifact whose internal identity disagrees with its index."""
    manifest = _write_bundle(tmp_path)
    relative = "resolved_options/legacy_svm.json"
    target = tmp_path / relative
    artifact = json.loads(target.read_text(encoding="utf-8"))
    artifact["name"] = "trace3_default"
    target.write_text(json.dumps(artifact), encoding="utf-8")
    _refresh_entry(root=tmp_path, entry=_entry_for(manifest, relative))
    _rewrite_manifest(tmp_path, manifest)

    with pytest.raises(ProvenanceError, match="artifact name"):
        validate_bundle(tmp_path, allow_diagnostic=True)


def test_resolved_variant_values_must_match_export_profile(tmp_path: Path) -> None:
    """Reject a structurally complete tree describing the wrong variant."""
    manifest = _write_bundle(tmp_path)
    relative = "resolved_options/trace3_pythia_skip.json"
    target = tmp_path / relative
    artifact = json.loads(target.read_text(encoding="utf-8"))
    artifact["options"]["pythia"]["skip"] = False
    target.write_text(json.dumps(artifact), encoding="utf-8")
    _refresh_entry(root=tmp_path, entry=_entry_for(manifest, relative))
    _rewrite_manifest(tmp_path, manifest)

    with pytest.raises(ProvenanceError, match="PYTHIA options mismatch"):
        validate_bundle(tmp_path, allow_diagnostic=True)


@pytest.mark.parametrize(
    ("variant", "field", "value"),
    [
        ("pilot_standard_numerical_3d_x0", "ntries", 3),
        ("pilot_pls_3d_grouped", "analytic", False),
        ("pilot_pls_3d_grouped", "alpha", 1.0),
        ("pilot_pls_3d_grouped", "viewGroups", [[1, 2]]),
    ],
)
def test_pilot_evidence_options_are_exact(
    tmp_path: Path,
    variant: str,
    field: str,
    value: object,
) -> None:
    """Reject evidence variants that no longer exercise their named branch."""
    manifest = _write_bundle(tmp_path)
    relative = f"resolved_options/{variant}.json"
    target = tmp_path / relative
    artifact = json.loads(target.read_text(encoding="utf-8"))
    artifact["options"]["pilot"][field] = value
    target.write_text(json.dumps(artifact), encoding="utf-8")
    _refresh_entry(tmp_path, _entry_for(manifest, relative))
    _rewrite_manifest(tmp_path, manifest)

    with pytest.raises(ProvenanceError, match="PILOT"):
        validate_bundle(tmp_path, allow_diagnostic=True)


@pytest.mark.parametrize(
    ("relative", "old", "new", "message"),
    [
        (
            "build_data/pilot/pilot_standard_analytic_3d/outputs/pilot_z.csv",
            "z_3",
            "wrong_dimension",
            "dimensional header",
        ),
        (
            "build_data/pilot/pilot_pls_3d_grouped/outputs/viewpoint_angles.csv",
            "1.5707963267948966",
            "1.0",
            "viewpoint angles",
        ),
        (
            "build_data/pilot/pilot_standard_numerical_3d_precalc/inputs/precalc_alpha.csv",
            "\n1.0\n",
            "\n1.05\n",
            "precalculated",
        ),
    ],
)
def test_pilot_evidence_artifact_mutations_are_rejected(
    tmp_path: Path,
    relative: str,
    old: str,
    new: str,
    message: str,
) -> None:
    """Hash-consistent scientific mutations must still violate v2 semantics."""
    manifest = _write_bundle(tmp_path)
    target = tmp_path / relative
    contents = target.read_text(encoding="utf-8")
    assert old in contents
    target.write_text(contents.replace(old, new, 1), encoding="utf-8")
    _refresh_entry(tmp_path, _entry_for(manifest, relative))
    _rewrite_manifest(tmp_path, manifest)

    with pytest.raises(ProvenanceError, match=message):
        validate_bundle(tmp_path, allow_diagnostic=True)


@pytest.mark.parametrize(
    ("filename", "message"),
    [
        ("pilot_eoptim.csv", "trial objective"),
        ("pilot_perf.csv", "trial topology"),
        ("pilot_alpha.csv", "trial objective"),
    ],
)
def test_pilot_trial_metrics_reject_hash_consistent_mutations(
    tmp_path: Path,
    filename: str,
    message: str,
) -> None:
    """Recompute every trial instead of trusting rehashed MATLAB diagnostics."""
    manifest = _write_bundle(tmp_path)
    relative = "build_data/pilot/pilot_standard_numerical_3d_x0/outputs/" + filename
    target = tmp_path / relative
    header, rows = _read_csv_for_mutation(target)
    if filename == "pilot_alpha.csv":
        perf_path = target.with_name("pilot_perf.csv")
        _, perf_rows = _read_csv_for_mutation(perf_path)
        best = max(range(len(perf_rows)), key=lambda index: float(perf_rows[index][0]))
        nonselected = next(index for index in range(len(rows[0])) if index != best)
        rows[0][nonselected] = "999.0"
    else:
        rows[0][0] = "999.0"
    _write_csv(target, header, rows)
    _refresh_entry(tmp_path, _entry_for(manifest, relative))
    _rewrite_manifest(tmp_path, manifest)

    with pytest.raises(ProvenanceError, match=message):
        validate_bundle(tmp_path, allow_diagnostic=True)


def test_pilot_stage_lineage_is_required(tmp_path: Path) -> None:
    """Reject an options-complete artifact that hides its 2D SIFTED lineage."""
    manifest = _write_bundle(tmp_path)
    relative = "build_data/pilot/pilot_pls_2d/inputs/stage_context.json"
    target = tmp_path / relative
    context = json.loads(target.read_text(encoding="utf-8"))
    context["sifted_effective_pilot_dims"] = 3
    target.write_text(json.dumps(context), encoding="utf-8")
    _refresh_entry(tmp_path, _entry_for(manifest, relative))
    _rewrite_manifest(tmp_path, manifest)

    with pytest.raises(ProvenanceError, match="upstream dimensions"):
        validate_bundle(tmp_path, allow_diagnostic=True)


def test_profile_rejects_path_metadata_mismatch(tmp_path: Path) -> None:
    """Require manifest phase, stage, and variant to agree with the path."""
    manifest = _write_bundle(tmp_path)
    entry = _entry_for(
        manifest,
        "explore_data/trace/trace3_default/outputs/membership.csv",
    )
    entry["phase"] = "build"
    _rewrite_manifest(tmp_path, manifest)

    with pytest.raises(ProvenanceError, match="Manifest metadata"):
        validate_bundle(tmp_path, allow_diagnostic=True)


def test_trace_geometry_schema_accepts_parts_and_holes(tmp_path: Path) -> None:
    """Accept reconstructable region/ring geometry without closure vertices."""
    manifest = _write_bundle(tmp_path)
    entry = _entry_for(
        manifest,
        "build_data/trace/trace3_default/outputs/good_NB.csv",
    )
    target = tmp_path / entry["path"]
    target.write_text(
        "part,ring,vertex,is_hole,z_1,z_2\n"
        "1,exterior,1,0,0,0\n"
        "1,exterior,2,0,3,0\n"
        "1,exterior,3,0,0,3\n"
        "1,hole_1,1,1,0.5,0.5\n"
        "1,hole_1,2,1,0.5,1\n"
        "1,hole_1,3,1,1,0.5\n",
        encoding="utf-8",
    )
    entry.update(
        sha256=sha256_file(target),
        size_bytes=target.stat().st_size,
        rows=6,
        empty=False,
    )
    _rewrite_manifest(tmp_path, manifest)

    validate_bundle(tmp_path, allow_diagnostic=True)


@pytest.mark.parametrize(
    "geometry",
    [
        "part,ring,vertex,z_1,z_2\n",
        (
            "part,ring,vertex,is_hole,z_1,z_2\n"
            "1,exterior,1,1,0,0\n"
            "1,exterior,2,1,1,0\n"
            "1,exterior,3,1,0,1\n"
        ),
        (
            "part,ring,vertex,is_hole,z_1,z_2\n"
            "1,exterior,1,0,0,0\n"
            "1,exterior,3,0,1,0\n"
            "1,exterior,4,0,0,1\n"
        ),
    ],
)
def test_trace_geometry_schema_rejects_ambiguous_data(
    tmp_path: Path,
    geometry: str,
) -> None:
    """Reject geometry that cannot be reconstructed without guessing."""
    manifest = _write_bundle(tmp_path)
    entry = _entry_for(
        manifest,
        "build_data/trace/trace3_default/outputs/good_NB.csv",
    )
    target = tmp_path / entry["path"]
    target.write_text(geometry, encoding="utf-8")
    with target.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.reader(stream))
    entry.update(
        sha256=sha256_file(target),
        size_bytes=target.stat().st_size,
        rows=max(0, len(rows) - 1),
        columns=len(rows[0]),
        empty=len(rows) == 1,
    )
    _rewrite_manifest(tmp_path, manifest)

    with pytest.raises(ProvenanceError, match="TRACE geometry"):
        validate_bundle(tmp_path, allow_diagnostic=True)


@pytest.mark.parametrize(
    ("relative", "message", "mutate"),
    [
        (
            "build_data/trace/pilot_standard_analytic_3d/outputs/good_NB_boundary_faces.csv",
            "outward oriented",
            "reverse_face",
        ),
        (
            "build_data/trace/pilot_standard_analytic_3d/outputs/good_NB_tetrahedra.csv",
            "connectivity index",
            "invalid_index",
        ),
        (
            "build_data/trace/pilot_standard_analytic_3d/outputs/good_NB_alpha_spectrum.csv",
            "strictly descending",
            "spectrum_order",
        ),
        (
            "build_data/trace/pilot_standard_analytic_3d/outputs/raw_metrics.csv",
            "volume mismatch",
            "volume",
        ),
        (
            "explore_data/trace/pilot_standard_analytic_3d/outputs/membership.csv",
            "membership mismatch",
            "membership",
        ),
        (
            "build_data/trace/pilot_standard_analytic_3d/outputs/hard_vertices.csv",
            "header-only",
            "empty_vertex",
        ),
        (
            "build_data/trace/pilot_standard_analytic_3d/outputs/summary.csv",
            "summary values must be finite",
            "summary_nan",
        ),
        (
            "build_data/trace/pilot_standard_analytic_3d/outputs/raw_metrics.csv",
            "nonnegative integer",
            "fractional_count",
        ),
        (
            "build_data/trace/pilot_standard_analytic_3d/outputs/raw_metrics.csv",
            "nonnegative integer",
            "negative_count",
        ),
        (
            "build_data/trace/pilot_standard_analytic_3d/outputs/raw_metrics.csv",
            "nonnegative integer",
            "nonfinite_count",
        ),
        (
            "build_data/trace/pilot_standard_analytic_3d/outputs/raw_metrics.csv",
            "RegionThreshold",
            "region_threshold",
        ),
        (
            "build_data/trace/pilot_standard_analytic_3d/outputs/raw_metrics.csv",
            "space",
            "space_state",
        ),
        (
            "build_data/trace/pilot_standard_analytic_3d/outputs/good_NB_vertices.csv",
            "identifiers are not contiguous",
            "fractional_identifier",
        ),
        (
            "build_data/trace/pilot_standard_analytic_3d/outputs/good_NB_alpha_spectrum.csv",
            "full support Delaunay spectrum",
            "extra_spectrum",
        ),
    ],
)
def test_trace3d_scientific_mutations_are_rejected(
    tmp_path: Path,
    relative: str,
    message: str,
    mutate: str,
) -> None:
    """Reject rehashed topology, alpha-state, metric, and membership corruption."""
    manifest = _write_bundle(tmp_path)
    target = tmp_path / relative
    header, rows = _read_csv_for_mutation(target)
    if mutate == "reverse_face":
        rows[0][1], rows[0][2] = rows[0][2], rows[0][1]
    elif mutate == "invalid_index":
        rows[0][-1] = "99"
    elif mutate == "spectrum_order":
        rows[:] = [["1", "0.5"], ["2", rows[0][1]]]
    elif mutate == "volume":
        good_nb = next(row for row in rows if row[:2] == ["good", "NB"])
        good_nb[14] = "9.0"
    elif mutate == "membership":
        rows[0][1] = "0"
    elif mutate == "empty_vertex":
        rows.append(["1", "0", "0", "0"])
    elif mutate == "summary_nan":
        rows[0][1] = "NaN"
    elif mutate in {"fractional_count", "negative_count", "nonfinite_count"}:
        good_nb = next(row for row in rows if row[:2] == ["good", "NB"])
        replacements = {
            "fractional_count": "1.5",
            "negative_count": "-1",
            "nonfinite_count": "Inf",
        }
        good_nb[header.index("region_count")] = replacements[mutate]
    elif mutate == "region_threshold":
        good_nb = next(row for row in rows if row[:2] == ["good", "NB"])
        good_nb[header.index("region_threshold")] = "0.123456789"
    elif mutate == "space_state":
        space = next(row for row in rows if row[:2] == ["space", ""])
        space[header.index("surface_area")] = "0"
    elif mutate == "fractional_identifier":
        rows[0][0] = "1.5"
    elif mutate == "extra_spectrum":
        rows[0][0] = "2"
        rows.insert(0, ["1", "1.0"])
        raw_target = target.parent / "raw_metrics.csv"
        raw_header, raw_rows = _read_csv_for_mutation(raw_target)
        good_nb = next(row for row in raw_rows if row[:2] == ["good", "NB"])
        good_nb[raw_header.index("alpha_spectrum_count")] = "2"
        _write_csv(raw_target, raw_header, raw_rows)
        _refresh_entry(
            tmp_path,
            _entry_for(manifest, raw_target.relative_to(tmp_path).as_posix()),
        )
    else:
        raise AssertionError(mutate)
    _write_csv(target, header, rows)
    _refresh_entry(tmp_path, _entry_for(manifest, relative))
    _rewrite_manifest(tmp_path, manifest)

    with pytest.raises(ProvenanceError, match=message):
        validate_bundle(tmp_path, allow_diagnostic=True)


def test_repository_inventory_classifies_every_historical_fixture() -> None:
    """Assign every historical data artifact one explicit trust class."""
    repo_root = Path(__file__).resolve().parents[1]

    report = validate_inventory(
        repo_root,
        repo_root / "tests" / "fixture_inventory.json",
    )

    assert report.file_count >= _MIN_HISTORICAL_FILES
    assert report.counts["legacy-unknown"] > 0
    assert report.counts["python-regression"] > 0
    assert report.counts["python-synthetic"] > 0


def test_inventory_rejects_an_unclassified_file(tmp_path: Path) -> None:
    """Reject an inventory that leaves any fixture unclassified."""
    (tmp_path / "fixtures").mkdir()
    (tmp_path / "fixtures" / "data.csv").write_text("x\n1\n", encoding="utf-8")
    inventory = {
        "schema_version": "pyinstancespace.fixture-inventory/v1",
        "roots": ["fixtures"],
        "ignore": [],
        "rules": [],
    }
    inventory_path = tmp_path / "inventory.json"
    inventory_path.write_text(json.dumps(inventory), encoding="utf-8")

    with pytest.raises(ProvenanceError, match="at least one"):
        validate_inventory(tmp_path, inventory_path)
