"""Scientific parity checks against the verified current MATLAB TRACE bundle."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import cast
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from numpy.typing import NDArray
from pandas.testing import assert_frame_equal
from shapely.geometry import MultiPolygon, Point, Polygon
from shapely.ops import unary_union

from instancespace.data.model import Footprint, TraceOut, pointwise_covers
from instancespace.data.options import (
    GeneralOptions,
    ParallelOptions,
    PythiaOptions,
    TraceOptions,
)
from instancespace.stages.trace import TraceInputs, TraceOutputs, TraceStage
from instancespace.utils.alpha_shape import (
    AlphaShape2D,
    Polygon2D,
    _alpha_region_mask,
)
from instancespace.utils.get_classifier_fcn import get_classifier_fcn

_CURRENT = Path(__file__).parent / "fixtures" / "matlab" / "current"
_TRACE3_VARIANTS = ("trace3_default", "trace3_pythia_skip")
_SCALAR_TOLERANCE = 1e-11
_GEOMETRY_TOLERANCE = 1e-10
_BOUNDARY_AMBIGUITIES = {
    "trace3_default": frozenset(
        {
            ("zoo", "in_good_RandF"),
            ("zoo", "in_best_RandF"),
        },
    ),
    "trace3_pythia_skip": frozenset(
        {
            ("wpbc_no_Nas", "in_good_LDA"),
            ("wpbc_no_Nas", "in_good_L_SVM"),
            ("wpbc_no_Nas", "in_best_L_SVM"),
        },
    ),
}
_BOUNDARY_SUMMARY_VALUES = {
    "trace3_default": {
        ("RandF", "Density_Good"): 22.111,
        ("RandF", "Density_Good_Normalized"): 2.046,
        ("RandF", "Purity_Good"): 0.651,
        ("RandF", "Density_Best"): 24.696,
        ("RandF", "Density_Best_Normalized"): 2.286,
        ("RandF", "Purity_Best"): 0.630,
    },
    "trace3_pythia_skip": {
        ("LDA", "Density_Good"): 43.362,
        ("LDA", "Density_Good_Normalized"): 4.013,
        ("LDA", "Purity_Good"): 0.647,
        ("L_SVM", "Density_Good"): 24.227,
        ("L_SVM", "Density_Good_Normalized"): 2.242,
        ("L_SVM", "Purity_Good"): 0.602,
        ("L_SVM", "Density_Best"): 25.862,
        ("L_SVM", "Density_Best_Normalized"): 2.393,
        ("L_SVM", "Purity_Best"): 0.568,
    },
}

pytestmark = pytest.mark.usefixtures("verified_current_matlab_bundle")


@dataclass(frozen=True)
class _AlphaCall:
    """One alpha-geometry evaluation made by the real TRACE3 stage."""

    shape: AlphaShape2D
    geometry: Polygon2D
    radius: float
    region_threshold: float
    inclusive: bool


@dataclass(frozen=True)
class _TraceCase:
    """Cached MATLAB inputs, expected artifacts, and real Python outputs."""

    variant: str
    pythia_skipped: bool
    build_root: Path
    explore_root: Path
    labels: tuple[str, ...]
    build_rows: tuple[str, ...]
    build_z: NDArray[np.double]
    built: TraceOut
    raw_metrics: pd.DataFrame
    alpha_calls: tuple[_AlphaCall, ...]
    explore_rows: tuple[str, ...]
    explore_z: NDArray[np.double]
    explore_y_bin: NDArray[np.bool_]
    explore_p: NDArray[np.int_]
    expected_membership: pd.DataFrame
    actual_membership: pd.DataFrame
    rescored: TraceOut


def _indexed_frame(path: Path) -> pd.DataFrame:
    """Read one row-labelled matrix from the current bundle."""
    return pd.read_csv(path, index_col=0, float_precision="round_trip")


def _double_matrix(path: Path) -> NDArray[np.double]:
    """Read a row-labelled floating-point matrix."""
    return np.asarray(_indexed_frame(path).to_numpy(dtype=np.double), dtype=np.double)


def _bool_matrix(path: Path) -> NDArray[np.bool_]:
    """Read a row-labelled Boolean matrix."""
    return np.asarray(_indexed_frame(path).to_numpy(dtype=np.bool_), dtype=np.bool_)


def _int_vector(path: Path) -> NDArray[np.int_]:
    """Read a row-labelled integer column as a vector."""
    return np.asarray(
        _indexed_frame(path).to_numpy(dtype=np.int_).reshape(-1),
        dtype=np.int_,
    )


def _labels(path: Path) -> tuple[str, ...]:
    """Read the ordered algorithm labels exported by MATLAB."""
    values = pd.read_csv(path, float_precision="round_trip")["algorithm_name"].tolist()
    return tuple(str(value) for value in values)


def _run_with_alpha_diagnostics(
    inputs: TraceInputs,
) -> tuple[TraceOutputs, tuple[_AlphaCall, ...]]:
    """Run TRACE3 while observing, but not changing, alpha-shape calls."""
    calls: list[_AlphaCall] = []
    implementation = AlphaShape2D.geometry

    def record_geometry(
        shape: AlphaShape2D,
        radius: float,
        *,
        region_threshold: float = 0.0,
        inclusive: bool = True,
    ) -> Polygon2D | None:
        geometry = implementation(
            shape,
            radius,
            region_threshold=region_threshold,
            inclusive=inclusive,
        )
        if geometry is not None:
            calls.append(
                _AlphaCall(
                    shape,
                    geometry,
                    float(radius),
                    float(region_threshold),
                    inclusive,
                ),
            )
        return geometry

    with patch.object(AlphaShape2D, "geometry", record_geometry):
        outputs = TraceStage._run(inputs)
    return outputs, tuple(calls)


def _membership(
    footprint: Footprint,
    z: NDArray[np.double],
) -> NDArray[np.bool_]:
    """Evaluate one trained footprint, including an empty-footprint fallback."""
    if footprint.polygon is None:
        return np.zeros(z.shape[0], dtype=np.bool_)
    return pointwise_covers(footprint.polygon, z)


@cache
def _case(variant: str) -> _TraceCase:
    """Run one current-gold TRACE3 variant once for the whole test module."""
    pythia_skipped = variant == "trace3_pythia_skip"
    build_root = _CURRENT / "build_data" / "trace" / variant
    explore_root = _CURRENT / "explore_data" / "trace" / variant
    build_inputs = build_root / "inputs"
    explore_inputs = explore_root / "inputs"

    labels = _labels(build_inputs / "algorithm_labels.csv")
    build_z_frame = _indexed_frame(build_inputs / "z.csv")
    build_z = np.asarray(build_z_frame.to_numpy(dtype=np.double), dtype=np.double)
    build_y_bin = _bool_matrix(build_inputs / "y_bin.csv")
    build_y_hat = _bool_matrix(build_inputs / "y_hat.csv")
    build_p = _int_vector(build_inputs / "p.csv")
    build_beta = _bool_matrix(build_inputs / "beta.csv").reshape(-1)

    trace_inputs = TraceInputs(
        z=build_z,
        selection0=np.zeros(build_z.shape[0], dtype=np.int_),
        p=build_p,
        beta=build_beta,
        algo_labels=list(labels),
        y_hat=build_y_hat,
        y_bin=build_y_bin,
        trace_options=TraceOptions.default(
            method="trace3",
            purity=0.60,
            contra=False,
            min_instances=4,
            min_area_frac=0.01,
        ),
        parallel_options=ParallelOptions.default(flag=False, n_cores=1),
        general_options=GeneralOptions.default(verbose=False, seed=42),
        pythia_options=PythiaOptions.default(
            classifier="knn",
            tuning="sobol",
            n_tuning_iter=20,
            skip=pythia_skipped,
        ),
    )
    trace_outputs, alpha_calls = _run_with_alpha_diagnostics(trace_inputs)
    built = TraceOut(
        trace_outputs.space,
        trace_outputs.good,
        trace_outputs.best,
        trace_outputs.hard,
        trace_outputs.trace_summary,
    )

    explore_z_frame = _indexed_frame(explore_inputs / "z.csv")
    explore_z = np.asarray(
        explore_z_frame.to_numpy(dtype=np.double),
        dtype=np.double,
    )
    explore_y_bin = _bool_matrix(explore_inputs / "y_bin.csv")
    explore_p = _int_vector(explore_inputs / "p.csv")
    explore_beta = _bool_matrix(explore_inputs / "beta.csv").reshape(-1)
    expected_membership = _indexed_frame(
        explore_root / "outputs" / "membership.csv",
    ).astype(bool)
    actual_values = np.column_stack(
        [
            *(_membership(footprint, explore_z) for footprint in built.good),
            *(_membership(footprint, explore_z) for footprint in built.best),
        ],
    )
    actual_membership = pd.DataFrame(
        actual_values,
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

    return _TraceCase(
        variant=variant,
        pythia_skipped=pythia_skipped,
        build_root=build_root,
        explore_root=explore_root,
        labels=labels,
        build_rows=tuple(str(value) for value in build_z_frame.index),
        build_z=build_z,
        built=built,
        raw_metrics=pd.read_csv(
            build_root / "outputs" / "raw_metrics.csv",
            float_precision="round_trip",
        ),
        alpha_calls=alpha_calls,
        explore_rows=tuple(str(value) for value in explore_z_frame.index),
        explore_z=explore_z,
        explore_y_bin=explore_y_bin,
        explore_p=explore_p,
        expected_membership=expected_membership,
        actual_membership=actual_membership,
        rescored=rescored,
    )


def _footprint(
    case: _TraceCase,
    kind: str,
    algorithm: str | None,
) -> Footprint:
    """Select the Python footprint named by a MATLAB raw-metric row."""
    if kind == "space":
        return case.built.space
    if kind == "hard":
        return case.built.hard
    if algorithm is None:
        raise AssertionError(f"{kind} footprint is missing its algorithm label")
    index = case.labels.index(algorithm)
    if kind == "good":
        return case.built.good[index]
    if kind == "best":
        return case.built.best[index]
    raise AssertionError(f"Unknown footprint kind: {kind}")


def _geometry_path(case: _TraceCase, kind: str, algorithm: str | None) -> Path:
    """Return the current-bundle geometry path for one footprint."""
    filename = "hard.csv" if kind == "hard" else f"{kind}_{algorithm}.csv"
    return case.build_root / "outputs" / filename


@cache
def _exported_geometry(path: Path) -> Polygon2D | None:
    """Reconstruct all exported parts and holes without false connecting edges."""
    frame = pd.read_csv(path, float_precision="round_trip")
    if frame.empty:
        return None

    polygons: list[Polygon] = []
    for _, part in frame.groupby("part", sort=False):
        exterior_rows = part.loc[~part["is_hole"].astype(bool)].sort_values("vertex")
        exterior = exterior_rows[["z_1", "z_2"]].to_numpy(dtype=np.double)
        holes = [
            hole.sort_values("vertex")[["z_1", "z_2"]].to_numpy(dtype=np.double)
            for _, hole in part.loc[part["is_hole"].astype(bool)].groupby(
                "ring",
                sort=False,
            )
        ]
        polygons.append(Polygon(exterior, holes))

    geometry = unary_union(polygons)
    if not isinstance(geometry, Polygon | MultiPolygon):
        raise AssertionError(f"Non-polygonal fixture geometry in {path}")
    return cast(Polygon2D, geometry)


def _polygon_parts(polygon: Polygon2D) -> list[Polygon]:
    """Return the disconnected Shapely polygon parts."""
    if isinstance(polygon, MultiPolygon):
        return list(polygon.geoms)
    return [polygon]


def _alpha_call(case: _TraceCase, footprint: Footprint) -> _AlphaCall:
    """Find the final alpha call whose exact geometry the footprint retains."""
    if footprint.polygon is None:
        raise AssertionError("An empty footprint has no final alpha diagnostic")
    matches = [call for call in case.alpha_calls if call.geometry is footprint.polygon]
    if len(matches) != 1:
        raise AssertionError(f"Expected one final alpha call, found {len(matches)}")
    return matches[0]


def _alpha_component_count(call: _AlphaCall) -> int:
    """Count MATLAB-style regions, where sharing one vertex connects simplices."""
    selected = (
        call.shape.circumradii <= call.radius
        if call.inclusive
        else call.shape.circumradii < call.radius
    )
    simplices = call.shape.simplices[selected]
    if call.region_threshold > 0:
        areas = np.asarray(
            [Polygon(call.shape.points[simplex]).area for simplex in simplices],
            dtype=np.double,
        )
        simplices = simplices[
            _alpha_region_mask(simplices, areas, call.region_threshold)
        ]
    if simplices.shape[0] == 0:
        return 0

    parents = list(range(simplices.shape[0]))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    owners: dict[int, int] = {}
    for simplex_index, simplex in enumerate(simplices):
        for vertex_value in simplex:
            vertex = int(vertex_value)
            owner = owners.setdefault(vertex, simplex_index)
            left = find(simplex_index)
            right = find(owner)
            if left != right:
                parents[right] = left
    return len({find(index) for index in range(simplices.shape[0])})


def _matlab_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Align MATLAB's row-label header without hiding scientific schema drift."""
    return frame.rename(columns={"Row": "Algorithm"})


@pytest.mark.parametrize("variant", _TRACE3_VARIANTS)
def test_current_matlab_trace3_raw_metrics_and_alpha_diagnostics(
    variant: str,
) -> None:
    """Match every representable MATLAB footprint metric and alpha diagnostic."""
    case = _case(variant)
    for row in case.raw_metrics.itertuples(index=False):
        kind = str(row.kind)
        algorithm = None if pd.isna(row.algorithm) else str(row.algorithm)
        footprint = _footprint(case, kind, algorithm)
        expected_empty = bool(row.empty)
        elements = int(cast(int | float, row.elements))
        good_elements = float(cast(int | float, row.good_elements))
        measure = float(cast(int | float, row.measure))
        density = float(cast(int | float, row.density))
        purity = float(cast(int | float, row.purity))
        component_count = int(cast(int | float, row.component_count))
        part_count = int(cast(int | float, row.geometry_part_count))
        hole_count = int(cast(int | float, row.hole_count))
        alpha_radius = float(cast(int | float, row.alpha_radius))
        region_threshold = float(cast(int | float, row.region_threshold))

        assert (footprint.polygon is None) is expected_empty
        assert footprint.elements == elements
        if np.isnan(good_elements):
            assert kind == "space"  # MATLAB does not populate this field.
        else:
            assert footprint.good_elements == int(good_elements)
        np.testing.assert_allclose(
            [footprint.area, footprint.density, footprint.purity],
            [measure, density, purity],
            rtol=0,
            atol=_SCALAR_TOLERANCE,
        )

        if expected_empty:
            assert component_count == 0
            assert part_count == 0
            assert hole_count == 0
            assert np.isnan(alpha_radius)
            assert np.isnan(region_threshold)
            continue

        polygon = cast(Polygon2D, footprint.polygon)
        parts = _polygon_parts(polygon)
        assert len(parts) == part_count
        assert sum(len(part.interiors) for part in parts) == hole_count

        alpha_call = _alpha_call(case, footprint)
        assert alpha_call.radius == pytest.approx(
            alpha_radius,
            rel=0,
            abs=_SCALAR_TOLERANCE,
        )
        assert alpha_call.region_threshold == pytest.approx(
            region_threshold,
            rel=0,
            abs=_SCALAR_TOLERANCE,
        )
        assert _alpha_component_count(alpha_call) == component_count


@pytest.mark.parametrize("variant", _TRACE3_VARIANTS)
def test_current_matlab_trace3_geometry_and_build_summary(variant: str) -> None:
    """Match current-gold TRACE3 topology, geometry, and rounded summary."""
    case = _case(variant)
    footprint_keys = [
        *((kind, label) for label in case.labels for kind in ("good", "best")),
        ("hard", None),
    ]
    for kind, algorithm in footprint_keys:
        actual = _footprint(case, kind, algorithm).polygon
        expected = _exported_geometry(_geometry_path(case, kind, algorithm))
        assert (actual is None) is (expected is None)
        if actual is None or expected is None:
            continue
        assert isinstance(actual, Polygon | MultiPolygon)
        assert actual.is_valid
        assert expected.is_valid
        assert actual.hausdorff_distance(expected) <= _GEOMETRY_TOLERANCE
        assert actual.symmetric_difference(expected).area <= _GEOMETRY_TOLERANCE

    expected_summary = _matlab_summary(
        pd.read_csv(
            case.build_root / "outputs" / "summary.csv",
            float_precision="round_trip",
        ),
    )
    actual_summary = _matlab_summary(case.built.summary)
    assert_frame_equal(
        actual_summary,
        expected_summary,
        check_dtype=False,
        check_exact=True,
    )


@pytest.mark.parametrize("variant", _TRACE3_VARIANTS)
def test_current_matlab_trace3_explore_membership(variant: str) -> None:
    """Require exact off-boundary membership and pin serialized ambiguities."""
    case = _case(variant)
    expected = case.expected_membership.to_numpy(dtype=np.bool_)
    actual = case.actual_membership.to_numpy(dtype=np.bool_)
    positions = np.argwhere(actual != expected)
    differences = {
        (case.explore_rows[int(row)], str(case.expected_membership.columns[int(col)]))
        for row, col in positions
    }
    expected_differences = _BOUNDARY_AMBIGUITIES[variant]
    assert differences == expected_differences

    for row, column in expected_differences:
        row_index = case.explore_rows.index(row)
        column_index = case.expected_membership.columns.get_loc(column)
        assert not expected[row_index, column_index]
        assert actual[row_index, column_index]

        kind_prefix, kind = (
            ("in_good_", "good")
            if column.startswith("in_good_")
            else ("in_best_", "best")
        )
        algorithm = column.removeprefix(kind_prefix)
        exported = _exported_geometry(_geometry_path(case, kind, algorithm))
        python_polygon = _footprint(case, kind, algorithm).polygon
        assert exported is not None
        assert isinstance(python_polygon, Polygon | MultiPolygon)
        point = Point(case.explore_z[row_index])

        # The round-trip CSV puts this repeated build/explore point exactly on
        # the exported vertex. MATLAB's original side-of-boundary result differs
        # from the boundary-inclusive serialized-geometry interpretation.
        assert exported.boundary.distance(point) == 0.0
        assert python_polygon.boundary.distance(point) == 0.0
        assert exported.touches(point)
        assert exported.covers(point)
        assert not exported.contains(point)
        build_row_index = case.build_rows.index(row)
        np.testing.assert_array_equal(
            case.build_z[build_row_index],
            case.explore_z[row_index],
        )


@pytest.mark.parametrize("variant", _TRACE3_VARIANTS)
def test_current_matlab_trace3_explore_rescore(variant: str) -> None:
    """Rescore trained geometry and account exactly for boundary evidence."""
    case = _case(variant)
    expected_membership = case.expected_membership
    zero_based_portfolio = case.explore_p - 1

    for index, label in enumerate(case.labels):
        expected_good_inside = expected_membership[f"in_good_{label}"].to_numpy(
            dtype=np.bool_,
        )
        expected_best_inside = expected_membership[f"in_best_{label}"].to_numpy(
            dtype=np.bool_,
        )
        good_boundary_rows = [
            case.explore_rows.index(row)
            for row, column in _BOUNDARY_AMBIGUITIES[variant]
            if column == f"in_good_{label}"
        ]
        best_boundary_rows = [
            case.explore_rows.index(row)
            for row, column in _BOUNDARY_AMBIGUITIES[variant]
            if column == f"in_best_{label}"
        ]

        good = case.rescored.good[index]
        best = case.rescored.best[index]
        assert good.polygon is case.built.good[index].polygon
        assert best.polygon is case.built.best[index].polygon
        assert good.elements == int(expected_good_inside.sum()) + len(
            good_boundary_rows,
        )
        assert good.good_elements == int(
            np.logical_and(
                expected_good_inside,
                case.explore_y_bin[:, index],
            ).sum(),
        ) + int(case.explore_y_bin[good_boundary_rows, index].sum())
        assert best.elements == int(expected_best_inside.sum()) + len(
            best_boundary_rows,
        )
        assert best.good_elements == int(
            np.logical_and(
                expected_best_inside,
                zero_based_portfolio == index,
            ).sum(),
        ) + int((zero_based_portfolio[best_boundary_rows] == index).sum())

    expected_summary = _matlab_summary(
        pd.read_csv(
            case.explore_root / "outputs" / "eval_summary.csv",
            float_precision="round_trip",
        ),
    ).set_index("Algorithm")
    actual_summary = _matlab_summary(case.rescored.summary).set_index("Algorithm")
    cells = np.argwhere(
        actual_summary.to_numpy(dtype=np.double)
        != expected_summary.to_numpy(dtype=np.double),
    )
    differences = {
        (str(actual_summary.index[int(row)]), str(actual_summary.columns[int(col)]))
        for row, col in cells
    }
    expected_values = _BOUNDARY_SUMMARY_VALUES[variant]
    expected_differences = set(expected_values)
    assert differences == expected_differences
    for (algorithm, column), expected_value in expected_values.items():
        assert actual_summary.loc[algorithm, column] == expected_value


def test_current_matlab_legacy_svm_hyperparameter_units() -> None:
    """Preserve BoxConstraint/KernelScale units at sklearn's SVM boundary."""
    root = _CURRENT / "build_data" / "pythia" / "legacy_svm"
    labels = _labels(root / "inputs" / "algorithm_labels.csv")
    parameters = pd.read_csv(
        root / "outputs" / "hyperparameters.csv",
        float_precision="round_trip",
    )
    assert tuple(str(value) for value in parameters["algo"]) == labels

    specification = get_classifier_fcn("svm")
    kernel_scale = specification.param2
    assert specification.param1.label == "BoxConstraint"
    assert kernel_scale is not None
    assert kernel_scale.label == "KernelScale"
    assert specification.param1.low == kernel_scale.low == 2**-10
    assert specification.param1.high == kernel_scale.high == 2**4

    for row in parameters.itertuples(index=False):
        box_constraint = float(cast(int | float, row.param1))
        matlab_kernel_scale = float(cast(int | float, row.param2))
        assert specification.param1.low <= box_constraint <= specification.param1.high
        assert kernel_scale.low <= matlab_kernel_scale <= kernel_scale.high

        estimator_c = specification.param1.from_precalc(box_constraint)
        estimator_gamma = kernel_scale.from_precalc(matlab_kernel_scale)
        assert float(estimator_c) == box_constraint
        assert float(estimator_gamma) == pytest.approx(
            1.0 / matlab_kernel_scale**2,
            rel=1e-15,
        )
        assert specification.param1.reported(estimator_c) == box_constraint
        assert kernel_scale.reported(estimator_gamma) == pytest.approx(
            matlab_kernel_scale,
            rel=1e-15,
        )
