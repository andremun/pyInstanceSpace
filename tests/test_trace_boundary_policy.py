"""Shared, analytically labelled MATLAB/Python boundary-contract examples."""

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import MultiPolygon, Polygon

from instancespace.data.model import Footprint, TraceOut, pointwise_covers
from instancespace.data.options import InstanceSpaceOptions, TraceOptions
from instancespace.stages.trace import TracePredictInput, TraceStage
from instancespace.utils.alpha_shape import AlphaShape3D

CASES = json.loads(
    (Path(__file__).parent / "contracts/trace_boundary_cases.json").read_text(),
)["cases"]


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["name"])
@pytest.mark.parametrize(
    ("scale", "offset"),
    [(1.0, 0.0), (16.0, 1000.0), (0.125, -16.0)],
)
def test_shared_boundary_contract(
    case: dict[str, Any],
    scale: float,
    offset: float,
) -> None:
    if case["kind"] == "polygon":
        geometry = Polygon(
            np.array(case["shell"]) * scale + offset,
            [np.array(ring) * scale + offset for ring in case["holes"]],
        )
    else:
        shape = AlphaShape3D.from_points(np.array(case["vertices"]) * scale + offset)
        assert shape is not None
        geometry = shape.geometry(np.inf)
        assert geometry is not None
    queries = np.array(case["queries"]) * scale + offset
    tolerance = case["tolerance"] * scale
    np.testing.assert_array_equal(pointwise_covers(geometry, queries), case["exact"])
    np.testing.assert_array_equal(
        pointwise_covers(geometry, queries, tolerance),
        case["tolerant"],
    )
    # Unrelated points cannot change the tolerance or an existing answer.
    expanded = np.vstack([queries, np.full(queries.shape[1], 1e12)])
    np.testing.assert_array_equal(
        pointwise_covers(geometry, expanded, tolerance)[:-1],
        case["tolerant"],
    )
    single = [pointwise_covers(geometry, row[None], tolerance)[0] for row in queries]
    np.testing.assert_array_equal(single, case["tolerant"])


@pytest.mark.parametrize("value", [-1.0, np.inf, np.nan, True])
def test_boundary_option_validation(value: float) -> None:
    with pytest.raises(ValueError, match="boundaryTolerance"):
        TraceOptions(boundary_tolerance=value)


def test_boundary_option_json_and_fitted_persistence(tmp_path: Path) -> None:
    opts = InstanceSpaceOptions.from_dict({"trace": {"boundaryTolerance": 0.001}})
    assert opts.trace.boundary_tolerance == 0.001
    assert TraceOptions().boundary_tolerance == 0
    polygon = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    query = np.array([[0.5, -0.0005], [0.5, -0.002]])
    footprint = Footprint.from_polygon(
        polygon,
        query,
        np.array([True, False]),
        boundary_tolerance=opts.trace.boundary_tolerance,
    )
    assert footprint.elements == footprint.good_elements == 1
    fitted = TraceOut(
        footprint,
        [footprint],
        [footprint],
        footprint,
        pd.DataFrame(),
        opts.trace.boundary_tolerance,
    )
    path = tmp_path / "trace.joblib"
    joblib.dump(fitted, path)
    loaded = joblib.load(path)
    predicted = TraceStage.predict(TracePredictInput(query), loaded)
    np.testing.assert_array_equal(predicted.in_good[:, 0], [True, False])
    rescored = TraceStage.rescore(
        loaded,
        query,
        np.ones((2, 1), dtype=bool),
        np.ones(2, dtype=int),
        np.ones(2, dtype=bool),
        ["a"],
    )
    assert rescored.good[0].elements == 1
    assert rescored.boundary_tolerance == loaded.boundary_tolerance


def test_fixture_validator_uses_the_same_distance_contract() -> None:
    from tools.fixture_provenance import _trace3d_covers, _Trace3DMesh

    case = CASES[1]
    mesh = _Trace3DMesh(
        [tuple(row) for row in case["vertices"]],
        [(0, 1, 2, 3)],
        [(0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3)],
        [],
    )
    assert [_trace3d_covers(mesh, q) for q in case["queries"]] == case["exact"]
    assert [
        _trace3d_covers(mesh, q, case["tolerance"]) for q in case["queries"]
    ] == case["tolerant"]


def test_empty_disconnected_and_nonfinite_queries() -> None:
    left = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    right = Polygon([(3, 0), (4, 0), (4, 1), (3, 1)])
    queries = np.array([[0.5, 0.5], [3.5, 0.5], [2, 0.5], [np.nan, 0], [np.inf, 0]])
    np.testing.assert_array_equal(
        pointwise_covers(MultiPolygon([left, right]), queries, 0.001),
        [True, True, False, False, False],
    )
    assert not pointwise_covers(Polygon(), queries, 0.001).any()
    assert pointwise_covers(left, np.empty((0, 2))).shape == (0,)
