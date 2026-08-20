"""Contract tests for the two-dimensional MATLAB TRACE3 port."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import pytest
from numpy.typing import NDArray
from shapely.geometry import Polygon

from instancespace.data.model import Footprint, TraceOut
from instancespace.data.options import (
    GeneralOptions,
    ParallelOptions,
    PythiaOptions,
    TraceOptions,
)
from instancespace.stages.trace import TraceInputs, TraceStage, _matlab_round
from instancespace.utils.alpha_shape import (
    AlphaShape2D,
    AlphaShape3D,
    TetrahedralMesh,
)

COLLAPSE_GEOMETRY_CALL = 3
EXPECTED_ALPHA_CALLS = 201
EXPECTED_TRACE_BUILD_CALLS = 5
SECOND_ALGORITHM_ONE_BASED = 2
GRID_SPLIT = 3.0
TRAINED_AREA = 4.0
EXPECTED_INSIDE_ELEMENTS = 2
THREE_DIMENSIONS = 3
EXPECTED_3D_INSIDE_ELEMENTS = 2
MATLAB_ROUNDED_TIE = 0.313


def _cube() -> NDArray[np.double]:
    return np.array(
        [[x, y, z] for x in (0.0, 1.0) for y in (0.0, 1.0) for z in (0.0, 1.0)],
        dtype=np.double,
    )


def _stage(
    z: NDArray[np.double],
    y_bin: NDArray[np.bool_],
    *,
    y_hat: NDArray[np.bool_] | None = None,
    p: NDArray[np.int_] | None = None,
    beta: NDArray[np.bool_] | None = None,
    purity: float = 0.6,
    min_instances: int = 4,
    min_area_frac: float = 0.01,
    parallel: bool = False,
    executor: ThreadPoolExecutor | None = None,
) -> TraceStage:
    n_instances, n_algorithms = y_bin.shape
    return TraceStage(
        z=z,
        y_bin=y_bin,
        p=(np.zeros(n_instances, dtype=np.int_) if p is None else p),
        beta=(np.zeros(n_instances, dtype=np.bool_) if beta is None else beta),
        algo_labels=[f"algo_{i}" for i in range(n_algorithms)],
        trace_opts=TraceOptions.default(
            method="trace3",
            purity=purity,
            min_instances=min_instances,
            min_area_frac=min_area_frac,
        ),
        parallel_opts=ParallelOptions.default(flag=parallel, n_cores=2),
        general_opts=GeneralOptions.default(),
        executor=executor,
        y_hat=y_hat,
    )


def _grid() -> NDArray[np.double]:
    return np.array(
        [[float(x), float(y)] for y in range(3) for x in range(3)],
        dtype=np.double,
    )


def test_trace_summary_rounds_decimal_ties_away_from_zero() -> None:
    """R2026a rounds exact three-decimal ties away from zero, not to even."""
    ties = np.array([0.3125, -0.3125, 0.3124999, -0.3124999], dtype=np.double)
    np.testing.assert_array_equal(
        _matlab_round(ties, 3),
        [MATLAB_ROUNDED_TIE, -MATLAB_ROUNDED_TIE, 0.312, -0.312],
    )
    tied_footprint = Footprint(None, 1.0, 16, 5, 1.0, 0.3125)
    space = Footprint(None, 4.0, 16, 16, 4.0, 1.0)

    summary = TraceStage._summary_table(  # noqa: SLF001
        [tied_footprint],
        [tied_footprint],
        ["tie"],
        space,
    )

    assert summary.loc[0, "Purity_Good"] == MATLAB_ROUNDED_TIE
    assert summary.loc[0, "Purity_Best"] == MATLAB_ROUNDED_TIE


def test_min_instances_is_an_exclusive_support_boundary() -> None:
    """The MATLAB default four requires five unique supporting points."""
    z = _grid()
    y_bin = np.zeros((z.shape[0], 1), dtype=np.bool_)
    y_bin[:4, 0] = True
    trace = _stage(z, y_bin, purity=0.0, min_area_frac=0.0)

    rejected = trace._build_trace3(y_bin[:, 0], None, 4.0)  # noqa: SLF001
    y_bin[4, 0] = True
    accepted = trace._build_trace3(y_bin[:, 0], None, 4.0)  # noqa: SLF001

    assert rejected.polygon is None
    assert accepted.polygon is not None


def test_prediction_filter_requires_truth_and_prediction_consensus() -> None:
    """Available PYTHIA predictions intersect rather than replace truth."""
    z = _grid()
    y_bin = np.zeros((z.shape[0], 1), dtype=np.bool_)
    y_bin[:5, 0] = True
    y_hat = y_bin.copy()
    y_hat[4, 0] = False
    trace = _stage(z, y_bin, y_hat=y_hat, purity=0.0, min_area_frac=0.0)

    filtered = trace._build_trace3(y_bin[:, 0], y_hat[:, 0], 4.0)  # noqa: SLF001
    fallback = trace._build_trace3(y_bin[:, 0], None, 4.0)  # noqa: SLF001

    assert filtered.polygon is None
    assert fallback.polygon is not None


def test_min_instances_counts_unique_supporting_points() -> None:
    """Duplicate supporting coordinates do not satisfy the footprint minimum."""
    z = np.array(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 1.0]],
        dtype=np.double,
    )
    y_bin = np.ones((z.shape[0], 1), dtype=np.bool_)
    trace = _stage(z, y_bin, purity=0.0, min_area_frac=0.0)

    footprint = trace._build_trace3(y_bin[:, 0], None, 1.0)  # noqa: SLF001

    assert footprint.polygon is None


class _RecordingAlphaShape:
    critical_radius = 10.0
    spectrum = np.array([10.0, 1.0], dtype=np.double)

    def __init__(self, *, collapse: bool = False) -> None:
        self.calls: list[tuple[float, float]] = []
        self.collapse = collapse

    def geometry(
        self,
        radius: float,
        *,
        region_threshold: float = 0.0,
        inclusive: bool = True,
    ) -> Polygon | None:
        del inclusive
        self.calls.append((radius, region_threshold))
        if self.collapse and len(self.calls) == COLLAPSE_GEOMETRY_CALL:
            return None
        return Polygon(
            [
                (-radius, -radius),
                (radius, -radius),
                (radius, radius),
                (-radius, radius),
            ],
        )


class _StaticAlphaShape:
    critical_radius = 1.0
    spectrum = np.array([1.0], dtype=np.double)

    def __init__(self) -> None:
        self.calls = 0

    def geometry(
        self,
        radius: float,
        *,
        region_threshold: float = 0.0,
        inclusive: bool = True,
    ) -> Polygon:
        del radius, region_threshold, inclusive
        self.calls += 1
        return Polygon([(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)])


class _RecordingAlphaShape3D:
    critical_radius = 10.0
    spectrum = np.array([10.0, 1.0], dtype=np.double)

    def __init__(self) -> None:
        self.calls: list[tuple[float, float]] = []

    def geometry(
        self,
        radius: float,
        *,
        region_threshold: float = 0.0,
        inclusive: bool = True,
    ) -> TetrahedralMesh:
        del inclusive
        self.calls.append((radius, region_threshold))
        return TetrahedralMesh(
            vertices=np.array(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.double,
            ),
            tetrahedra=np.array([[0, 1, 2, 3]], dtype=np.int_),
            boundary_faces=np.array(
                [[1, 2, 3], [0, 3, 2], [0, 1, 3], [0, 2, 1]],
                dtype=np.int_,
            ),
            alpha=radius,
            region_threshold=region_threshold,
            region_count=1,
            volume=radius,
            surface_area=1.0,
        )


def test_single_alpha_spectrum_returns_initial_shape_below_purity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A one-value spectrum returns the initial footprint without tightening."""
    z = np.array([[i / 20.0, i / 30.0] for i in range(10)], dtype=np.double)
    y_bin = np.zeros((10, 1), dtype=np.bool_)
    y_bin[:5, 0] = True
    trace = _stage(z, y_bin, purity=1.0, min_area_frac=0.0)
    fake = _StaticAlphaShape()
    monkeypatch.setattr(
        AlphaShape2D,
        "from_points",
        classmethod(lambda _cls, _points: fake),
    )

    footprint = trace._build_trace3(y_bin[:, 0], None, 1.0)  # noqa: SLF001

    assert fake.calls == 1
    assert footprint.polygon is not None
    assert footprint.purity < trace.opts.purity


def test_minimum_area_fraction_rejects_only_strictly_smaller_shapes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A footprint exactly on the configured area boundary remains valid."""
    z = np.array([[i / 20.0, i / 30.0] for i in range(5)], dtype=np.double)
    y_bin = np.ones((5, 1), dtype=np.bool_)
    trace = _stage(z, y_bin, purity=0.0, min_area_frac=0.01)
    monkeypatch.setattr(
        AlphaShape2D,
        "from_points",
        classmethod(lambda _cls, _points: _StaticAlphaShape()),
    )

    boundary = trace._build_trace3(y_bin[:, 0], None, 400.0)  # noqa: SLF001
    undersized = trace._build_trace3(y_bin[:, 0], None, 401.0)  # noqa: SLF001

    assert boundary.polygon is not None
    assert undersized.polygon is None


def test_trace3_evaluates_exactly_100_radii_with_stateful_threshold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The tightening loop evaluates 100 radii and carries threshold state."""
    z = np.array([[i / 20.0, i / 30.0] for i in range(10)], dtype=np.double)
    y_bin = np.zeros((10, 1), dtype=np.bool_)
    y_bin[:5, 0] = True
    trace = _stage(z, y_bin, purity=1.0, min_area_frac=0.0)
    fake = _RecordingAlphaShape()
    monkeypatch.setattr(
        AlphaShape2D,
        "from_points",
        classmethod(lambda _cls, _points: fake),
    )

    footprint = trace._build_trace3(y_bin[:, 0], None, 1.0)  # noqa: SLF001

    assert len(fake.calls) == EXPECTED_ALPHA_CALLS
    expected_radii = np.linspace(10.0, 1.0, 101)[1:]
    np.testing.assert_allclose([call[0] for call in fake.calls[1::2]], expected_radii)
    assert fake.calls[1][1] == 0.0
    assert fake.calls[2][1] == pytest.approx(
        Polygon([(-9.91, -9.91), (9.91, -9.91), (9.91, 9.91), (-9.91, 9.91)]).area
        / 20.0,
    )
    assert fake.calls[3][1] == fake.calls[2][1]
    assert footprint.area == pytest.approx(4.0)


def test_trace3_stateful_threshold_uses_volume_for_three_dimensions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The same 100-step loop carries prior volume into 3D region filtering."""
    values = np.linspace(0.02, 0.2, 10, dtype=np.double)
    z = np.column_stack((values, values / 2.0, values / 3.0))
    y_bin = np.zeros((z.shape[0], 1), dtype=np.bool_)
    y_bin[:5, 0] = True
    trace = _stage(z, y_bin, purity=1.0, min_area_frac=0.0)
    fake = _RecordingAlphaShape3D()
    monkeypatch.setattr(
        AlphaShape3D,
        "from_points",
        classmethod(lambda _cls, _points: fake),
    )

    footprint = trace._build_trace3(y_bin[:, 0], None, 1.0)  # noqa: SLF001

    assert len(fake.calls) == EXPECTED_ALPHA_CALLS
    expected_radii = np.linspace(10.0, 1.0, 101)[1:]
    np.testing.assert_allclose([call[0] for call in fake.calls[1::2]], expected_radii)
    assert fake.calls[1][1] == 0.0
    assert fake.calls[2][1] == pytest.approx(expected_radii[0] / 20.0)
    assert fake.calls[3][1] == fake.calls[2][1]
    assert footprint.area == pytest.approx(1.0)
    assert footprint.dimension == THREE_DIMENSIONS


def test_later_shape_collapse_returns_empty_not_previous_geometry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An invalid later radius discards the earlier valid footprint."""
    z = np.array([[i / 20.0, i / 30.0] for i in range(10)], dtype=np.double)
    y_bin = np.zeros((10, 1), dtype=np.bool_)
    y_bin[:5, 0] = True
    trace = _stage(z, y_bin, purity=1.0, min_area_frac=0.0)
    fake = _RecordingAlphaShape(collapse=True)
    monkeypatch.setattr(
        AlphaShape2D,
        "from_points",
        classmethod(lambda _cls, _points: fake),
    )

    footprint = trace._build_trace3(y_bin[:, 0], None, 1.0)  # noqa: SLF001

    assert footprint.polygon is None
    assert footprint.area == 0


@pytest.mark.parametrize("pythia_skipped", [False, True])
def test_trace3_orchestration_uses_truth_portfolio_and_optional_predictions(
    monkeypatch: pytest.MonkeyPatch,
    *,
    pythia_skipped: bool,
) -> None:
    """TRACE3 uses true labels and P while PYTHIA remains only a filter."""
    z = np.array(
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
        dtype=np.double,
    )
    y_bin = np.array(
        [[True, False], [True, True], [False, True], [False, False]],
        dtype=np.bool_,
    )
    y_hat = np.array(
        [[True, True], [False, True], [True, False], [False, False]],
        dtype=np.bool_,
    )
    observed: list[tuple[NDArray[np.bool_], NDArray[np.bool_] | None]] = []

    def capture_build(
        self: TraceStage,
        labels: NDArray[np.bool_],
        predictions: NDArray[np.bool_] | None,
        space_area: float,
    ) -> Footprint:
        del self, space_area
        observed.append(
            (labels.copy(), None if predictions is None else predictions.copy()),
        )
        return Footprint(None, 0, 0, 0, 0, 0)

    monkeypatch.setattr(TraceStage, "_build_trace3", capture_build)
    monkeypatch.setattr(
        TraceStage,
        "_remove_contradictions",
        lambda *_args: pytest.fail("TRACE3 ran legacy contradiction removal"),
    )
    inputs = TraceInputs(
        z=z,
        selection0=np.array([1, 1, 0, 0], dtype=np.int_),
        p=np.array([1, 2, 2, 1], dtype=np.int_),
        beta=np.array([True, True, False, False], dtype=np.bool_),
        algo_labels=["a", "b"],
        y_hat=y_hat,
        y_bin=y_bin,
        trace_options=TraceOptions.default(method="trace3", use_sim=True),
        parallel_options=ParallelOptions.default(flag=False),
        general_options=GeneralOptions.default(),
        pythia_options=PythiaOptions.default(skip=pythia_skipped),
    )

    TraceStage._run(inputs)  # noqa: SLF001

    assert len(observed) == EXPECTED_TRACE_BUILD_CALLS
    np.testing.assert_array_equal(observed[0][0], y_bin[:, 0])
    np.testing.assert_array_equal(observed[1][0], inputs.p == 1)
    np.testing.assert_array_equal(observed[2][0], y_bin[:, 1])
    np.testing.assert_array_equal(
        observed[3][0],
        inputs.p == SECOND_ALGORITHM_ONE_BASED,
    )
    np.testing.assert_array_equal(observed[4][0], ~inputs.beta)
    assert observed[4][1] is None
    if pythia_skipped:
        assert all(prediction is None for _, prediction in observed)
    else:
        for observed_index, algorithm_index in ((0, 0), (1, 0), (2, 1), (3, 1)):
            prediction = observed[observed_index][1]
            assert prediction is not None
            np.testing.assert_array_equal(prediction, y_hat[:, algorithm_index])


def test_parallel_and_sequential_trace3_outputs_match() -> None:
    """Parallel scheduling cannot alter TRACE3 geometry or metrics."""
    z = np.array(
        [[float(x), float(y)] for y in range(5) for x in range(5)],
        dtype=np.double,
    )
    y_bin = np.column_stack(
        (
            z[:, 0] <= GRID_SPLIT,
            z[:, 1] >= 1.0,
        ),
    ).astype(np.bool_)
    y_hat = np.ones_like(y_bin)
    p = np.where(z[:, 0] <= z[:, 1], 0, 1).astype(np.int_)
    beta = np.logical_or(y_bin[:, 0], y_bin[:, 1])

    sequential = _stage(z, y_bin, y_hat=y_hat, p=p, beta=beta)._trace()  # noqa: SLF001
    with ThreadPoolExecutor(max_workers=2) as executor:
        parallel_stage = _stage(
            z,
            y_bin,
            y_hat=y_hat,
            p=p,
            beta=beta,
            parallel=True,
            executor=executor,
        )
        parallel = parallel_stage._trace()  # noqa: SLF001

    pd.testing.assert_frame_equal(sequential.trace_summary, parallel.trace_summary)
    for sequential_fp, parallel_fp in zip(
        [*sequential.good, *sequential.best, sequential.hard],
        [*parallel.good, *parallel.best, parallel.hard],
        strict=True,
    ):
        assert sequential_fp == parallel_fp


def test_trace3_builds_native_three_dimensional_footprints() -> None:
    """A cube flows through build, metrics, and dimension-aware summary labels."""
    z = _cube()
    y_bin = np.ones((z.shape[0], 1), dtype=np.bool_)
    trace = _stage(z, y_bin, purity=0.0, min_area_frac=0.0)

    output = trace._trace()  # noqa: SLF001

    assert output.space.dimension == THREE_DIMENSIONS
    assert output.space.area == pytest.approx(1.0)
    assert isinstance(output.good[0].polygon, TetrahedralMesh)
    assert output.good[0].measure == pytest.approx(1.0)
    assert output.good[0].elements == z.shape[0]
    assert output.good[0].purity == 1.0
    assert "Volume_Good" in output.trace_summary
    assert "Area_Good" not in output.trace_summary


def test_trace3_degenerate_three_dimensional_support_is_canonical_empty() -> None:
    """Coplanar/duplicate support produces dimension-aware empty footprints."""
    z = np.ones((5, 3), dtype=np.double)
    y_bin = np.ones((5, 1), dtype=np.bool_)
    trace = _stage(z, y_bin)

    output = trace._trace()  # noqa: SLF001

    assert output.space.area == 0.0
    assert output.good[0] == Footprint(
        None,
        0,
        0,
        0,
        0,
        0,
        THREE_DIMENSIONS,
    )
    assert output.best[0] == Footprint(
        None,
        0,
        0,
        0,
        0,
        0,
        THREE_DIMENSIONS,
    )
    assert output.hard == Footprint(None, 0, 0, 0, 0, 0, THREE_DIMENSIONS)


def test_legacy_method_warns_and_dispatches_3d_to_trace3(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """MATLAB never sends a 3D projection through polygonal legacy TRACE."""
    z = _cube()
    y_bin = np.ones((z.shape[0], 1), dtype=np.bool_)
    warnings: list[str] = []
    monkeypatch.setattr(
        "instancespace.stages.trace.logger.warning",
        warnings.append,
    )
    inputs = TraceInputs(
        z=z,
        selection0=np.full(z.shape[0], -1, dtype=np.int_),
        p=np.ones(z.shape[0], dtype=np.int_),
        beta=np.zeros(z.shape[0], dtype=np.bool_),
        algo_labels=["algo"],
        y_hat=np.ones_like(y_bin),
        y_bin=y_bin,
        trace_options=TraceOptions.default(
            method="legacy",
            purity=0.0,
            min_instances=4,
            min_area_frac=0.0,
        ),
        parallel_options=ParallelOptions.default(flag=False),
        general_options=GeneralOptions.default(),
    )

    output = TraceStage._run(inputs)  # noqa: SLF001

    assert len(warnings) == 1
    assert "dispatching" in warnings[0]
    assert isinstance(output.good[0].polygon, TetrahedralMesh)


def test_legacy_3d_dispatch_uses_trace3_truth_portfolio_and_predictions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Legacy use_sim inputs cannot replace MATLAB's 3D Ybin/P TRACE3 contract."""
    z = _cube()
    y_bin = np.column_stack((z[:, 0] == 0.0, z[:, 1] == 1.0))
    y_hat = np.column_stack((z[:, 2] == 0.0, z[:, 0] == 1.0))
    p = np.where(z[:, 0] == 0.0, 1, 2).astype(np.int_)
    selection0 = np.full(z.shape[0], -1, dtype=np.int_)
    observed: list[tuple[NDArray[np.bool_], NDArray[np.bool_] | None]] = []

    def capture_build(
        self: TraceStage,
        labels: NDArray[np.bool_],
        predictions: NDArray[np.bool_] | None,
        space_area: float,
    ) -> Footprint:
        del self, space_area
        observed.append(
            (labels.copy(), None if predictions is None else predictions.copy()),
        )
        return Footprint(None, 0, 0, 0, 0, 0, THREE_DIMENSIONS)

    monkeypatch.setattr(TraceStage, "_build_trace3", capture_build)
    monkeypatch.setattr("instancespace.stages.trace.logger.warning", lambda _msg: None)
    inputs = TraceInputs(
        z=z,
        selection0=selection0,
        p=p,
        beta=np.zeros(z.shape[0], dtype=np.bool_),
        algo_labels=["a", "b"],
        y_hat=y_hat,
        y_bin=y_bin,
        trace_options=TraceOptions.default(method="legacy", use_sim=True),
        parallel_options=ParallelOptions.default(flag=False),
        general_options=GeneralOptions.default(),
    )

    TraceStage._run(inputs)  # noqa: SLF001

    assert len(observed) == EXPECTED_TRACE_BUILD_CALLS
    for good_index, best_index, algorithm_index in ((0, 1, 0), (2, 3, 1)):
        np.testing.assert_array_equal(
            observed[good_index][0],
            y_bin[:, algorithm_index],
        )
        np.testing.assert_array_equal(observed[best_index][0], p == algorithm_index + 1)
        good_prediction = observed[good_index][1]
        best_prediction = observed[best_index][1]
        assert good_prediction is not None
        assert best_prediction is not None
        np.testing.assert_array_equal(
            good_prediction,
            y_hat[:, algorithm_index],
        )
        np.testing.assert_array_equal(
            best_prediction,
            y_hat[:, algorithm_index],
        )
    np.testing.assert_array_equal(observed[-1][0], np.ones(z.shape[0], dtype=np.bool_))
    assert observed[-1][1] is None


def test_rescore_keeps_geometry_and_adds_empty_new_algorithms(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Rescore preserves trained geometry and pads test-only algorithms."""
    polygon = Polygon([(0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0)])
    trained_fp = Footprint(polygon, 4.0, 99, 98, 97.0, 0.1)
    empty = Footprint(None, 0, 0, 0, 0, 0)
    trained = TraceOut(
        space=Footprint(None, 9.0, 10, 10, 10 / 9, 1.0),
        good=[trained_fp],
        best=[trained_fp],
        hard=trained_fp,
        summary=pd.DataFrame(),
    )
    z = np.array([[0.0, 0.0], [1.0, 1.0], [3.0, 3.0]], dtype=np.double)
    y_bin = np.array(
        [[True, False], [False, True], [True, True]],
        dtype=np.bool_,
    )
    monkeypatch.setattr(
        AlphaShape2D,
        "from_points",
        classmethod(
            lambda _cls, _points: pytest.fail("rescore rebuilt trained geometry"),
        ),
    )

    rescored = TraceStage.rescore(
        trained,
        z,
        y_bin,
        np.array([1, 2, 2], dtype=np.int_),
        np.array([False, True, False], dtype=np.bool_),
        ["trained", "new"],
    )

    assert rescored.space is trained.space
    assert rescored.good[0].polygon is polygon
    assert rescored.good[0].area == TRAINED_AREA
    assert rescored.good[0].elements == EXPECTED_INSIDE_ELEMENTS
    assert rescored.good[0].good_elements == 1
    assert rescored.good[1] == empty
    assert rescored.best[1] == empty
    assert rescored.hard.polygon is polygon
    assert rescored.summary["Algorithm"].tolist() == ["trained", "new"]


def test_three_dimensional_rescore_reuses_mesh_membership() -> None:
    """3D evaluation changes evidence metrics without rebuilding trained geometry."""
    cube = _cube()
    shape = AlphaShape3D.from_points(cube)
    assert shape is not None
    mesh = shape.geometry(shape.critical_radius)
    assert mesh is not None
    trained_footprint = Footprint.from_polygon(
        mesh,
        cube,
        np.ones(cube.shape[0], dtype=np.bool_),
    )
    trained = TraceOut(
        space=Footprint(None, 1.0, cube.shape[0], cube.shape[0], 8.0, 1.0, 3),
        good=[trained_footprint],
        best=[trained_footprint],
        hard=trained_footprint,
        summary=pd.DataFrame(),
    )
    z = np.array(
        [[0.5, 0.5, 0.5], [0.0, 0.5, 0.5], [1.1, 0.5, 0.5]],
        dtype=np.double,
    )
    y_bin = np.array([[True], [False], [True]], dtype=np.bool_)

    rescored = TraceStage.rescore(
        trained,
        z,
        y_bin,
        np.ones(z.shape[0], dtype=np.int_),
        np.array([False, True, False], dtype=np.bool_),
        ["algo"],
    )

    assert rescored.good[0].polygon is mesh
    assert rescored.good[0].area == trained_footprint.area
    assert rescored.good[0].elements == EXPECTED_3D_INSIDE_ELEMENTS
    assert rescored.good[0].good_elements == 1
    assert rescored.good[0].density == pytest.approx(2.0)
    assert rescored.good[0].purity == pytest.approx(0.5)
    assert "Volume_Good" in rescored.summary


def test_rescore_rejects_trained_and_explored_dimension_mismatch() -> None:
    """A 3D trained model cannot silently discard an explored coordinate."""
    empty = Footprint(None, 0, 0, 0, 0, 0, THREE_DIMENSIONS)
    trained = TraceOut(empty, [empty], [empty], empty, pd.DataFrame())

    with pytest.raises(ValueError, match="coordinate mismatch"):
        TraceStage.rescore(
            trained,
            np.zeros((2, 2), dtype=np.double),
            np.ones((2, 1), dtype=np.bool_),
            np.ones(2, dtype=np.int_),
            np.zeros(2, dtype=np.bool_),
            ["algo"],
        )
