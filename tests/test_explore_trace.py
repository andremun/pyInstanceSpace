# ruff: noqa: ANN001, COM812, D103, PLR2004, PT018, SLF001
"""Tests for TRACE stage-owned explore-time inference.

Unit tests exercise footprint-membership logic directly, plus TraceStage's
pool-reuse hook (Q6) - the actual mechanism `compute_algorithm_qualities` uses to
decide whether to submit work to a caller-supplied pool or create its own, not the
footprint computation itself (covered by the validation test below).

The historical regression test loads legacy-unknown footprint polygons
(training_artifacts/trace/good_<algo>.csv, best_<algo>.csv) together with the
stored 2D test coordinates and verifies that stage inference reproduces the
stored boolean membership matrix exactly. Authoritative MATLAB compatibility is
covered separately by the provenance-verified R2026a readers.

Scope note. step5's ``in_space`` column is sourced from CLOISTER's
``model.cloist.Zecorr`` via the extract script's ``inpolygon`` call, not
from exploreIS. exploreIS does not compute per-instance in_space, and
CLOISTER training is out of scope for this port. The validation here covers
the ``in_good_*`` and ``in_best_*`` columns only.

No disagreement budget is used: the legacy snapshot currently has no ambiguous
boundary row, while verified-current tests characterize boundary cases explicitly.
"""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest
from numpy.typing import NDArray
from shapely.geometry import MultiPolygon, Polygon

from instancespace.data.model import Footprint, TraceOut
from instancespace.data.options import GeneralOptions, ParallelOptions
from instancespace.instance_space import InstanceSpace
from instancespace.stages.trace import TracePredictInput, TracePredictOutput, TraceStage
from instancespace.utils.alpha_shape import AlphaShape3D, TetrahedralMesh

REFERENCE_DIR = Path("tests/matlab_reference")
ARTIFACTS_DIR = REFERENCE_DIR / "training_artifacts" / "trace"
OUTPUTS_DIR = REFERENCE_DIR / "explore_outputs"

ALGO_ORDER = [
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


def make_footprint(
    polygon: Polygon | TetrahedralMesh | None,
    dimension: int = 2,
) -> Footprint:
    return cast(Footprint, SimpleNamespace(polygon=polygon, dimension=dimension))


def make_instance_space(
    good_polys: list[Polygon | TetrahedralMesh | None],
    best_polys: list[Polygon | TetrahedralMesh | None],
    trained_dimensions: int = 2,
) -> InstanceSpace:
    trace = Mock(spec=TraceOut)
    trace.space = make_footprint(None, trained_dimensions)
    trace.good = [make_footprint(p, trained_dimensions) for p in good_polys]
    trace.best = [make_footprint(p, trained_dimensions) for p in best_polys]
    trace.hard = make_footprint(None, trained_dimensions)
    model = Mock()
    model.trace = trace
    model.pilot = SimpleNamespace(
        z=np.zeros((1, trained_dimensions), dtype=np.double),
    )
    instance_space = Mock(spec=InstanceSpace)
    instance_space._model = model
    instance_space._require_model = Mock(return_value=model)
    return instance_space


def _predict_trace(
    instance_space: InstanceSpace,
    z: NDArray[np.double],
    n_new_algos: int = 0,
) -> TracePredictOutput:
    """Call the stage contract with fitted state held by InstanceSpace."""
    model = cast(Any, instance_space)._require_model()
    fitted = cast(TraceOut, model.trace)
    return TraceStage.predict(TracePredictInput(z, n_new_algos), fitted)


def test_trace_output_shapes() -> None:
    square = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    space = make_instance_space([square, square, square], [square, square, square])
    z = np.random.default_rng().random((5, 2))
    in_good, in_best = _predict_trace(space, z)
    assert in_good.shape == (5, 3)
    assert in_best.shape == (5, 3)
    assert in_good.dtype == np.bool_
    assert in_best.dtype == np.bool_


def test_trace_inside_outside_and_matlab_boundary_semantics() -> None:
    """MATLAB ``isinterior`` counts exact boundary points as inside."""
    square = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    space = make_instance_space([square], [square])
    z = np.array([[0.5, 0.5], [2.0, 2.0], [0.0, 0.0]])
    in_good, in_best = _predict_trace(space, z)
    assert in_good[0, 0]
    assert not in_good[1, 0]
    assert in_good[2, 0]
    assert in_best[0, 0]
    assert not in_best[1, 0]
    assert in_best[2, 0]


@pytest.mark.parametrize(
    ("trained_dimensions", "explored_dimensions"),
    [(2, 3), (3, 2)],
)
def test_trace_rejects_trained_and_explored_dimension_mismatch(
    trained_dimensions: int,
    explored_dimensions: int,
) -> None:
    """Neither trained nor explored coordinates may be silently discarded."""
    square = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    space = make_instance_space(
        [square],
        [square],
        trained_dimensions=trained_dimensions,
    )
    z = np.zeros((2, explored_dimensions), dtype=np.double)

    with pytest.raises(ValueError, match="coordinate mismatch"):
        _predict_trace(space, z)


def test_trace_three_dimensional_mesh_membership_is_inclusive() -> None:
    """Explore uses trained tetrahedra directly, including their boundaries."""
    points = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.double,
    )
    shape = AlphaShape3D.from_points(points)
    assert shape is not None
    mesh = shape.geometry(shape.critical_radius)
    assert mesh is not None
    space = make_instance_space([mesh], [mesh], trained_dimensions=3)
    z = np.array(
        [[0.1, 0.1, 0.1], [0.5, 0.5, 0.0], [0.34, 0.34, 0.34]],
        dtype=np.double,
    )

    in_good, in_best = _predict_trace(space, z)

    np.testing.assert_array_equal(in_good[:, 0], [True, True, False])
    np.testing.assert_array_equal(in_best[:, 0], [True, True, False])


def test_trace_none_polygon_returns_false() -> None:
    space = make_instance_space([None, None], [None, None])
    z = np.array([[0.5, 0.5], [1.0, 1.0]])
    in_good, in_best = _predict_trace(space, z)
    assert not in_good.any()
    assert not in_best.any()


def test_trace_per_algo_independent() -> None:
    # Two algos with disjoint polygons: instance inside one is outside the other.
    left = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    right = Polygon([(2, 0), (3, 0), (3, 1), (2, 1)])
    space = make_instance_space([left, right], [left, right])
    z = np.array([[0.5, 0.5], [2.5, 0.5]])
    in_good, in_best = _predict_trace(space, z)
    assert in_good[0, 0] and not in_good[0, 1]
    assert not in_good[1, 0] and in_good[1, 1]
    assert in_best[0, 0] and not in_best[0, 1]
    assert not in_best[1, 0] and in_best[1, 1]


def test_trace_good_and_best_independent() -> None:
    # good polygon contains point, best polygon does not.
    good = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
    best = Polygon([(5, 5), (6, 5), (6, 6), (5, 6)])
    space = make_instance_space([good], [best])
    z = np.array([[1.0, 1.0]])
    in_good, in_best = _predict_trace(space, z)
    assert in_good[0, 0]
    assert not in_best[0, 0]


def test_trace_widens_output_for_new_algorithms() -> None:
    """F9 full MATLAB parity: `n_new_algos` pads in_good/in_best with `False`.

    Matches MATLAB's `TRACEthrow3` empty-footprint placeholder for
    algorithms present in the test set but absent from training - there is
    no trained footprint to test membership against.
    """
    square = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    space = make_instance_space([square], [square])
    z = np.array([[0.5, 0.5], [2.0, 2.0]])

    in_good, in_best = _predict_trace(space, z, n_new_algos=2)

    assert in_good.shape == (2, 3)
    assert in_best.shape == (2, 3)
    # Trained column (0) is unaffected by the widening.
    assert in_good[0, 0] and not in_good[1, 0]
    # New-algorithm columns (1, 2) are always False - no trained footprint.
    assert not in_good[:, 1:].any()
    assert not in_best[:, 1:].any()


def test_trace_predict_does_not_rebuild_or_mutate_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Membership reads trained geometry without rebuilding footprints."""
    square = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    space = make_instance_space([square], [square])
    fitted = cast(TraceOut, cast(Any, space)._require_model().trace)
    z = np.array([[0.5, 0.5], [2.0, 2.0]])
    z_before = z.copy()
    good_before = tuple(fitted.good)
    best_before = tuple(fitted.best)
    rebuild = Mock(side_effect=AssertionError("TRACE training was called"))
    monkeypatch.setattr(TraceStage, "trace", rebuild)

    TraceStage.predict(TracePredictInput(z), fitted)

    rebuild.assert_not_called()
    np.testing.assert_array_equal(z, z_before)
    assert tuple(fitted.good) == good_before
    assert tuple(fitted.best) == best_before
    assert fitted.good[0].polygon is square
    assert fitted.best[0].polygon is square


def test_trace_instance_space_wrapper_remains_compatible() -> None:
    """Keep the private wrapper compatible while orchestration migrates."""
    square = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    space = make_instance_space([square], [square])
    z = np.array([[0.5, 0.5], [2.0, 2.0]])

    expected = _predict_trace(space, z)
    actual = InstanceSpace._explore_trace(space, z)

    for actual_field, expected_field in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(actual_field, expected_field)


def test_trace_predict_rejects_invalid_fitted_dimension() -> None:
    """A corrupt trained footprint dimension cannot be silently treated as 2D."""
    square = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    space = make_instance_space([square], [square], trained_dimensions=4)

    with pytest.raises(ValueError, match="fitted geometry"):
        _predict_trace(space, np.zeros((2, 2), dtype=np.double))


@pytest.mark.parametrize(
    "z",
    [
        np.zeros(2, dtype=np.double),
        np.zeros((2, 1), dtype=np.double),
        np.zeros((2, 4), dtype=np.double),
    ],
)
def test_trace_predict_rejects_invalid_explored_shape(
    z: NDArray[np.double],
) -> None:
    """Only two-dimensional matrices with two or three coordinates are valid."""
    space = make_instance_space([None], [None])

    with pytest.raises(ValueError, match="two-dimensional matrix|coordinates"):
        _predict_trace(space, z)


def _bare_trace_stage(n_algos, executor) -> TraceStage:  # type: ignore[no-untyped-def]
    # Params deliberately untyped (see test_instance_space_executor.py's
    # _bare_instance_space for why): a typed signature makes mypy check this
    # body, which then rejects the intentional attribute-monkeypatching below.
    stage = TraceStage.__new__(TraceStage)
    stage.algo_labels = [f"algo{i}" for i in range(n_algos)]
    stage.y_bin = np.zeros((3, n_algos), dtype=np.bool_)
    stage.p = np.zeros(3, dtype=np.int_)
    stage.parallel_opts = ParallelOptions(True, 2)
    stage.executor = executor
    stage.general_opts = cast(GeneralOptions, SimpleNamespace(verbose=False))
    stage.process_algorithm = lambda i: (  # type: ignore[method-assign]
        i,
        Footprint(None, 0, 0, 0, 0, 0),
        Footprint(None, 0, 0, 0, 0, 0),
    )
    return stage


def test_compute_algorithm_qualities_reuses_a_supplied_executor() -> None:
    shared_executor = ThreadPoolExecutor(max_workers=2)
    stage = _bare_trace_stage(n_algos=3, executor=shared_executor)

    with patch(
        "instancespace.stages.trace.ThreadPoolExecutor",
    ) as mock_pool_class:
        good, best = stage.compute_algorithm_qualities(3)

    mock_pool_class.assert_not_called()
    assert len(good) == 3
    assert len(best) == 3
    shared_executor.shutdown(wait=True)


def test_compute_algorithm_qualities_stays_sequential_when_parallel_disabled() -> None:
    """A supplied pool is ignored when the shared parallel flag is false."""
    n_algorithms = 3
    supplied_executor = Mock(spec=ThreadPoolExecutor)
    stage = _bare_trace_stage(n_algos=n_algorithms, executor=supplied_executor)
    stage.parallel_opts = ParallelOptions(False, 2)

    with patch(
        "instancespace.stages.trace.ThreadPoolExecutor",
    ) as mock_pool_class:
        good, best = stage.compute_algorithm_qualities(n_algorithms)

    mock_pool_class.assert_not_called()
    supplied_executor.submit.assert_not_called()
    assert len(good) == n_algorithms
    assert len(best) == n_algorithms


def test_compute_algorithm_qualities_creates_its_own_pool_when_none_supplied() -> None:
    stage = _bare_trace_stage(n_algos=2, executor=None)

    good, best = stage.compute_algorithm_qualities(2)

    assert len(good) == 2
    assert len(best) == 2


def test_compute_algorithm_qualities_output_identical_with_and_without_reuse() -> None:
    # Same inputs, only the pool-sourcing differs - results must match exactly.
    own_pool_stage = _bare_trace_stage(n_algos=4, executor=None)
    own_good, own_best = own_pool_stage.compute_algorithm_qualities(4)

    shared_executor = ThreadPoolExecutor(max_workers=2)
    shared_stage = _bare_trace_stage(n_algos=4, executor=shared_executor)
    shared_good, shared_best = shared_stage.compute_algorithm_qualities(4)
    shared_executor.shutdown(wait=True)

    assert own_good == shared_good
    assert own_best == shared_best


def load_polygon(path: Path) -> Polygon | MultiPolygon | None:
    """Reconstruct a shapely (Multi)Polygon from a MATLAB polyshape CSV.

    MATLAB exports multi-region polyshapes as a single CSV with NaN rows
    delimiting the regions; rebuild each region as a Polygon and combine.
    """
    if not path.exists():
        return None
    df = pd.read_csv(path)
    vertices = df[["x", "y"]].to_numpy(dtype=np.double)
    regions = []
    current: list[tuple[float, float]] = []
    for row in vertices:
        if np.isnan(row).any():
            if len(current) >= 3:
                regions.append(Polygon(current))
            current = []
        else:
            current.append(tuple(row))
    if len(current) >= 3:
        regions.append(Polygon(current))
    if not regions:
        return None
    return regions[0] if len(regions) == 1 else MultiPolygon(regions)


def build_trace_from_artifacts() -> TraceOut:
    good_polys = [load_polygon(ARTIFACTS_DIR / f"good_{a}.csv") for a in ALGO_ORDER]
    best_polys = [load_polygon(ARTIFACTS_DIR / f"best_{a}.csv") for a in ALGO_ORDER]
    trace = Mock(spec=TraceOut)
    trace.space = make_footprint(None)
    trace.good = [make_footprint(p) for p in good_polys]
    trace.best = [make_footprint(p) for p in best_polys]
    trace.hard = make_footprint(None)
    return trace


def test_trace_matches_legacy_snapshot() -> None:
    """Reproduce all stored legacy good/best memberships exactly."""
    trace = build_trace_from_artifacts()

    instance_space = Mock(spec=InstanceSpace)
    instance_space._model = Mock()
    instance_space._model.trace = trace
    instance_space._require_model = Mock(return_value=instance_space._model)

    z = pd.read_csv(OUTPUTS_DIR / "step3_after_pilot.csv", index_col=0)
    instance_space._model.pilot = SimpleNamespace(z=z.to_numpy(dtype=np.double))
    in_good, in_best = _predict_trace(instance_space, z.to_numpy())

    ref = pd.read_csv(OUTPUTS_DIR / "step5_trace_membership.csv", index_col=0)

    for j, algo in enumerate(ALGO_ORDER):
        ref_good = ref[f"in_good_{algo}"].to_numpy(dtype=np.bool_)
        ref_best = ref[f"in_best_{algo}"].to_numpy(dtype=np.bool_)
        np.testing.assert_array_equal(in_good[:, j], ref_good)
        np.testing.assert_array_equal(in_best[:, j], ref_best)
