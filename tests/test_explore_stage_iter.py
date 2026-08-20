# ruff: noqa: ANN001, ARG005, D103, SLF001
"""Unit tests for the staged explore entry points.

These exercise the orchestration added by ``explore_stage_iter`` and the refactored
``explore`` in isolation: the per-stage inference methods are stubbed, so the tests
check only that the stages run in the right order, that each stage's output feeds the
next, and that ``explore`` maps those outputs onto the right ``ExploreResult`` fields.
The stage methods' numerical fidelity is covered against MATLAB by the per-stage
validation suites.
"""

from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest
from numpy.typing import NDArray

from instancespace.data.metadata import Metadata
from instancespace.data.model import Footprint, TraceOut
from instancespace.instance_space import ExploreStage, InstanceSpace, _EvaluationResult
from instancespace.stages.trace import TraceStage

# A ground-truth-free stand-in for `Metadata`: these tests exercise stage
# orchestration only (the per-stage methods are stubbed below), and F9's
# ExploreStage.EVALUATION branch reads `test_metadata.algorithm_names`
# directly (unlike the other stages, which only see it through the stubbed
# methods) - so it must be a real value, not `None`, even though nothing
# else about this fake object needs to look like a `Metadata`.
_FakeMetadata = type("_FakeMetadata", (), {"algorithm_names": []})
_NO_GROUND_TRUTH = cast(Metadata, cast(object, _FakeMetadata()))
_FakeData = type("_FakeData", (), {"algo_labels": ["a0", "a1"]})


def _stub_stages(space) -> None:  # type: ignore[no-untyped-def]
    """Replace the pieces the staged pipeline calls with cheap sentinels.

    ``space`` is deliberately left unannotated: a typed signature makes mypy
    check this body, which then rejects the intentional attribute-monkeypatching
    below (real stage methods return arrays, these stubs return plain strings).
    """
    space._validate_for_explore = lambda _md: None
    space._extract_features = lambda _md: "xraw"
    space._explore_prelim = lambda x: f"prelim({x})"
    space._explore_sifted = lambda x: f"sifted({x})"
    space._explore_pilot = lambda _x: "Z"
    space._explore_pythia = lambda _z, n_new_algos=0: ("yhat", "pr0", "sel")
    space._explore_trace = lambda _z, n_new_algos=0: ("ingood", "inbest")
    space._validate_explore_trace_dimensions = lambda _z: None
    space._find_new_algorithms = lambda _md, _algo_labels: []
    # Only reached when test_metadata carries ground truth (has_ground_truth
    # branch in explore_stage_iter) - stubbed here too so that path doesn't
    # need a real trained Model.
    space._require_model = lambda: type(
        "_FakeModel",
        (),
        {"data": _FakeData(), "trace": "trained_trace"},
    )()


def test_explore_stage_iter_yields_the_five_stages_in_order() -> None:
    space = InstanceSpace.__new__(InstanceSpace)
    _stub_stages(space)

    yielded = list(space.explore_stage_iter(_NO_GROUND_TRUTH))

    assert [annotated.stage for annotated in yielded] == [
        ExploreStage.PRELIM,
        ExploreStage.SIFTED,
        ExploreStage.PILOT,
        ExploreStage.PYTHIA,
        ExploreStage.TRACE,
    ]
    stages = {annotated.stage: annotated.output for annotated in yielded}
    # Each geometric stage feeds the next; PILOT's output feeds PYTHIA and TRACE.
    assert stages[ExploreStage.PRELIM] == "prelim(xraw)"
    assert stages[ExploreStage.SIFTED] == "sifted(prelim(xraw))"
    assert stages[ExploreStage.PILOT] == "Z"
    assert stages[ExploreStage.PYTHIA] == ("yhat", "pr0", "sel")
    assert stages[ExploreStage.TRACE] == ("ingood", "inbest")
    # No ExploreStage.EVALUATION (F9) when the test metadata has no ground truth.
    assert ExploreStage.EVALUATION not in stages


def test_3d_stage_iter_yields_native_trace_membership() -> None:
    """Matching 3D projections now advance through the lazy TRACE boundary."""
    z = np.array([[1.0, 2.0, 3.0]], dtype=np.double)
    model = SimpleNamespace(
        data=_FakeData(),
        pilot=SimpleNamespace(z=np.zeros((2, 3), dtype=np.double)),
        trace=SimpleNamespace(good=[], best=[]),
    )
    space = InstanceSpace.__new__(InstanceSpace)
    stubbed = cast(Any, space)
    stubbed._validate_for_explore = lambda _md: None
    stubbed._extract_features = lambda _md: "xraw"
    stubbed._explore_prelim = lambda x: f"prelim({x})"
    stubbed._explore_sifted = lambda x: f"sifted({x})"
    stubbed._explore_pilot = lambda _x: z
    stubbed._explore_pythia = lambda _z, n_new_algos=0: (
        "yhat",
        "pr0",
        "sel",
    )
    stubbed._require_model = lambda: model

    stages = space.explore_stage_iter(_NO_GROUND_TRUTH)

    assert next(stages).stage is ExploreStage.PRELIM
    assert next(stages).stage is ExploreStage.SIFTED
    pilot = next(stages)
    assert pilot.stage is ExploreStage.PILOT
    np.testing.assert_array_equal(pilot.output, z)
    assert next(stages).stage is ExploreStage.PYTHIA
    trace = next(stages)
    assert trace.stage is ExploreStage.TRACE
    in_good, in_best = trace.output
    assert in_good.shape == (1, 0)
    assert in_best.shape == (1, 0)
    with pytest.raises(StopIteration):
        next(stages)


def test_projection_mismatch_fails_only_when_lazy_trace_is_requested() -> None:
    """Earlier explore stages remain inspectable before TRACE rejects a mismatch."""
    z = np.array([[1.0, 2.0, 3.0]], dtype=np.double)
    model = SimpleNamespace(
        data=_FakeData(),
        pilot=SimpleNamespace(z=np.zeros((2, 2), dtype=np.double)),
        trace=SimpleNamespace(good=[], best=[]),
    )
    space = InstanceSpace.__new__(InstanceSpace)
    stubbed = cast(Any, space)
    stubbed._validate_for_explore = lambda _md: None
    stubbed._extract_features = lambda _md: "xraw"
    stubbed._explore_prelim = lambda value: value
    stubbed._explore_sifted = lambda value: value
    stubbed._explore_pilot = lambda _value: z
    stubbed._explore_pythia = lambda _z, n_new_algos=0: ("yhat", "pr0", "sel")
    stubbed._require_model = lambda: model
    stages = space.explore_stage_iter(_NO_GROUND_TRUTH)

    assert [next(stages).stage for _ in range(4)] == [
        ExploreStage.PRELIM,
        ExploreStage.SIFTED,
        ExploreStage.PILOT,
        ExploreStage.PYTHIA,
    ]
    with pytest.raises(ValueError, match="coordinate mismatch"):
        next(stages)


def test_explore_maps_stage_outputs_onto_the_result() -> None:
    space = InstanceSpace.__new__(InstanceSpace)
    space._explore_results = []
    _stub_stages(space)

    def fake_extract_instance_labels(_md: object) -> list[str]:
        return ["i1", "i2"]

    space._extract_instance_labels = fake_extract_instance_labels  # type: ignore[method-assign,assignment]

    result = space.explore(_NO_GROUND_TRUTH, dataset_id="d1")

    assert result.dataset_id == "d1"
    # x is the post-SIFTED features, z is the PILOT projection.
    assert result.x == "sifted(prelim(xraw))"
    assert result.z == "Z"
    assert (result.y_hat, result.pr0_hat, result.selection0) == (  # type: ignore[comparison-overlap]
        "yhat",
        "pr0",
        "sel",
    )
    assert (result.in_good, result.in_best) == (  # type: ignore[comparison-overlap]
        "ingood",
        "inbest",
    )
    assert result.inst_labels == ["i1", "i2"]
    assert space.explore_results == [result]
    # No ground truth in this fixture -> all F9 evaluation fields stay None,
    # preserving pre-F9 behaviour exactly for feature-only test metadata.
    assert result.y_actual is None
    assert result.y_best_actual is None
    assert result.p_actual is None
    assert result.beta_actual is None
    assert result.accuracy_actual is None
    assert result.precision_actual is None
    assert result.recall_actual is None
    assert result.cvcmat_actual is None
    assert result.trace_out is None


def test_explore_stage_iter_yields_evaluation_when_ground_truth_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """F9: ExploreStage.EVALUATION is yielded (after TRACE) only with ground truth.

    Orchestration-only: `_explore_evaluate` itself is stubbed here, so this
    checks wiring (the conditional yield, and that it receives PYTHIA's
    already-computed `y_hat`), not evaluation numerics - those are covered
    by `tests/test_explore_evaluate.py`.
    """
    with_ground_truth = type("_FakeMetadata", (), {"algorithm_names": ["algo1"]})()

    space = InstanceSpace.__new__(InstanceSpace)
    _stub_stages(space)

    evaluation = _EvaluationResult(
        y_actual=np.zeros((1, 1), dtype=np.bool_),
        y_best_actual=np.zeros(1, dtype=np.double),
        p_actual=np.ones(1, dtype=np.int_),
        beta_actual=np.zeros(1, dtype=np.bool_),
        accuracy_actual=np.zeros(1, dtype=np.double),
        precision_actual=np.zeros(1, dtype=np.double),
        recall_actual=np.zeros(1, dtype=np.double),
        cvcmat_actual=np.zeros((1, 4), dtype=np.double),
        algo_labels=["algo1"],
    )

    def fake_explore_evaluate(
        test_metadata: Metadata,
        y_hat: NDArray[np.bool_],
        new_algo_labels: list[str],
    ) -> _EvaluationResult:
        del test_metadata, y_hat, new_algo_labels
        return evaluation

    space._explore_evaluate = fake_explore_evaluate  # type: ignore[method-assign]
    empty = Footprint(None, 0, 0, 0, 0, 0)
    rescored = TraceOut(empty, [], [], empty, pd.DataFrame())
    monkeypatch.setattr(
        TraceStage,
        "rescore",
        lambda *_args: rescored,
    )

    yielded = list(space.explore_stage_iter(cast(Metadata, with_ground_truth)))

    assert [annotated.stage for annotated in yielded] == [
        ExploreStage.PRELIM,
        ExploreStage.SIFTED,
        ExploreStage.PILOT,
        ExploreStage.PYTHIA,
        ExploreStage.TRACE,
        ExploreStage.EVALUATION,
    ]
    stages = {annotated.stage: annotated.output for annotated in yielded}
    # _explore_pythia's stub returns ("yhat", "pr0", "sel") - y_hat is [0].
    assert stages[ExploreStage.EVALUATION].trace_out is rescored


def test_explore_stage_iter_defers_evaluation_until_after_trace_yield(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stopping at TRACE cannot run the later evaluation or rescore work."""
    with_ground_truth = type("_FakeMetadata", (), {"algorithm_names": ["algo1"]})()
    space = InstanceSpace.__new__(InstanceSpace)
    _stub_stages(space)
    events: list[str] = []

    evaluation = _EvaluationResult(
        y_actual=np.zeros((1, 1), dtype=np.bool_),
        y_best_actual=np.zeros(1, dtype=np.double),
        p_actual=np.ones(1, dtype=np.int_),
        beta_actual=np.zeros(1, dtype=np.bool_),
        accuracy_actual=np.zeros(1, dtype=np.double),
        precision_actual=np.zeros(1, dtype=np.double),
        recall_actual=np.zeros(1, dtype=np.double),
        cvcmat_actual=np.zeros((1, 4), dtype=np.double),
        algo_labels=["algo1"],
    )

    def fake_explore_trace(
        z: NDArray[np.double],
        n_new_algos: int = 0,
    ) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
        del z, n_new_algos
        events.append("trace")
        memberships = np.zeros((1, 1), dtype=np.bool_)
        return memberships, memberships

    def fake_explore_evaluate(
        test_metadata: Metadata,
        y_hat: NDArray[np.bool_],
        new_algo_labels: list[str],
    ) -> _EvaluationResult:
        del test_metadata, y_hat, new_algo_labels
        events.append("evaluate")
        return evaluation

    def fake_rescore(*_args: object) -> TraceOut:
        events.append("rescore")
        empty = Footprint(None, 0, 0, 0, 0, 0)
        return TraceOut(empty, [], [], empty, pd.DataFrame())

    space._explore_trace = fake_explore_trace  # type: ignore[method-assign]
    space._explore_evaluate = fake_explore_evaluate  # type: ignore[method-assign]
    monkeypatch.setattr(TraceStage, "rescore", fake_rescore)

    stages = space.explore_stage_iter(cast(Metadata, with_ground_truth))
    for expected in (
        ExploreStage.PRELIM,
        ExploreStage.SIFTED,
        ExploreStage.PILOT,
        ExploreStage.PYTHIA,
    ):
        assert next(stages).stage is expected

    assert next(stages).stage is ExploreStage.TRACE
    assert events == ["trace"]
    assert next(stages).stage is ExploreStage.EVALUATION
    assert events == ["trace", "evaluate", "rescore"]


def test_3d_stage_iter_rescores_after_trace_membership(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ground-truth evaluation reuses trained native 3D TRACE geometry."""
    with_ground_truth = type("_FakeMetadata", (), {"algorithm_names": ["algo1"]})()
    z = np.array([[1.0, 2.0, 3.0]], dtype=np.double)
    model = SimpleNamespace(
        data=_FakeData(),
        pilot=SimpleNamespace(z=np.zeros((2, 3), dtype=np.double)),
        trace="trained_trace",
    )
    space = InstanceSpace.__new__(InstanceSpace)
    _stub_stages(space)
    stubbed = cast(Any, space)
    stubbed._explore_pilot = lambda _x: z
    stubbed._require_model = lambda: model
    stubbed._validate_explore_trace_dimensions = (
        lambda value: InstanceSpace._validate_explore_trace_dimensions(space, value)
    )
    evaluation = _EvaluationResult(
        y_actual=np.zeros((1, 1), dtype=np.bool_),
        y_best_actual=np.zeros(1, dtype=np.double),
        p_actual=np.ones(1, dtype=np.int_),
        beta_actual=np.zeros(1, dtype=np.bool_),
        accuracy_actual=np.zeros(1, dtype=np.double),
        precision_actual=np.zeros(1, dtype=np.double),
        recall_actual=np.zeros(1, dtype=np.double),
        cvcmat_actual=np.zeros((1, 4), dtype=np.double),
        algo_labels=["algo1"],
    )
    space._explore_evaluate = lambda *_args: evaluation  # type: ignore[method-assign]

    rescored_dimensions: list[int] = []

    def fake_rescore(
        _trained: object,
        explored_z: NDArray[np.double],
        *_args: object,
    ) -> TraceOut:
        rescored_dimensions.append(explored_z.shape[1])
        empty = Footprint(None, 0, 0, 0, 0, 0, 3)
        return TraceOut(empty, [empty], [empty], empty, pd.DataFrame())

    monkeypatch.setattr(TraceStage, "rescore", fake_rescore)
    stages = space.explore_stage_iter(cast(Metadata, with_ground_truth))

    assert [next(stages).stage for _ in range(5)] == [
        ExploreStage.PRELIM,
        ExploreStage.SIFTED,
        ExploreStage.PILOT,
        ExploreStage.PYTHIA,
        ExploreStage.TRACE,
    ]
    evaluation_stage = next(stages)
    assert evaluation_stage.stage is ExploreStage.EVALUATION
    assert rescored_dimensions == [3]


def test_explore_maps_rescored_trace_onto_ground_truth_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ground-truth explore results expose metrics on the trained geometry."""
    with_ground_truth = type("_FakeMetadata", (), {"algorithm_names": ["algo1"]})()
    space = InstanceSpace.__new__(InstanceSpace)
    space._explore_results = []
    _stub_stages(space)

    def fake_extract_instance_labels(metadata: Metadata) -> pd.Series:  # type: ignore[type-arg]
        del metadata
        return pd.Series(["i1"])

    space._extract_instance_labels = fake_extract_instance_labels  # type: ignore[method-assign]

    evaluation = _EvaluationResult(
        y_actual=np.zeros((1, 1), dtype=np.bool_),
        y_best_actual=np.zeros(1, dtype=np.double),
        p_actual=np.ones(1, dtype=np.int_),
        beta_actual=np.zeros(1, dtype=np.bool_),
        accuracy_actual=np.zeros(1, dtype=np.double),
        precision_actual=np.zeros(1, dtype=np.double),
        recall_actual=np.zeros(1, dtype=np.double),
        cvcmat_actual=np.zeros((1, 4), dtype=np.double),
        algo_labels=["algo1"],
    )
    space._explore_evaluate = lambda *_args: evaluation  # type: ignore[method-assign]
    empty = Footprint(None, 0, 0, 0, 0, 0)
    rescored = TraceOut(empty, [], [], empty, pd.DataFrame())
    monkeypatch.setattr(TraceStage, "rescore", lambda *_args: rescored)

    result = space.explore(cast(Metadata, with_ground_truth), dataset_id="d1")

    assert result.trace_out is rescored
    assert result.algo_labels == ["algo1"]
    assert space.explore_results == [result]
