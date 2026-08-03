"""Unit tests for the staged explore entry points.

These exercise the orchestration added by ``explore_stage_iter`` and the refactored
``explore`` in isolation: the per-stage inference methods are stubbed, so the tests
check only that the stages run in the right order, that each stage's output feeds the
next, and that ``explore`` maps those outputs onto the right ``ExploreResult`` fields.
The stage methods' numerical fidelity is covered against MATLAB by the per-stage
validation suites.
"""

from typing import cast

from instancespace.data.metadata import Metadata
from instancespace.instance_space import ExploreStage, InstanceSpace

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
    space._find_new_algorithms = lambda _md, _algo_labels: []
    # Only reached when test_metadata carries ground truth (has_ground_truth
    # branch in explore_stage_iter) - stubbed here too so that path doesn't
    # need a real trained Model.
    space._require_model = lambda: type("_FakeModel", (), {"data": _FakeData()})()


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


def test_explore_stage_iter_yields_evaluation_when_ground_truth_present() -> None:
    """F9: ExploreStage.EVALUATION is yielded (after TRACE) only with ground truth.

    Orchestration-only: `_explore_evaluate` itself is stubbed here, so this
    checks wiring (the conditional yield, and that it receives PYTHIA's
    already-computed `y_hat`), not evaluation numerics - those are covered
    by `tests/test_explore_evaluate.py`.
    """
    with_ground_truth = type("_FakeMetadata", (), {"algorithm_names": ["algo1"]})()

    space = InstanceSpace.__new__(InstanceSpace)
    _stub_stages(space)
    space._explore_evaluate = (  # type: ignore[method-assign,assignment,return-value]
        lambda _md, y_hat, _new_algo_labels: f"evaluation({y_hat})"
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
    assert stages[ExploreStage.EVALUATION] == "evaluation(yhat)"
