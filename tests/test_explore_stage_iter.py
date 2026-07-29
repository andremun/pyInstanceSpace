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
    space._explore_pythia = lambda _z: ("yhat", "pr0", "sel")
    space._explore_trace = lambda _z: ("ingood", "inbest")


def test_explore_stage_iter_yields_the_five_stages_in_order() -> None:
    space = InstanceSpace.__new__(InstanceSpace)
    _stub_stages(space)

    yielded = list(space.explore_stage_iter(cast(Metadata, None)))

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


def test_explore_maps_stage_outputs_onto_the_result() -> None:
    space = InstanceSpace.__new__(InstanceSpace)
    space._explore_results = []
    _stub_stages(space)

    def fake_extract_instance_labels(_md: object) -> list[str]:
        return ["i1", "i2"]

    space._extract_instance_labels = fake_extract_instance_labels  # type: ignore[method-assign,assignment]

    result = space.explore(cast(Metadata, None), dataset_id="d1")

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
