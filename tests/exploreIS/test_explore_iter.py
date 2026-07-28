"""Unit tests for the staged explore entry points.

These exercise the orchestration added by ``explore_iter`` and the refactored
``explore`` in isolation: the per-stage inference methods are stubbed, so the tests
check only that the stages run in the right order, that each stage's output feeds the
next, and that ``explore`` maps those outputs onto the right ``ExploreResult`` fields.
The stage methods' numerical fidelity is covered against MATLAB by the per-stage
validation suites.
"""

from instancespace.instance_space import InstanceSpace


def _stub_stages(space):
    """Replace the pieces the staged pipeline calls with cheap sentinels."""
    space._validate_for_explore = lambda _md: None
    space._extract_features = lambda _md: "xraw"
    space._explore_prelim = lambda x: f"prelim({x})"
    space._explore_sifted = lambda x: f"sifted({x})"
    space._explore_pilot = lambda _x: "Z"
    space._explore_pythia = lambda _z: ("yhat", "pr0", "sel")
    space._explore_trace = lambda _z: ("ingood", "inbest")


def test_explore_iter_yields_the_five_stages_in_order():
    space = InstanceSpace.__new__(InstanceSpace)
    _stub_stages(space)

    yielded = list(space.explore_iter(None))

    assert [name for name, _ in yielded] == [
        "prelim", "sifted", "pilot", "pythia", "trace",
    ]
    stages = dict(yielded)
    # Each geometric stage feeds the next; PILOT's output feeds PYTHIA and TRACE.
    assert stages["prelim"] == "prelim(xraw)"
    assert stages["sifted"] == "sifted(prelim(xraw))"
    assert stages["pilot"] == "Z"
    assert stages["pythia"] == ("yhat", "pr0", "sel")
    assert stages["trace"] == ("ingood", "inbest")


def test_explore_maps_stage_outputs_onto_the_result():
    space = InstanceSpace.__new__(InstanceSpace)
    space._explore_results = []
    _stub_stages(space)
    space._extract_instance_labels = lambda _md: ["i1", "i2"]

    result = space.explore(None, dataset_id="d1")

    assert result.dataset_id == "d1"
    # x is the post-SIFTED features, z is the PILOT projection.
    assert result.x == "sifted(prelim(xraw))"
    assert result.z == "Z"
    assert (result.y_hat, result.pr0_hat, result.selection0) == ("yhat", "pr0", "sel")
    assert (result.in_good, result.in_best) == ("ingood", "inbest")
    assert result.inst_labels == ["i1", "i2"]
    assert space.explore_results == [result]
