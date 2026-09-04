# SPDX-License-Identifier: LicenseRef-PolyForm-Noncommercial-1.0.0
# Copyright (c) 2024-2026 Mario Andrés Muñoz
"""T2 (end-to-end build() test) and Q8 (stage-rerun invalidation regression test).

T2 is the Python equivalent of MATLAB's `test_integration.m` - construct a real
InstanceSpace with real metadata + options, run the full 7-stage pipeline, and
assert every stage's output actually landed on the resulting Model. Before this
test, no test in this repo called `.build()` end to end - the existing
`instance_space_from_files` callers all stop at preprocessing/prelim.

Q8 depends on this same fixture: a real, fully-built 7-stage pipeline is what
`_rollback_to_schedule_index()`'s wave-position invalidation needs to be
exercised against meaningfully. Both share one module-scoped build rather than
paying its cost twice.
"""

from collections.abc import Iterator
from pathlib import Path
from unittest.mock import Mock

import pytest

from instancespace.data.metadata import from_csv_file
from instancespace.instance_space import InstanceSpace, instance_space_from_files
from instancespace.stages.cloister import CloisterStage
from instancespace.stages.pilot import PilotStage
from instancespace.stages.prelim import PrelimStage
from instancespace.stages.pythia import PythiaStage
from instancespace.stages.sifted import SiftedStage
from instancespace.stages.trace import TraceStage


@pytest.fixture(scope="module")
def built_instance_space() -> Iterator[InstanceSpace]:
    """Build one complete instance space for the integration assertions."""
    script_dir = Path(__file__).resolve().parent
    metadata_path = script_dir / "test_data/preprocessing/metadata.csv"
    options_path = script_dir / "test_data/preprocessing/options.json"

    instance_space = instance_space_from_files(metadata_path, options_path)
    assert instance_space is not None
    instance_space.build()

    yield instance_space

    instance_space.close()


def test_build_produces_a_fully_populated_model(
    built_instance_space: InstanceSpace,
) -> None:
    """Expose every completed stage, including the optional 2D viewpoint."""
    model = built_instance_space.model
    n_algos = len(model.data.algo_labels)

    assert model.data.x.shape[0] > 0
    assert model.prelim.mu_x.shape[0] > 0
    assert model.sifted.selvars.size > 0
    assert model.pilot.z.shape[1] == model.opts.pilot.dims
    assert model.pilot.viewpoint is None
    assert model.cloister.z_edge.size > 0
    assert len(model.pythia.svm) == n_algos
    assert not model.pythia.summary.empty
    assert len(model.trace.good) == n_algos
    assert len(model.trace.best) == n_algos
    assert not model.trace.summary.empty


def test_explore_applies_every_fitted_stage_without_retraining(
    built_instance_space: InstanceSpace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise the real build-to-explore path, including truth-aware rescore."""
    metadata_path = Path(__file__).parent / "test_data/preprocessing/metadata.csv"
    metadata = from_csv_file(metadata_path)
    assert metadata is not None

    for stage, method_name in (
        (PrelimStage, "prelim"),
        (SiftedStage, "sifted"),
        (PilotStage, "pilot"),
        (PythiaStage, "pythia"),
        (TraceStage, "trace"),
    ):
        monkeypatch.setattr(
            stage,
            method_name,
            Mock(side_effect=AssertionError(f"{method_name} retrained during explore")),
        )

    result = built_instance_space.explore(metadata, dataset_id="training-replay")
    n_instances = metadata.features.shape[0]
    n_algorithms = len(built_instance_space.model.data.algo_labels)
    dimensions = built_instance_space.model.pilot.a.shape[0]

    assert result.y_hat is not None
    assert result.pr0_hat is not None
    assert result.selection0 is not None
    assert result.in_good is not None
    assert result.in_best is not None
    assert result.x.shape[0] == n_instances
    assert result.z.shape == (n_instances, dimensions)
    assert result.y_hat.shape == (n_instances, n_algorithms)
    assert result.pr0_hat.shape == (n_instances, n_algorithms)
    assert result.selection0.shape == (n_instances,)
    assert result.in_good.shape == (n_instances, n_algorithms)
    assert result.in_best.shape == (n_instances, n_algorithms)
    assert result.y_actual is not None
    assert result.y_actual.shape == (n_instances, n_algorithms)
    assert result.trace_out is not None
    assert len(result.trace_out.good) == n_algorithms
    assert len(result.trace_out.best) == n_algorithms


def test_rerunning_cloister_does_not_invalidate_pythias_output(
    built_instance_space: InstanceSpace,
) -> None:
    """Q8's literal check: PythiaStage and CloisterStage share a schedule wave.

    `_rollback_to_schedule_index()` invalidates every wave *after* the one
    being rolled back to - since Pythia and Cloister are in the same wave,
    rerunning Cloister must not touch Pythia's already-computed output at all.
    """
    runner = built_instance_space._runner
    svm_before = runner._available_arguments["svm"]
    y_hat_before = runner._available_arguments["y_hat"]

    built_instance_space.run_stage(CloisterStage)

    assert runner._available_arguments["svm"] is svm_before
    assert runner._available_arguments["y_hat"] is y_hat_before


def test_rerunning_cloister_leaves_trace_runnable(
    built_instance_space: InstanceSpace,
) -> None:
    """TraceStage has no real dependency on CloisterStage's output at all.

    (`TraceInputs` never references `z_edge`/`z_ecorr`.) After rerunning
    CloisterStage, `run_stage(TraceStage)` must still be immediately callable
    - not blocked behind re-running any earlier stage - confirming the
    wave-position invalidation this pipeline relies on doesn't wrongly gate a
    stage that Cloister was never a prerequisite for.
    """
    built_instance_space.run_stage(CloisterStage)

    trace_output = built_instance_space.run_stage(TraceStage)

    assert trace_output is not None
