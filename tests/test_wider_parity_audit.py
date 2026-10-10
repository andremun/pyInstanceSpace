# ruff: noqa: D103, PLR2004, SLF001
"""Controlled end-to-end regressions for the current-master parity audit."""

from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from numpy.typing import NDArray

from instancespace.data.metadata import Metadata
from instancespace.data.options import (
    InstanceSpaceOptions,
    ParallelOptions,
    PilotOptions,
    PythiaOptions,
    SelvarsOptions,
    SiftedOptions,
)
from instancespace.instance_space import InstanceSpace
from instancespace.model import Model


def _metadata() -> Metadata:
    rng = np.random.default_rng(72)
    x = rng.normal(size=(32, 5))
    y = np.column_stack((0.1 + 0.1 * x[:, 0] ** 2, 0.1 + 0.1 * x[:, 2] ** 2))
    return Metadata(
        ["a", "unused", "b", "c", "washed"],
        ["one", "two"],
        pd.Series([f"i{i}" for i in range(len(x))]),
        None,
        x,
        y,
    )


def _space(metadata: Metadata, *, manual: bool) -> InstanceSpace:
    opts = InstanceSpaceOptions.default(
        parallel=ParallelOptions.default(flag=False),
        selvars=SelvarsOptions.default(
            feats=["a", "b", "c"] if manual else None,
            small_scale_flag=False,
        ),
        sifted=SiftedOptions.default(flag=False),
        pilot=PilotOptions.default(method="pls"),
        pythia=PythiaOptions.default(tuning="none", cv_folds=2, params=np.ones((2, 2))),
    )
    space = InstanceSpace(metadata, opts)
    space.build()
    return space


@pytest.mark.parametrize("manual", [True, False])
def test_feature_replay_after_training_column_removal(
    tmp_path: Path,
    manual: bool,
) -> None:
    metadata = _metadata()
    if not manual:
        x = metadata.features.copy()
        x[:, [1, 4]] = np.nan
        metadata = replace(metadata, features=x)
    space = _space(metadata, manual=manual)
    # Supply only retained columns, reordered by name. Omitted training columns
    # were never consumed by fitted PRELIM and must not be required at inference.
    query = replace(
        metadata,
        feature_names=["c", "a", "b"],
        features=metadata.features[:, [3, 0, 2]],
        algorithm_names=[],
        algorithms=np.empty((32, 0)),
    )
    result = space.explore(query)
    np.testing.assert_allclose(result.z, space.model.pilot.z, atol=1e-12)
    assert space.model.preprocessing_features == ("a", "b", "c")
    np.testing.assert_allclose(space.explore(metadata).z, result.z, atol=1e-12)
    path = tmp_path / "model.joblib"
    space.model.save(path)
    space._model = Model.load(path)
    np.testing.assert_allclose(space.explore(query).z, result.z, atol=1e-12)


def test_predictions_are_independent_of_test_outcomes() -> None:
    metadata = _metadata()
    space = _space(metadata, manual=False)
    without = replace(metadata, algorithm_names=[], algorithms=np.empty((32, 0)))
    altered = replace(metadata, algorithms=metadata.algorithms[:, ::-1] * 100)
    missing = replace(metadata, algorithms=np.full((32, 2), np.nan))
    extra = replace(
        metadata,
        algorithm_names=["one", "two", "new"],
        algorithms=np.column_stack((altered.algorithms, np.zeros(32))),
    )
    expected = space.explore(without)
    for query in [metadata, altered, missing, extra]:
        actual = space.explore(query)
        assert actual.y_hat is not None
        assert expected.y_hat is not None
        assert actual.pr0_hat is not None
        assert expected.pr0_hat is not None
        assert actual.selection0 is not None
        assert expected.selection0 is not None
        assert actual.in_good is not None
        assert expected.in_good is not None
        assert actual.in_best is not None
        assert expected.in_best is not None
        np.testing.assert_array_equal(actual.z, expected.z)
        np.testing.assert_array_equal(actual.y_hat[:, :2], expected.y_hat)
        np.testing.assert_array_equal(actual.pr0_hat[:, :2], expected.pr0_hat)
        np.testing.assert_array_equal(actual.selection0, expected.selection0)
        np.testing.assert_array_equal(actual.in_good[:, :2], expected.in_good)
        np.testing.assert_array_equal(actual.in_best[:, :2], expected.in_best)


def test_manual_portfolio_selection_precedes_washing_and_labels() -> None:
    from instancespace.data.options import GeneralOptions, PrelimOptions
    from instancespace.stages.prelim import PrelimInput, PrelimStage
    from instancespace.stages.preprocessing import (
        PreprocessingInput,
        PreprocessingStage,
    )

    metadata = _metadata()
    y = metadata.algorithms.copy()
    y[0] = [np.nan, 1]
    y[1] = [2, np.nan]
    metadata = replace(metadata, algorithms=y)
    opts = SelvarsOptions.default(algos=["one"])
    source = PreprocessingInput(
        metadata.feature_names,
        metadata.algorithm_names,
        metadata.instance_labels,
        None,
        metadata.features,
        metadata.algorithms,
        opts,
    )
    selected = PreprocessingStage._run(source)
    isolated = PreprocessingStage._run(
        source._replace(
            algorithm_names=["one"],
            algorithms=y[:, :1],
            selvars_options=SelvarsOptions.default(),
        ),
    )
    assert selected.inst_labels.tolist() == metadata.instance_labels.iloc[1:].tolist()
    assert selected.algo_labels == ["one"]
    outputs = []
    for value in [selected, isolated]:
        outputs.append(
            PrelimStage._run(
                PrelimInput(
                    value.x,
                    value.y,
                    value.x_raw,
                    value.y_raw,
                    value.s,
                    value.inst_labels,
                    PrelimOptions.default(),
                    SelvarsOptions.default(),
                    GeneralOptions.default(),
                    value.algo_labels,
                ),
            ),
        )
    for name in ["x", "y", "y_bin", "y_best", "p", "num_good_algos", "beta"]:
        np.testing.assert_array_equal(
            getattr(outputs[0], name),
            getattr(outputs[1], name),
        )


def test_sifted_cache_isolated_between_runs_and_decodes_original_columns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from instancespace.data.options import GeneralOptions
    from instancespace.stages.sifted import SiftedStage

    rng = np.random.default_rng(19)
    x = rng.normal(size=(30, 6))
    y = rng.normal(size=(30, 1))
    run_scores: list[float] = []
    caches: list[dict[bytes, float]] = []

    class ControlledGA:
        """Fix chromosomes while exercising production fitness and cache wiring."""

        cost_cache: dict[bytes, float]
        selfx: NDArray[np.double]

        def __init__(self, **kwargs: Any) -> None:  # noqa: ANN401
            self.fitness = kwargs["fitness_func"]

        def run(self) -> None:
            assert self.cost_cache == {}
            self.score = self.fitness(self, np.array([0, 0]), 0)
            # Corrupting inputs after the first call makes an accidental cache
            # miss observable, without relying on timing or optimizer output.
            original = self.selfx
            self.selfx = np.full_like(original, np.nan)
            assert self.fitness(self, np.array([0, 0]), 1) == self.score
            self.selfx = original
            caches.append(self.cost_cache)
            run_scores.append(self.score)

        def best_solution(self) -> tuple[NDArray[np.int_], float, int]:
            return np.array([0, 0]), self.score, 0

    monkeypatch.setattr("instancespace.stages.sifted.pygad.GA", ControlledGA)
    # Same feature bitmask, but different labels between runs must be rescored.
    for good in [
        np.ones((30, 1), dtype=bool),
        (x[:, :1] > 0),
        np.ones((30, 1), dtype=bool),
    ]:
        stage = SiftedStage(
            x,
            y,
            good,
            x.copy(),
            y.copy(),
            np.zeros(30, dtype=bool),
            good.sum(axis=1),
            y[:, 0],
            np.ones(30, dtype=int),
            pd.Series(range(30)),
            None,
            [f"f{i}" for i in range(6)],
            SiftedOptions.default(k=2),
            ParallelOptions.default(flag=False),
            GeneralOptions.default(),
        )
        selected, indices = stage._find_best_combination(
            x[:, [1, 3, 5]],
            np.array([[True, False], [True, False], [False, True]]),
            np.array([1, 3, 5]),
            np.random.default_rng(12),
        )
        np.testing.assert_array_equal(indices, [1, 5])
        np.testing.assert_array_equal(selected, x[:, [1, 5]])
    assert run_scores[0] == run_scores[2] == 1
    assert run_scores[1] < 1
    assert len({id(cache) for cache in caches}) == 3


@pytest.mark.parametrize("maximize", [False, True])
def test_automatic_pruning_refits_retained_portfolio(maximize: bool) -> None:
    from instancespace.data.options import PerformanceOptions

    metadata = _metadata()
    rng = np.random.default_rng(5)
    retained = rng.uniform(0.1, 5, size=(32, 2))
    retained[0] = [4, 5]  # Removed algorithm wins here, but is never good.
    retained[1] = [0.5, 0.5]  # Tie and beta must use the retained portfolio.
    y = np.column_stack((retained, np.full(32, 2.0)))
    if maximize:
        y = 6 - y
    metadata = replace(
        metadata,
        algorithm_names=["one", "never", "two"],
        algorithms=y[:, [0, 2, 1]],
    )
    opts = InstanceSpaceOptions.default(
        parallel=ParallelOptions.default(flag=False),
        perf=PerformanceOptions.default(
            max_perf=maximize,
            abs_perf=True,
            epsilon=5 if maximize else 1,
            beta_threshold=0.75,
        ),
        sifted=SiftedOptions.default(flag=False),
        pilot=PilotOptions.default(method="pls"),
        pythia=PythiaOptions.default(tuning="none", cv_folds=2, params=np.ones((2, 2))),
    )
    full = InstanceSpace(metadata, opts)
    actual = full.build()
    expected = InstanceSpace(
        replace(metadata, algorithm_names=["one", "two"], algorithms=y[:, :2]),
        opts,
    ).build()
    assert actual.data.algo_labels == ["one", "two"]
    for name in ["x", "y", "y_raw", "y_bin", "y_best", "p", "beta", "num_good_algos"]:
        np.testing.assert_array_equal(
            getattr(actual.data, name),
            getattr(expected.data, name),
        )
    for name in ["min_y", "lambda_y", "mu_y", "sigma_y"]:
        np.testing.assert_array_equal(
            getattr(actual.prelim, name),
            getattr(expected.prelim, name),
        )
    assert actual.data.p[0] == 1
    assert actual.data.beta[1]
    # The discarded algorithm is reconciled as test-only, never recommended.
    result = full.explore(metadata)
    assert result.algo_labels == ["one", "two", "never"]
    assert result.y_hat is not None
    assert result.selection0 is not None
    assert not result.y_hat[:, 2].any()
    assert not (result.selection0 == 2).any()


def test_automatic_pruning_rejects_empty_portfolio() -> None:
    metadata = _metadata()
    metadata = replace(metadata, algorithms=np.full((32, 2), 100.0))
    with pytest.raises(ValueError, match="no good algorithms"):
        _space(metadata, manual=False)
