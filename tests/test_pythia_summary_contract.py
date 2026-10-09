# SPDX-License-Identifier: LicenseRef-PolyForm-Noncommercial-1.0.0
# Copyright (c) 2024-2026 Mario Andrés Muñoz
# ruff: noqa: SLF001
"""Hand-computable summary and regret contracts from current MATLAB master."""

import numpy as np
import pandas as pd
import pytest
from numpy.typing import NDArray

from instancespace.data.options import GeneralOptions, ParallelOptions, PythiaOptions
from instancespace.stages.pythia import PythiaStage


def _summary(
    y: NDArray[np.double],
    labels: NDArray[np.bool_],
    selection: NDArray[np.int_],
    fallback: NDArray[np.int_] | None = None,
) -> pd.DataFrame:
    nalgos = y.shape[1]
    return PythiaStage._generate_summary(
        nalgos=nalgos,
        algo_labels=[f"a{i}" for i in range(nalgos)],
        y=y,
        y_hat=labels,
        y_bin=labels,
        y_best=np.zeros(y.shape[0]),
        selection0=selection,
        selection1=selection if fallback is None else fallback,
        accuracy=[np.nan] * nalgos,
        precision=[np.nan] * nalgos,
        recall=[np.nan] * nalgos,
        box_consnt=[np.nan] * nalgos,
        k_scale=[np.nan] * nalgos,
        param1_label=None,
        param2_label=None,
    ).set_index("Algorithms")


@pytest.mark.filterwarnings("error::RuntimeWarning")
def test_summary_observed_denominators_and_disjoint_recall() -> None:
    """Unobserved selections are not failures; successes are never also misses."""
    y = np.array(
        [
            [1, 1, np.nan],
            [2, 1, np.nan],
            [2, 2, np.nan],
            [np.nan, 1, np.nan],
            [np.nan, np.nan, np.nan],
        ],
        dtype=np.double,
    )
    # Deliberately true labels at missing outcomes must not create successes.
    labels = np.array(
        [
            [True, True, True],
            [False, True, True],
            [False, False, True],
            [True, True, True],
            [True, True, True],
        ],
    )
    before = y.copy()
    result = _summary(y, labels, np.array([0, 0, 0, 0, 0]))
    np.testing.assert_allclose(
        result["Probability_of_good"],
        [0.333, 0.75, np.nan, 0.75, 0.333],
        equal_nan=True,
    )
    assert result.loc["Selector", "CV_model_precision"] == 33.3
    assert result.loc["Selector", "CV_model_recall"] == 33.3
    np.testing.assert_array_equal(y, before)


@pytest.mark.parametrize("good", [True, False])
def test_summary_complete_all_good_or_all_bad(good: bool) -> None:
    result = _summary(
        np.ones((2, 2)),
        np.full((2, 2), good),
        np.array([0, 1]),
    )
    np.testing.assert_array_equal(result["Probability_of_good"], float(good))
    assert result.loc["Selector", "CV_model_precision"] == (100.0 if good else 0.0)
    if good:
        assert result.loc["Selector", "CV_model_recall"] == 100.0
    else:
        assert pd.isna(result.loc["Selector", "CV_model_recall"])


@pytest.mark.filterwarnings("error::RuntimeWarning")
@pytest.mark.parametrize("nrows", [0, 2])
def test_summary_no_observations(nrows: int) -> None:
    result = _summary(
        np.full((nrows, 2), np.nan),
        np.ones((nrows, 2), dtype=np.bool_),
        np.zeros(nrows, dtype=np.int_),
    )
    assert result["Probability_of_good"].isna().all()
    assert pd.isna(result.loc["Selector", "CV_model_precision"])
    assert pd.isna(result.loc["Selector", "CV_model_recall"])


def test_summary_fallback_probability_does_not_credit_abstention() -> None:
    result = _summary(
        np.ones((2, 2)),
        np.ones((2, 2), dtype=np.bool_),
        np.array([-1, -1]),
        np.array([0, 1]),
    )
    assert result.loc["Selector", "Probability_of_good"] == 1.0
    assert pd.isna(result.loc["Selector", "CV_model_precision"])
    assert result.loc["Selector", "CV_model_recall"] == 0.0
    without_fallback = _summary(
        np.ones((2, 2)),
        np.ones((2, 2), dtype=np.bool_),
        np.array([-1, -1]),
    )
    assert pd.isna(without_fallback.loc["Selector", "Probability_of_good"])


@pytest.mark.parametrize(
    ("outcomes", "best", "expected"),
    [
        ([[1.0, 4.0], [10.0, 12.0]], [1.0, 10.0], [[2.0, 3.0], [2.0, 2.0]]),
        ([[1.0, 4.0], [10.0, 12.0]], [4.0, 12.0], [[3.0, 2.0], [2.0, 2.0]]),
        ([[1.0, 1.0], [10.0, 10.0]], [1.0, 10.0], [[1.0, 1.0], [1.0, 1.0]]),
        (
            [[np.nan, np.nan], [np.nan, np.nan]],
            [np.nan, np.nan],
            [[1.0, 1.0], [1.0, 1.0]],
        ),
        ([[1.0, np.nan], [10.0, 12.0]], [1.0, 10.0], [[2.0, 2.0], [2.0, 2.0]]),
    ],
)
def test_training_uses_per_instance_regret(
    outcomes: list[list[float]],
    best: list[float],
    expected: list[list[float]],
) -> None:
    """Exercise weight plumbing without numerical fitting via constant labels."""
    y = np.array(outcomes)
    y_before = y.copy()
    result = PythiaStage.pythia(
        np.array([[0.0, 0.0], [1.0, 1.0]]),
        y,
        np.ones((2, 2), dtype=np.bool_),
        np.array(best),
        ["a", "b"],
        PythiaOptions.default(
            cv_folds=2,
            use_weights=True,
            tuning="none",
            params=np.ones((2, 2)),
        ),
        ParallelOptions.default(),
        GeneralOptions(verbose=False, seed=0),
    )
    np.testing.assert_array_equal(result.w, expected)
    np.testing.assert_array_equal(y, y_before)
