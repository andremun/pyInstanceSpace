# SPDX-License-Identifier: LicenseRef-PolyForm-Noncommercial-1.0.0
# Copyright (c) 2024-2026 Mario Andrés Muñoz
# ruff: noqa: SLF001
"""Tests for stage-owned PYTHIA ground-truth evaluation.

Covers the numerics `_explore_evaluate`/`_build_test_algo_matrix` add on top of
`compute_binary_performance`: reindexing a test set's `algo_*` columns to the trained
algorithm order, preserving MATLAB's all-false truth for an absent trained algorithm,
and deriving accuracy/precision/recall/confusion matrices from fitted-slot presence
and PYTHIA predictions. Orchestration is covered by `test_explore_stage_iter.py`.
"""

import warnings
from typing import cast
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
from numpy.typing import NDArray

from instancespace.data.metadata import Metadata
from instancespace.data.model import PythiaOut
from instancespace.data.options import InstanceSpaceOptions
from instancespace.instance_space import InstanceSpace
from instancespace.stages.pythia import (
    PythiaEvaluateInput,
    PythiaEvaluateOutput,
    PythiaStage,
)


def _make_metadata(
    algo_names: list[str],
    algo_perf: NDArray[np.double],
) -> Metadata:
    n_inst = algo_perf.shape[0]
    return Metadata(
        feature_names=["f1"],
        algorithm_names=algo_names,
        instance_labels=pd.Series([f"i{i}" for i in range(n_inst)]),
        instance_sources=None,
        features=np.zeros((n_inst, 1)),
        algorithms=algo_perf,
    )


def _fitted_with_slots(slots: list[object | None]) -> PythiaOut:
    """Create the fitted-state fields needed by stage-owned evaluation."""
    fitted = Mock(spec=PythiaOut)
    fitted.svm = slots
    fitted.accuracy = [1.0] * len(slots)
    fitted.precision = [1.0] * len(slots)
    fitted.recall = [1.0] * len(slots)
    return cast(PythiaOut, fitted)


def _make_space(algo_labels: list[str]) -> InstanceSpace:
    space = InstanceSpace.__new__(InstanceSpace)
    model = Mock()
    model.data = Mock()
    model.data.algo_labels = algo_labels
    model.pythia = _fitted_with_slots([Mock() for _ in algo_labels])
    space._model = model
    space._options = InstanceSpaceOptions.default(*([None] * 12))
    space._require_model = Mock(return_value=model)  # type: ignore[method-assign]
    return space


def _evaluate_pythia(
    y_true: NDArray[np.bool_],
    y_pred: NDArray[np.bool_],
    slots: list[object | None] | None = None,
) -> PythiaEvaluateOutput:
    """Call the stage-owned MATLAB confusion-count formulas."""
    fitted = _fitted_with_slots(
        [Mock() for _ in range(y_pred.shape[1])] if slots is None else slots,
    )
    return PythiaStage.evaluate(
        PythiaEvaluateInput(
            y_true=y_true,
            y_pred=y_pred,
        ),
        fitted,
    )


def test_build_test_algo_matrix_reindexes_case_insensitively() -> None:
    """Case-insensitive name matching mirrors MATLAB's `strcmpi`."""
    space = _make_space(["Alg1", "Alg2"])
    # Test set's columns are in the opposite order and different case.
    test_metadata = _make_metadata(
        ["alg2", "alg1"],
        np.array([[10.0, 20.0], [30.0, 40.0]]),
    )

    y_raw, has_gt = space._build_test_algo_matrix(
        test_metadata,
        ["Alg1", "Alg2"],
        [],
    )

    # Column 0 (Alg1) should come from the test set's "alg1" column (index 1).
    np.testing.assert_allclose(y_raw[:, 0], [20.0, 40.0])
    # Column 1 (Alg2) should come from the test set's "alg2" column (index 0).
    np.testing.assert_allclose(y_raw[:, 1], [10.0, 30.0])
    assert np.all(has_gt)


def test_build_test_algo_matrix_nans_algorithm_absent_from_test_set() -> None:
    """An algorithm in training but absent from the test set is "not evaluated".

    Per the roadmap F9 pathway: it becomes an all-NaN column rather than
    aborting evaluation for the other, present algorithms.
    """
    space = _make_space(["Alg1", "Alg2"])
    test_metadata = _make_metadata(["Alg1"], np.array([[10.0], [30.0]]))

    y_raw, has_gt = space._build_test_algo_matrix(
        test_metadata,
        ["Alg1", "Alg2"],
        [],
    )

    np.testing.assert_allclose(y_raw[:, 0], [10.0, 30.0])
    assert np.all(np.isnan(y_raw[:, 1]))
    assert has_gt.tolist() == [True, False]


def test_build_test_algo_matrix_excludes_new_algorithm_when_not_requested() -> None:
    """A test-set-only algorithm is excluded unless named in `new_algo_labels`.

    `_build_test_algo_matrix` doesn't decide what's "new" itself - that's
    `_find_new_algorithms`'s job - so passing an empty `new_algo_labels`
    here (as if the caller chose not to widen) must not include it, and the
    returned width must match `algo_labels` alone.
    """
    space = _make_space(["Alg1"])
    test_metadata = _make_metadata(
        ["Alg1", "BrandNewAlgo"],
        np.array([[10.0, 99.0], [30.0, 99.0]]),
    )

    y_raw, has_gt = space._build_test_algo_matrix(test_metadata, ["Alg1"], [])

    assert y_raw.shape == (2, 1)
    np.testing.assert_allclose(y_raw[:, 0], [10.0, 30.0])
    assert has_gt.tolist() == [True]


def test_build_test_algo_matrix_appends_new_algorithm_columns() -> None:
    """A test-set-only algorithm named in `new_algo_labels` is appended.

    Full MATLAB parity (F9): matches MATLAB's `Yaux` widening in
    `evaluateTestSet` - the new algorithm's real performance data is placed
    in an extra trailing column, not dropped.
    """
    space = _make_space(["Alg1"])
    test_metadata = _make_metadata(
        ["Alg1", "BrandNewAlgo"],
        np.array([[10.0, 99.0], [30.0, 88.0]]),
    )

    y_raw, has_gt = space._build_test_algo_matrix(
        test_metadata,
        ["Alg1"],
        ["BrandNewAlgo"],
    )

    assert y_raw.shape == (2, 2)
    np.testing.assert_allclose(y_raw[:, 0], [10.0, 30.0])
    np.testing.assert_allclose(y_raw[:, 1], [99.0, 88.0])
    assert has_gt.tolist() == [True]  # only covers the *trained* column


def test_find_new_algorithms_case_insensitive_and_deduplicated() -> None:
    """The internal helper remains defensive against duplicate legacy metadata."""
    space = _make_space(["Alg1"])
    # Public Metadata construction now rejects this ambiguous schema. A mock
    # keeps coverage for legacy objects loaded from an older checkpoint.
    test_metadata = Mock(spec=Metadata)
    test_metadata.algorithm_names = ["alg1", "BrandNew", "brandnew"]

    new_algos = space._find_new_algorithms(test_metadata, ["Alg1"])

    assert new_algos == ["BrandNew"]


def test_pythia_evaluate_computes_metrics_against_ground_truth() -> None:
    """Perfect classifier predictions against ground truth -> perfect metrics.

    Default `PerformanceOptions` (`max_perf=False`, `abs_perf=True`,
    `epsilon=0.20`): "good" means performance <= 0.20. Instance 0 is good for
    Alg1 only; instance 1 is good for Alg2 only. `y_hat` (PYTHIA's
    predictions, already computed - not recomputed by `_explore_evaluate`)
    matches the ground truth exactly.
    """
    y_true = np.array([[True, False], [False, True]])
    y_hat = np.array([[True, False], [False, True]])

    result = _evaluate_pythia(y_true, y_hat)

    np.testing.assert_allclose(result.accuracy, [1.0, 1.0])
    np.testing.assert_allclose(result.precision, [1.0, 1.0])
    np.testing.assert_allclose(result.recall, [1.0, 1.0])
    # MATLAB stores cm(:)' in column-major order: [tn, fn, fp, tp].
    # Each algorithm has one instance of each class, both correctly predicted.
    np.testing.assert_allclose(result.cvcmat, [[1, 0, 0, 1], [1, 0, 0, 1]])


def test_pythia_evaluate_scores_trained_algorithm_without_test_truth() -> None:
    """MATLAB scores a fitted slot against its reconciled all-false truth column."""
    y_true = np.array([[True, False], [False, False]])
    y_hat = np.array([[True, True], [False, True]])

    result = _evaluate_pythia(y_true, y_hat)

    assert not np.isnan(result.accuracy[0])
    assert result.accuracy[1] == 0.0
    assert result.precision[1] == 0.0
    assert np.isnan(result.recall[1])
    np.testing.assert_array_equal(result.cvcmat[1], [0, 0, 2, 0])


def test_pythia_evaluate_uses_matlab_column_major_confusion_order() -> None:
    """An asymmetric confusion matrix distinguishes MATLAB's stored column order.

    Cross-checks `_explore_evaluate`'s accuracy/precision/recall against an
    independently computed `sklearn.metrics.confusion_matrix` on the same
    ground truth/prediction pair, rather than trusting the implementation's
    own arithmetic.
    """
    from sklearn.metrics import confusion_matrix

    y_true = np.array([[False], [False], [False], [True]])
    y_hat = np.array([[False], [True], [True], [True]])

    result = _evaluate_pythia(y_true, y_hat)

    y_true_vector = y_true[:, 0]
    y_pred = y_hat[:, 0]
    tn, fp, fn, tp = confusion_matrix(
        y_true_vector,
        y_pred,
        labels=[False, True],
    ).ravel()
    expected_cm = np.array([tn, fn, fp, tp])

    np.testing.assert_allclose(result.cvcmat[0], expected_cm)
    expected_accuracy = (y_true_vector == y_pred).mean()
    np.testing.assert_allclose(result.accuracy[0], expected_accuracy)


def test_pythia_evaluate_empty_trained_slot_matches_matlab_skip() -> None:
    """An empty model slot keeps zero counts, zero accuracy, and undefined rates."""
    y_true = np.array([[True], [False]])
    y_pred = np.zeros((2, 1), dtype=np.bool_)

    result = _evaluate_pythia(y_true, y_pred, slots=[None])

    np.testing.assert_array_equal(result.cvcmat[0], [0, 0, 0, 0])
    assert result.accuracy[0] == 0.0
    assert np.isnan(result.precision[0])
    assert np.isnan(result.recall[0])


def test_pythia_evaluate_undefined_rates_are_nan_without_warnings() -> None:
    """MATLAB's 0/0 precision and recall are NaN, with no sklearn warning."""
    y_true = np.zeros((3, 1), dtype=np.bool_)
    y_pred = np.zeros((3, 1), dtype=np.bool_)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = _evaluate_pythia(y_true, y_pred)

    np.testing.assert_array_equal(result.cvcmat[0], [3, 0, 0, 0])
    assert result.accuracy[0] == 1.0
    assert np.isnan(result.precision[0])
    assert np.isnan(result.recall[0])


def test_pythia_evaluate_preserves_inputs() -> None:
    """Evaluation is a read-only calculation over truth and predictions."""
    y_true = np.array([[True], [False]])
    y_pred = np.array([[False], [False]])
    truth_before = y_true.copy()
    prediction_before = y_pred.copy()

    _evaluate_pythia(y_true, y_pred)

    np.testing.assert_array_equal(y_true, truth_before)
    np.testing.assert_array_equal(y_pred, prediction_before)


def test_instance_space_evaluate_wrapper_remains_compatible() -> None:
    """Keep the private wrapper compatible while orchestration migrates."""
    space = _make_space(["Alg1", "Alg2"])
    test_metadata = _make_metadata(
        ["Alg1", "Alg2"],
        np.array([[0.1, 5.0], [5.0, 0.1]]),
    )
    y_hat = np.array([[True, False], [False, True]])

    wrapped = space._explore_evaluate(test_metadata, y_hat, [])
    stage = _evaluate_pythia(wrapped.y_actual, y_hat)

    np.testing.assert_array_equal(wrapped.accuracy_actual, stage.accuracy)
    np.testing.assert_array_equal(wrapped.precision_actual, stage.precision)
    np.testing.assert_array_equal(wrapped.recall_actual, stage.recall)
    np.testing.assert_array_equal(wrapped.cvcmat_actual, stage.cvcmat)


@pytest.mark.parametrize(
    ("y_true", "y_pred", "slots"),
    [
        (
            np.zeros((2, 1), dtype=np.bool_),
            np.zeros((3, 1), dtype=np.bool_),
            [Mock()],
        ),
        (
            np.zeros((2, 1), dtype=np.bool_),
            np.zeros((2, 1), dtype=np.bool_),
            [Mock(), Mock()],
        ),
    ],
)
def test_pythia_evaluate_rejects_inconsistent_shapes(
    y_true: NDArray[np.bool_],
    y_pred: NDArray[np.bool_],
    slots: list[object | None],
) -> None:
    """Inconsistent truth/prediction metadata is rejected before scoring."""
    with pytest.raises(ValueError, match="matching shapes|more trained classifiers"):
        _evaluate_pythia(
            y_true,
            y_pred,
            slots,
        )


def test_explore_evaluate_new_algorithm_full_parity() -> None:
    """Full MATLAB parity (F9): a new algorithm participates in y_best/p/beta.

    It reports NaN rates because no trained classifier exists, while MATLAB's
    preallocated confusion row remains zero.
    """
    space = _make_space(["Alg1"])
    # Instance 0: BrandNew (0.05) beats Alg1 (5.0) outright -> best algorithm
    # for instance 0 should be the new algorithm (index 1), not Alg1.
    test_metadata = _make_metadata(
        ["Alg1", "BrandNew"],
        np.array([[5.0, 0.05], [0.1, 9.0]]),
    )
    # y_hat already widened by _explore_pythia(n_new_algos=1): BrandNew's
    # column is all-False (no classifier).
    y_hat = np.array([[True, False], [True, False]])

    result = space._explore_evaluate(test_metadata, y_hat, ["BrandNew"])
    stage_result = _evaluate_pythia(
        result.y_actual,
        y_hat,
        slots=[Mock()],
    )

    assert result.algo_labels == ["Alg1", "BrandNew"]
    assert result.y_actual.shape == (2, 2)
    # Instance 0's best algorithm is BrandNew (index 1, 0-based -> p=2, 1-based).
    assert result.p_actual[0] == 2
    assert not np.isnan(result.accuracy_actual[0])  # Alg1: real classifier
    assert np.isnan(result.accuracy_actual[1])  # BrandNew: no classifier
    assert np.isnan(result.precision_actual[1])
    assert np.isnan(result.recall_actual[1])
    np.testing.assert_array_equal(result.cvcmat_actual[1], [0, 0, 0, 0])
    np.testing.assert_array_equal(result.accuracy_actual, stage_result.accuracy)
    np.testing.assert_array_equal(result.precision_actual, stage_result.precision)
    np.testing.assert_array_equal(result.recall_actual, stage_result.recall)
    np.testing.assert_array_equal(result.cvcmat_actual, stage_result.cvcmat)
