# SPDX-License-Identifier: LicenseRef-PolyForm-Noncommercial-1.0.0
# Copyright (c) 2024-2026 Mario Andrés Muñoz
# ruff: noqa: SLF001
"""Tests for F9's explore()-time ground-truth evaluation (_explore_evaluate).

Covers the numerics `_explore_evaluate`/`_build_test_algo_matrix` add on top of
`compute_binary_performance` (F9's shared extraction, also unit-tested directly in
`test_build_prelim.py`): reindexing a test set's `algo_*` columns to the trained
algorithm order (case-insensitively), padding a training algorithm absent from the
test set with NaN (and reporting NaN metrics for it, not a confusion matrix computed
against a fabricated label), and deriving accuracy/precision/recall/confusion-matrix
from PYTHIA's predictions against that ground truth. Orchestration (when
`ExploreStage.EVALUATION` is yielded at all) is covered by `test_explore_stage_iter.py`.
"""

from unittest.mock import Mock

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from instancespace.data.metadata import Metadata
from instancespace.data.options import InstanceSpaceOptions
from instancespace.instance_space import InstanceSpace


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


def _make_space(algo_labels: list[str]) -> InstanceSpace:
    space = InstanceSpace.__new__(InstanceSpace)
    model = Mock()
    model.data = Mock()
    model.data.algo_labels = algo_labels
    space._model = model
    space._options = InstanceSpaceOptions.default(*([None] * 12))
    space._require_model = Mock(return_value=model)  # type: ignore[method-assign]
    return space


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


def test_explore_evaluate_computes_metrics_against_ground_truth() -> None:
    """Perfect classifier predictions against ground truth -> perfect metrics.

    Default `PerformanceOptions` (`max_perf=False`, `abs_perf=True`,
    `epsilon=0.20`): "good" means performance <= 0.20. Instance 0 is good for
    Alg1 only; instance 1 is good for Alg2 only. `y_hat` (PYTHIA's
    predictions, already computed - not recomputed by `_explore_evaluate`)
    matches the ground truth exactly.
    """
    space = _make_space(["Alg1", "Alg2"])
    test_metadata = _make_metadata(
        ["Alg1", "Alg2"],
        np.array([[0.1, 5.0], [5.0, 0.1]]),
    )
    y_hat = np.array([[True, False], [False, True]])

    result = space._explore_evaluate(test_metadata, y_hat, [])

    np.testing.assert_array_equal(
        result.y_actual,
        np.array([[True, False], [False, True]]),
    )
    np.testing.assert_allclose(result.accuracy_actual, [1.0, 1.0])
    np.testing.assert_allclose(result.precision_actual, [1.0, 1.0])
    np.testing.assert_allclose(result.recall_actual, [1.0, 1.0])
    # cvcmat columns are [tn, fp, fn, tp]; each algo has one good, one bad
    # instance, both correctly predicted -> tn=1, fp=0, fn=0, tp=1.
    np.testing.assert_allclose(result.cvcmat_actual, [[1, 0, 0, 1], [1, 0, 0, 1]])
    assert result.algo_labels == ["Alg1", "Alg2"]


def test_explore_evaluate_reports_nan_for_algorithm_without_ground_truth() -> None:
    """An algorithm absent from the test set gets NaN metrics, not a fabricated one."""
    space = _make_space(["Alg1", "Alg2"])
    # Only Alg1 has ground truth in the test set.
    test_metadata = _make_metadata(["Alg1"], np.array([[0.1], [5.0]]))
    y_hat = np.array([[True, True], [False, True]])

    result = space._explore_evaluate(test_metadata, y_hat, [])

    assert not np.isnan(result.accuracy_actual[0])
    assert np.isnan(result.accuracy_actual[1])
    assert np.isnan(result.precision_actual[1])
    assert np.isnan(result.recall_actual[1])
    assert np.all(np.isnan(result.cvcmat_actual[1]))


def test_explore_evaluate_imperfect_predictions_match_hand_computed_confusion() -> None:
    """A deliberately imperfect prediction reproduces sklearn's own confusion matrix.

    Cross-checks `_explore_evaluate`'s accuracy/precision/recall against an
    independently computed `sklearn.metrics.confusion_matrix` on the same
    ground truth/prediction pair, rather than trusting the implementation's
    own arithmetic.
    """
    from sklearn.metrics import confusion_matrix

    space = _make_space(["Alg1"])
    # 4 instances, "good" iff performance <= 0.20 (default epsilon).
    test_metadata = _make_metadata(
        ["Alg1"],
        np.array([[0.1], [0.1], [5.0], [5.0]]),
    )
    y_hat = np.array([[True], [False], [False], [True]])  # 1 correct, 1 wrong each way

    result = space._explore_evaluate(test_metadata, y_hat, [])

    y_true = np.array([True, True, False, False])
    y_pred = y_hat[:, 0]
    expected_cm = confusion_matrix(y_true, y_pred, labels=[False, True]).ravel()

    np.testing.assert_allclose(result.cvcmat_actual[0], expected_cm)
    expected_accuracy = (y_true == y_pred).mean()
    np.testing.assert_allclose(result.accuracy_actual[0], expected_accuracy)


def test_explore_evaluate_new_algorithm_full_parity() -> None:
    """Full MATLAB parity (F9): a new algorithm participates in y_best/p/beta.

    It still reports NaN accuracy/precision/recall/cvcmat (no trained
    classifier exists to score it against its own, real ground truth) -
    matching
    MATLAB's `PYTHIAevalMode` "no CV model" convention.
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

    assert result.algo_labels == ["Alg1", "BrandNew"]
    assert result.y_actual.shape == (2, 2)
    # Instance 0's best algorithm is BrandNew (index 1, 0-based -> p=2, 1-based).
    assert result.p_actual[0] == 2  # noqa: PLR2004
    assert not np.isnan(result.accuracy_actual[0])  # Alg1: real classifier
    assert np.isnan(result.accuracy_actual[1])  # BrandNew: no classifier
    assert np.isnan(result.precision_actual[1])
    assert np.isnan(result.recall_actual[1])
    assert np.all(np.isnan(result.cvcmat_actual[1]))
