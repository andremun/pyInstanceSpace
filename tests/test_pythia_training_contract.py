# ruff: noqa: D103, PLR2004, SLF001
"""Training summaries use held-out predictions and replayable normalization."""

from unittest.mock import patch

import numpy as np

from instancespace.data.options import GeneralOptions, ParallelOptions, PythiaOptions
from instancespace.stages.pythia import (
    PythiaStage,
    _ClassifierResult,
    _ConstantClassifier,
)


def test_constant_coordinate_uses_unit_scale() -> None:
    z = np.array([[2.0, 1.0], [2.0, 3.0], [2.0, 5.0]])
    mu, sigma, normalized = PythiaStage._compute_znorm(z)
    np.testing.assert_array_equal(normalized[:, 0], 0)
    assert sigma[0] == 1
    np.testing.assert_allclose(normalized, (z - mu) / sigma)


def test_summary_uses_out_of_fold_predictions() -> None:
    truth = np.array([True, True, False, False])
    held_out = np.array([True, False, True, False])
    result = _ClassifierResult(
        _ConstantClassifier(True),
        held_out,
        np.zeros(4),
        truth,
        np.zeros(4),
        1.0,
        1.0,
    )
    with patch.object(PythiaStage, "_fit_classifier", return_value=result):
        output = PythiaStage.pythia(
            np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0], [3.0, 1.0]]),
            np.ones((4, 1)),
            truth[:, None],
            np.ones(4),
            ["algorithm"],
            PythiaOptions.default(cv_folds=2, tuning="none", params=np.ones((1, 2))),
            ParallelOptions.default(),
            GeneralOptions.default(),
        )
    # In-sample labels are perfect; held-out labels have precision 1/2.
    np.testing.assert_array_equal(output.y_hat[:, 0], truth)
    summary = output.pythia_summary.set_index("Algorithms")
    assert summary.loc["Selector", "CV_model_precision"] == 50.0
