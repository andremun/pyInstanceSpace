# ruff: noqa: D103, PLR2004, SLF001
"""Tests for PRELIM stage's explore-time inference.

Unit tests verify ``PrelimStage.predict()`` with edge cases and error conditions,
independent of MATLAB reference data. The validation test loads MATLAB-trained
PRELIM parameters (bounds, Box-Cox lambda, z-score mu/sigma) and verifies that
stage inference reproduces MATLAB's per-feature transformations on the 235-instance
test set (threshold: max relative error < 1% - PRELIM is a deterministic pipeline
[bounding -> Box-Cox -> z-score], so Python should match MATLAB to floating-point
precision when fed the same parameters).
"""

from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any, cast
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
from numpy.typing import NDArray

from instancespace.data.model import PrelimOut
from instancespace.instance_space import InstanceSpace
from instancespace.stages.prelim import PrelimPredictInput, PrelimStage

REFERENCE_DIR = Path("tests/matlab_reference")
ARTIFACTS_DIR = REFERENCE_DIR / "training_artifacts"
OUTPUTS_DIR = REFERENCE_DIR / "explore_outputs"
INPUT_DIR = REFERENCE_DIR / "input"


def _predict_prelim(
    instance_space: InstanceSpace,
    x: NDArray[np.double],
) -> NDArray[np.double]:
    """Call the stage contract with the fitted state held by InstanceSpace."""
    options = cast(Any, instance_space)._options
    fitted = cast(PrelimOut, cast(Any, instance_space)._require_model().prelim)
    return PrelimStage.predict(
        PrelimPredictInput(
            x=x,
            auto_preproc=options.auto.preproc,
            bound_enabled=options.bound.flag,
            norm_enabled=options.norm.flag,
        ),
        fitted,
    )


@pytest.fixture()
def mock_prelim_params() -> PrelimOut:
    """Create mock PRELIM parameters for testing."""
    n_features = 3
    return PrelimOut(
        med_val=np.zeros(n_features),
        iq_range=np.zeros(n_features),
        hi_bound=np.array([10.0, 10.0, 10.0]),
        lo_bound=np.array([0.0, 0.0, 0.0]),
        min_x=np.array([0.0, 0.0, 0.0]),
        lambda_x=np.array([1.0, 1.0, 1.0]),  # No Box-Cox transform
        mu_x=np.array([5.0, 5.0, 5.0]),
        sigma_x=np.array([2.0, 2.0, 2.0]),
        min_y=0.0,
        lambda_y=np.array([]),
        sigma_y=np.array([]),
        mu_y=np.array([]),
    )


@pytest.fixture()
def mock_instance_space(mock_prelim_params: PrelimOut) -> InstanceSpace:
    """Create mock InstanceSpace with PRELIM parameters."""
    mock_is = Mock(spec=InstanceSpace)
    mock_is._model = Mock()
    mock_is._model.prelim = mock_prelim_params
    mock_is._require_model = Mock(return_value=mock_is._model)
    # All tests in this file exercise both steps (matches this fixture's
    # historical, pre-flag-check behaviour) - the bound=False/norm=False
    # cases have their own dedicated tests below.
    mock_is._options = Mock()
    mock_is._options.auto.preproc = True
    mock_is._options.bound.flag = True
    mock_is._options.norm.flag = True
    return mock_is


def test_prelim_basic_functionality(mock_instance_space: InstanceSpace) -> None:
    """Test basic PRELIM transformation."""
    # Input data: 5 instances, 3 features
    x_raw = np.array(
        [
            [5.0, 5.0, 5.0],
            [1.0, 2.0, 3.0],
            [9.0, 8.0, 7.0],
            [0.5, 5.5, 10.5],
            [5.0, 5.0, 5.0],
        ],
    )

    x_transformed = _predict_prelim(mock_instance_space, x_raw)

    # Check output shape
    assert x_transformed.shape == x_raw.shape
    assert x_transformed.shape == (5, 3)

    # Check output is float array
    assert x_transformed.dtype == np.float64


def test_prelim_preserves_input(mock_instance_space: InstanceSpace) -> None:
    """Test that PRELIM doesn't modify input array."""
    x_raw = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    x_raw_copy = x_raw.copy()

    _predict_prelim(mock_instance_space, x_raw)

    # Original input should be unchanged
    np.testing.assert_array_equal(x_raw, x_raw_copy)


def test_prelim_predict_preserves_fitted_state(
    mock_instance_space: InstanceSpace,
    mock_prelim_params: PrelimOut,
) -> None:
    """Inference reads fitted arrays without mutating them."""
    field_names = ("lo_bound", "hi_bound", "min_x", "lambda_x", "mu_x", "sigma_x")
    before = {name: getattr(mock_prelim_params, name).copy() for name in field_names}

    _predict_prelim(mock_instance_space, np.array([[1.0, 2.0, 3.0]]))

    for name, expected in before.items():
        np.testing.assert_array_equal(getattr(mock_prelim_params, name), expected)


def test_prelim_predict_does_not_run_training(
    mock_instance_space: InstanceSpace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Inference must not enter PRELIM's fitting path."""
    train = Mock(side_effect=AssertionError("PRELIM training was called"))
    monkeypatch.setattr(PrelimStage, "prelim", train)

    _predict_prelim(mock_instance_space, np.array([[1.0, 2.0, 3.0]]))

    train.assert_not_called()


def test_prelim_instance_space_wrapper_remains_compatible(
    mock_instance_space: InstanceSpace,
) -> None:
    """Keep the private wrapper compatible while orchestration migrates."""
    x_raw = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    expected = _predict_prelim(mock_instance_space, x_raw)
    actual = InstanceSpace._explore_prelim(mock_instance_space, x_raw)

    np.testing.assert_array_equal(actual, expected)


def test_prelim_instance_space_wrapper_delegates_to_stage(
    mock_instance_space: InstanceSpace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The compatibility wrapper contains no independent PRELIM science."""
    sentinel = np.array([[42.0]], dtype=np.double)
    predict = Mock(return_value=sentinel)
    monkeypatch.setattr(PrelimStage, "predict", predict)

    actual = InstanceSpace._explore_prelim(
        mock_instance_space,
        np.array([[1.0, 2.0, 3.0]], dtype=np.double),
    )

    assert actual is sentinel
    predict.assert_called_once()


def test_prelim_single_instance(mock_instance_space: InstanceSpace) -> None:
    """Test PRELIM with single instance."""
    x_raw = np.array([[5.0, 5.0, 5.0]])

    x_transformed = _predict_prelim(mock_instance_space, x_raw)

    assert x_transformed.shape == (1, 3)


def test_prelim_bounding(mock_instance_space: InstanceSpace) -> None:
    """Test that PRELIM applies bounding correctly."""
    # Values outside bounds [0, 10]
    x_raw = np.array(
        [[-5.0, 15.0, 5.0], [5.0, 5.0, 5.0]],  # Out of bounds  # Within bounds
    )

    x_transformed = _predict_prelim(mock_instance_space, x_raw)

    # All values should be finite (no inf/nan from out-of-bounds)
    assert np.all(np.isfinite(x_transformed))


def test_prelim_handles_nan_input(mock_instance_space: InstanceSpace) -> None:
    """Test PRELIM handles NaN values in input."""
    x_raw = np.array([[5.0, np.nan, 5.0], [1.0, 2.0, np.nan]])

    x_transformed = _predict_prelim(mock_instance_space, x_raw)

    # Output shape should be preserved
    assert x_transformed.shape == (2, 3)

    # NaN values should remain NaN (not transformed)
    assert np.isnan(x_transformed[0, 1])
    assert np.isnan(x_transformed[1, 2])


def test_prelim_all_nan_feature(mock_instance_space: InstanceSpace) -> None:
    """Test PRELIM with a feature that is all NaN."""
    x_raw = np.array([[5.0, np.nan, 5.0], [1.0, np.nan, 3.0], [9.0, np.nan, 7.0]])

    x_transformed = _predict_prelim(mock_instance_space, x_raw)

    # Output should have same NaN pattern
    assert np.all(np.isnan(x_transformed[:, 1]))
    # Other features should be transformed
    assert np.all(np.isfinite(x_transformed[:, 0]))
    assert np.all(np.isfinite(x_transformed[:, 2]))


def test_prelim_dimension_consistency(mock_instance_space: InstanceSpace) -> None:
    """Test PRELIM with various input dimensions."""
    # Test with different numbers of instances
    for n_instances in [1, 10, 100]:
        x_raw = np.random.default_rng().random((n_instances, 3)) * 10

        x_transformed = _predict_prelim(mock_instance_space, x_raw)

        assert x_transformed.shape == (n_instances, 3)


def test_prelim_deterministic(mock_instance_space: InstanceSpace) -> None:
    """Test that PRELIM is deterministic (same input → same output)."""
    x_raw = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    result1 = _predict_prelim(mock_instance_space, x_raw)
    result2 = _predict_prelim(mock_instance_space, x_raw)

    np.testing.assert_array_equal(result1, result2)


def test_prelim_numerical_stability(mock_instance_space: InstanceSpace) -> None:
    """Test PRELIM with very small and very large values."""
    x_raw = np.array([[1e-10, 1e10, 5.0], [5.0, 5.0, 5.0]])

    # Should not raise errors
    x_transformed = _predict_prelim(mock_instance_space, x_raw)

    # After bounding, values should be clipped to [0, 10]
    # After transformation, should be finite
    assert np.all(np.isfinite(x_transformed))


def _collect_warnings(fn: Callable[..., object], *args: object) -> list[str]:
    """Run fn(*args) and return the loguru WARNING-level messages it emitted."""
    from loguru import logger

    messages: list[str] = []
    sink_id = logger.add(
        lambda msg: messages.append(msg.record["message"]),
        level="WARNING",
    )
    try:
        fn(*args)
    finally:
        logger.remove(sink_id)
    return messages


def test_prelim_ood_warning_fires_above_threshold(
    mock_instance_space: InstanceSpace,
) -> None:
    """Regression test for #250: explore() warns when >5% of instances are clipped."""
    # Bounds are [0, 10] on all 3 features (mock_prelim_params). 2 of 10 instances
    # (20%) have a feature outside bounds -> above the 5% threshold.
    x_raw = np.vstack(
        [
            np.full((8, 3), 5.0),
            np.array([[-5.0, 5.0, 5.0], [5.0, 15.0, 5.0]]),
        ],
    )

    messages = _collect_warnings(_predict_prelim, mock_instance_space, x_raw)

    assert any("clipped" in m for m in messages)


def test_prelim_ood_warning_silent_below_threshold(
    mock_instance_space: InstanceSpace,
) -> None:
    """No warning when no instance needs clipping."""
    x_raw = np.full((10, 3), 5.0)  # well within [0, 10] on all features

    messages = _collect_warnings(_predict_prelim, mock_instance_space, x_raw)

    assert messages == []


def test_prelim_skips_clipping_when_bound_flag_is_false(
    mock_instance_space: InstanceSpace,
) -> None:
    """Regression: explore() must not clip when the trained model used bound=False.

    Previously `_explore_prelim` always clipped to `lo_bound`/`hi_bound`
    regardless of `BoundOptions.flag` - wrong whenever a model was trained
    with bounding disabled, since values genuinely outside the training
    range would then be silently clamped anyway.

    MATLAB still clamps the min-shifted Box-Cox input to one, independently
    of raw bound clipping. With this fixture's lambda=1, min_x=0, mu=5 and
    sigma=2, that makes the expected raw-domain value ``max(x, 0)`` before
    z-scoring.
    """
    mock_instance_space._options.bound.flag = False  # type: ignore[misc]
    x_raw = np.array([[-0.5, 15.0, 5.0], [5.0, 5.0, 5.0]])

    messages = _collect_warnings(
        _predict_prelim,
        mock_instance_space,
        x_raw,
    )
    result = _predict_prelim(mock_instance_space, x_raw)

    assert messages == []  # no OOD warning either, since nothing was clipped
    expected = (np.maximum(x_raw, 0.0) - 5.0) / 2.0
    np.testing.assert_allclose(result, expected)


def test_prelim_clamps_shifted_values_below_one_before_boxcox(
    mock_prelim_params: PrelimOut,
) -> None:
    """Match MATLAB below/equal/above-min behavior when bounds are disabled."""
    fitted = replace(
        mock_prelim_params,
        min_x=np.full(3, 5.0),
        mu_x=np.zeros(3),
        sigma_x=np.ones(3),
    )
    x_raw = np.array(
        [
            [0.0, 4.0, 5.0],
            [5.0, 6.0, np.nan],
        ],
        dtype=np.double,
    )

    actual = PrelimStage.predict(
        PrelimPredictInput(x_raw, True, False, True),
        fitted,
    )

    expected = np.array([[0.0, 0.0, 0.0], [0.0, 1.0, np.nan]])
    np.testing.assert_allclose(actual, expected, equal_nan=True)


def test_prelim_skips_normalisation_when_norm_flag_is_false(
    mock_instance_space: InstanceSpace,
) -> None:
    """Regression: explore() must not apply Box-Cox/z-score when norm=False.

    Previously `_explore_prelim` always applied Box-Cox + z-score
    regardless of `NormOptions.flag`; with `norm=False` a real trained
    model's `lambda_x`/`mu_x`/`sigma_x` are all-zero (never fit), which
    would have produced `inf`/`nan` for every instance (Box-Cox at
    `lambda=0` is a log transform, then dividing by `sigma_x=0`).
    """
    mock_instance_space._options.norm.flag = False  # type: ignore[misc]
    # Zero out lambda_x/mu_x/sigma_x to match a real norm=False-trained
    # model exactly (the fixture's mock_prelim_params otherwise sets
    # non-zero values that would coincidentally survive the bug).
    prelim = mock_instance_space._model.prelim  # type: ignore[union-attr]
    prelim.lambda_x[:] = 0.0
    prelim.mu_x[:] = 0.0
    prelim.sigma_x[:] = 0.0
    x_raw = np.array([[5.0, 5.0, 5.0], [1.0, 2.0, 3.0]])

    result = _predict_prelim(mock_instance_space, x_raw)

    assert np.all(np.isfinite(result))
    # bound=True still applies (unaffected by norm=False); only the
    # normalisation step is skipped, so output equals the clipped input.
    expected = np.clip(x_raw, prelim.lo_bound, prelim.hi_bound)
    np.testing.assert_array_equal(result, expected)


def test_prelim_skips_all_transforms_when_auto_preproc_is_false(
    mock_instance_space: InstanceSpace,
) -> None:
    """Explore must mirror a training run with automatic preprocessing off."""
    cast(Any, mock_instance_space)._options.auto.preproc = False
    x_raw = np.array([[-5.0, 15.0, 5.0], [1.0, 2.0, 3.0]])

    messages = _collect_warnings(
        _predict_prelim,
        mock_instance_space,
        x_raw,
    )
    result = _predict_prelim(mock_instance_space, x_raw)

    assert messages == []
    np.testing.assert_array_equal(result, x_raw)


def load_prelim_params() -> PrelimOut:
    df = pd.read_csv(ARTIFACTS_DIR / "prelim" / "prelim_params.csv")
    n = len(df)
    return PrelimOut(
        med_val=np.zeros(n),
        iq_range=np.zeros(n),
        hi_bound=df["hi_bound"].to_numpy(),
        lo_bound=df["lo_bound"].to_numpy(),
        min_x=df["min_x"].to_numpy(),
        lambda_x=df["lambda_x"].to_numpy(),
        mu_x=df["mu_x"].to_numpy(),
        sigma_x=df["sigma_x"].to_numpy(),
        min_y=0.0,
        lambda_y=np.array([]),
        sigma_y=np.array([]),
        mu_y=np.array([]),
    )


def test_prelim_predict_matches_current_matlab_oracle(
    verified_current_matlab_bundle: Path,
) -> None:
    """Replay R2026a's fitted PRELIM transformation with round-trip reads."""
    root = verified_current_matlab_bundle / "build_data" / "prelim" / "default"
    outputs = root / "outputs"
    inputs = root / "inputs"
    params = pd.read_csv(
        outputs / "prelim_feature_params.csv",
        float_precision="round_trip",
    )
    x_raw = pd.read_csv(
        inputs / "x_raw.csv",
        float_precision="round_trip",
    ).iloc[:, 1:]
    expected = pd.read_csv(
        inputs / "x_processed.csv",
        float_precision="round_trip",
    ).iloc[:, 1:]
    fitted = PrelimOut(
        med_val=params["medval"].to_numpy(dtype=np.double),
        iq_range=params["iqrange"].to_numpy(dtype=np.double),
        hi_bound=params["hi_bound"].to_numpy(dtype=np.double),
        lo_bound=params["lo_bound"].to_numpy(dtype=np.double),
        min_x=params["min_x"].to_numpy(dtype=np.double),
        lambda_x=params["lambda_x"].to_numpy(dtype=np.double),
        mu_x=params["mu_x"].to_numpy(dtype=np.double),
        sigma_x=params["sigma_x"].to_numpy(dtype=np.double),
        min_y=0.0,
        lambda_y=np.array([], dtype=np.double),
        sigma_y=np.array([], dtype=np.double),
        mu_y=np.array([], dtype=np.double),
    )

    actual = PrelimStage.predict(
        PrelimPredictInput(
            x=x_raw.to_numpy(dtype=np.double),
            auto_preproc=True,
            bound_enabled=True,
            norm_enabled=True,
        ),
        fitted,
    )

    np.testing.assert_allclose(
        actual,
        expected.to_numpy(dtype=np.double),
        atol=2e-13,
        rtol=0,
    )


def test_prelim_matches_legacy_snapshot_outside_current_clamp_cases() -> None:
    """Keep the unverified snapshot diagnostic separate from current-gold behavior."""
    test_df = pd.read_csv(INPUT_DIR / "metadata_test.csv")
    x_raw = test_df.iloc[:, 1:11].to_numpy(dtype=np.double)

    instance_space = Mock(spec=InstanceSpace)
    instance_space._model = Mock()
    fitted = load_prelim_params()
    instance_space._model.prelim = fitted
    instance_space._require_model = Mock(return_value=instance_space._model)
    instance_space._options = Mock()
    instance_space._options.auto.preproc = True
    instance_space._options.bound.flag = True
    instance_space._options.norm.flag = True

    result = _predict_prelim(instance_space, x_raw)

    expected = pd.read_csv(
        OUTPUTS_DIR / "step1_after_prelim.csv",
        index_col="instance_id",
    ).to_numpy(dtype=np.double)

    assert result.shape == expected.shape
    bounded = np.minimum(np.maximum(x_raw, fitted.lo_bound), fitted.hi_bound)
    current_gold_clamps = np.isfinite(bounded) & (bounded - fitted.min_x + 1 < 1)
    assert np.any(current_gold_clamps)
    np.testing.assert_allclose(
        result[~current_gold_clamps],
        expected[~current_gold_clamps],
        rtol=0.01,
        atol=1e-12,
    )
