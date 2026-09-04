# ruff: noqa: D103, PLR2004, SLF001
"""Tests for PILOT stage's explore-time inference.

Unit tests exercise ``PilotStage.predict()`` with mocked/stubbed dependencies,
independent of MATLAB reference data. PILOT inference is the dimension-generic linear
projection ``z = x @ A.T`` used by MATLAB explore, including its deliberate lack of
the centering used by the PLS build projection.
"""

from pathlib import Path
from typing import Any, cast
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
from numpy.typing import NDArray

from instancespace.data.model import PilotOut
from instancespace.instance_space import InstanceSpace
from instancespace.stages.pilot import PilotPredictInput, PilotStage

REFERENCE_DIR = Path("tests/matlab_reference")
ARTIFACTS_DIR = REFERENCE_DIR / "training_artifacts"
OUTPUTS_DIR = REFERENCE_DIR / "explore_outputs"
CURRENT_PILOT_VARIANTS = (
    "pilot_standard_analytic_3d",
    "pilot_standard_numerical_3d_x0",
    "pilot_standard_numerical_3d_precalc",
    "pilot_pls_2d",
    "pilot_pls_3d_grouped",
)

_rng = np.random.default_rng()


def _predict_pilot(
    instance_space: InstanceSpace,
    x: NDArray[np.double],
) -> NDArray[np.double]:
    """Call the stage contract with the fitted state held by InstanceSpace."""
    fitted = cast(PilotOut, cast(Any, instance_space)._require_model().pilot)
    return PilotStage.predict(PilotPredictInput(x), fitted)


def make_instance_space(a: NDArray[np.double]) -> InstanceSpace:
    pilot = Mock(spec=PilotOut)
    pilot.a = a
    model = Mock()
    model.pilot = pilot
    instance_space = Mock(spec=InstanceSpace)
    instance_space._model = model
    instance_space._require_model = Mock(return_value=model)
    return instance_space


def test_pilot_output_shape() -> None:
    a = np.eye(2, 5)  # (2, 5): projects 5 features → 2 dimensions
    x = _rng.random((10, 5))
    result = _predict_pilot(make_instance_space(a), x)
    assert result.shape == (10, 2)


def test_pilot_correct_projection() -> None:
    # Z = X @ A.T — verify with known values
    a = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])  # (2, 3)
    x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    result = _predict_pilot(make_instance_space(a), x)
    expected = np.array([[1.0, 2.0], [4.0, 5.0]])
    np.testing.assert_array_almost_equal(result, expected)


def test_pilot_3d_projection_preserves_matlab_explore_centering_asymmetry() -> None:
    """Explore uses exact uncentred X @ A.T even when a PLS build was centred."""
    a = np.array(
        [
            [1.0, 0.0, 0.0, 0.5],
            [0.0, 1.0, -1.0, 0.0],
            [0.25, 0.0, 0.5, 1.0],
        ],
    )
    x = np.array(
        [
            [2.0, 4.0, 6.0, 8.0],
            [4.0, 8.0, 12.0, 16.0],
            [6.0, 12.0, 18.0, 24.0],
        ],
    )

    result = _predict_pilot(make_instance_space(a), x)
    uncentred = x @ a.T
    centred = (x - np.mean(x, axis=0)) @ a.T

    assert result.shape == (x.shape[0], a.shape[0])
    np.testing.assert_array_equal(result, uncentred)
    assert not np.array_equal(result, centred)


def test_pilot_single_instance() -> None:
    a = _rng.random((2, 4))
    x = _rng.random((1, 4))
    result = _predict_pilot(make_instance_space(a), x)
    assert result.shape == (1, 2)


def test_pilot_preserves_input() -> None:
    a = _rng.random((2, 3))
    x = _rng.random((5, 3))
    x_copy = x.copy()
    _predict_pilot(make_instance_space(a), x)
    np.testing.assert_array_equal(x, x_copy)


def test_pilot_predict_preserves_fitted_state() -> None:
    """Inference reads the fitted projection without mutating it."""
    a = _rng.random((2, 3))
    before = a.copy()

    _predict_pilot(make_instance_space(a), _rng.random((5, 3)))

    np.testing.assert_array_equal(a, before)


def test_pilot_predict_does_not_run_training(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Inference must not enter PILOT's fitting path."""
    train = Mock(side_effect=AssertionError("PILOT training was called"))
    monkeypatch.setattr(PilotStage, "pilot", train)

    _predict_pilot(make_instance_space(np.eye(2)), np.eye(2))

    train.assert_not_called()


def test_pilot_instance_space_wrapper_remains_compatible() -> None:
    """Keep the private wrapper compatible while orchestration migrates."""
    a = _rng.random((2, 3))
    x = _rng.random((5, 3))
    instance_space = make_instance_space(a)

    expected = _predict_pilot(instance_space, x)
    actual = InstanceSpace._explore_pilot(instance_space, x)

    np.testing.assert_array_equal(actual, expected)


def test_pilot_deterministic() -> None:
    a = _rng.random((2, 6))
    x = _rng.random((20, 6))
    r1 = _predict_pilot(make_instance_space(a), x)
    r2 = _predict_pilot(make_instance_space(a), x)
    np.testing.assert_array_equal(r1, r2)


def test_pilot_zero_matrix() -> None:
    a = np.zeros((2, 4))
    x = _rng.random((5, 4))
    result = _predict_pilot(make_instance_space(a), x)
    np.testing.assert_array_equal(result, np.zeros((5, 2)))


@pytest.mark.parametrize("variant", CURRENT_PILOT_VARIANTS)
def test_pilot_predict_matches_current_matlab_oracle(
    variant: str,
    verified_current_matlab_bundle: Path,
) -> None:
    """Replay R2026a's uncentred 2D/3D explore projection directly."""
    root = verified_current_matlab_bundle / "explore_data" / "pilot" / variant
    x = pd.read_csv(
        root / "inputs" / "x.csv",
        float_precision="round_trip",
    ).iloc[:, 1:]
    a = pd.read_csv(
        root / "inputs" / "projection_a.csv",
        float_precision="round_trip",
    ).iloc[:, 1:]
    expected = pd.read_csv(
        root / "outputs" / "pilot_z.csv",
        float_precision="round_trip",
    ).iloc[:, 1:]

    actual = PilotStage.predict(
        PilotPredictInput(x.to_numpy(dtype=np.double)),
        cast(PilotOut, Mock(a=a.to_numpy(dtype=np.double))),
    )

    np.testing.assert_allclose(
        actual,
        expected.to_numpy(dtype=np.double),
        atol=2e-13,
        rtol=0,
    )


def load_pilot_matrix() -> Mock:
    df = pd.read_csv(ARTIFACTS_DIR / "pilot" / "pilot_matrix.csv")
    a = df[["z1_coef", "z2_coef"]].to_numpy(dtype=np.double).T

    pilot = Mock(spec=PilotOut)
    pilot.a = a
    model = Mock()
    model.pilot = pilot
    return model


def test_pilot_matches_legacy_snapshot() -> None:
    """Reproduce the unverified historical step-3 projection narrowly."""
    x_input = pd.read_csv(
        OUTPUTS_DIR / "step2_after_sifted.csv",
        index_col="instance_id",
    ).to_numpy(dtype=np.double)

    instance_space = Mock(spec=InstanceSpace)
    instance_space._model = load_pilot_matrix()
    instance_space._require_model = Mock(return_value=instance_space._model)

    result = _predict_pilot(instance_space, x_input)

    expected = pd.read_csv(
        OUTPUTS_DIR / "step3_after_pilot.csv",
        index_col=0,
    ).to_numpy(dtype=np.double)

    np.testing.assert_allclose(result, expected, rtol=0, atol=2e-13)
