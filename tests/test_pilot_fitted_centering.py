# ruff: noqa: D103, PLR2004, SLF001
"""PLS inference must reuse the fitted mean, including after persistence."""

from pathlib import Path

import joblib
import numpy as np
import pytest

from instancespace.data.model import PilotOut
from instancespace.data.options import CloisterOptions, GeneralOptions, PilotOptions
from instancespace.stages.cloister import CloisterInput, CloisterStage
from instancespace.stages.pilot import PilotPredictInput, PilotStage


@pytest.mark.parametrize("dims", [2, 3])
def test_pls_fitted_centering_and_persistence(tmp_path: Path, dims: int) -> None:
    rng = np.random.default_rng(20261009)
    x = rng.normal(size=(30, 4)) + np.array([10, 20, -30, 40])
    y = x @ rng.normal(size=(4, 3)) + rng.normal(size=(30, 3))
    output = PilotStage.pilot(
        x,
        y,
        ["a", "b", "c", "d"],
        PilotOptions.default(method="pls", dims=dims),
        GeneralOptions.default(),
        _do_output=False,
    )
    fitted = PilotOut.from_stage_runner_output(output._asdict())
    actual = PilotStage.predict(PilotPredictInput(x), fitted)
    np.testing.assert_allclose(actual, output.z, atol=1e-12)
    # A single row and a shifted batch must reuse the training mean.
    query = x[:4] + 7
    expected = (query - x.mean(axis=0)) @ fitted.a.T
    np.testing.assert_allclose(
        PilotStage.predict(PilotPredictInput(query), fitted),
        expected,
    )
    np.testing.assert_allclose(
        PilotStage.predict(PilotPredictInput(query[:1]), fitted),
        expected[:1],
    )
    path = tmp_path / "pilot.joblib"
    joblib.dump(fitted, path)
    restored = joblib.load(path)
    np.testing.assert_allclose(
        PilotStage.predict(PilotPredictInput(query), restored),
        expected,
    )
    np.testing.assert_array_equal(restored.x_mean, x.mean(axis=0))


@pytest.mark.parametrize("max_features", [2, 20])
def test_cloister_uses_fitted_pilot_mean(max_features: int) -> None:
    from itertools import product

    x = np.array(list(product([-1.0, 1.0], repeat=3))) + np.array([10, 20, 30])
    mean = x.mean(axis=0)
    opts = CloisterOptions.default(max_features=max_features)
    expected = CloisterStage.cloister(x - mean, np.eye(3), opts)
    actual = CloisterStage._run(CloisterInput(x, np.eye(3), opts, mean))
    np.testing.assert_allclose(actual.z_edge, expected.z_edge)
    np.testing.assert_allclose(actual.z_ecorr, expected.z_ecorr)
