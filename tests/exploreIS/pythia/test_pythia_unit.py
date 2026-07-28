"""Unit tests for PYTHIA stage (_explore_pythia).

Post-S1, _explore_pythia calls scikit-learn's own predict()/predict_proba() on
whatever's in model.pythia.svm, so these tests fit small real SVCs rather than
hand-computing kernel arithmetic by hand — that arithmetic is sklearn's own
responsibility now, not ours to re-verify. What's still ours to test: correctly
deriving the z-score normalisation from model.pilot.z (not the stage's own,
already-normalised, mu/sigma), and the precision-weighted selection logic.
"""

from unittest.mock import Mock

import numpy as np
from sklearn.svm import SVC

from instancespace.data.model import PythiaOut
from instancespace.instance_space import InstanceSpace


def _fit_svc(rng: np.random.Generator, *, kernel: str = "rbf") -> SVC:
    """Fit a tiny SVC on two well-separated clusters (bad=False, good=True).

    ``stages/pythia.py`` only ever trains "rbf" or "poly" (`kernel = "poly" if
    is_poly_krnl else "rbf"` - "linear" is unreachable in production), matching
    PYTHIA's poly hyperparameters (degree=2, coef0=1) when kernel="poly".
    """
    z = np.vstack([
        rng.normal(-3.0, 0.5, size=(20, 2)),
        rng.normal(3.0, 0.5, size=(20, 2)),
    ])
    y = np.array([False] * 20 + [True] * 20)
    kwargs = {"degree": 2, "coef0": 1} if kernel == "poly" else {}
    svc = SVC(kernel=kernel, probability=True, random_state=0, **kwargs)
    svc.fit(z, y)
    return svc


def make_instance_space(
    pilot_z: np.ndarray,  # type: ignore[type-arg]
    svms: list[SVC],
    precision: np.ndarray,  # type: ignore[type-arg]
) -> InstanceSpace:
    model = Mock()
    model.pythia = Mock(spec=PythiaOut)
    model.pythia.svm = svms
    model.pythia.precision = precision
    model.pilot.z = pilot_z
    instance_space = Mock(spec=InstanceSpace)
    instance_space._model = model
    instance_space._require_model = Mock(return_value=model)
    return instance_space


def _two_point_pilot_z(
    mu: np.ndarray,  # type: ignore[type-arg]
    sigma: np.ndarray,  # type: ignore[type-arg]
) -> np.ndarray:  # type: ignore[type-arg]
    """Build a 2-row array whose mean/std (ddof=1) exactly reproduce mu/sigma."""
    offset = sigma / np.sqrt(2.0)
    return np.vstack([mu - offset, mu + offset])


def test_pythia_output_shapes():
    rng = np.random.default_rng(0)
    svc = _fit_svc(rng)
    pilot_z = _two_point_pilot_z(np.array([0.0, 0.0]), np.array([1.0, 1.0]))
    space = make_instance_space(
        pilot_z=pilot_z,
        svms=[svc, svc, svc],
        precision=np.array([0.9, 0.8, 0.7]),
    )
    z = rng.normal(size=(10, 2))
    y_hat, pr0_hat, selection0 = InstanceSpace._explore_pythia(space, z)
    assert y_hat.shape == (10, 3)
    assert pr0_hat.shape == (10, 3)
    assert selection0.shape == (10,)
    assert y_hat.dtype == np.bool_


def test_pythia_matches_direct_svc_calls():
    """y_hat/pr0_hat must exactly match calling predict()/predict_proba() directly."""
    rng = np.random.default_rng(0)
    svc = _fit_svc(rng)
    pilot_z = _two_point_pilot_z(np.array([0.0, 0.0]), np.array([1.0, 1.0]))
    space = make_instance_space(
        pilot_z=pilot_z,
        svms=[svc],
        precision=np.array([1.0]),
    )
    z = rng.normal(size=(15, 2))
    y_hat, pr0_hat, _ = InstanceSpace._explore_pythia(space, z)

    # mu=(0,0), sigma=(1,1) -> normalisation is a no-op here.
    expected_proba = svc.predict_proba(z)
    expected_pred = svc.predict(z)

    np.testing.assert_array_equal(y_hat[:, 0], expected_pred)
    np.testing.assert_allclose(pr0_hat[:, 0], expected_proba[:, 0])


def test_pythia_matches_direct_svc_calls_poly_kernel():
    """Same as above, for PYTHIA's other trainable kernel (poly, degree=2/coef0=1)."""
    rng = np.random.default_rng(0)
    svc = _fit_svc(rng, kernel="poly")
    pilot_z = _two_point_pilot_z(np.array([0.0, 0.0]), np.array([1.0, 1.0]))
    space = make_instance_space(
        pilot_z=pilot_z,
        svms=[svc],
        precision=np.array([1.0]),
    )
    z = rng.normal(size=(15, 2))
    y_hat, pr0_hat, _ = InstanceSpace._explore_pythia(space, z)

    expected_proba = svc.predict_proba(z)
    expected_pred = svc.predict(z)

    np.testing.assert_array_equal(y_hat[:, 0], expected_pred)
    np.testing.assert_allclose(pr0_hat[:, 0], expected_proba[:, 0])


def test_pythia_zscore_normalization_uses_pilot_z():
    """z-score normalisation must be derived from model.pilot.z, not elsewhere.

    mu=(1,2), sigma=(2,4) via pilot_z; a raw input of exactly (1,2) should
    normalise to (0,0) and therefore match calling the SVC directly on (0,0).
    """
    rng = np.random.default_rng(0)
    svc = _fit_svc(rng)
    mu = np.array([1.0, 2.0])
    sigma = np.array([2.0, 4.0])
    pilot_z = _two_point_pilot_z(mu, sigma)
    space = make_instance_space(
        pilot_z=pilot_z,
        svms=[svc],
        precision=np.array([1.0]),
    )
    z = np.array([[1.0, 2.0]])
    _, pr0_hat, _ = InstanceSpace._explore_pythia(space, z)

    expected = svc.predict_proba(np.array([[0.0, 0.0]]))[:, 0]
    np.testing.assert_allclose(pr0_hat[:, 0], expected)


def test_pythia_selection0_picks_highest_precision_positive():
    # Two algos both predict "good"; higher precision wins.
    rng = np.random.default_rng(0)
    svc = _fit_svc(rng)
    pilot_z = _two_point_pilot_z(np.array([0.0, 0.0]), np.array([1.0, 1.0]))
    space = make_instance_space(
        pilot_z=pilot_z,
        svms=[svc, svc],
        precision=np.array([0.5, 0.9]),
    )
    z = np.array([[3.0, 3.0]])  # deep in the "good" cluster
    y_hat, _, selection0 = InstanceSpace._explore_pythia(space, z)
    assert y_hat[0, 0] and y_hat[0, 1]
    assert selection0[0] == 1  # higher precision algo


def test_pythia_selection0_none_when_no_positive():
    # All algos predict "bad"; selection0 = -1
    rng = np.random.default_rng(0)
    svc = _fit_svc(rng)
    pilot_z = _two_point_pilot_z(np.array([0.0, 0.0]), np.array([1.0, 1.0]))
    space = make_instance_space(
        pilot_z=pilot_z,
        svms=[svc, svc],
        precision=np.array([0.9, 0.9]),
    )
    z = np.array([[-3.0, -3.0]])  # deep in the "bad" cluster
    y_hat, _, selection0 = InstanceSpace._explore_pythia(space, z)
    assert not y_hat[0, 0] and not y_hat[0, 1]
    assert selection0[0] == -1
