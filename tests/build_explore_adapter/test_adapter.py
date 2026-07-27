"""Unit tests for the build->explore adapter's PYTHIA translation.

These train a tiny SVM inline (no full build()) so they run in milliseconds. They
check the two halves of the conversion separately: the decision function transfers
*exactly* (support vectors, signed coefficients, bias, kernel scale), while the Platt
probability transfers up to the small difference between the clean Platt sigmoid the
artifact form uses and scikit-learn's internal posterior.
"""

import numpy as np
import pytest
from sklearn.svm import SVC

from instancespace.build_explore_adapter import _svc_to_artifact, adapt_for_explore
from instancespace.instance_space import InstanceSpace


def _toy_svc():
    rng = np.random.default_rng(0)
    z = np.vstack([rng.normal(-1.0, 0.8, size=(40, 2)),
                   rng.normal(1.0, 0.8, size=(40, 2))])
    y = np.array([False] * 40 + [True] * 40)
    mu, sigma = z.mean(axis=0), z.std(axis=0, ddof=1)
    z_norm = (z - mu) / sigma
    svc = SVC(kernel="rbf", gamma="scale", probability=True, random_state=0)
    svc.fit(z_norm, y)
    return svc, z, z_norm, mu, sigma


def _toy_svc_poly():
    rng = np.random.default_rng(0)
    z = np.vstack([rng.normal(-1.0, 0.8, size=(40, 2)),
                   rng.normal(1.0, 0.8, size=(40, 2))])
    y = np.array([False] * 40 + [True] * 40)
    mu, sigma = z.mean(axis=0), z.std(axis=0, ddof=1)
    z_norm = (z - mu) / sigma
    # Matches stages/pythia.py's poly-kernel training: degree=2, coef0=1, tuned gamma.
    svc = SVC(
        kernel="poly", degree=2, coef0=1, gamma=0.7, probability=True, random_state=0,
    )
    svc.fit(z_norm, y)
    return svc, z, z_norm, mu, sigma


def test_decision_function_transfers_exactly():
    svc, _z, z_norm, _mu, _sigma = _toy_svc()
    art = _svc_to_artifact(svc)
    dist_sq = (np.sum(z_norm**2, axis=1, keepdims=True)
               + np.sum(art.support_vectors**2, axis=1)
               - 2.0 * z_norm @ art.support_vectors.T)
    decision = np.exp(-dist_sq / art.kernel_param**2) @ art.alphas + art.bias
    # Same support vectors / coefficients / bias / kernel scale -> identical decision.
    np.testing.assert_allclose(decision, svc.decision_function(z_norm), atol=1e-9)


def test_kernel_scale_matches_gamma():
    svc, *_ = _toy_svc()
    art = _svc_to_artifact(svc)
    # exp(-||.||^2 / s^2) == exp(-gamma ||.||^2)  =>  s = gamma**-0.5
    assert art.kernel_param == pytest.approx(svc._gamma**-0.5)


def test_explore_pythia_reproduces_sklearn_posterior():
    svc, z, z_norm, mu, sigma = _toy_svc()
    art = _svc_to_artifact(svc)
    from types import SimpleNamespace

    model = SimpleNamespace(pythia=SimpleNamespace(
        mu=mu, sigma=sigma, precision=np.array([1.0]), svm=[art]))
    holder = SimpleNamespace(_explore_model=model)
    _y_hat, pr0_hat, _sel = InstanceSpace._explore_pythia(holder, z)

    post_good = 1.0 - pr0_hat[:, 0]
    sklearn_post = svc.predict_proba(z_norm)[:, 1]
    # Ranking is preserved; the small residual is the clean Platt sigmoid (artifact /
    # MATLAB convention) versus scikit-learn's internal posterior, not a conversion error.
    assert abs(np.corrcoef(post_good, sklearn_post)[0, 1]) > 0.99


def test_poly_decision_function_transfers_exactly():
    svc, _z, z_norm, _mu, _sigma = _toy_svc_poly()
    art = _svc_to_artifact(svc)
    # art.support_vectors is pre-scaled by gamma, so (z . sv + 1)**degree reproduces
    # scikit-learn's actual (gamma * z . sv + coef0)**degree.
    k = (z_norm @ art.support_vectors.T + 1.0) ** art.kernel_param
    decision = k @ art.alphas + art.bias
    np.testing.assert_allclose(decision, svc.decision_function(z_norm), atol=1e-9)


def test_poly_kernel_param_matches_degree():
    svc, *_ = _toy_svc_poly()
    art = _svc_to_artifact(svc)
    assert art.kernel_fn == "polynomial"
    assert art.kernel_param == svc.degree


def test_explore_pythia_reproduces_sklearn_posterior_poly():
    svc, z, z_norm, mu, sigma = _toy_svc_poly()
    art = _svc_to_artifact(svc)
    from types import SimpleNamespace

    model = SimpleNamespace(pythia=SimpleNamespace(
        mu=mu, sigma=sigma, precision=np.array([1.0]), svm=[art]))
    holder = SimpleNamespace(_explore_model=model)
    y_hat, pr0_hat, _sel = InstanceSpace._explore_pythia(holder, z)

    assert np.all(np.isfinite(pr0_hat))
    post_good = 1.0 - pr0_hat[:, 0]
    sklearn_post = svc.predict_proba(z_norm)[:, 1]
    assert abs(np.corrcoef(post_good, sklearn_post)[0, 1]) > 0.99
    assert y_hat.dtype == np.bool_


def test_unsupported_kernel_raises():
    rng = np.random.default_rng(1)
    z = rng.normal(size=(20, 2))
    y = np.array([False] * 10 + [True] * 10)
    svc = SVC(kernel="sigmoid", probability=True, random_state=0).fit(z, y)
    with pytest.raises(NotImplementedError):
        _svc_to_artifact(svc)


def test_adapt_for_explore_passes_through_and_translates():
    from types import SimpleNamespace

    svc, _z, _zn, _mu, _sigma = _toy_svc()
    sentinel_prelim, sentinel_sifted, sentinel_trace = object(), object(), object()
    built = SimpleNamespace(
        prelim=sentinel_prelim, sifted=sentinel_sifted, trace=sentinel_trace,
        pilot=SimpleNamespace(z=np.zeros((5, 2)), a=np.zeros((2, 3))),
        pythia=SimpleNamespace(precision=[0.5], svm=[svc]),
    )
    adapted = adapt_for_explore(built, ["a", "b", "c"])
    # PRELIM/SIFTED/PILOT/TRACE are passed through unchanged...
    assert adapted.prelim is sentinel_prelim
    assert adapted.sifted is sentinel_sifted
    assert adapted.trace is sentinel_trace
    # ...the SVC is translated, and the original feature labels are restored.
    assert hasattr(adapted.pythia.svm[0], "alphas")
    assert adapted.data.feat_labels == ["a", "b", "c"]


def _built_model_holder():
    from types import SimpleNamespace

    svc, *_ = _toy_svc()
    built = SimpleNamespace(
        prelim=object(), sifted=object(), trace=object(),
        pilot=SimpleNamespace(z=np.zeros((5, 2)), a=np.zeros((2, 3))),
        pythia=SimpleNamespace(precision=[0.5], svm=[svc]),
    )
    return SimpleNamespace(
        _model=built,
        _explore_model=None,
        _metadata=SimpleNamespace(feature_names=["a", "b", "c"]),
    )


def test_ensure_explore_model_detects_and_converts_built_model():
    holder = _built_model_holder()
    built = holder._model
    InstanceSpace._ensure_explore_model(holder)
    # The built model is detected by shape and converted...
    assert hasattr(holder._explore_model.pythia.svm[0], "alphas")
    # ...while the unconverted scikit-learn model stays in place, untouched.
    assert holder._model is built
    assert holder._model.pythia.svm[0].kernel == "rbf"


def test_ensure_explore_model_converts_only_once():
    holder = _built_model_holder()
    InstanceSpace._ensure_explore_model(holder)
    converted = holder._explore_model
    InstanceSpace._ensure_explore_model(holder)
    assert holder._explore_model is converted


def test_ensure_explore_model_uses_artifact_model_directly():
    from types import SimpleNamespace

    svc, *_ = _toy_svc()
    artifact = SimpleNamespace(pythia=SimpleNamespace(svm=[_svc_to_artifact(svc)]))
    holder = SimpleNamespace(_model=artifact, _explore_model=None, _metadata=None)
    InstanceSpace._ensure_explore_model(holder)
    assert holder._explore_model is artifact
