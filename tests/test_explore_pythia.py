"""Tests for PYTHIA stage's explore()-time inference (_explore_pythia).

Post-S1, _explore_pythia calls scikit-learn's own predict()/predict_proba() on
whatever's in model.pythia.svm, so the unit tests fit small real SVCs rather than
hand-computing kernel arithmetic by hand - that arithmetic is sklearn's own
responsibility now, not ours to re-verify. What's still ours to test: correctly
deriving the z-score normalisation from model.pilot.z (not the stage's own,
already-normalised, mu/sigma), and the precision-weighted selection logic.

The validation test loads MATLAB-trained SVM artifacts (pythia/zscore.csv,
pythia/precision.csv, pythia/svm_<algo>.csv) together with the MATLAB-projected 2D
coordinates (explore_outputs/step3_after_pilot.csv) and verifies that
_explore_pythia reproduces MATLAB's binary predictions and posterior probabilities.
Post-S1, _explore_pythia only knows how to call .predict()/.predict_proba() on
whatever is in model.pythia.svm - there's no live scikit-learn SVC trained on
MATLAB's data to hand it, only MATLAB's exported numbers (support vectors, signed
alphas, bias, kernel params, Platt A/B). _MatlabArtifactSvm wraps those numbers
behind sklearn's calling convention, replicating the same kernel + Platt-sigmoid
math _explore_pythia used before S1, so this MATLAB-fidelity check keeps working
without needing to reconstruct a real fitted SVC's full internal state.

Validation thresholds (per-stage port-fidelity check, MATLAB inputs in, MATLAB
outputs out):
- Binary prediction agreement >= 99%.
- Probability Pearson |r| mean >= 0.99 across algorithms.
"""

from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.svm import SVC

from instancespace.data.model import PythiaOut
from instancespace.instance_space import InstanceSpace

REFERENCE_DIR = Path("tests/matlab_reference")
ARTIFACTS_DIR = REFERENCE_DIR / "training_artifacts" / "pythia"
OUTPUTS_DIR = REFERENCE_DIR / "explore_outputs"


_DECISION_THRESHOLD = 0.5


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


def test_pythia_output_shapes() -> None:
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


def test_pythia_matches_direct_svc_calls() -> None:
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


def test_pythia_matches_direct_svc_calls_poly_kernel() -> None:
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


def test_pythia_zscore_normalization_uses_pilot_z() -> None:
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


def test_pythia_selection0_picks_highest_precision_positive() -> None:
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


def test_pythia_selection0_none_when_no_positive() -> None:
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


class _MatlabArtifactSvm:
    """A MATLAB-exported SVM artifact, callable like a fitted scikit-learn SVC.

    Implements just enough of the SVC interface (`classes_`, `predict`,
    `predict_proba`) for `_explore_pythia` to consume it, using the same
    kernel-evaluation + Platt-sigmoid math the pre-S1 hand-rolled adapter path used.
    """

    classes_ = np.array([False, True])

    def __init__(
        self,
        support_vectors: NDArray[np.double],
        alphas: NDArray[np.double],
        bias: float,
        kernel_fn: str,
        kernel_param: float,
        platt_a: float,
        platt_b: float,
    ) -> None:
        self.support_vectors = support_vectors
        self.alphas = alphas
        self.bias = bias
        self.kernel_fn = kernel_fn
        self.kernel_param = kernel_param
        self.platt_a = platt_a
        self.platt_b = platt_b

    def _post_good(self, z_norm: NDArray[np.double]) -> NDArray[np.double]:
        svs = self.support_vectors
        kernel_fn = self.kernel_fn.lower()

        if kernel_fn in ("gaussian", "rbf"):
            scale = self.kernel_param
            dist_sq = (
                np.sum(z_norm**2, axis=1, keepdims=True)
                + np.sum(svs**2, axis=1)
                - 2.0 * (z_norm @ svs.T)
            )
            k = np.exp(-dist_sq / scale**2)
        elif kernel_fn == "polynomial":
            k = (z_norm @ svs.T + 1.0) ** self.kernel_param
        else:
            k = z_norm @ svs.T

        decision = k @ self.alphas + self.bias
        post_good: NDArray[np.double] = 1.0 / (
            1.0 + np.exp(self.platt_a * decision + self.platt_b)
        )
        return post_good

    def predict_proba(self, z_norm: NDArray[np.double]) -> NDArray[np.double]:
        """Return [P(bad), P(good)] columns, matching sklearn's classes_ order."""
        post_good = self._post_good(z_norm)
        return np.column_stack([1.0 - post_good, post_good])

    def predict(self, z_norm: NDArray[np.double]) -> NDArray[np.bool_]:
        """Return the predicted class, matching sklearn's >=0.5 threshold."""
        return self._post_good(z_norm) >= _DECISION_THRESHOLD


def load_svm(path: Path) -> _MatlabArtifactSvm:
    """Reconstruct a per-algorithm SVM from its CSV.

    Each row is one support vector; row-1 columns also carry the per-SVM
    scalars (kernel_fn, kernel_param, bias, Platt A/B). Alphas are already
    signed (Alpha * SupportVectorLabels) on export.
    """
    df = pd.read_csv(path)
    return _MatlabArtifactSvm(
        support_vectors=df[["sv_z1", "sv_z2"]].to_numpy(dtype=np.double),
        alphas=df["alpha"].to_numpy(dtype=np.double),
        kernel_fn=df["kernel_fn"].iloc[0],
        kernel_param=float(df["kernel_param"].iloc[0]),
        bias=float(df["bias"].iloc[0]),
        platt_a=float(df["platt_A"].iloc[0]),
        platt_b=float(df["platt_B"].iloc[0]),
    )


def build_pythia_from_artifacts() -> tuple[PythiaOut, NDArray[np.double]]:
    """Return a mocked PythiaOut plus a synthetic pilot.z with matching mu/sigma.

    _explore_pythia derives its z-score normalisation from model.pilot.z (see its
    docstring for why PYTHIA's own stored mu/sigma aren't usable for this), so this
    builds a 2-point array whose mean and std reproduce MATLAB's exported zscore.csv
    values exactly, rather than a large synthetic training set.
    """
    zscore = pd.read_csv(ARTIFACTS_DIR / "zscore.csv").iloc[0]
    precision_df = pd.read_csv(ARTIFACTS_DIR / "precision.csv")
    algo_order = precision_df["algo"].tolist()
    svms = [load_svm(ARTIFACTS_DIR / f"svm_{algo}.csv") for algo in algo_order]

    mu = np.array([zscore["mu_z1"], zscore["mu_z2"]], dtype=np.double)
    sigma = np.array([zscore["sigma_z1"], zscore["sigma_z2"]], dtype=np.double)
    offset = sigma / np.sqrt(2.0)
    pilot_z = np.vstack([mu - offset, mu + offset])

    pythia = Mock(spec=PythiaOut)
    pythia.svm = svms
    pythia.precision = precision_df["precision"].to_numpy(dtype=np.double)
    pythia._algo_order = algo_order
    return pythia, pilot_z


def test_pythia_matches_matlab() -> None:
    """PYTHIA binary agreement >= 99%, probability |r| mean >= 0.99."""
    pythia, pilot_z = build_pythia_from_artifacts()
    algo_order = pythia._algo_order  # type: ignore[attr-defined]

    z = pd.read_csv(OUTPUTS_DIR / "step3_after_pilot.csv", index_col=0)

    instance_space = Mock(spec=InstanceSpace)
    instance_space._model = Mock()
    instance_space._model.pythia = pythia
    instance_space._model.pilot.z = pilot_z
    instance_space._require_model = Mock(return_value=instance_space._model)

    y_hat, pr0_hat, _ = InstanceSpace._explore_pythia(instance_space, z.to_numpy())

    ref_pred = pd.read_csv(OUTPUTS_DIR / "step4_pythia_predictions.csv", index_col=0)
    ref_prob = pd.read_csv(OUTPUTS_DIR / "step4_pythia_probabilities.csv", index_col=0)

    ref_pred = ref_pred[algo_order].to_numpy(dtype=np.bool_)
    ref_prob = ref_prob[algo_order].to_numpy(dtype=np.double)

    assert y_hat.shape == ref_pred.shape
    assert pr0_hat.shape == ref_prob.shape

    agreement = (y_hat == ref_pred).mean()
    per_algo_r = np.array([
        np.corrcoef(pr0_hat[:, i], ref_prob[:, i])[0, 1]
        for i in range(len(algo_order))
    ])
    mean_abs_r = np.mean(np.abs(per_algo_r))

    print(f"\nInput:    {z.shape[0]} instances x 2 coordinates")
    print(f"Algorithms: {len(algo_order)}")
    print(f"Binary agreement with MATLAB: {agreement * 100:.2f}%")
    print(f"Probability Pearson r (per algo): {per_algo_r}")
    print(f"Mean |r|: {mean_abs_r:.4f}")

    assert agreement >= 0.99, (
        f"Binary agreement {agreement * 100:.2f}% < 99% threshold"
    )
    assert mean_abs_r >= 0.99, (
        f"Mean |Pearson r| {mean_abs_r:.4f} < 0.99 threshold"
    )
    print(
        f"[PASS] PYTHIA validation: {agreement * 100:.2f}% agreement, "
        f"|r|={mean_abs_r:.3f}",
    )
