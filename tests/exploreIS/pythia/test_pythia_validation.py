"""Validation test for PYTHIA stage against MATLAB reference data.

Loads MATLAB-trained SVM artifacts (pythia/zscore.csv, pythia/precision.csv,
pythia/svm_<algo>.csv) together with the MATLAB-projected 2D coordinates
(explore_outputs/step3_after_pilot.csv) and verifies that _explore_pythia
reproduces MATLAB's binary predictions and posterior probabilities.

Thresholds (per-stage port-fidelity check, MATLAB inputs in, MATLAB outputs out):
- Binary prediction agreement >= 99%.
- Probability Pearson |r| mean >= 0.99 across algorithms.
"""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd

from instancespace.data.model import PythiaOut
from instancespace.instance_space import InstanceSpace

REFERENCE_DIR = Path("tests/matlab_reference")
ARTIFACTS_DIR = REFERENCE_DIR / "training_artifacts" / "pythia"
OUTPUTS_DIR = REFERENCE_DIR / "explore_outputs"


def load_svm(path: Path) -> SimpleNamespace:
    """Reconstruct a per-algorithm SVM from its CSV.

    Each row is one support vector; row-1 columns also carry the per-SVM
    scalars (kernel_fn, kernel_param, bias, Platt A/B). Alphas are already
    signed (Alpha * SupportVectorLabels) on export.
    """
    df = pd.read_csv(path)
    return SimpleNamespace(
        support_vectors=df[["sv_z1", "sv_z2"]].to_numpy(dtype=np.double),
        alphas=df["alpha"].to_numpy(dtype=np.double),
        kernel_fn=df["kernel_fn"].iloc[0],
        kernel_param=float(df["kernel_param"].iloc[0]),
        bias=float(df["bias"].iloc[0]),
        platt_A=float(df["platt_A"].iloc[0]),
        platt_B=float(df["platt_B"].iloc[0]),
    )


def build_pythia_from_artifacts() -> PythiaOut:
    zscore = pd.read_csv(ARTIFACTS_DIR / "zscore.csv").iloc[0]
    precision_df = pd.read_csv(ARTIFACTS_DIR / "precision.csv")
    algo_order = precision_df["algo"].tolist()
    svms = [load_svm(ARTIFACTS_DIR / f"svm_{algo}.csv") for algo in algo_order]

    pythia = Mock(spec=PythiaOut)
    pythia.mu = np.array([zscore["mu_z1"], zscore["mu_z2"]], dtype=np.double)
    pythia.sigma = np.array(
        [zscore["sigma_z1"], zscore["sigma_z2"]], dtype=np.double,
    )
    pythia.svm = svms
    pythia.precision = precision_df["precision"].to_numpy(dtype=np.double)
    pythia._algo_order = algo_order  # type: ignore[attr-defined]
    return pythia


def test_pythia_matches_matlab():
    """PYTHIA binary agreement >= 99%, probability |r| mean >= 0.99."""
    pythia = build_pythia_from_artifacts()
    algo_order = pythia._algo_order

    z = pd.read_csv(OUTPUTS_DIR / "step3_after_pilot.csv", index_col=0)

    instance_space = Mock(spec=InstanceSpace)
    instance_space._model = Mock()
    instance_space._model.pythia = pythia

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
