"""Unit tests for PYTHIA stage (_explore_pythia)."""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from instancespace.data.model import PythiaOut
from instancespace.instance_space import InstanceSpace


def make_svm(
    *,
    sv: np.ndarray,
    alpha: np.ndarray,
    bias: float,
    kernel_fn: str,
    kernel_param: float,
    platt_a: float,
    platt_b: float,
) -> SimpleNamespace:
    return SimpleNamespace(
        support_vectors=sv,
        alphas=alpha,
        bias=bias,
        kernel_fn=kernel_fn,
        kernel_param=kernel_param,
        platt_A=platt_a,
        platt_B=platt_b,
    )


def make_instance_space(
    mu: np.ndarray,
    sigma: np.ndarray,
    svms: list,
    precision: np.ndarray,
) -> InstanceSpace:
    pythia = Mock(spec=PythiaOut)
    pythia.mu = mu
    pythia.sigma = sigma
    pythia.svm = svms
    pythia.precision = precision
    model = Mock()
    model.pythia = pythia
    instance_space = Mock(spec=InstanceSpace)
    instance_space._model = model
    return instance_space


def test_pythia_output_shapes():
    svm = make_svm(
        sv=np.array([[0.0, 0.0]]),
        alpha=np.array([1.0]),
        bias=0.0,
        kernel_fn="gaussian",
        kernel_param=1.0,
        platt_a=-1.0,
        platt_b=0.0,
    )
    space = make_instance_space(
        mu=np.array([0.0, 0.0]),
        sigma=np.array([1.0, 1.0]),
        svms=[svm, svm, svm],
        precision=np.array([0.9, 0.8, 0.7]),
    )
    z = np.random.rand(10, 2)
    y_hat, pr0_hat, selection0 = InstanceSpace._explore_pythia(space, z)
    assert y_hat.shape == (10, 3)
    assert pr0_hat.shape == (10, 3)
    assert selection0.shape == (10,)
    assert y_hat.dtype == np.bool_


def test_pythia_gaussian_decision_arithmetic():
    # MATLAB fitcsvm gaussian: K(x,y) = exp(-||x-y||^2 / KernelScale^2).
    # Single SV at origin, alpha=1, bias=0, scale=1 → decision = exp(-||z||^2).
    # Platt A=-1, B=0 → post_good = 1/(1+exp(-decision)).
    svm = make_svm(
        sv=np.array([[0.0, 0.0]]),
        alpha=np.array([1.0]),
        bias=0.0,
        kernel_fn="gaussian",
        kernel_param=1.0,
        platt_a=-1.0,
        platt_b=0.0,
    )
    space = make_instance_space(
        mu=np.array([0.0, 0.0]),
        sigma=np.array([1.0, 1.0]),
        svms=[svm],
        precision=np.array([1.0]),
    )
    z = np.array([[0.0, 0.0], [1.0, 0.0], [10.0, 0.0]])
    _, pr0_hat, _ = InstanceSpace._explore_pythia(space, z)

    # z=(0,0): decision=exp(0)=1,   post_good=1/(1+e^-1)≈0.7311
    # z=(1,0): decision=exp(-1),    post_good=1/(1+e^-exp(-1))
    # z=(10,0): decision≈0,         post_good≈0.5
    expected_pr0 = np.array([
        1.0 - 1.0 / (1.0 + np.exp(-1.0)),
        1.0 - 1.0 / (1.0 + np.exp(-np.exp(-1.0))),
        1.0 - 1.0 / (1.0 + np.exp(-np.exp(-100.0))),
    ])
    np.testing.assert_allclose(pr0_hat[:, 0], expected_pr0, rtol=1e-10)


def test_pythia_zscore_normalization():
    # Confirm z is z-score normalised before kernel eval.
    # With mu=(1,2), sigma=(2,4), input z=(1,2) → normalised (0,0) → decision=1
    svm = make_svm(
        sv=np.array([[0.0, 0.0]]),
        alpha=np.array([1.0]),
        bias=0.0,
        kernel_fn="gaussian",
        kernel_param=1.0,
        platt_a=-1.0,
        platt_b=0.0,
    )
    space = make_instance_space(
        mu=np.array([1.0, 2.0]),
        sigma=np.array([2.0, 4.0]),
        svms=[svm],
        precision=np.array([1.0]),
    )
    z = np.array([[1.0, 2.0]])
    _, pr0_hat, _ = InstanceSpace._explore_pythia(space, z)
    # Normalised point is (0,0), so decision = exp(0) = 1
    expected = 1.0 - 1.0 / (1.0 + np.exp(-1.0))
    np.testing.assert_allclose(pr0_hat[0, 0], expected, rtol=1e-10)


def test_pythia_selection0_picks_highest_precision_positive():
    # Two algos both predict "good"; higher precision wins.
    strong = make_svm(
        sv=np.array([[0.0, 0.0]]),
        alpha=np.array([10.0]),
        bias=0.0,
        kernel_fn="gaussian",
        kernel_param=1.0,
        platt_a=-1.0,
        platt_b=0.0,
    )
    space = make_instance_space(
        mu=np.array([0.0, 0.0]),
        sigma=np.array([1.0, 1.0]),
        svms=[strong, strong],
        precision=np.array([0.5, 0.9]),
    )
    z = np.array([[0.0, 0.0]])
    y_hat, _, selection0 = InstanceSpace._explore_pythia(space, z)
    assert y_hat[0, 0] and y_hat[0, 1]
    assert selection0[0] == 1  # higher precision algo


def test_pythia_selection0_none_when_no_positive():
    # All algos predict "bad" (post_good < 0.5); selection0 = -1
    weak = make_svm(
        sv=np.array([[0.0, 0.0]]),
        alpha=np.array([-10.0]),  # strongly negative decision
        bias=0.0,
        kernel_fn="gaussian",
        kernel_param=1.0,
        platt_a=-1.0,
        platt_b=0.0,
    )
    space = make_instance_space(
        mu=np.array([0.0, 0.0]),
        sigma=np.array([1.0, 1.0]),
        svms=[weak, weak],
        precision=np.array([0.9, 0.9]),
    )
    z = np.array([[0.0, 0.0]])
    y_hat, _, selection0 = InstanceSpace._explore_pythia(space, z)
    assert not y_hat[0, 0] and not y_hat[0, 1]
    assert selection0[0] == -1
