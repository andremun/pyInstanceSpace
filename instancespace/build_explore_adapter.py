"""Translate a Python-trained model into the form ``InstanceSpace.explore()`` consumes.

``build()`` and ``explore()`` were written against two different model
representations. ``explore()`` is an inference-only port of MATLAB ``exploreIS.m``
and therefore reads the MATLAB-exported *artifact* form: each PYTHIA SVM is a small
record of signed support-vector coefficients, a kernel scale, a bias and Platt
``A``/``B`` coefficients, and the model carries the original feature labels.
``build()`` instead stores each SVM as a fitted scikit-learn ``SVC`` and keeps only
the post-SIFTED feature labels. The two are not interchangeable, so a model produced
by ``build()`` cannot be fed straight into ``explore()``.

This module is the middleware that bridges them. Four of the five inference stages
already match and are passed through unchanged (PRELIM, SIFTED, PILOT, TRACE); only
PYTHIA is translated:

* each ``SVC`` is rewritten into the artifact record ``explore()`` expects, and
* the z-score normalisation constants are recomputed from the trained projection,
  because ``build()`` stores the *post*-normalisation constants (mean ~ 0, std ~ 1),
  which would turn the explore-time normalisation into a no-op.

``explore()`` itself is left untouched, so its validated 1:1 fidelity to MATLAB is
preserved; the adapter is a separate layer on top.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Sequence

import numpy as np


def _svc_to_artifact(svc: Any) -> SimpleNamespace:
    """Rewrite one fitted scikit-learn ``SVC`` into the explore() artifact record.

    The Gaussian kernel scale is converted so that the artifact kernel
    ``exp(-||x-y||^2 / s^2)`` equals scikit-learn's ``exp(-gamma ||x-y||^2)``, i.e.
    ``s = gamma**-0.5``. ``dual_coef_`` already carries the signed Lagrange
    multipliers, and ``intercept_`` is the bias, so both transfer directly.
    """
    if svc.kernel == "rbf":
        kernel_fn = "gaussian"
        kernel_param = 1.0 / np.sqrt(svc._gamma)
    elif svc.kernel == "linear":
        kernel_fn = "linear"
        kernel_param = 0.0
    else:
        raise NotImplementedError(
            f"the build->explore adapter does not support the {svc.kernel!r} kernel; "
            "only the Gaussian and linear PYTHIA kernels are handled"
        )
    return SimpleNamespace(
        support_vectors=np.asarray(svc.support_vectors_, dtype=np.double),
        alphas=np.asarray(svc.dual_coef_, dtype=np.double).ravel(),
        bias=float(svc.intercept_[0]),
        kernel_fn=kernel_fn,
        kernel_param=float(kernel_param),
        platt_A=float(svc.probA_[0]),
        platt_B=float(svc.probB_[0]),
    )


def adapt_for_explore(model: Any, feature_names: Sequence[str]) -> SimpleNamespace:
    """Return an ``explore()``-compatible model from a ``build()`` model.

    Parameters
    ----------
    model:
        The model returned by ``InstanceSpace.build()`` (or ``InstanceSpace.model``).
    feature_names:
        The original, pre-SIFTED training feature labels in the order PRELIM was
        fitted, e.g. the ``feature_names`` of the metadata the model was built from.
        ``build()`` overwrites the model's feature labels with the post-SIFTED
        subset, which would make ``explore()``'s feature extraction select the wrong
        columns.

    Notes
    -----
    Assign the result to the instance space before exploring::

        space.build()
        space._model = adapt_for_explore(space.model, train_metadata.feature_names)
        result = space.explore(test_metadata)
    """
    z = np.asarray(model.pilot.z, dtype=np.double)
    pythia = SimpleNamespace(
        mu=z.mean(axis=0),
        sigma=z.std(axis=0, ddof=1),
        precision=np.asarray(model.pythia.precision, dtype=np.double),
        svm=[_svc_to_artifact(svc) for svc in model.pythia.svm],
    )
    return SimpleNamespace(
        prelim=model.prelim,
        sifted=model.sifted,
        pilot=model.pilot,
        pythia=pythia,
        trace=model.trace,
        data=SimpleNamespace(feat_labels=list(feature_names)),
    )
