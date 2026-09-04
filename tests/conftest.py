# SPDX-License-Identifier: LicenseRef-PolyForm-Noncommercial-1.0.0
# Copyright (c) 2024-2026 Mario Andrés Muñoz
"""Shared pytest fixtures.

Purely additive infrastructure: existing test files that build their own
local datasets/options don't need to change to keep working. New tests are
free to use these instead of re-deriving the same small synthetic dataset or
default-options boilerplate that's currently duplicated across ~35 files.
"""

from pathlib import Path
from typing import NamedTuple

import numpy as np
import pytest
from numpy.typing import NDArray

from instancespace.data.options import GeneralOptions, InstanceSpaceOptions
from tools.fixture_provenance import validate_bundle

_CURRENT_MATLAB_FILE_COUNT = 423


class SmallPythiaDataset(NamedTuple):
    """A tiny synthetic dataset shaped like PYTHIA's own inputs."""

    z: NDArray[np.double]
    y: NDArray[np.double]
    y_bin: NDArray[np.bool_]
    y_best: NDArray[np.double]
    algo_labels: list[str]


@pytest.fixture(scope="session")
def verified_current_matlab_bundle() -> Path:
    """Authenticate the installed R2026a oracle before any reader consumes it."""
    bundle = Path(__file__).parent / "fixtures" / "matlab" / "current"
    report = validate_bundle(bundle)
    assert report.trust == "matlab-verified"
    assert report.matlab_release == "R2026a"
    assert report.file_count == _CURRENT_MATLAB_FILE_COUNT
    return bundle


@pytest.fixture()
def small_pythia_dataset() -> SmallPythiaDataset:
    """Build a tiny synthetic dataset for fast, classifier-agnostic PYTHIA tests."""
    rng = np.random.default_rng(0)
    ninst = 20
    nalgos = 2
    coin_flip_threshold = 0.5
    return SmallPythiaDataset(
        z=rng.random((ninst, 2)),
        y=rng.random((ninst, nalgos)),
        y_bin=rng.random((ninst, nalgos)) > coin_flip_threshold,
        y_best=rng.random(ninst),
        algo_labels=["a0", "a1"],
    )


@pytest.fixture()
def deterministic_general_options() -> GeneralOptions:
    """`GeneralOptions` with logging quiet and a fixed seed, for reproducible tests."""
    return GeneralOptions(verbose=False, seed=0)


@pytest.fixture()
def valid_options() -> InstanceSpaceOptions:
    """Return a fully-defaulted, valid `InstanceSpaceOptions` to mutate a field on."""
    return InstanceSpaceOptions.default(*([None] * 12))
