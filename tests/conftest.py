# SPDX-License-Identifier: LicenseRef-PolyForm-Noncommercial-1.0.0
# Copyright (c) 2024-2026 Mario Andrés Muñoz
"""Shared fixtures for validated options and authenticated MATLAB references."""

from pathlib import Path

import pytest

from instancespace.data.options import InstanceSpaceOptions
from tools.fixture_provenance import validate_bundle

_CURRENT_MATLAB_FILE_COUNT = 423


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
def valid_options() -> InstanceSpaceOptions:
    """Return a fully-defaulted, valid `InstanceSpaceOptions` to mutate a field on."""
    return InstanceSpaceOptions.default(*([None] * 12))
