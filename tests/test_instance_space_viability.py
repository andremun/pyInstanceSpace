# ruff: noqa: SLF001
"""InstanceSpace boundary tests for standard-build and explore viability."""

from typing import cast

import numpy as np
import pandas as pd
import pytest

from instancespace.data.metadata import Metadata
from instancespace.instance_space import InstanceSpace
from instancespace.model import Model
from tests.utils.option_creator import create_option


def _metadata(*, features: int, algorithms: int) -> Metadata:
    """Create shape-consistent metadata with the requested column counts."""
    return Metadata(
        feature_names=[f"feature_{index}" for index in range(features)],
        algorithm_names=[f"algorithm_{index}" for index in range(algorithms)],
        instance_labels=pd.Series(["instance"]),
        instance_sources=None,
        features=np.ones((1, features), dtype=np.double),
        algorithms=np.ones((1, algorithms), dtype=np.double),
    )


def test_instance_space_rejects_nonviable_standard_build_metadata() -> None:
    """The public build boundary rejects fewer than three features eagerly."""
    metadata = _metadata(features=2, algorithms=1)

    with pytest.raises(ValueError, match="Build metadata.*at least three features"):
        InstanceSpace(metadata, create_option())


def test_explore_accepts_feature_only_metadata_with_viable_features() -> None:
    """Explore does not require test-set algorithm performance columns."""
    space = InstanceSpace.__new__(InstanceSpace)
    space._metadata = _metadata(features=3, algorithms=1)
    space._require_model = lambda: cast(Model, object())  # type: ignore[method-assign]

    space._validate_for_explore(_metadata(features=3, algorithms=0))


def test_explore_rejects_nonviable_feature_dimensions() -> None:
    """The public explore boundary rejects fewer than three test features."""
    space = InstanceSpace.__new__(InstanceSpace)
    space._metadata = _metadata(features=2, algorithms=1)
    space._require_model = lambda: cast(Model, object())  # type: ignore[method-assign]

    with pytest.raises(ValueError, match="Explore metadata.*at least three features"):
        space._validate_for_explore(_metadata(features=2, algorithms=0))
