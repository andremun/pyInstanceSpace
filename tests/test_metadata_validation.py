# SPDX-License-Identifier: LicenseRef-PolyForm-Noncommercial-1.0.0
# Copyright (c) 2024-2026 Mario Andrés Muñoz
"""Tests for strict metadata schema and shape validation."""

from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import pytest

from instancespace.data.metadata import Metadata, from_csv_file


def _valid_frame() -> pd.DataFrame:
    """Return the smallest generally valid parsed metadata frame."""
    return pd.DataFrame(
        {
            "instances": [1],
            "feature_one": [0.1],
            "feature_two": [0.2],
            "feature_three": [0.3],
            "algo_solver": [1.0],
        },
    )


@pytest.mark.parametrize(
    "columns",
    [
        ["feature_one"],
        ["instances", "INSTANCES", "feature_one"],
        ["instances", "source", "SOURCE", "feature_one"],
    ],
)
def test_metadata_rejects_ambiguous_reserved_columns(columns: list[str]) -> None:
    """Reserved columns have exact case-insensitive cardinalities."""
    data = pd.DataFrame([[1.0] * len(columns)], columns=columns)

    with pytest.raises(ValueError, match="instances|source"):
        Metadata.from_data_frame(data)


def test_one_row_metadata_keeps_series_and_numeric_labels_become_text() -> None:
    """A one-row frame must not collapse labels to a scalar."""
    metadata = Metadata.from_data_frame(_valid_frame())

    assert isinstance(metadata.instance_labels, pd.Series)
    labels = cast(list[str], metadata.instance_labels.tolist())
    assert labels == ["1"]
    assert metadata.features.shape == (1, 3)
    assert metadata.algorithms.shape == (1, 1)


@pytest.mark.parametrize(
    ("column", "values"),
    [
        ("feature_one", ["not numeric"]),
        ("algo_solver", ["not numeric"]),
        ("feature_one", [True]),
        ("algo_solver", [False]),
        ("feature_one", [np.inf]),
        ("algo_solver", [-np.inf]),
    ],
)
def test_metadata_rejects_invalid_feature_and_algorithm_values(
    column: str,
    values: list[object],
) -> None:
    """Feature and algorithm columns must be finite-or-missing real values."""
    data = _valid_frame()
    data[column] = values

    with pytest.raises(ValueError, match="numeric|finite"):
        Metadata.from_data_frame(data)


def test_metadata_allows_nan_values() -> None:
    """Missing numeric observations remain valid metadata values."""
    data = _valid_frame()
    data.loc[0, "feature_one"] = np.nan
    data.loc[0, "algo_solver"] = np.nan

    metadata = Metadata.from_data_frame(data)

    assert np.isnan(metadata.features[0, 0])
    assert np.isnan(metadata.algorithms[0, 0])


@pytest.mark.parametrize(
    "columns",
    [
        ["instances", "feature_A", "Feature_a"],
        ["instances", "algo_A", "ALGO_a", "feature_one"],
        ["instances", "feature_", "algo_A"],
    ],
)
def test_metadata_rejects_empty_or_duplicate_stripped_names(
    columns: list[str],
) -> None:
    """Names must remain nonempty and unique after prefix removal."""
    data = pd.DataFrame([[1.0] * len(columns)], columns=columns)

    with pytest.raises(ValueError, match="nonempty|unique"):
        Metadata.from_data_frame(data)


def test_feature_only_metadata_is_valid_for_explore() -> None:
    """The generic metadata type permits an empty algorithm matrix."""
    data = _valid_frame().drop(columns="algo_solver")

    metadata = Metadata.from_data_frame(data)

    assert metadata.algorithm_names == []
    assert metadata.algorithms.shape == (1, 0)


@pytest.mark.parametrize(
    "change",
    [
        {"feature_names": ["one"]},
        {"algorithm_names": []},
        {"instance_labels": pd.Series(["a", "b"])},
        {"instance_sources": pd.Series(["a", "b"])},
        {"algorithms": np.ones((2, 1))},
    ],
)
def test_direct_metadata_construction_rejects_inconsistent_shapes(
    change: dict[str, object],
) -> None:
    """Direct construction follows the same shape contract as CSV parsing."""
    values: dict[str, object] = {
        "feature_names": ["one", "two", "three"],
        "algorithm_names": ["solver"],
        "instance_labels": pd.Series(["a"]),
        "instance_sources": pd.Series(["source"]),
        "features": np.ones((1, 3)),
        "algorithms": np.ones((1, 1)),
    }
    values.update(change)

    with pytest.raises(ValueError, match="match|same number"):
        Metadata(**values)  # type: ignore[arg-type]


def test_csv_duplicate_header_is_rejected_before_pandas_mangles_it(
    tmp_path: Path,
) -> None:
    """Raw duplicate headers must not become synthetic '.1' columns."""
    path = tmp_path / "metadata.csv"
    path.write_text(
        "instances,feature_one,feature_one,algo_solver\n" "one,1.0,2.0,3.0\n",
        encoding="utf-8",
    )

    assert from_csv_file(path) is None
