"""Tests for instancespace.data.metadata.Metadata.

Covers the "feature_"/"algo_" column-naming convention: metadata.csv columns
carry that prefix to disambiguate column types, but the prefix must not leak
into feature_names/algorithm_names, since those are used verbatim in graph
labels, exported CSV headers, and filenames (see issue #222).
"""

import pandas as pd

from instancespace.data.metadata import Metadata


def test_feature_and_algorithm_names_strip_prefix() -> None:
    """feature_/algo_ column prefixes must not leak into the parsed names."""
    data = pd.DataFrame(
        {
            "Instances": ["inst1", "inst2"],
            "feature_Entropy": [0.1, 0.2],
            "feature_ErrorRate": [0.3, 0.4],
            "algo_CART": [0.5, 0.6],
            "algo_KNN": [0.7, 0.8],
        },
    )

    metadata = Metadata.from_data_frame(data)

    assert metadata.feature_names == ["Entropy", "ErrorRate"]
    assert metadata.algorithm_names == ["CART", "KNN"]


def test_feature_and_algorithm_names_strip_prefix_case_insensitively() -> None:
    """Column-type detection is case-insensitive; stripping must match it."""
    data = pd.DataFrame(
        {
            "instances": ["inst1"],
            "Feature_Entropy": [0.1],
            "ALGO_CART": [0.5],
        },
    )

    metadata = Metadata.from_data_frame(data)

    assert metadata.feature_names == ["Entropy"]
    assert metadata.algorithm_names == ["CART"]
