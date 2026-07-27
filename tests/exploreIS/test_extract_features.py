"""Regression test for #253: explore()'s feature extraction matches columns by name.

Q5 confirmed this is deliberate, permanent behaviour (not an accidental divergence from
MATLAB's stricter ``featureOrderMismatch`` error) -- test metadata's feature columns may
be supplied in any order, since ``_extract_features`` matches them by name, not position.
"""

from types import SimpleNamespace

import numpy as np

from instancespace.instance_space import InstanceSpace


def test_extract_features_matches_columns_by_name_not_position():
    space = InstanceSpace.__new__(InstanceSpace)
    model_data = SimpleNamespace(feat_labels=["a", "b", "c"])
    space._explore_model = SimpleNamespace(data=model_data)

    # Test metadata supplies columns in a different order than training.
    metadata = SimpleNamespace(
        feature_names=["c", "a", "b"],
        features=np.array([
            [30.0, 10.0, 20.0],
            [31.0, 11.0, 21.0],
        ]),
    )

    x = space._extract_features(metadata)

    # Regardless of the input column order, output columns must be [a, b, c].
    expected = np.array([
        [10.0, 20.0, 30.0],
        [11.0, 21.0, 31.0],
    ])
    np.testing.assert_array_equal(x, expected)
