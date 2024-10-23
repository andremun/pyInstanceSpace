"""
Contains test cases for the remove_instances_with_many_missing_values function.

These testing codes are tested by artificial data
(the data that I generated, rather than read from CSV)
and check against with the logic of original codes of BuildIS

"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from matilda.stages.preprocessing import PreprocessingStage

path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))


def test_remove_instances_with_two_row_missing() -> None:
    """
    Test case.

    expected outcome: 1. 2 rows(0,1) will be removed in both X and Y
                      2. For X, 1 column( feature0 ) will be removed
                      3. inst_labels content will change
                      4. feat_labels will change
    """
    rng = np.random

    # Create sample data with missing values (10 rows)
    large_x = rng.random((10, 10))
    large_x[0, :] = np.nan  # First row all NaN
    large_x[1, :5] = np.nan  # Second row first 5 columns NaN
    large_x[:, 0] = np.nan  # First column all NaN (> 20% missing)

    large_y = rng.random((10, 5))
    large_y[1, :] = np.nan  # second row all NaN

    inst_labels = pd.Series(["inst" + str(i) for i in range(10)])
    feat_labels = [f"feature{i}" for i in range(10)]

    new_x, new_y, new_inst_labels, new_feat_labels, _ = (
        PreprocessingStage.remove_instances_with_many_missing_values(
            x=large_x,
            y=large_y,
            inst_labels=inst_labels,
            feat_labels=feat_labels,
            s=None,
        )
    )

    expected_rows = 8  # two rows (instances) should be removed
    expected_x_columns = 9  # first column should be removed
    expected_y_columns = 5

    # Assertions to verify correct modifications
    assert new_x.shape == (
        expected_rows,
        expected_x_columns,
    ), "X dimensions are incorrect after removal"
    assert new_y.shape == (
        expected_rows,
        expected_y_columns,
    ), "Y dimensions are incorrect after removal"
    assert (
        new_inst_labels.size == expected_rows
    ), "Incorrect number of instance labels after removal"
    assert (
        len(new_feat_labels) == expected_x_columns
    ), "Feature labels were not updated correctly"

    # Check instance labels content
    assert new_inst_labels.tolist() == [
        "inst" + str(i) for i in range(2, 10)
    ], "Instance labels content incorrect"
    assert new_feat_labels == [
        f"feature{i}" for i in range(1, 10)
    ], "Feature labels content incorrect"


def test_remove_instances_with_3_row_missing() -> None:
    """
    Test case.

    expected outcome: 1. 3 rows(2,3,4) will be removed in both X and Y
                      2. For X, columns will be unchanged
                      3. inst_labels content will change
                      4. feat_labels will unchanged
    """
    rng = np.random

    # Create sample data with missing values (10 rows)
    large_x = rng.random((10, 10))
    large_x[2, :] = np.nan  # third row all NaN
    large_x[1, :5] = np.nan  # Second row first 5 columns NaN

    large_y = rng.random((10, 5))
    large_y[4, :] = np.nan  # fifth row all NaN
    large_y[3, :] = np.nan  # forth row all NaN

    inst_labels = pd.Series(["inst" + str(i) for i in range(10)])
    feat_labels = [f"feature{i}" for i in range(10)]
    s = pd.Series(["source" + str(i) for i in range(10)])  # Generate content for s

    new_x, new_y, new_inst_labels, new_feat_labels, new_s = (
        PreprocessingStage.remove_instances_with_many_missing_values(
            x=large_x,
            y=large_y,
            inst_labels=inst_labels,
            feat_labels=feat_labels,
            s=s,
        )
    )
    expected_rows = 7

    expected_x_columns = 10
    expected_y_columns = 5

    # Assertions to verify correct modifications
    assert new_x.shape == (
        expected_rows,
        expected_x_columns,
    ), "X dimensions are incorrect after removal"
    assert new_y.shape == (
        expected_rows,
        expected_y_columns,
    ), "Y dimensions are incorrect after removal"
    assert (
        new_inst_labels.size == expected_rows
    ), "Incorrect number of instance labels after removal"
    assert new_feat_labels == feat_labels, "Feature labels should not change"

    # Check instance labels content
    assert new_inst_labels.tolist() == [
        "inst" + str(i) for i in range(10) if i not in [2, 3, 4]
    ], "Instance labels content incorrect"

    # Check source series content
    assert new_s is not None
    assert new_s.tolist() == [
        "source" + str(i) for i in range(10) if i not in [2, 3, 4]
    ], "Source content incorrect"


def test_remove_instances_keep_same() -> None:
    """
    Test case.

    expected outcome: 1. No rows will be removed in both X and Y
                      2. For X, only 5 columns will remained
                      3. inst_labels content unchanged
                      4. feat_labels will only contain feature 5 to 9
    """
    rng = np.random

    # Create sample data with missing values (10 rows)
    large_x = rng.random((10, 10))

    large_x[1, :5] = np.nan  # Second row first 5 columns NaN

    large_x[4, :5] = np.nan  # 5th row first 5 columns NaN

    large_y = rng.random((10, 5))
    large_y[6, :2] = np.nan  # 7th row first 2columns NaN

    inst_labels = pd.Series(["inst" + str(i) for i in range(10)])
    feat_labels = [f"feature{i}" for i in range(10)]
    s = pd.Series(["source" + str(i) for i in range(10)])  # Generate content for s

    new_x, new_y, new_inst_labels, new_feat_labels, new_s = (
        PreprocessingStage.remove_instances_with_many_missing_values(
            x=large_x,
            y=large_y,
            inst_labels=inst_labels,
            feat_labels=feat_labels,
            s=s,
        )
    )

    expected_rows = 10
    expected_x_columns = 5
    expected_y_columns = 5

    # Assertions to verify correct modifications
    assert new_x.shape == (
        expected_rows,
        expected_x_columns,
    ), "X dimensions are incorrect after removal"
    assert new_y.shape == (
        expected_rows,
        expected_y_columns,
    ), "Y dimensions are incorrect; all should remain"
    assert new_inst_labels.size == expected_rows, "Instance labels should not change"
    assert (
        len(new_feat_labels) == expected_x_columns
    ), "Feature labels should be updated correctly"

    # Check instance labels content
    assert new_inst_labels.tolist() == [
        "inst" + str(i) for i in range(10)
    ], "Instance labels content incorrect"

    # Check feature labels content
    assert new_feat_labels == [
        f"feature{i}" for i in range(5, 10)
    ], "Feature labels content incorrect"

    # Check source series content
    assert new_s is not None
    assert new_s.tolist() == [
        "source" + str(i) for i in range(10)
    ], "Source content incorrect"


def test_duplicated_data_edge() -> None:
    """
    Test case.

    expected outcome: 1. No rows will be removed in both X and Y
                      2. For X, only 5 columns will remained
                      3. inst_labels content unchanged
                      4. feat_labels will only contain feature 5 to 9
                      5. not print: too many repeated instances
    """
    rng = np.random

    # Create sample data with missing values and repeated instances (10 rows)
    large_x = np.array(
        [
            [0.72, 0.45, 0.18, 0.79, 0.65, 0.33, 0.31, 0.07, 0.54, 0.42],
            [np.nan, np.nan, np.nan, np.nan, np.nan, 0.09, 0.76, 0.91, 0.62, 0.17],
            [0.72, 0.45, 0.18, 0.79, 0.65, 0.33, 0.31, 0.07, 0.54, 0.42],
            [0.72, 0.45, 0.18, 0.79, 0.65, 0.33, 0.31, 0.07, 0.54, 0.42],
            [np.nan, np.nan, np.nan, np.nan, np.nan, 0.87, 0.52, 0.63, 0.29, 0.71],
            [0.72, 0.45, 0.18, 0.79, 0.65, 0.33, 0.31, 0.07, 0.54, 0.42],
            [0.35, 0.64, 0.97, 0.82, 0.46, 0.30, 0.01, 0.50, 0.28, 0.13],
            [0.75, 0.89, 0.52, 0.40, 0.60, 0.77, 0.92, 0.34, 0.08, 0.44],
            [0.72, 0.45, 0.18, 0.79, 0.65, 0.33, 0.31, 0.07, 0.54, 0.42],
            [0.72, 0.45, 0.18, 0.79, 0.65, 0.33, 0.31, 0.07, 0.54, 0.42],
        ],
    )

    large_y = rng.random((10, 5))
    large_y[6, :2] = np.nan  # 7th row first 2 columns NaN

    inst_labels = pd.Series(["inst" + str(i) for i in range(10)])
    feat_labels = [f"feature{i}" for i in range(10)]
    s = pd.Series(["source" + str(i) for i in range(10)])  # Generate content for s

    new_x, new_y, new_inst_labels, new_feat_labels, new_s = (
        PreprocessingStage.remove_instances_with_many_missing_values(
            x=large_x,
            y=large_y,
            inst_labels=inst_labels,
            feat_labels=feat_labels,
            s=s,
        )
    )

    expected_rows = 10
    expected_x_columns = 5
    expected_y_columns = 5

    # Check instances removal
    assert new_x.shape == (
        expected_rows,
        expected_x_columns,
    ), "X dimensions are incorrect after removal"
    assert new_y.shape == (
        expected_rows,
        expected_y_columns,
    ), "Y dimensions are incorrect; all should remain"
    assert new_inst_labels.size == expected_rows, "Instance labels should not change"
    assert (
        len(new_feat_labels) == expected_x_columns
    ), "Feature labels should be updated correctly"
    assert new_inst_labels.tolist() == [
        "inst" + str(i) for i in range(10)
    ], "Instance labels content incorrect"
    assert new_feat_labels == [
        f"feature{i}" for i in range(5, 10)
    ], "Feature labels content incorrect"
    assert new_s is not None
    assert new_s.tolist() == [
        "source" + str(i) for i in range(10)
    ], "Source content incorrect"


def test_duplicated_data() -> None:
    """
    Test case.

    expected outcome: 1. No rows will be removed in both X and Y
                      2. For X, all columns will remained
                      3. inst_labels content unchanged
                      4. feat_labels will be unchanged
                      5. print: too many repeated instances
    """
    rng = np.random

    # Create sample data with missing values and repeated instances (10 rows)
    large_x = np.array(
        [
            [0.72, 0.45, 0.18, 0.79, 0.65, 0.33, 0.31, 0.07, 0.54, 0.42],
            [0.72, 0.45, 0.18, 0.79, 0.65, 0.33, 0.31, 0.07, 0.54, 0.42],
            [0.72, 0.45, 0.18, 0.79, 0.65, 0.33, 0.31, 0.07, 0.54, 0.42],
            [0.35, 0.64, 0.97, 0.82, 0.46, 0.30, 0.01, 0.50, 0.28, 0.13],
            [np.nan, np.nan, np.nan, np.nan, np.nan, 0.87, 0.52, 0.63, 0.29, 0.71],
            [0.72, 0.45, 0.18, 0.79, 0.65, 0.33, 0.31, 0.07, 0.54, 0.42],
            [0.35, 0.64, 0.97, 0.82, 0.46, 0.30, 0.01, 0.50, 0.28, 0.13],
            [0.75, 0.89, 0.52, 0.40, 0.60, 0.77, 0.92, 0.34, 0.08, 0.44],
            [0.72, 0.45, 0.18, 0.79, 0.65, 0.33, 0.31, 0.07, 0.54, 0.42],
            [0.72, 0.45, 0.18, 0.79, 0.65, 0.33, 0.31, 0.07, 0.54, 0.42],
        ],
    )

    large_y = rng.random((10, 5))
    large_y[6, :2] = np.nan  # 7th row first 2 columns NaN

    s = pd.Series(["string" + str(i) for i in range(10)])

    inst_labels = pd.Series(["inst" + str(i) for i in range(10)])
    feat_labels = [f"feature{i}" for i in range(10)]

    new_x, new_y, new_inst_labels, new_feat_labels, new_s = (
        PreprocessingStage.remove_instances_with_many_missing_values(
            x=large_x,
            y=large_y,
            inst_labels=inst_labels,
            feat_labels=feat_labels,
            s=s,
        )
    )

    expected_rows = 10
    expected_x_columns = 10
    expected_y_columns = 5

    # Check instances removal
    assert (
        new_x.shape[0] == expected_rows
    ), "Instances with all NaN values not removed correctly"
    assert (
        new_y.shape[0] == expected_rows
    ), "Instances with all NaN values not removed correctly"
    assert (
        new_inst_labels.shape[0] == expected_rows
    ), "Instance labels not updated after removal"

    # Check feature dimensions are unchanged
    assert new_x.shape[1] == expected_x_columns, "x dimensions should not change"
    assert new_y.shape[1] == expected_y_columns, "y dimensions should not change"

    # Check inst_labels content
    assert new_inst_labels.tolist() == [
        "inst" + str(i) for i in range(0, 10)
    ], "inst_labels content not right"

    # Check feat_labels content
    assert new_feat_labels == [
        "feature" + str(i) for i in range(0, 10)
    ], "feat_labels content not right"

    assert new_s is not None, "s content should be valid"

    # Check s content
    assert new_s.tolist() == [
        "string" + str(i) for i in range(10)
    ], "s content not right"


"""
Contains test cases for the remove_instances_with_many_missing_values function.

These testing codes are tested by artificial data
(the data that I generated, rather than read from CSV)
and check against with the logic of original codes of BuildIS

"""
