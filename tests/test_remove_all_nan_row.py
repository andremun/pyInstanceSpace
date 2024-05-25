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

from matilda.data.model import (
    Data,
)
from matilda.stages.pre_processing import PrePro

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

    data = Data(
        inst_labels=pd.Series(["inst" + str(i) for i in range(10)]),
        feat_labels=[f"feature{i}" for i in range(10)],
        algo_labels=[f"algo{i}" for i in range(5)],
        x=large_x,
        y=large_y,
        x_raw=np.array([], dtype=np.double),
        y_raw=np.array([], dtype=np.double),
        y_bin=np.array([], dtype=np.bool_),
        y_best=np.array([], dtype=np.double),
        p=np.array([], dtype=np.double),
        num_good_algos=np.array([], dtype=np.double),
        beta=np.array([], dtype=np.bool_),
        s=None,
        uniformity=None,
    )

    out = PrePro.remove_instances_with_many_missing_values(data)

    expected_rows = 8  # two rows (instances) should be removed
    expected_x_columns = 9  # first column should be removed
    expected_y_columns = 5

    # Check instances removal
    assert (
            out.x.shape[0] == expected_rows
    ), "Instances with all NaN values not removed correctly"
    assert (
            out.y.shape[0] == expected_rows
    ), "Instances with all NaN values not removed correctly"
    assert (
            out.inst_labels.shape[0] == expected_rows
    ), "Instance labels not updated after removal"

    # Check feature dimensions
    assert out.x.shape[1] == expected_x_columns, "x dimensions should not change"
    assert out.y.shape[1] == expected_y_columns, "y dimensions should not change"

    # Check inst_labels content
    assert out.inst_labels.tolist() == [
        "inst" + str(i) for i in range(2, 10)
    ], "inst_labels content not right"

    assert out.feat_labels == [f"feature{i}" for i in range(1, 10)], \
        "feat_labels content not right"


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
    s = ["source" + str(i) for i in range(10)]  # Generate content for s

    data = Data(
        inst_labels=pd.Series(["inst" + str(i) for i in range(10)]),
        feat_labels=[f"feature{i}" for i in range(10)],
        algo_labels=[f"algo{i}" for i in range(5)],
        x=large_x,
        y=large_y,
        x_raw=np.array([], dtype=np.double),
        y_raw=np.array([], dtype=np.double),
        y_bin=np.array([], dtype=np.bool_),
        y_best=np.array([], dtype=np.double),
        p=np.array([], dtype=np.double),
        num_good_algos=np.array([], dtype=np.double),
        beta=np.array([], dtype=np.bool_),
        s=s,
        uniformity=None,
    )

    out = PrePro.remove_instances_with_many_missing_values(data)

    expected_rows = 7

    # Check instances removal
    assert (
            out.x.shape[0] == expected_rows
    ), "Instances with all NaN values not removed correctly"
    assert (
            out.y.shape[0] == expected_rows
    ), "Instances with all NaN values not removed correctly"
    assert (
            out.inst_labels.shape[0] == expected_rows
    ), "Instance labels not updated after removal"

    expected_x_columns = 10
    expected_y_columns = 5

    # Check feature dimensions are unchanged
    assert out.x.shape[1] == expected_x_columns, "x dimensions should not change"
    assert out.y.shape[1] == expected_y_columns, "y dimensions should not change"

    # Check inst_labels content
    assert out.inst_labels.tolist() == [
        "inst0",
        "inst1",
        "inst5",
        "inst6",
        "inst7",
        "inst8",
        "inst9",
    ], "inst_labels content not right"

    # Check feat_labels content
    assert out.feat_labels == [f"feature{i}" for i in range(10)], \
        "feat_labels content not right"

    assert out.s is not None, "s content should be valid"
    assert out.s == [
        "source0",
        "source1",
        "source5",
        "source6",
        "source7",
        "source8",
        "source9",
    ], "s content not right"


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

    s = ["source" + str(i) for i in range(10)]  # Generate content for s

    data = Data(
        inst_labels=pd.Series(["inst" + str(i) for i in range(10)]),
        feat_labels=[f"feature{i}" for i in range(10)],
        algo_labels=[f"algo{i}" for i in range(5)],
        x=large_x,
        y=large_y,
        x_raw=np.array([], dtype=np.double),
        y_raw=np.array([], dtype=np.double),
        y_bin=np.array([], dtype=np.bool_),
        y_best=np.array([], dtype=np.double),
        p=np.array([], dtype=np.double),
        num_good_algos=np.array([], dtype=np.double),
        beta=np.array([], dtype=np.bool_),
        s=s,
        uniformity=None,
    )

    out = PrePro.remove_instances_with_many_missing_values(data)

    expected_rows = 10
    expected_x_columns = 5
    expected_y_columns = 5

    # Check instances removal
    assert (
            out.x.shape[0] == expected_rows
    ), "Instances with all NaN values not removed correctly"
    assert (
            out.y.shape[0] == expected_rows
    ), "Instances with all NaN values not removed correctly"
    assert (
            out.inst_labels.shape[0] == expected_rows
    ), "Instance labels not updated after removal"

    # Check feature dimensions are unchanged
    assert out.x.shape[1] == expected_x_columns, "x dimensions should not change"
    assert out.y.shape[1] == expected_y_columns, "y dimensions should not change"

    # Check inst_labels content
    assert out.inst_labels.tolist() == [
        "inst" + str(i) for i in range(0, 10)
    ], "inst_labels content not right"

    # Check feat_labels content
    assert out.feat_labels == [
        "feature" + str(i) for i in range(5, 10)
    ], "feat_labels content not right"


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
    large_x = np.array([
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
    ])

    large_y = rng.random((10, 5))
    large_y[6, :2] = np.nan  # 7th row first 2 columns NaN
    s = ["source" + str(i) for i in range(10)]  # Generate content for s

    data = Data(
        inst_labels=pd.Series(["inst" + str(i) for i in range(10)]),
        feat_labels=[f"feature{i}" for i in range(10)],
        algo_labels=[f"algo{i}" for i in range(5)],
        x=large_x,
        y=large_y,
        x_raw=np.array([], dtype=np.double),
        y_raw=np.array([], dtype=np.double),
        y_bin=np.array([], dtype=np.bool_),
        y_best=np.array([], dtype=np.double),
        p=np.array([], dtype=np.double),
        num_good_algos=np.array([], dtype=np.double),
        beta=np.array([], dtype=np.bool_),
        s=s,
        uniformity=None,
    )

    out = PrePro.remove_instances_with_many_missing_values(data)

    expected_rows = 10
    expected_x_columns = 5
    expected_y_columns = 5

    # Check instances removal
    assert (
            out.x.shape[0] == expected_rows
    ), "Instances with all NaN values not removed correctly"
    assert (
            out.y.shape[0] == expected_rows
    ), "Instances with all NaN values not removed correctly"
    assert (
            out.inst_labels.shape[0] == expected_rows
    ), "Instance labels not updated after removal"

    # Check feature dimensions are unchanged
    assert out.x.shape[1] == expected_x_columns, "x dimensions should not change"
    assert out.y.shape[1] == expected_y_columns, "y dimensions should not change"

    # Check inst_labels content
    assert out.inst_labels.tolist() == [
        "inst" + str(i) for i in range(0, 10)
    ], "inst_labels content not right"

    # Check feat_labels content
    assert out.feat_labels == [
        "feature" + str(i) for i in range(5, 10)
    ], "feat_labels content not right"

    assert out.s is not None, "s content should be valid"

    assert out.s == ["source" + str(i) for i in range(0, 10)], \
        "s content not right"


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
    large_x = np.array([
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
    ])

    large_y = rng.random((10, 5))
    large_y[6, :2] = np.nan  # 7th row first 2 columns NaN

    x_raw = rng.random((10, 10))
    y_raw = rng.random((10, 5))
    y_bin = rng.choice([True, False], size=(10, 5))
    y_best = rng.random((10, 5))
    p = rng.random((10, 5))
    num_good_algos = np.array([], dtype=np.double)
    beta = rng.choice([True, False], size=(10, 5))
    s = ["string" + str(i) for i in range(10)]

    data = Data(
        inst_labels=pd.Series(["inst" + str(i) for i in range(10)]),
        feat_labels=[f"feature{i}" for i in range(10)],
        algo_labels=[f"algo{i}" for i in range(5)],
        x=large_x,
        y=large_y,
        x_raw=x_raw,
        y_raw=y_raw,
        y_bin=y_bin,
        y_best=y_best,
        p=p,
        num_good_algos=num_good_algos,
        beta=beta,
        s=s,
        uniformity=None,
    )

    out = PrePro.remove_instances_with_many_missing_values(data)

    expected_rows = 10
    expected_x_columns = 10
    expected_y_columns = 5

    # Check instances removal
    assert (
            out.x.shape[0] == expected_rows
    ), "Instances with all NaN values not removed correctly"
    assert (
            out.y.shape[0] == expected_rows
    ), "Instances with all NaN values not removed correctly"
    assert (
            out.inst_labels.shape[0] == expected_rows
    ), "Instance labels not updated after removal"

    # Check feature dimensions are unchanged
    assert out.x.shape[1] == expected_x_columns, "x dimensions should not change"
    assert out.y.shape[1] == expected_y_columns, "y dimensions should not change"

    # Check inst_labels content
    assert out.inst_labels.tolist() == [
        "inst" + str(i) for i in range(0, 10)
    ], "inst_labels content not right"

    # Check feat_labels content
    assert out.feat_labels == [
        "feature" + str(i) for i in range(0, 10)
    ], "feat_labels content not right"

    # Check algo_labels content
    assert out.algo_labels == [
        "algo" + str(i) for i in range(0, 5)
    ], "algo_labels content not right"

    # Check x_raw content
    assert np.array_equal(out.x_raw, x_raw), "x_raw content not right"

    # Check y_raw content
    assert np.array_equal(out.y_raw, y_raw), "y_raw content not right"

    # Check y_bin content
    assert np.array_equal(out.y_bin, y_bin), "y_bin content not right"

    # Check y_best content
    assert np.array_equal(out.y_best, y_best), "y_best content not right"

    # Check p content
    assert np.array_equal(out.p, p), "p content not right"

    # Check num_good_algos content
    assert np.array_equal(out.num_good_algos, num_good_algos), \
        "num_good_algos content not right"

    # Check beta content
    assert np.array_equal(out.beta, beta), "beta content not right"

    assert out.s is not None, "s content should be valid"

    # Check s content
    assert out.s == ["string" + str(i) for i in range(10)], \
        "s content not right"
