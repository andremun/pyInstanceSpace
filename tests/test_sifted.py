"""Test module for Sifted stage to verify its functionality.

The file contains multiple unit tests to ensure that the `Sifted` class corretly
perform its tasks. The basic mechanism of the test is to compare its output against
output from MATLAB and check if the outputs are the same or reasonable similar. The
tests also include some boundary test where appropriate to test the boundary of the
statement within the methods to ensure they are implemented appropriately.

Tests includes:
- For the function select_features_by_performance, we check xaux value, check if
   features selected are the same
- For the function select_features_by_clustering, we check if number of elements in the
   same clusters for both matlab and python are over given threshold %
- For the function ga, check if the filtered x value, for each row and column, only one
   instances with high correlation, others are low correlation. Test passed if more than
   1 columns/rows don't fulfil this condition

"""

from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from matilda.data.options import ParallelOptions, SiftedOptions
from matilda.stages.sifted import Sifted

# prepare input required for testing
script_dir = Path(__file__).parent
csv_path_x = script_dir / "test_data/sifted/input/0-input_X.csv"
csv_path_y = script_dir / "test_data/sifted/input/0-input_Y.csv"
csv_path_ybin = script_dir / "test_data/sifted/input/0-input_Ybin.csv"
csv_path_feat_labels = script_dir / "test_data/sifted/input/0-input_featlabels.csv"
input_x = np.genfromtxt(csv_path_x, delimiter=",")
input_y = np.genfromtxt(csv_path_y, delimiter=",")
input_ybin = np.genfromtxt(csv_path_ybin, delimiter=",")
feat_labels = np.genfromtxt(csv_path_feat_labels, delimiter=",", dtype=str).tolist()
opts = SiftedOptions.default()
par_opts = ParallelOptions.default(n_cores=10)


def test_select_features_by_performance() -> None:
    """Test performance selection against MATLAB's performance selection output.

    Ensures that `xaux` after filtering by correlation performance is exactly the
    same as MATLAB's output.
    """
    csv_path_xaux = script_dir / "test_data/sifted/output/correlation_matlab.csv"
    xaux_matlab = np.genfromtxt(csv_path_xaux, delimiter=",")

    sifted = Sifted(input_x, input_y, input_ybin, feat_labels, opts, par_opts)
    xaux_python, _, _, _ = sifted.select_features_by_performance()

    assert np.allclose(xaux_matlab, xaux_python, atol=1e-04)


def test_select_features_by_clustering() -> None:
    """Test cluster selection against MATLAB's cluster selection output.

    Despite the difference in cluster labels, we ensure that the number of items in
    python's cluster are 80% same as items in matlab's cluster.
    """
    csv_path_cluster = script_dir / "test_data/sifted/output/clusters_matlab.csv"
    cluster_matlab = np.genfromtxt(csv_path_cluster, delimiter=",")

    rng = np.random.default_rng(seed=0)

    sifted = Sifted(input_x, input_y, input_ybin, feat_labels, opts, par_opts)
    x_aux, _, _, _ = sifted.select_features_by_performance()
    sifted.evaluate_cluster(x_aux, rng)
    _, cluster_python = sifted.select_features_by_clustering(x_aux, rng)

    assert are_same_clusters(cluster_matlab, cluster_python)


def are_same_clusters(
    cluster_a: NDArray[np.intc],
    cluster_b: NDArray[np.intc],
    threshold: float = 0.8,
) -> bool:
    """Check if two clusters have same number of elements more than threshold set.

    Parameters
    ----------
    cluster_a : NDArray[np.intc]
        The first cluster.
    cluster_b : NDArray[np.intc]
        The second cluster.
    threshold : float, optional
        The min ratio of matching elements between the two clusters (default is 0.8).

    Returns
    -------
    bool
        True if the number of matching elements exceeds the threshold, False otherwise.
    """
    cluster_a = np.array(cluster_a)
    cluster_b = np.array(cluster_b)

    unique_labels_a = np.unique(cluster_a)
    total_elements = len(cluster_a)
    matching_elements = 0

    for label in unique_labels_a:
        indices_a = np.where(cluster_a == label)[0]

        # Find the corresponding label in B for the same indices
        label_in_b = cluster_b[indices_a[0]]

        # Count the number of matching labels in B for these indices
        matches = np.sum(cluster_b[indices_a] == label_in_b)
        matching_elements += matches

    match_ratio = matching_elements / total_elements

    return bool(match_ratio >= threshold)


def test_run() -> None:
    """Test the run method of Sifted class.

    Given the output of sifted stage of matlab and python, compute the correlation
    between them. Check for each column and row, there's only one value that has high
    correlation (>0.9) and other correlation values are low (<0.9)
    """
    import time

    start = time.time()
    csv_path_x = script_dir / "test_data/sifted/output/x_matlab.csv"
    x_matlab = pd.read_csv(csv_path_x, header=None)

    data_change, _ = Sifted.run(
        input_x,
        input_y,
        input_ybin,
        feat_labels,
        opts,
        par_opts,
    )
    x_python = pd.DataFrame(data_change.x)

    print(f"runtime {time.time() - start}")

    # compute correlation matrix that has been categorised into high, normal and low
    correlation_matrix = compute_correlation(x_python, x_matlab)

    print(f"correlation {time.time() - start}")

    # test case pass if 70%
    assert correlation_matrix_check(correlation_matrix, threshold=0.7)


def compute_correlation(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    """Compute correlation matrix and categorise them into high, normal and low.

    Correlation values are categorised as high, normal, or low.

    Parameters
    ----------
    df1 : pd.DataFrame
        The first dataframe.
    df2 : pd.DataFrame
        The second dataframe.

    Returns
    -------
    pd.DataFrame
        A dataframe where the correlation values are categorised into high (1),
        normal (0), and low (-1).
    """
    upper_bound = 0.7
    lower_bound = 0.3

    def categorise_value(x: float) -> int:
        """Categorise correlation value into high, normal and low."""
        if x > upper_bound:
            return 1
        if x < lower_bound:
            return -1
        return 0

    # given two dataframe, compute correlation matrix
    correlation_matrix = pd.DataFrame(index=df1.columns, columns=df2.columns)
    for col1 in df1.columns:
        for col2 in df2.columns:
            correlation_matrix.loc[col1, col2] = df1[col1].corr(df2[col2])
    correlation_matrix = correlation_matrix.abs()

    # categorise correlation matrix's value to high and low
    return correlation_matrix.map(categorise_value)


def correlation_matrix_check(df: pd.DataFrame, threshold: float) -> bool:
    """Check if at least threshold percentage of both rows and columns fulfil condition.

    The condition is fulfilled if only one value in a row or column has a high
    correlation (categorised as 1).

    Parameters
    ----------
    df : pd.DataFrame
        The correlation matrix with categorised values.
    threshold : float
        The minimum percentage of rows and columns that must fulfill the condition.

    Returns
    -------
    bool
        True if the condition is satisfied for at least the threshold percentage,
        False otherwise.
    """
    # for every row, calculate percentage of only one value has modified correlation
    # equals to 1
    row_condition = (df == 1).sum(axis=1) == 1
    row_percentage = row_condition.mean()

    # for every column, calculate percentage of only one value has modified correlation
    # equals to 1
    col_condition = (df == 1).sum(axis=0) == 1
    col_percentage = col_condition.mean()

    total_percentage = (row_percentage + col_percentage) / 2

    return total_percentage >= threshold
