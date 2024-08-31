""" Test module for Sifted stage to verify its functionality

The file contains multiple unit tests to ensure that the `Sifted` class corretly
perform its tasks. The basic mechanism of the test is to compare its output against
output from MATLAB and check if the outputs are the same or reasonable similar. The
tests also include some boundary test where appropriate to test the boundary of the
statement within the methods to ensure they are implemented appropriately.

Tests includes:
- For the function select_features_by_performance, we check xaux value, check if features selected are the same 
- For the function select_features_by_clustering, we check if number of elements in the same clusters for both matlab and python
    are over given threshold %
- For the function ga, check if the filtered x value, for each row and column, only one instances with high correlation, others are low correlation. 
    Test passed if more than 1 columns/rows don't fulfil this condition

"""

import numpy as np
import pandas as pd

from pathlib import Path
from matilda.stages.sifted import Sifted
from matilda.data.options import SiftedOptions
from sklearn.preprocessing import MinMaxScaler

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

def test_select_features_by_performance() -> None:
    ''' Ensures xaux after filtered by correlation performance are exactly the same'''
    csv_path_xaux = script_dir / "test_data/sifted/output/correlation_matlab.csv"
    xaux_matlab = np.genfromtxt(csv_path_xaux, delimiter=",")
    
    sifted = Sifted(input_x, input_y, input_ybin, feat_labels, opts)
    xaux_python = sifted.select_features_by_performance()
    
    assert np.allclose(xaux_matlab, xaux_python, atol=1e-04)

def test_select_features_by_clustering() -> None:
    """
    Despite the difference in cluster labels, we ensure that the number of items in python's cluster
    are 80% same as items in matlab's cluster
    """
    csv_path_cluster = script_dir / "test_data/sifted/output/clusters_matlab.csv"
    cluster_matlab = np.genfromtxt(csv_path_cluster, delimiter=",")
    
    sifted = Sifted(input_x, input_y, input_ybin, feat_labels, opts)
    sifted.select_features_by_performance()
    cluster_python = sifted.select_features_by_clustering()
    
    assert are_same_clusters(cluster_matlab, cluster_python)
    
def are_same_clusters(A, B, threshold=0.8) -> bool:
    """
    Helper function to check if two clusters have same number of elements more than threshold set
    """
    A = np.array(A)
    B = np.array(B)
    
    unique_labels_A = np.unique(A)
    total_elements = len(A)
    matching_elements = 0
    
    for label in unique_labels_A:
        indices_A = np.where(A == label)[0]
        
        # Find the corresponding label in B for the same indices
        label_in_B = B[indices_A[0]]
        
        # Count the number of matching labels in B for these indices
        matches = np.sum(B[indices_A] == label_in_B)
        matching_elements += matches
        
    match_ratio = matching_elements / total_elements

    return match_ratio >= threshold

def test_run() -> None:
    """
    Given the output of sifted stage of matlab and python, compute the correlation between them. Check for each
    column and row, there's only one value that has high correlation (>0.9) and other correlation values are low (<0.9)
    """
    csv_path_x = script_dir / "test_data/sifted/output/x_matlab.csv"
    x_matlab = pd.read_csv(csv_path_x, header = None)

    data_change, sifted_output = Sifted.run(input_x, input_y, input_ybin, feat_labels, opts)
    x_python = pd.DataFrame(data_change.x)
    
    # compute correlation matrix that has been categorised into high, normal and low
    correlation_matrix = compute_correlation(x_python, x_matlab)
    # test case pass if 70% 
    assert correlation_matrix_check(correlation_matrix, threshold=0.7)

def compute_correlation(df1, df2):
    """
    Compute correlation value given two dataframe, and categorise them into high, normal and low
    """
    def categorise_value(x):
        if x > 0.7:
            return 1
        elif x < 0.3:
            return -1
        else:
            return 0
    
    # given two dataframe, compute correlation matrix
    correlation_matrix = pd.DataFrame(index=df1.columns, columns=df2.columns)
    for col1 in df1.columns:
        for col2 in df2.columns:
            correlation_matrix.loc[col1, col2] = df1[col1].corr(df2[col2])
    correlation_matrix = correlation_matrix.abs()
            
    # categorise correlation matrix's value to high and low
    correlation_matrix_transform = correlation_matrix.map(categorise_value)
    
    return correlation_matrix_transform


def correlation_matrix_check(df, threshold):
    """
    Return true if at least threshold percentage of both rows and columns fulfil condition, condition
    fulfill if only 1 value in row/column contains high value (1)
    """
    # for every row, calculate percentage of only one value has modified correlation equals to 1
    row_condition = (df == 1).sum(axis=1) == 1
    row_percentage = row_condition.mean() 
    
    # for every column, calculate percentage of only one value has modified correlation equals to 1
    col_condition = (df==1).sum(axis=0) == 1
    col_percentage = col_condition.mean()
    
    total_percentage = (row_percentage + col_percentage) / 2
    
    return total_percentage >= threshold
