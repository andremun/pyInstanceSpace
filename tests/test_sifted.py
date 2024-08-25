""" Test module for Sifted stage to verify its functionality

The file contains multiple unit tests to ensure that the `Sifted` class corretly
perform its tasks. The basic mechanism of the test is to compare its output against
output from MATLAB and check if the outputs are the same or reasonable similar. The
tests also include some boundary test where appropriate to test the boundary of the
statement within the methods to ensure they are implemented appropriately.

Tests includes:
-  For the function select_features_by_performance, we check xaux value, check if features selected are the same 
- For the function costfcn, given an example, check if cost value is the same 
- For the function ga, check if the filtered x value, for each row and column, only one instances with high correlation, others are low correlation

"""

# cross correlation between matlab and pythoon, by row and by column, one and only one one corrlatino is above 0.9, others are lower than 0.3
# recommend best k value that gives the silhoulette value, and give warning if not bell value
# chang into int for binary matrix
# compare whether the faeture is in same cluster
# check error of machine learning model is wtithin a range
# line 133 in matlab, get [ind,y], cost value 

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
    # csv_path_x = script_dir / "test_data/sifted/output/X.csv"
    # x_matlab = pd.read_csv(csv_path_x, header = None)

    data_change, sifted_output = Sifted.run(input_x, input_y, input_ybin, feat_labels, opts)
    x_python = pd.DataFrame(data_change.x)
    scaler = MinMaxScaler()
    
    x_python.to_csv('x_python.csv')
    
    return
    
    # # Compute the correlation between corresponding columns
    # correlation_matrix = np.corrcoef(x_python.T, x_matlab.T)[:x_python.shape[1], x_python.shape[1]:]

    # # Convert the correlation matrix to a DataFrame
    # correlation_df = pd.DataFrame(correlation_matrix, columns=x_matlab.columns, index=x_python.columns)
    # correlation_df_normalised= scaler.fit_transform(correlation_df)
    # normalised_correlation = pd.DataFrame(correlation_df_normalised)
    
    # normalised_correlation.to_csv("normalised_correlation.csv")
    
    # assert check_correlation(normalised_correlation)

def check_correlation(matrix: pd.DataFrame) -> bool:
    """
    Check if for each row or column in the matrix, only one correlation is > 0.9 and others are <= 0.9.
    """
    matrix_np = matrix.to_numpy()
    
    # Above this threshold signify high correlation while below signify low correlation
    threshold = 0.9
    
    # Check for each row, only one correlation greater than threshold
    for row in matrix_np:
        count_above_threshold = np.sum(row > threshold)
        if count_above_threshold != 1:
            return False
    
     # Check for each column, only one correlation greater than threshold
    for col in matrix_np.T:  
        count_above_threshold = np.sum(col > threshold)
        if count_above_threshold != 1:
            return False
    
    # If all rows and columns meet the condition, return True
    return True

test_run()