"""Test input parameters, particularly metrics, are accurately parsed and stored. """
import pytest
import csv
import numpy as np
from matilda.pythia import pythia
from  matilda.data.option import *

pythia_opts = PythiaOptions(
    cv_folds=5,
    is_poly_krnl=False, 
    use_weights=False,
    use_lib_svm=False
)

CSV_Z = 'tests/pythia/test_pythia_input/z.csv'
CSV_Y = 'tests/pythia/test_pythia_input/y.csv'
CSV_YBIN = 'tests/pythia/test_pythia_input/ybin.csv'
CSV_YBEST = 'tests/pythia/test_pythia_input/ybest.csv'
CSV_ALGO = 'tests/pythia/test_pythia_input/algolabels.csv'

try:
    z = pd.read_csv(CSV_Z, header=None, dtype=np.float64)
    y = np.loadtxt(CSV_Y, delimiter=',')
    y_bin = np.loadtxt(CSV_YBIN, delimiter=',', skiprows=1)
    y_best = np.loadtxt(CSV_YBEST, delimiter=',')

    with open(CSV_ALGO, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            algolabels = row

except Exception as e:
    print(f"Failed to load data: {e}")


def test_input():
    assert z is not None, "Z is None, data not loaded."
    assert y is not None, "Y is None, data not loaded."
    assert y_bin is not None, "YBin is None, data not loaded."
    assert y_best is not None, "YBest is None, data not loaded."
    assert algolabels is not None

    # expected_first_row = np.array([0.78943, 1.9664])
    # np.testing.assert_array_almost_equal(Z[0], expected_first_row, decimal=5, err_msg="First row does not match expected values.")
    # print(algolabels)

"""T01: Check the equivalence of 'soft' normalization of a dataset Z """

# SVMs assume that the data it works with is in a standard range, usually either 0 to 1, or -1 to 1 
# (roughly). So the normalization of feature vectors prior to feeding them to the SVM is required. 
def test_znorm_svm_input():

    z_norm_M = pd.read_csv('tests/pythia/test_pythia_input/z_norm.csv', header=None, dtype=np.float64)

    res =  pythia(z, y, y_bin, y_best, algolabels, pythia_opts)

    z_norm_P = pd.read_csv('tests/pythia/test_pythia_output/z_norm.csv', header=None, dtype=np.float64)

    # assert pd.testing.assert_frame_equal(z_norm_M, z_norm_P, atol=1e-5)
    # Small numerical differences may arise due to how each language handles 
    # floating-point arithmetic. atol is appropriate tolerance

    difference = np.isclose(z_norm_M.values, z_norm_P.values, atol=1e-7)
    assert difference.all(), f"Mismatch found in elements: {np.where(~difference)}"

    

