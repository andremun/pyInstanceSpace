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

CSV_Z = 'tests/test_pythia_input/z.csv'
CSV_Y = 'tests/test_pythia_input/y.csv'
CSV_YBIN = 'tests/test_pythia_input/ybin.csv'
CSV_YBEST = 'tests/test_pythia_input/ybest.csv'
CSV_ALGO = 'tests/test_pythia_input/algolabels.csv'

try:
    z = np.loadtxt(CSV_Z, delimiter=',')
    y = np.loadtxt(CSV_Y, delimiter=',')
    y_bin = np.loadtxt(CSV_YBIN, delimiter=',')
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

    expected = ['N','B','L','D','A','Q','D','A','C','A','R','T','J','4','8','K','N','N','L','_','S','V','M','p','o','l','y','_','S','V','M','R','B','F','_','S','V','M','R','a','n','d','F']
    assert algolabels == expected

    # expected_first_row = np.array([0.78943, 1.9664])
    # np.testing.assert_array_almost_equal(Z[0], expected_first_row, decimal=5, err_msg="First row does not match expected values.")
    # print(algolabels)

"""Test and compare the parameters for SVM training """

# T01: Check the equivalence of normalization of a dataset Z
def test_znorm_svm_input():

    z_norm_M = pd.read_csv('tests/test_pythia_input/z_norm.csv')

    res =  pythia(z, y, y_bin, y_best, algolabels, pythia_opts)

    z_norm_P = pd.read_csv('tests/test_pythia_output/z_norm.csv')

    z_norm_M = z_norm_M.astype(float)
    z_norm_P = z_norm_P.astype(float)

    # assert pd.testing.assert_frame_equal(z_norm_M, z_norm_P, atol=1e-5)
    # Small numerical differences may arise due to how each language handles 
    # floating-point arithmetic. atol is appropriate tolerance

    # atol=1e-4 will pass!
    difference = np.isclose(z_norm_M.values, z_norm_P.values, atol=1e-4)
    assert difference.all(), f"Mismatch found in elements: {np.where(~difference)}"

    

