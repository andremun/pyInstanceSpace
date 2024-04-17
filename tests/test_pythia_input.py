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
opts = Opts(
    parallel=None,
    perf=None,
    auto=None,
    bound=None,
    norm=None,
    selvars=None,
    sifted=None,
    pilot=None,
    cloister=None,
    pythia=pythia_opts,
    trace=None,
    outputs=None
)
CSV_Z = 'tests/testData_pythia/z.csv'
CSV_Y = 'tests/testData_pythia/y.csv'
CSV_YBIN = 'tests/testData_pythia/ybin.csv'
CSV_YBEST = 'tests/testData_pythia/ybest.csv'
CSV_ALGO = 'tests/testData_pythia/algolabels.csv'

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

def test_cv_indices():
    nalgos = y_bin.shape[1]
    cv_folds = opts.pythia.cv_folds

    res =  pythia(z, y, y_bin, y_best, algolabels, opts)
    i = 0
    
    for fold in range(nalgos):
        # Load indices from Python-generated CSV files
        python_data = pd.read_csv(f'python_cv_indices_{i}.csv')
        matlab_data = pd.read_csv(f'tests/testData_pythia/matlab_cv_indices_{i + 1}.csv')

        # Convert DataFrame rows to sets for unordered comparison
        python_sets = [set(row.dropna().astype(int)) for _, row in python_data.iterrows()]
        matlab_sets = [set(row.dropna().astype(int)) for _, row in matlab_data.iterrows()]

        # Ensure each fold's test indices match
        for p_set, m_set in zip(python_sets, matlab_sets):
            assert p_set == m_set, f"Mismatch in fold indices for algorithm {i}"
        
