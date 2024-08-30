import csv

import numpy as np

from matilda.data.option import *
from matilda.pythia_tao import pythia

CSV_Z = "pythia/test_pythia_input/z_M.csv"
CSV_Y = "pythia/test_pythia_input/y.csv"
CSV_YBIN = "pythia/test_pythia_input/ybin.csv"
CSV_YBEST = "pythia/test_pythia_input/ybest.csv"
CSV_ALGO = "pythia/test_pythia_input/algolabels.csv"

pythia_opts = PythiaOptions(
    cv_folds=5,
    is_poly_krnl=False,
    use_weights=False,
    use_lib_svm=False,
)

z = np.loadtxt(CSV_Z, delimiter=",")
y = np.loadtxt(CSV_Y, delimiter=",")
y_bin = np.loadtxt(CSV_YBIN, delimiter=",", skiprows=1)
y_best = np.loadtxt(CSV_YBEST, delimiter=",")

with open(CSV_ALGO, newline="") as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        algolabels = row

res = pythia(z, y, y_bin, y_best, algolabels, pythia_opts)
