import pandas as pd
from matilda.data.options import TraceOptions
from matilda.stages.trace import Trace
import numpy as np

def test_trace_pythia():
    # Reading algo_labels from a text file with comma delimiter
    with open('test_data/trace_csvs/algolabels.txt', 'r') as f:
        algo_labels = f.read().split(',')


    # Reading z from z.csv
    z = np.genfromtxt('test_data/trace_csvs/Z.csv', delimiter=',', dtype=np.double)

    # Reading y_bin from y_bin.csv
    y_bin = np.genfromtxt('test_data/trace_csvs/yhat.csv', delimiter=',', dtype=np.int8).astype(np.bool_)

    # Reading p from p.csv
    p = np.genfromtxt('test_data/trace_csvs/selection0.csv', delimiter=',', dtype=np.integer)
    p = p - 1
    # Reading beta from beta.csv
    beta = np.genfromtxt('test_data/trace_csvs/beta.csv', delimiter=',', dtype=np.int8).astype(np.bool_)

    trace_options = TraceOptions(True, 0.55)

    trace = Trace()
    trace.run(z, y_bin, p, beta, algo_labels, trace_options)
def test_trace_simulation():
    # Reading algo_labels from a text file with comma delimiter
    with open('test_data/trace_csvs/algolabels.txt', 'r') as f:
        algo_labels = f.read().split(',')


    # Reading z from z.csv
    z = np.genfromtxt('test_data/trace_csvs/Z.csv', delimiter=',', dtype=np.double)

    # Reading y_bin from y_bin.csv
    y_bin2 = np.genfromtxt('test_data/trace_csvs/yhat2.csv', delimiter=',', dtype=np.int8).astype(np.bool_)

    # Reading p from p.csv
    p2 = np.genfromtxt('test_data/trace_csvs/dataP.csv', delimiter=',', dtype=np.integer)
    p2 = p2 - 1
    # Reading beta from beta.csv
    beta = np.genfromtxt('test_data/trace_csvs/beta.csv', delimiter=',', dtype=np.int8).astype(np.bool_)

    trace_options = TraceOptions(False, 0.55)

    trace = Trace()
    trace.run(z, y_bin2, p2, beta, algo_labels, trace_options)



