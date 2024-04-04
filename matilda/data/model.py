import numpy as np
import pandas as pd

from dataclasses import dataclass, field

from typing import List, Set

from numpy import double
from numpy.typing import NDArray

from matilda.data.option import Opts


@dataclass
class Data:
    instlabels: pd.Series
    featlabels: List[str] # could be 1-row Datagram(corresponding to cell in Matlab) comment by Chen
    algolabels: List[str] # could be 1-row Datagram, comment by Chen
    S: Set[str] = None
    X: NDArray[np.double]
    Y: NDArray[np.double]
    Xraw: NDArray[np.double]
    Yraw: NDArray[np.double]
    Ybin: NDArray[np.bool_]
    Ybest: NDArray[np.double]
    P: NDArray[np.double]
    numGoodAlgos: NDArray[np.double]
    beta: NDArray[np.bool_]

@dataclass
class Featsel:
    idx: NDArray[np.intc]

@dataclass
class PrelimOut:
    medval: NDArray[np.double] 
    iqrange: NDArray[np.double]
    hibound: NDArray[np.double]
    lobound: NDArray[np.double]
    minX: NDArray[np.double]
    lambdaX: NDArray[np.double]
    muX: NDArray[np.double]
    sigmaX: NDArray[np.double] 
    minY: NDArray[np.double]
    lambdaY: NDArray[np.double]
    muY: np.double = 0.0
    sigmaY: NDArray[np.double] 

@dataclass
class SiftedOut:
    flag: np.int #not sure datatype, confirm it later
    rho: np.double
    K: np.int
    NTREES: np.int
    Maxlter: np.int
    Replicates: np.int


@dataclass
class PilotOut:
    """By Chen"""
    X0: NDArray[np.double] # not sure about the dimensions
    """
    size has two version:
        [2*m+2*n, opts.ntries]
        row       column
        Note: Xbar = [X Y];
            m = size(Xbar, 2);
            n = size(X, 2); % Number of features
    or
    ...
    """

    alpha: NDArray[np.double]
    eoptim: NDArray[np.double]
    perf: NDArray[np.double]
    A: NDArray[np.double]
    Z: NDArray[np.double]
    C: NDArray[np.double]
    B: NDArray[np.double]
    error: NDArray[np.double] #or just the double
    R2: NDArray[np.double]
    summary: pd.DataFrame

@dataclass
class CloistOut:
    pass

@dataclass
class PythiaOut:
    mu:
    sigma:
    cp:
    svm:
    cvcmat:
    Ysub
    Yhat: NDArray(bool)
    Pr0sub
    Pr0hat
    boxconsnt
    kscale
    prcision
    recall
    accuracy
    selection0:NDArray(double)
    selection1
    summary

@dataclass
class polyshape:
    # polyshape is the builtin Matlab Data structure,
    # may find a similar one in python
    pass

@dataclass
class Footprint:
    """By Chen """
    polygon: polyshape
    area: double;
    elements: double;
    goodElements: double;
    density: double;
    purity: double;


@dataclass
class TraceOut:
    """By Chen """
    space: Footprint
    good: NDArray(Footprint)
    best: NDArray(Footprint)
    hard: Footprint
    summary: pd.DataFrame # for the dataform that looks like the
                          # Excel spreadsheet(rownames and column names are mixed with data),
                          # I decide to use DataFrame



class Featsel:
    idx: NDArray[double]

@dataclass
class Model:
    data: Data
    data_dense: Data 
    featsel: Featsel 
    prelim: PrelimOut
    sifted: SiftedOut
    pilot: PilotOut
    cloist: CloistOut
    pythia: PythiaOut
    trace: TraceOut
    opts: Opts