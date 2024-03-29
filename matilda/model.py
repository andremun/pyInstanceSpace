from dataclasses import dataclass, field
from typing import List, Set
from numpy.typing import NDArray
import numpy as np
import pandas as pd

@dataclass
class Data:
    instlabels: pd.Series
    featlabels: List[str]
    algolabels: List[str]
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
    pass

@dataclass
class PilotOut:
    pass

@dataclass
class CloistOut:
    pass

@dataclass
class PythiaOut:
    pass

@dataclass
class TraceOut:
    pass

@dataclass
class Opts:
    pass

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