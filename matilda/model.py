from dataclasses import dataclass, field
from typing import Dict, Any, List
from numpy.typing import NDArray
import numpy as np
import pandas as pd

@dataclass
class Data:
    instlabels: pd.Series = field(default_factory=lambda: pd.Series(dtype=str))
    featlabels: List[str] = field(default_factory=list)
    algolabels: List[str] = field(default_factory=list)
    X: NDArray[np.double] = field(default_factory=lambda: np.array([], dtype=np.double))
    Y: NDArray[np.double] = field(default_factory=lambda: np.array([], dtype=np.double))
    Xraw: NDArray[np.double] = field(default_factory=lambda: np.array([], dtype=np.double))
    Yraw: NDArray[np.double] = field(default_factory=lambda: np.array([], dtype=np.double))
    Ybin: NDArray[np.bool_] = field(default_factory=lambda: np.array([], dtype=np.bool_))
    Ybest: NDArray[np.double] = field(default_factory=lambda: np.array([], dtype=np.double))
    P: NDArray[np.double] = field(default_factory=lambda: np.array([], dtype=np.double))
    numGoodAlgos: NDArray[np.double] = field(default_factory=lambda: np.array([], dtype=np.double))
    beta: NDArray[np.bool_] = field(default_factory=lambda: np.array([], dtype=np.bool_))

@dataclass
class Featsel:
    idx: NDArray[np.intc] = field(default_factory=lambda: np.array([], dtype=np.intc))

@dataclass
class PrelimOut:
    medval: NDArray[np.double] = field(default_factory=lambda: np.array([], dtype=np.double))
    iqrange: NDArray[np.double] = field(default_factory=lambda: np.array([], dtype=np.double))
    hibound: NDArray[np.double] = field(default_factory=lambda: np.array([], dtype=np.double))
    lobound: NDArray[np.double] = field(default_factory=lambda: np.array([], dtype=np.double))
    minX: NDArray[np.double] = field(default_factory=lambda: np.array([], dtype=np.double))
    lambdaX: NDArray[np.double] = field(default_factory=lambda: np.array([], dtype=np.double))
    muX: NDArray[np.double] = field(default_factory=lambda: np.array([], dtype=np.double))
    sigmaX: NDArray[np.double] = field(default_factory=lambda: np.array([], dtype=np.double))
    minY: NDArray[np.double] = field(default_factory=lambda: np.array([], dtype=np.double))
    lambdaY: NDArray[np.double] = field(default_factory=lambda: np.array([], dtype=np.double))
    muY: np.double = 0.0
    sigmaY: NDArray[np.double] = field(default_factory=lambda: np.array([], dtype=np.double))

@dataclass
class SiftedOut:
    a: NDArray[np.double]
    b: np.double = 0.0
    c: NDArray[np.double]

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
    data: Data = field(default_factory=Data)
    data_dense: Data = field(default_factory=Data)
    featsel: Featsel = field(default_factory=Featsel)
    prelim: PrelimOut = field(default_factory=PrelimOut)
    sifted: SiftedOut = field(default_factory=SiftedOut)
    pilot: PilotOut = field(default_factory=PilotOut)
    cloist: CloistOut = field(default_factory=CloistOut)
    pythia: PythiaOut = field(default_factory=PythiaOut)
    trace: TraceOut = field(default_factory=TraceOut)
    opts: Opts = field(default_factory=Opts)