from typing import Dict, Any, List
from numpy.typing import NDArray
import numpy as np
import pandas as pd

from matilda import PRELIM

model = {
    "data": {
        "instlabels": pd.Series(dtype=str),  # row names for instances
        "featlabels": List[str],             # column names for features
        "algolabels": List[str],             # column names for algorithms
        "X": NDArray[np.double],             # 2D numpy array for feature values
        "Y": NDArray[np.double],             # 2D numpy array for algorithm values
        "Xraw": NDArray[np.double],          # 2D numpy array for raw feature values
        "Yraw": NDArray[np.double],          # 2D numpy array for raw algorithm values
        "Ybin": NDArray[np.bool_],           # 2D numpy array for binary performance measure
        "Ybest": NDArray[np.double],         # 1D numpy array for best algo value
        "P": NDArray[np.double],             # 1D numpy array 
        "numGoodAlgos": NDArray[np.double],  # 1D numpy array for number of good algo for instances
        "beta": NDArray[np.bool_],           # 1D numpy array 
    },
    "data_dense": {},                        # Copy of modale["data"]
    "featsel": {
        "idx": NDArray[np.intc],             # 1D numpy array for index
    },
    "prelim": Dict[str, Any],
    "sifted": None,
    "pilot": None,
    "cloist": None,
    "pythia": None,
    "trace": None,
    "opts": {
        "selvars": {},
    }
}

def buildIS(rootdir: str) -> Dict[str, Any]:
    # TODO: Rewrite buildIS logic in Python
    
    return model