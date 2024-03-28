from typing import Dict, Any, List
from numpy.typing import NDArray
import numpy as np
import pandas as pd

out = {
    "medval": NDArray[np.double],
    "iqrange": NDArray[np.double],
    "hibound": NDArray[np.double],
    "lobound": NDArray[np.double],
    "minX": NDArray[np.double],
    "lambdaX": NDArray[np.double],
    "muX": NDArray[np.double],
    "sigmaX": NDArray[np.double],
    "minY": NDArray[np.double],
    "lambdaY": NDArray[np.double],
    "muY": np.double,
    "sigmaY": NDArray[np.double],
}

def PRELIM(X: NDArray[np.double], Y: NDArray[np.double], opts: Dict[str, Any]) -> List[Any]:
    # TODO: Rewrite PRELIM logic in python

    # Remove dummy data after implementing logic
    X = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]])
    Y = np.array([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8], [0.9, 1.0, 1.1, 1.2]])
    Ybest = np.array([0.4, 0.8, 1.2])
    Ybin = np.array([[False, False, False, True], [True, True, True, True], [True, True, True, True]])
    P = np.array([0.1, 0.2, 0.3])
    numGoodAlgos = np.array([1, 4, 4])
    beta = np.array([True, False, True])
    
    return [X,Y,Ybest,Ybin,P,numGoodAlgos,beta,out]