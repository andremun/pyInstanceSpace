import stage    

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import ConvexHull, QhullError
from scipy.stats import pearsonr

from matilda.data.model import BoundaryResult, CloisterDataChanged, CloisterOut
from matilda.data.options import CloisterOptions

class cloisterStage(stage):
    def __init__(self, x: NDArray[np.double],
        a: NDArray[np.double]) -> None:
        self.x = x
        self.a = a

    @staticmethod
    def _inputs() -> list[tuple[str, type]]:
        return [
            ["x", NDArray[np.double]],
            ["a", NDArray[np.double]]
        ]

    @staticmethod
    def _outputs() -> list[tuple[str, type]]:
        return [
            ["z_edge", NDArray[np.double]],
            ["z_ecorr", NDArray[np.double]]
        ]
    
    def _run(options: CloisterOptions) -> tuple[NDArray[np.double], NDArray[np.double]]:
        #Implement code that goes into buildIS here
        raise NotImplementedError

    @staticmethod
    def cloister(x: NDArray[np.double],
        a: NDArray[np.double]) -> tuple[NDArray[np.double], NDArray[np.double]]:
        #Implement code from CLOISTER.m here
        raise NotImplementedError
