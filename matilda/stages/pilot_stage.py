import stage

import numpy as np
import pandas as pd
import scipy.linalg as la
import scipy.optimize as optim
from numpy.typing import NDArray
from scipy.spatial.distance import pdist
from scipy.stats import pearsonr

from matilda.data.model import PilotDataChanged, PilotOut
from matilda.data.options import PilotOptions

class pilotStage(stage):
    def __init__(self, x: NDArray[np.double],
        y: NDArray[np.double],
        feat_labels: list[str]) -> None:
        self.x = x
        self.y = y
        self.feat_labels = feat_labels

    @staticmethod
    def _inputs() -> list[tuple[str, type]]:
        return [
            ["x", NDArray[np.double]],
            ["y", NDArray[np.double]],
            ["feat_labels", list[str]]
            ]
    
    @staticmethod
    def _outputs() -> list[tuple[str, type]]:
        return [
                ["X0", NDArray[np.double] | None],  # not sure about the dimensions
                ["alpha", NDArray[np.double] | None],
                ["eoptim", NDArray[np.double] | None],
                ["perf", NDArray[np.double] | None],
                ["a", NDArray[np.double]],
                ["z", NDArray[np.double]],
                ["c", NDArray[np.double]],
                ["b", NDArray[np.double]],
                ["error", NDArray[np.double]],  # or just the double
                ["r2", NDArray[np.double]],
                ["summary", pd.DataFrame]
        ]

    def _run(x: NDArray[np.double],
        y: NDArray[np.double],
        feat_labels: list[str]) -> tuple[NDArray[np.double] | None,
                                         NDArray[np.double] | None,
                                         NDArray[np.double] | None,
                                         NDArray[np.double] | None, 
                                         NDArray[np.double], NDArray[np.double], 
                                         NDArray[np.double], NDArray[np.double], 
                                         NDArray[np.double], NDArray[np.double], 
                                         pd.DataFrame]:
        
        #Implement all the code in and around this class in buildIS
        raise NotImplementedError
    
    @staticmethod
    def pilot(x: NDArray[np.double],
        y: NDArray[np.double],
        feat_labels: list[str]) -> tuple[NDArray[np.double] | None,
                                         NDArray[np.double] | None,
                                         NDArray[np.double] | None,
                                         NDArray[np.double] | None, 
                                         NDArray[np.double], NDArray[np.double], 
                                         NDArray[np.double], NDArray[np.double], 
                                         NDArray[np.double], NDArray[np.double], 
                                         pd.DataFrame]:
        #Implement all the code in PILOT.py here
        raise NotImplementedError