import Stage

from __future__ import annotations
import pandas as pd

import numpy as np
from numpy.typing import NDArray
from shapely.geometry import MultiPolygon, Polygon

from matilda.data.model import Footprint, TraceDataChanged, TraceOut
from matilda.data.options import TraceOptions

class traceStage(Stage):
    def __init__(self, z: NDArray[np.double], y_bin: NDArray[np.bool_], 
                 p: NDArray[np.double], beta: NDArray[np.bool_], 
                 algo_labels: list[str]) -> None:
        self.z = z
        self.y_bin = y_bin
        self.p = p
        self.beta = beta
        self.algo_labels = algo_labels
    
    @staticmethod
    def _inputs() -> list[tuple[str, type]]:
        return [
            ["z", NDArray[np.double]], 
            ["y_bin", NDArray[np.bool_]], 
            ["p", NDArray[np.double]], 
            ["beta", NDArray[np.bool_]], 
            ["algo_labels", list[str]]
        ]
    
    @staticmethod
    def _outputs() -> list[tuple[str, type]]:
        return [
            ["space", Footprint],
            ["good", list[Footprint]],
            ["best", list[Footprint]],
            ["hard", Footprint],
            ["summary", pd.Dataframe]
        ]
    
    def _run(options: TraceOptions) -> tuple[Footprint, list[Footprint], list[Footprint],
                                             Footprint, pd.DataFrame]:
        # All the code including the code in the buildIS should be here
        raise NotImplementedError
    
    @staticmethod
    def trace(z: NDArray[np.double], y_bin: NDArray[np.bool_], 
                 p: NDArray[np.double], beta: NDArray[np.bool_], 
                 algo_labels: list[str]) -> tuple[Footprint, list[Footprint], 
                                                  list[Footprint], Footprint, 
                                                  pd.DataFrame]:
        # This has code specific to TRACE.m
        raise NotImplementedError


