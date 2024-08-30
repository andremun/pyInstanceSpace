import pandas as pd
import stage

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import optimize, stats

class prelimStage(stage):
    def __init__(self, 
                 
        x: NDArray[np.double],
        y: NDArray[np.double],
        max_perf: bool,
        abs_perf: bool,
        epsilon: float,
        beta_threshold: float,
        bound: bool,
        norm: bool,
        
        small_scale_flag: bool,
        small_scale: float,
        file_idx_flag: bool,
        file_idx: str,
        feats: pd.DataFrame | None,
        algos: pd.DataFrame | None,
        selvars_type: str,
        min_distance: float,
        density_flag: bool
        ) -> None:
        
        self.x = x
        self.y = y
        self.max_perf = max_perf
        self.abs_perf = abs_perf
        self.epsilon = epsilon
        self.beta_threshold = beta_threshold
        self.bound = bound
        self.norm = norm
        
        self.small_scale_flag = small_scale_flag
        self.small_scale = small_scale
        self.file_idx_flag = file_idx_flag
        self.file_idx = file_idx
        self.feats = feats
        self.algos = algos
        self.selvars_type = selvars_type
        self.min_distance = min_distance
        self.density_flag = density_flag
        
    @staticmethod
    def _inputs() -> list[tuple[str, type]]:
        return [
            ["x", NDArray[np.double]],
            ["y", NDArray[np.double]],
            ["max_perf", bool],
            ["abs_perf", bool],
            ["epsilon", float],
            ["beta_threshold", float],
            ["bound", bool],
            ["norm", bool],
            ["small_scale_flag", bool],
            ["small_scale", float],
            ["file_idx_flag", bool],
            ["file_idx", str],
            ["feats", pd.DataFrame | None],
            ["algos", pd.DataFrame | None],
            ["selvars_type", str],
            ["min_distance", float],
            ["density_flag", bool]
            ]
        
    @staticmethod 
    def _outputs() -> List[Tuple[str, type]]:
        return [
            ("med_val", NDArray[np.double]),
            ("iq_range", NDArray[np.double]),
            ("hi_bound", NDArray[np.double]),
            ("lo_bound", NDArray[np.double]),
            ("min_x", NDArray[np.double]),
            ("lambda_x", NDArray[np.double]),
            ("mu_x", NDArray[np.double]),
            ("sigma_x", NDArray[np.double]),
            ("min_y", float),
            ("lambda_y", NDArray[np.double]),
            ("sigma_y", NDArray[np.double]),
            ("mu_y", NDArray[np.double])
        ]
        
    def _run(self,
        x: NDArray[np.double],
        y: NDArray[np.double],
        max_perf: bool,
        abs_perf: bool,
        epsilon: float,
        beta_threshold: float,
        bound: bool,
        norm: bool,
        small_scale_flag: bool,
        small_scale: float,
        file_idx_flag: bool,
        file_idx: str,
        feats: pd.DataFrame | None,
        algos: pd.DataFrame | None,
        selvars_type: str,
        min_distance: float,
        density_flag: bool) -> tuple[
            NDArray[np.double], NDArray[np.double], NDArray[np.double], NDArray[np.double], 
            NDArray[np.double], NDArray[np.double], NDArray[np.double], NDArray[np.double], 
            float, NDArray[np.double], NDArray[np.double], NDArray[np.double]]:
            
            raise NotImplementedError
        
    def prelim(self, 
        x: NDArray[np.double],
        y: NDArray[np.double],
        max_perf: bool,
        abs_perf: bool,
        epsilon: float,
        beta_threshold: float,
        bound: bool,
        norm: bool) -> tuple[
            NDArray[np.double], NDArray[np.double], NDArray[np.double], NDArray[np.double], 
            NDArray[np.double], NDArray[np.double], NDArray[np.double], NDArray[np.double], 
            float, NDArray[np.double], NDArray[np.double], NDArray[np.double]]:
        
        raise NotImplementedError