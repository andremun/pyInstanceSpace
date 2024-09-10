"""Performing preliminary data processing.

The main focus is on the `prelim` function, which prepares the input data for further
analysis and modeling.

The `prelim` function takes feature and performance data matrices along with a set of
processing options, and performs various preprocessing tasks such as normalization,
outlier detection and removal, and binary performance classification. These tasks are
guided by the options specified in the `InstanceSpaceOptions` object.
"""

from typing import NamedTuple

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from matilda.stages.stage import Stage


class _PrelimInputs(NamedTuple):
    x: NDArray[np.double]
    y: NDArray[np.double]
    max_perf: bool
    abs_perf: bool
    epsilon: float
    beta_threshold: float
    bound: bool
    norm: bool
    small_scale_flag: bool
    small_scale: float
    file_idx_flag: bool
    file_idx: str
    feats: pd.DataFrame | None
    algos: pd.DataFrame | None
    selvars_type: str
    min_distance: float
    density_flag: bool


class _PrelimOutputs(NamedTuple):
    med_val: NDArray[np.double]
    iq_range: NDArray[np.double]
    hi_bound: NDArray[np.double]
    lo_bound: NDArray[np.double]
    min_x: NDArray[np.double]
    lambda_x: NDArray[np.double]
    mu_x: NDArray[np.double]
    sigma_x: NDArray[np.double]
    min_y: float
    lambda_y: NDArray[np.double]
    sigma_y: NDArray[np.double]
    mu_y: NDArray[np.double]


class PrelimStage(Stage):
    """See file docstring."""

    def __init__(
        self,
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
        density_flag: bool,
    ) -> None:
        """See file docstring."""
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
    def _inputs() -> type[NamedTuple]:
        """Return inputs of the STAGE (run method)."""
        return _PrelimInputs

    @staticmethod
    def _outputs() -> type[NamedTuple]:
        """Return outputs of the STAGE (run method)."""
        return _PrelimOutputs

    @staticmethod
    def _run(inputs: _PrelimInputs) -> _PrelimOutputs:
        """See file docstring."""
        raise NotImplementedError

    def prelim(
        self,
        x: NDArray[np.double],
        y: NDArray[np.double],
        max_perf: bool,
        abs_perf: bool,
        epsilon: float,
        beta_threshold: float,
        bound: bool,
        norm: bool,
    ) -> tuple[
        NDArray[np.double],
        NDArray[np.double],
        NDArray[np.double],
        NDArray[np.double],
        NDArray[np.double],
        NDArray[np.double],
        NDArray[np.double],
        NDArray[np.double],
        float,
        NDArray[np.double],
        NDArray[np.double],
        NDArray[np.double],
    ]:
        """See file docstring."""
        raise NotImplementedError
