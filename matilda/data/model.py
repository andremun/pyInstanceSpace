"""
Defines a comprehensive set of data classes used in the instance space analysis.

These classes are designed to encapsulate various aspects of the data and the results
of different analytical processes, facilitating a structured and organized approach
to data analysis and model building.
"""

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from matilda.data.option import Options


@dataclass
class Data:
    """Holds initial dataset from metadata and processed data after operations."""

    inst_labels: pd.Series
    feat_labels: list[str]
    algo_labels: list[str]
    x: NDArray[np.double]
    y: NDArray[np.double]
    x_raw: NDArray[np.double]
    y_raw: NDArray[np.double]
    y_bin: NDArray[np.bool_]
    y_best: NDArray[np.double]
    p: NDArray[np.double]
    num_good_algos: NDArray[np.double]
    beta: NDArray[np.bool_]
    s: set[str] | None


@dataclass
class FeatSel:
    """Holds indices for feature selection."""

    idx: NDArray[np.intc]


@dataclass
class AlgorithmSummary:
    """Provides a summary of an algorithm's performance across different metrics."""

    name: str
    avg_perf_all_instances: float | None
    std_perf_all_instances: float | None
    probability_of_good: float | None
    avg_perf_selected_instances: float | None
    std_perf_selected_instances: float | None
    cv_model_accuracy: float | None
    cv_model_precision: float | None
    cv_model_recall: float | None
    box_constraint: float | None
    kernel_scale: float | None


@dataclass
class PrelimOut:
    """Contains preliminary output metrics calculated from the data."""

    med_val: NDArray[np.double]
    iq_range: NDArray[np.double]
    hi_bound: NDArray[np.double]
    lo_bound: NDArray[np.double]
    min_x: NDArray[np.double]
    lambda_x: NDArray[np.double]
    mu_x: NDArray[np.double]
    sigma_x: NDArray[np.double]
    min_y: NDArray[np.double]
    lambda_y: NDArray[np.double]
    sigma_y: NDArray[np.double]
    mu_y: float = 0.0


@dataclass
class SiftedOut:
    """Results of the sifting process in the data analysis pipeline."""

    flag: int  # not sure datatype, confirm it later
    rho: np.double
    k: int
    n_trees: int
    max_lter: int
    replicates: int


@dataclass
class PilotOut:
    """Results of the Pilot process in the data analysis pipeline."""

    X0: NDArray[np.double]  # not sure about the dimensions
    alpha: NDArray[np.double]
    eoptim: NDArray[np.double]
    perf: NDArray[np.double]
    a: NDArray[np.double]
    z: NDArray[np.double]
    c: NDArray[np.double]
    b: NDArray[np.double]
    error: NDArray[np.double]  # or just the double
    r2: NDArray[np.double]
    summary: pd.DataFrame


@dataclass
class BoundaryResult:
    """Results of generating boundaries from Cloister process."""

    x_edge: NDArray[np.double]
    remove: NDArray[np.double]

    def __iter__(self: "BoundaryResult") -> Iterator[NDArray[np.double]]:
        """Allow unpacking directly."""
        return iter((self.x_edge, self.remove))

@dataclass
class CloisterOut:
    """Results of the Cloister process in the data analysis pipeline."""

    z_edge: NDArray[np.double]
    z_ecorr:NDArray[np.double]

    def __iter__(self: "CloisterOut") -> Iterator[NDArray[np.double]]:
        """Allow unpacking directly."""
        return iter((self.z_edge, self.z_ecorr))


@dataclass
class PythiaOut:
    """Results of the Pythia process in the data analysis pipeline."""

    mu: list[float]
    sigma: list[float]
    cp: Any  # Change it to proper type
    svm: Any  # Change it to proper type
    cvcmat: NDArray[np.double]
    y_sub: NDArray[np.bool_]
    y_hat: NDArray[np.bool_]
    pr0_sub: NDArray[np.double]
    pr0_hat: NDArray[np.double]
    box_consnt: list[float]
    k_scale: list[float]
    precision: list[float]
    recall: list[float]
    accuracy: list[float]
    selection0: NDArray[np.double]
    selection1: Any  # Change it to proper type
    summary: pd.DataFrame


@dataclass
class PolyShape:
    """Represent Polygon shape for footprint."""

    # polyshape is the builtin Matlab Data structure,
    # may find a similar one in python
    pass


@dataclass
class Footprint:
    """Represent the geometric and quality attributes of a spatial footprint."""

    polygon: PolyShape
    area: float
    elements: float
    good_elements: float
    density: float
    purity: float


@dataclass
class TraceOut:
    """Results of the Trace process in the data analysis pipeline."""

    space: Footprint
    good: list[Footprint]
    best: list[Footprint]
    hard: Footprint
    summary: pd.DataFrame  # for the dataform that looks like the
    # Excel spreadsheet(rownames and column names are mixed with data),
    # I decide to use DataFrame


@dataclass
class Model:
    """
    Contain data and output.

    Combines all components into a full model representation, including data and
    analysis results.
    """

    data: Data
    data_dense: Data
    feat_sel: FeatSel
    prelim: PrelimOut
    sifted: SiftedOut
    pilot: PilotOut
    cloist: CloisterOut
    pythia: PythiaOut
    trace: TraceOut
    opts: Options
