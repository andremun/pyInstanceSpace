"""
Defines a comprehensive set of data classes used in the instance space analysis.

These classes are designed to encapsulate various aspects of the data and the results
of different analytical processes, facilitating a structured and organized approach
to data analysis and model building.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from matilda.data.option import Opts


@dataclass
class Data:
    """Holds initial dataset from metadata and processed data after operations."""

    inst_labels: pd.Series
    feat_labels: list[str]
    algo_labels: list[str]
    s: set[str] = None
    x: NDArray[np.double]
    y: NDArray[np.double]
    x_raw: NDArray[np.double]
    y_raw: NDArray[np.double]
    y_bin: NDArray[np.bool_]
    y_best: NDArray[np.double]
    p: NDArray[np.double]
    num_good_algos: NDArray[np.double]
    beta: NDArray[np.bool_]


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
    mu_y: np.double = 0.0
    sigma_y: NDArray[np.double]


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
    X0: NDArray[np.double]  # not sure about the dimensions
    """
    size has two version:
        [2*m+2*n, opts.ntries]
        row       column
        Note: Xbar = [X Y];
            m = size(Xbar, 2);
            n = size(X, 2); % Number of features
    or
    ...
    """

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
class CloistOut:
    """Output from the cloistering."""

    Zedge: NDArray[np.double]
    Zecorr:NDArray[np.double]

    pass


@dataclass
class PythiaOut:
    """Output from the Pythia."""

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
    """Placeholder for a polygon shape."""

    # polyshape is the builtin Matlab Data structure,
    # may find a similar one in python
    pass


@dataclass
class Footprint:

    polygon: PolyShape
    area: float
    elements: float
    good_elements: float
    density: float
    purity: float


@dataclass
class TraceOut:

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
    cloist: CloistOut
    pythia: PythiaOut
    trace: TraceOut
    opts: Opts
