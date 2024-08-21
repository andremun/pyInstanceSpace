"""Defines a comprehensive set of data classes used in the instance space analysis.

These classes are designed to encapsulate various aspects of the data and the results
of different analytical processes, facilitating a structured and organized approach
to data analysis and model building.
"""

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from shapely.geometry import Polygon

from matilda.data.options import InstanceSpaceOptions


@dataclass(frozen=True)
class Data:
    """Holds initial dataset from metadata and processed data after operations."""

    inst_labels: pd.Series  # type: ignore[type-arg]
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
    uniformity: float | None


T = TypeVar("T")


@dataclass(frozen=True)
class StageState(Generic[T]):
    """The state of the data at the end of a Stage."""

    data: Data
    out: T


@dataclass(frozen=True)
class FeatSel:
    """Holds indices for feature selection."""

    idx: NDArray[np.intc]


@dataclass(frozen=True)
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


@dataclass(frozen=True)
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
    min_y: float
    lambda_y: NDArray[np.double]
    sigma_y: NDArray[np.double]
    mu_y: NDArray[np.double]


@dataclass(frozen=True)
class PrelimDataChanged:
    """The fields of Data that the Prelim stage changes."""

    x: NDArray[np.double]
    y: NDArray[np.double]
    y_bin: NDArray[np.bool_]
    y_best: NDArray[np.double]
    p: NDArray[np.double]
    num_good_algos: NDArray[np.double]
    beta: NDArray[np.bool_]

    def merge_with(self, data: Data) -> Data:
        """Merge changed fields of data with a Data object."""
        return Data(
            inst_labels=data.inst_labels,
            feat_labels=data.feat_labels,
            algo_labels=data.algo_labels,
            uniformity=data.uniformity,
            x=self.x,
            x_raw=data.x_raw,
            y=self.y,
            y_raw=data.y_raw,
            y_bin=self.y_bin,
            y_best=self.y_best,
            p=self.p,
            num_good_algos=self.num_good_algos,
            beta=self.beta,
            s=data.s,
        )


@dataclass(frozen=True)
class SiftedOut:
    """Results of the sifting process in the data analysis pipeline."""

    flag: int  # not sure datatype, confirm it later
    rho: np.double
    k: int
    n_trees: int
    max_lter: int
    replicates: int
    idx: NDArray[np.int_]


@dataclass(frozen=True)
class SiftedDataChanged:
    """The fields of Data that the Sifted stage changes."""

    def merge_with(self, data: Data) -> Data:
        """Merge changed fields of data with a Data object."""
        raise NotImplementedError


@dataclass(frozen=True)
class PilotOut:
    """Results of the Pilot process in the data analysis pipeline."""

    X0: NDArray[np.double] | None  # not sure about the dimensions
    alpha: NDArray[np.double] | None
    eoptim: NDArray[np.double] | None
    perf: NDArray[np.double] | None
    a: NDArray[np.double]
    z: NDArray[np.double]
    c: NDArray[np.double]
    b: NDArray[np.double]
    error: NDArray[np.double]  # or just the double
    r2: NDArray[np.double]
    summary: pd.DataFrame


@dataclass(frozen=True)
class PilotDataChanged:
    """The fields of Data that the Pilot stage changes."""

    def merge_with(self, data: Data) -> Data:
        """Merge changed fields of data with a Data object."""
        raise NotImplementedError


@dataclass(frozen=True)
class BoundaryResult:
    """Results of generating boundaries from Cloister process."""

    x_edge: NDArray[np.double]
    remove: NDArray[np.double]

    def __iter__(self) -> Iterator[NDArray[np.double]]:
        """Allow unpacking directly."""
        return iter((self.x_edge, self.remove))


@dataclass(frozen=True)
class CloisterOut:
    """Results of the Cloister process in the data analysis pipeline."""

    z_edge: NDArray[np.double]
    z_ecorr: NDArray[np.double]

    def __iter__(self) -> Iterator[NDArray[np.double]]:
        """Allow unpacking directly."""
        return iter((self.z_edge, self.z_ecorr))


@dataclass(frozen=True)
class CloisterDataChanged:
    """The fields of Data that the Cloister stage changes."""

    def merge_with(self, data: Data) -> Data:
        """Merge changed fields of data with a Data object."""
        raise NotImplementedError


@dataclass(frozen=True)
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


@dataclass(frozen=True)
class PythiaDataChanged:
    """The fields of Data that the Pythia stage changes."""

    def merge_with(self, data: Data) -> Data:
        """Merge changed fields of data with a Data object."""
        raise NotImplementedError


@dataclass(frozen=True)
class Footprint:
    """A class to represent a footprint with geometric and statistical properties.

    Attributes:
    ----------
    polygon : Polygon
        The geometric shape of the footprint.
    area : float
        The area of the footprint.
    elements : int
        The number of data points within the footprint.
    good_elements : int
        The number of "good" data points within the footprint (as defined by specific
        criteria).
    density : float
        The density of points within the footprint.
    purity : float
        The purity of "good" elements in relation to all elements in the footprint.
    """

    polygon: Polygon
    area: float
    elements: int
    good_elements: int
    density: float
    purity: float

    def __init__(self, polygon: Polygon) -> None:
        """Initialise a Footprint."""
        # This is a kinda hacky way to get around the frozen problem.
        # A nicer way would be a static method to construct it from a polygon rust style
        # from_polygon().

        object.__setattr__(self, "polygon", polygon if polygon else None)
        object.__setattr__(self, "area", self.polygon.area if polygon else None)
        object.__setattr__(self, "elements", 0)
        object.__setattr__(self, "good_elements", 0)
        object.__setattr__(self, "density", 0)
        object.__setattr__(self, "purity", 0)


@dataclass(frozen=True)
class TraceOut:
    """Results of the Trace process in the data analysis pipeline."""

    space: Footprint
    good: list[Footprint]
    best: list[Footprint]
    hard: Footprint
    summary: pd.DataFrame  # for the dataform that looks like the
    # Excel spreadsheet(rownames and column names are mixed with data),
    # I decide to use DataFrame


@dataclass(frozen=True)
class TraceDataChanged:
    """The fields of Data that the Trace stage changes."""

    def merge_with(self, data: Data) -> Data:
        """Merge changed fields of data with a Data object."""
        raise NotImplementedError


@dataclass(frozen=True)
class Model:
    """Contain data and output.

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
    opts: InstanceSpaceOptions
