"""Defines a comprehensive set of data classes used in the instance space analysis.

These classes are designed to encapsulate various aspects of the data and the results
of different analytical processes, facilitating a structured and organized approach
to data analysis and model building.
"""

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, TypeVar

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from shapely.geometry import MultiPoint, Polygon


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

    T = TypeVar("T", bound="Data")

    @classmethod
    def from_stage_runner_output(
        cls: type[T],
        stage_runner_output: dict[str, Any],
    ) -> T:
        """Initialise a Data object from the output of an InstanceSpace StageRunner.

        Args
        ----
            cls (type[T]): the class
            stage_runner_output (dict[str, Any]): output of StageRunner for an
                InstanceSpace

        Returns
        -------
            Data: a Data object
        """
        return cls(
            inst_labels=stage_runner_output["inst_labels"],
            feat_labels=stage_runner_output["feat_labels"],
            algo_labels=stage_runner_output["algo_labels"],
            x=stage_runner_output["x"],
            y=stage_runner_output["y"],
            x_raw=stage_runner_output["x_raw"],
            y_raw=stage_runner_output["y_raw"],
            y_bin=stage_runner_output["y_bin"],
            y_best=stage_runner_output["y_best"],
            p=stage_runner_output["p"],
            num_good_algos=stage_runner_output["num_good_algos"],
            beta=stage_runner_output["beta"],
            s=stage_runner_output["s"],
            uniformity=stage_runner_output["uniformity"],
        )


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

    T = TypeVar("T", bound="PrelimOut")

    @classmethod
    def from_stage_runner_output(
        cls: type[T],
        stage_runner_output: dict[str, Any],
    ) -> T:
        """
        Initialise a PrelimOut object from the output of an InstanceSpace StageRunner.

        Args
        ----
            cls (type[T]): the class
            stage_runner_output (dict[str, Any]): output of StageRunner for an
                InstanceSpace

        Returns
        -------
            PrelimOut: a PrelimOut object
        """
        return cls(
            med_val=stage_runner_output["med_val"],
            iq_range=stage_runner_output["iq_range"],
            hi_bound=stage_runner_output["hi_bound"],
            lo_bound=stage_runner_output["lo_bound"],
            min_x=stage_runner_output["min_x"],
            lambda_x=stage_runner_output["lambda_x"],
            mu_x=stage_runner_output["mu_x"],
            sigma_x=stage_runner_output["sigma_x"],
            min_y=stage_runner_output["min_y"],
            lambda_y=stage_runner_output["lambda_y"],
            sigma_y=stage_runner_output["sigma_y"],
            mu_y=stage_runner_output["mu_y"],
        )


@dataclass(frozen=True)
class SiftedOut:
    """Results of the sifting process in the data analysis pipeline."""

    selvars: NDArray[np.intc]
    idx: NDArray[np.intc]
    rho: NDArray[np.double] | None
    pval: NDArray[np.double] | None
    silhouette_scores: list[float] | None
    clust: NDArray[np.bool_] | None

    T = TypeVar("T", bound="SiftedOut")

    @classmethod
    def from_stage_runner_output(
        cls: type[T],
        stage_runner_output: dict[str, Any],
    ) -> T:
        """
        Initialise a SiftedOut object from the output of an InstanceSpace StageRunner.

        Args
        ----
            cls (type[T]): the class
            stage_runner_output (dict[str, Any]): output of StageRunner for an
                InstanceSpace

        Returns
        -------
            SiftedOut: a SiftedOut object
        """
        return cls(
            flag=stage_runner_output["flag"],
            rho=stage_runner_output["rho"],
            k=stage_runner_output["k"],
            n_trees=stage_runner_output["n_trees"],
            max_lter=stage_runner_output["max_lter"],
            replicates=stage_runner_output["replicates"],
            idx=stage_runner_output["idx"],
        )


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

    T = TypeVar("T", bound="PilotOut")

    @classmethod
    def from_stage_runner_output(
        cls: type[T],
        stage_runner_output: dict[str, Any],
    ) -> T:
        """Initialise a PilotOut object from the output of an InstanceSpace StageRunner.

        Args
        ----
            cls (type[T]): the class
            stage_runner_output (dict[str, Any]): output of StageRunner for an
                InstanceSpace

        Returns
        -------
            PilotOut: a PilotOut object
        """
        return cls(
            X0=stage_runner_output["X0"],
            alpha=stage_runner_output["alpha"],
            eoptim=stage_runner_output["eoptim"],
            perf=stage_runner_output["perf"],
            a=stage_runner_output["a"],
            z=stage_runner_output["z"],
            c=stage_runner_output["c"],
            b=stage_runner_output["b"],
            error=stage_runner_output["error"],
            r2=stage_runner_output["r2"],
            summary=stage_runner_output["summary"],
        )


@dataclass(frozen=True)
class CloisterOut:
    """Results of the Cloister process in the data analysis pipeline."""

    z_edge: NDArray[np.double]
    z_ecorr: NDArray[np.double]

    def __iter__(self) -> Iterator[NDArray[np.double]]:
        """Allow unpacking directly."""
        return iter((self.z_edge, self.z_ecorr))

    T = TypeVar("T", bound="CloisterOut")

    @classmethod
    def from_stage_runner_output(
        cls: type[T],
        stage_runner_output: dict[str, Any],
    ) -> T:
        """
        Initialise a CloisterOut object from the output of an InstanceSpace StageRunner.

        Args
        ----
            cls (type[T]): the class
            stage_runner_output (dict[str, Any]): output of StageRunner for an
                InstanceSpace

        Returns
        -------
            CloisterOut: a CloisterOut object
        """
        return cls(
            z_edge=stage_runner_output["z_edge"],
            z_ecorr=stage_runner_output["z_ecorr"],
        )


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

    T = TypeVar("T", bound="PythiaOut")

    @classmethod
    def from_stage_runner_output(
        cls: type[T],
        stage_runner_output: dict[str, Any],
    ) -> T:
        """
        Initialise a PythiaOut object from the output of an InstanceSpace StageRunner.

        Args
        ----
            cls (type[T]): the class
            stage_runner_output (dict[str, Any]): output of StageRunner for an
                InstanceSpace

        Returns
        -------
            PythiaOut: a PythiaOut object
        """
        return cls(
            mu=stage_runner_output["mu"],
            sigma=stage_runner_output["sigma"],
            cp=stage_runner_output["cp"],
            svm=stage_runner_output["svm"],
            cvcmat=stage_runner_output["cvcmat"],
            y_sub=stage_runner_output["y_sub"],
            y_hat=stage_runner_output["y_hat"],
            pr0_sub=stage_runner_output["pr0_sub"],
            pr0_hat=stage_runner_output["pr0_hat"],
            box_consnt=stage_runner_output["box_consnt"],
            k_scale=stage_runner_output["k_scale"],
            precision=stage_runner_output["precision"],
            recall=stage_runner_output["recall"],
            accuracy=stage_runner_output["accuracy"],
            selection0=stage_runner_output["selection0"],
            selection1=stage_runner_output["selection1"],
            summary=stage_runner_output["summary"],
        )


@dataclass(frozen=True)
class Footprint:
    """A class to represent a footprint with geometric and statistical properties.

    Attributes:
    ----------
    polygon : Polygon | None
        The geometric shape of the footprint.
    area : float
        The area of the footprint.
    elements : int
        The number of data points within the footprint.
    good_elements : int
        The number of "good" data points within the footprint.
    density : float
        The density of points within the footprint.
    purity : float
        The purity of "good" elements in relation to all elements in the footprint.
    """

    polygon: Polygon | None
    area: float
    elements: int
    good_elements: int
    density: float
    purity: float

    @classmethod
    def from_polygon(
        cls: type["Footprint"],
        polygon: Polygon | None,
        z: NDArray[np.double],
        y_bin: NDArray[np.bool_],
        smoothen: bool = False,
    ) -> "Footprint":
        """Create a Footprint object based on the given polygon.

        Parameters:
        ----------
        polygon : Polygon
            The polygon to create the footprint from.
        z : NDArray[np.double]
            The space of instances, represented as an array of data points (features).
        y_bin : NDArray[np.bool_]
            Binary array indicating the points corresponding to the footprint.
        smoothen : bool, optional
            Indicates if the polygon borders need to be smoothened, by default False.

        Returns:
        -------
        Footprint:
            The created footprint, or an empty one if the polygon is empty.
        """
        if polygon is None:
            return cls(None, 0, 0, 0, 0, 0)

        if smoothen:
            polygon = polygon.buffer(0.01).buffer(-0.01)

        elements = np.sum(
            [polygon.contains(point) for point in MultiPoint(z).geoms],
        )
        good_elements = np.sum(
            [polygon.contains(point) for point in MultiPoint(z[y_bin]).geoms],
        )
        density = elements / polygon.area if polygon.area != 0 else 0
        purity = good_elements / elements if elements != 0 else 0

        return cls(
            polygon=polygon,
            area=polygon.area,
            elements=elements,
            good_elements=good_elements,
            density=density,
            purity=purity,
        )


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

    T = TypeVar("T", bound="TraceOut")

    @classmethod
    def from_stage_runner_output(
        cls: type[T],
        stage_runner_output: dict[str, Any],
    ) -> T:
        """Initialise a TraceOut object from the output of an InstanceSpace StageRunner.

        Args
        ----
            cls (type[T]): the class
            stage_runner_output (dict[str, Any]): output of StageRunner for an
                InstanceSpace

        Returns
        -------
            TraceOut: a TraceOut object
        """
        return cls(
            space=stage_runner_output["space"],
            good=stage_runner_output["good"],
            best=stage_runner_output["best"],
            hard=stage_runner_output["hard"],
            summary=stage_runner_output["summary"],
        )


@dataclass(frozen=True)
class FeatSel:
    """Holds indices for feature selection."""

    idx: NDArray[np.intc]

    T = TypeVar("T", bound="FeatSel")

    @classmethod
    def from_stage_runner_output(
        cls: type[T],
        stage_runner_output: dict[str, Any],
    ) -> T:
        """Initialise a FeatSel object from the output of an InstanceSpace StageRunner.

        Args
        ----
            cls (type[T]): the class
            stage_runner_output (dict[str, Any]): output of StageRunner for an
                InstanceSpace

        Returns
        -------
            FeatSel: a FeatSel object
        """
        return cls(
            idx=stage_runner_output["idx"],
        )
