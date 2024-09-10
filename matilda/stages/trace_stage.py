"""TRACE: Calculating the algorithm footprints.

Triangulation with Removal of Areas with Contradicting Evidence (TRACE)
is an algorithm used to estimate the area of good performance of an
algorithm within the space.
For more details, please read the original Matlab code and liveDemo.

"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from matilda.data.model import Footprint
from matilda.data.options import TraceOptions
from matilda.stages.stage import Stage


class TraceStage(Stage):
    """The class for the Trace stage."""

    def __init__(
        self,
        z: NDArray[np.double],
        y_bin: NDArray[np.bool_],
        p: NDArray[np.double],
        beta: NDArray[np.bool_],
        algo_labels: list[str],
    ) -> None:
        """Initialise the Trace stage.

            Args
            ----
        z (NDArray[np.double]): The space of instances
        y_bin (NDArray[np.bool_]): Binary indicators of performance
        p (NDArray[np.double]): Performance metrics for algorithms
        beta (NDArray[np.bool_]): Specific beta threshold for footprint calculation
        algo_labels (list[str]): Labels for each algorithm. Note that the datatype
            is still in deciding

        Returns
        -------
        None
        """
        self.z = z
        self.y_bin = y_bin
        self.p = p
        self.beta = beta
        self.algo_labels = algo_labels

    @staticmethod
    def _inputs() -> list[tuple[str, type]]:
        """Use the method for determining the inputs for trace.

        Args
        ----
            None

        Returns
        -------
            list[tuple[str, type]]
                List of inputs for the stage
        """
        return [
            ["z", NDArray[np.double]],
            ["y_bin", NDArray[np.bool_]],
            ["p", NDArray[np.double]],
            ["beta", NDArray[np.bool_]],
            ["algo_labels", list[str]],
        ]

    @staticmethod
    def _outputs() -> list[tuple[str, type]]:
        """Use the method for determining the outputs for trace.

        Args
        ----
            None

        Returns
        -------
            list[tuple[str, type]]
                List of outputs for the stage
        """
        return [
            ["space", Footprint],
            ["good", list[Footprint]],
            ["best", list[Footprint]],
            ["hard", Footprint],
            ["summary", pd.Dataframe],
        ]

    def _run(
        self,
        options: TraceOptions,
    ) -> tuple[Footprint, list[Footprint], list[Footprint], Footprint, pd.DataFrame]:
        """Use the method for running the trace stage as well as surrounding buildIS.

        Args
        ----
            options (TraceOptions): Configuration options for TRACE and its subroutines

        Returns
        -------
            tuple[Footprint, list[Footprint], list[Footprint], Footprint, pd.DataFrame]
                The results of the trace stage
        """
        # All the code including the code in the buildIS should be here
        raise NotImplementedError

    @staticmethod
    def trace(
        z: NDArray[np.double],
        y_bin: NDArray[np.bool_],
        p: NDArray[np.double],
        beta: NDArray[np.bool_],
        algo_labels: list[str],
    ) -> tuple[Footprint, list[Footprint], list[Footprint], Footprint, pd.DataFrame]:
        """Use the method for running the trace stage.

        Args
        ----
            z (NDArray[np.double]): The space of instances
            y_bin (NDArray[np.bool_]): Binary indicators of performance
            p (NDArray[np.double]): Performance metrics for algorithms
            beta (NDArray[np.bool_]): Specific beta threshold for footprint calculation
            algo_labels (list[str]): Labels for each algorithm. Note that the datatype
                is still in deciding

        Returns
        -------
            tuple[Footprint, list[Footprint], list[Footprint], Footprint, pd.DataFrame]
                The results of the trace stage
        """
        # This has code specific to TRACE.m
        raise NotImplementedError
