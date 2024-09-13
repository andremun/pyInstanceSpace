"""PILOT: Obtaining a two-dimensional projection.

Projecting Instances with Linearly Observable Trends (PILOT)
is a dimensionality reduction algorithm which aims to facilitate
the identification of relationships between instances and
algorithms by unveiling linear trends in the data, increasing
from one edge of the space to the opposite.

"""

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from matilda.stages.stage import Stage


class PilotStage(Stage):
    """Pilot stage class."""

    def __init__(
        self,
        x: NDArray[np.double],
        y: NDArray[np.double],
        feat_labels: list[str],
    ) -> None:
        """Initialize the Pilot stage.

        The Initialize functon is used to create a Pilot class.

        Args
        ----
            x (NDArray[np.double]): The feature matrix (instances x features) to
                process.
            y (NDArray[np.double]): The data points for the selected feature
            feat_labels (list[str]): List feature names

        Returns
        -------
            None
        """
        self.x = x
        self.y = y
        self.feat_labels = feat_labels

    @staticmethod
    def _inputs() -> list[tuple[str, type]]:
        """Use the method for determining the inputs for pilot.

        Args
        ----
            None

        Returns
        -------
            list[tuple[str, type]]
                List of inputs for the stage
        """
        return [
            ["x", NDArray[np.double]],
            ["y", NDArray[np.double]],
            ["feat_labels", list[str]],
        ]

    @staticmethod
    def _outputs() -> list[tuple[str, type]]:
        """Use the method for determining the outputs for pilot.

        Args
        ----
            None

        Returns
        -------
            list[tuple[str, type]]
                List of outputs for the stage
        """
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
            ["summary", pd.DataFrame],
        ]

    def _run(
        self,
        x: NDArray[np.double],
        y: NDArray[np.double],
        feat_labels: list[str],
    ) -> tuple[
        NDArray[np.double] | None,
        NDArray[np.double] | None,
        NDArray[np.double] | None,
        NDArray[np.double] | None,
        NDArray[np.double],
        NDArray[np.double],
        NDArray[np.double],
        NDArray[np.double],
        NDArray[np.double],
        NDArray[np.double],
        pd.DataFrame,
    ]:

        # Implement all the code in and around this class in buildIS
        raise NotImplementedError

    @staticmethod
    def pilot(
        x: NDArray[np.double],
        y: NDArray[np.double],
        feat_labels: list[str],
    ) -> tuple[
        NDArray[np.double] | None,
        NDArray[np.double] | None,
        NDArray[np.double] | None,
        NDArray[np.double] | None,
        NDArray[np.double],
        NDArray[np.double],
        NDArray[np.double],
        NDArray[np.double],
        NDArray[np.double],
        NDArray[np.double],
        pd.DataFrame,
    ]:
        """Use the method used for running the pilot stage only."""
        # Implement all the code in PILOT.py here
        raise NotImplementedError
