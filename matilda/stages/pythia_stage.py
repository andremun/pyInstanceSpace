"""PYTHIA function for algorithm selection and performance evaluation using SVM."""

from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from stages.stage import Stage

from matilda.data.options import PythiaOptions


class PythiaStage(Stage):
    """See file docstring."""

    def __init__(
        self,
        z: NDArray[np.double],
        y: NDArray[np.double],
        y_bin: NDArray[np.bool_],
        y_best: NDArray[np.double],
        algo_labels: list[str],
    ) -> None:
        """See file docstring."""
        self.z = z
        self.y = y
        self.y_bin = y_bin
        self.y_best = y_best
        self.algo_labels = algo_labels

    @staticmethod
    def _inputs() -> list[tuple[str, type]]:
        """See file docstring."""
        return [
            ["z", NDArray[np.double]],
            ["y", NDArray[np.double]],
            ["y_bin", NDArray[np.bool_]],
            ["y_best", NDArray[np.double]],
            ["algo_labels", list[str]],
        ]

    @staticmethod
    def _outputs() -> list[tuple[str, type]]:
        """See file docstring."""
        return [
            ["mu", list[float]],
            ["sigma", list[float]],
            ["cp", Any],  # Change it to proper type
            ["svm", Any],  # Change it to proper type
            ["cvcmat", NDArray[np.double]],
            ["y_sub", NDArray[np.bool_]],
            ["y_hat", NDArray[np.bool_]],
            ["pr0_sub", NDArray[np.double]],
            ["pr0_hat", NDArray[np.double]],
            ["box_consnt", list[float]],
            ["k_scale", list[float]],
            ["precision", list[float]],
            ["recall", list[float]],
            ["accuracy", list[float]],
            ["selection0", NDArray[np.double]],
            ["selection1", Any],  # Change it to proper type
            ["summary", pd.DataFrame],
        ]

    def _run(
        self,
        options: PythiaOptions,
    ) -> tuple[
        NDArray[np.double],
        NDArray[np.double],
        NDArray[np.bool_],
        NDArray[np.double],
        list[str],
    ]:
        """See file docstring."""
        # Implement all the code related to pythia executed in buildIS
        raise NotImplementedError

    @staticmethod
    def pythia(
        z: NDArray[np.double],
        y: NDArray[np.double],
        y_bin: NDArray[np.bool_],
        y_best: NDArray[np.double],
        algo_labels: list[str],
    ) -> tuple[
        NDArray[np.double],
        NDArray[np.double],
        NDArray[np.bool_],
        NDArray[np.double],
        list[str],
    ]:
        """See file docstring."""
        # Implement all the code in PYTHIA.m
        raise NotImplementedError
