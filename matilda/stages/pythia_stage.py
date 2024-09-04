import stage

import pandas as pd
import numpy as np
from numpy.typing import NDArray

from matilda.data.model import PythiaDataChanged, PythiaOut
from matilda.data.options import PythiaOptions

class pythiaStage(stage):
    def __init__(self, z: NDArray[np.double],  # noqa: ARG004
        y: NDArray[np.double],  # noqa: ARG004
        y_bin: NDArray[np.bool_],  # noqa: ARG004
        y_best: NDArray[np.double],  # noqa: ARG004
        algo_labels: list[str]) -> None:
        self.z = z
        self.y = y
        self.y_bin = y_bin
        self.y_best = y_best
        self.algo_labels = algo_labels

    @staticmethod
    def _inputs() -> list[tuple[str, type]]:
        return [
            ["z", NDArray[np.double]],  # noqa: ARG004
        ["y", NDArray[np.double]],  # noqa: ARG004
        ["y_bin", NDArray[np.bool_]],  # noqa: ARG004
        ["y_best", NDArray[np.double]],  # noqa: ARG004
        ["algo_labels", list[str]]
        ]

    @staticmethod
    def _outputs() -> list[tuple[str, type]]:
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
                ["summary", pd.DataFrame]
        ]

    def _run(options: PythiaOptions) -> tuple[NDArray[np.double],  # noqa: ARG004
        NDArray[np.double],  # noqa: ARG004
        NDArray[np.bool_],  # noqa: ARG004
        NDArray[np.double],  # noqa: ARG004
        list[str]]:
        
        #Implement all the code related to pythia executed in buildIS
        raise NotImplementedError 

    @staticmethod
    def pythia(z: NDArray[np.double],  # noqa: ARG004
        y: NDArray[np.double],  # noqa: ARG004
        y_bin: NDArray[np.bool_],  # noqa: ARG004
        y_best: NDArray[np.double],  # noqa: ARG004
        algo_labels: list[str]) -> tuple[NDArray[np.double],  # noqa: ARG004
        NDArray[np.double],  # noqa: ARG004
        NDArray[np.bool_],  # noqa: ARG004
        NDArray[np.double],  # noqa: ARG004
        list[str]]:

        #Implement all the code in PYTHIA.m
        raise NotImplementedError