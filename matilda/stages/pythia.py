"""PYTHIA function for algorithm selection and performance evaluation using SVM."""

import numpy as np
from numpy.typing import NDArray

from matilda.data.model import PythiaOut
from matilda.data.option import Options


class SvmRes:
    """Resent data resulting from SVM."""

    svm: None
    Ysub: NDArray[np.double]
    Psub: NDArray[np.double]
    Yhat: NDArray[np.double]
    Phat: NDArray[np.double]
    C: float
    g: float

class Pythia:
    """See file docstring."""

    @staticmethod
    def run(
        z: NDArray[np.double],  # noqa: ARG004
        y: NDArray[np.double],  # noqa: ARG004
        y_bin: NDArray[np.bool_],  # noqa: ARG004
        y_best: NDArray[np.double],  # noqa: ARG004
        algo_labels: list[str],  # noqa: ARG004
        opts: Options,  # noqa: ARG004
    ) -> PythiaOut:
        """PYTHIA function for algorithm selection and performance evaluation using SVM.

        Args
            z: Feature matrix (instances x features).
            y: Target variable vector (not used directly in this
                function, but part
            y_bin: Binary matrix indicating success/failure of
                algorithms.
            y_best: Vector containing the best performance of each
                instance.
            algo_labels: List of algorithm labels.
            opts: Dictionary of options.
        of the interface).

        Returns
        -------
            Summary of performance for each algorithm.
        """
        print("  -> Initializing PYTHIA.")

        # TODO Section 1: Initialize and standardize the dataset.

        # TODO Section 2: Configure the SVM training process.
        # (Including kernel function selection, library usage, hyperparameter strategy,
        # and cost-sensitive classification.)

        # TODO Section 3: Train SVM model for each algorithm & Evaluate performance.

        # TODO Section 4: SVM model selection.

        # TODO Section 5: SVM model selection.

        # TODO Section 6: Generate output

        raise NotImplementedError


    @staticmethod
    def fitlibsvm(
        z: NDArray[np.double],
        y_bin: NDArray[np.double],
        n_folds: int,
        kernel: str,
        params: NDArray[np.double],
    ) -> SvmRes:
        """Train a SVM model using the LIBSVM library."""
        raise NotImplementedError


    @staticmethod
    def fitmatsvm(
        z: NDArray[np.double],
        y_bin: NDArray[np.double],
        w: NDArray[np.double],
        cp: NDArray[np.double], # Actually its an array and the type is dynamic
        k: str,
        params: NDArray[np.double],
    ) -> SvmRes:
        """Train a SVM model using MATLAB's 'fitcsvm' function."""
        raise NotImplementedError
