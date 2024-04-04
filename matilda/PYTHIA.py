
import numpy as np
from numpy.typing import NDArray
from sklearn.svm import SVC

from matilda.data.model import AlgorithmSummary
from matilda.data.option import Opts


def pythia(
    z: np.NDArray[np.double],
    y: NDArray[np.double],
    y_bin: NDArray[np.double],
    y_best: NDArray[np.double],
    algo_labels: list[str],
    opts: Opts,
) -> list[AlgorithmSummary]:
    """
    PYTHIA function for algorithm selection and performance evaluation using SVM.

    :param Z: Feature matrix (instances x features).
    :param Y: Target variable vector (not used directly in this function, but part of 
              the interface).
    :param Ybin: Binary matrix indicating success/failure of algorithms.
    :param Ybest: Vector containing the best performance of each instance.
    :param algolabels: List of algorithm labels.
    :param opts: Dictionary of options.

    :return: Summary of performance for each algorithm.
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


class SvmRes:
    svm: SVC
    Ysub: NDArray[np.double]
    Psub: NDArray[np.double]
    Yhat: NDArray[np.double]
    Phat: NDArray[np.double]
    C: float
    g: float


def fitlibsvm(
    z: NDArray[np.double],
    y_bin: NDArray[np.double],
    n_folds: int,
    kernel: str,
    params: NDArray[np.double],
) -> SvmRes:
    """Train a SVM model using the LIBSVM library."""
    raise NotImplementedError


def fitmatsvm(
    z: NDArray[np.double],
    y_bin: NDArray[np.double],
    w: NDArray[np.double],
    cp,
    k: str,
    params: NDArray[np.double],
) -> SvmRes:
    """
    Train a SVM model using MATLAB's 'fitcsvm' function.

    :param cp: Cross-validation splitting strategy from package/lib
    """
    raise NotImplementedError
