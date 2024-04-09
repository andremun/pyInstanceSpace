"""PYTHIA function for algorithm selection and performance evaluation using SVM."""

import numpy as np
from numpy.typing import NDArray
from scipy.stats import zscore
from sklearn import SVC

from matilda.data.model import AlgorithmSummary
from matilda.data.option import Opts


def pythia(
    z: NDArray[np.double],
    y: NDArray[np.double],
    y_bin: NDArray[np.double],
    y_best: NDArray[np.double],
    algo_labels: list[str],
    opts: Opts,
) -> list[AlgorithmSummary]:
    """
    PYTHIA function for algorithm selection and performance evaluation using SVM.

    :param Z: Feature matrix (instances x features).
    :param Y: Target variable vector (not used directly in this function, but part
    of the interface).
    :param Ybin: Binary matrix indicating success/failure of algorithms.
    :param Ybest: Vector containing the best performance of each instance.
    :param algolabels: List of algorithm labels.
    :param opts: Dictionary of options.

    :return: Summary of performance for each algorithm.
    """
    print("  -> Initializing PYTHIA.")

    # Test case Required: Same result from MATLAB
    z_norm = zscore(z, axis = 0, ddof = 1)
    mu, sigma = np.mean(z, axis=0), np.std(z, axis=0, ddof=1) 

    ninst, nalgos = y_bin.shape

    cp, svm = [None] * nalgos, [None] * nalgos

    cvcmat = np.zeros((nalgos, 4))

    y_sub, y_hat = np.zeros_like(y_bin, dtype=bool), np.zeros_like(y_bin, dtype=bool)

    pro_sub, pro_hat = np.zeros_like(y_bin, dtype=float), np.zeros_like(y_bin, dtype=float)

    box_const, k_scale = np.zeros(nalgos), np.zeros(nalgos)

    """ 
    Section 2: Configure the SVM training process.
    # (Including kernel function selection, library usage, hyperparameter strategy,
    # and cost-sensitive classification.)
    """

    print("-------------------------------------------------------------------------")
    # No params in opt?
    precalcparams = (
        hasattr(opts, 'params') and
        isinstance(opts.params, (list, np.ndarray)) and # only np.ndarrays
        np.array(opts.params).shape == (nalgos, 2)
    )
    params = np.full((nalgos, 2), np.nan)

    if opts.pythia.is_poly_krnl:
        kernel_fcn = "polynomial"
    else:
        if ninst > 1000:
           print("  -> For datasets larger than 1K Instances, PYTHIA works better with a Polynomial kernel.")
           print("  -> Consider changing the kernel if the results are unsatisfactory.")
           print("-------------------------------------------------------------------------")
        kernel_fcn = "gaussian"
    
    print(f"  -> PYTHIA is using a {kernel_fcn} kernel ")
    print("-------------------------------------------------------------------------")
    
    # TODO Section 3: Train SVM model for each algorithm & Evaluate performance.

    # TODO Section 4: SVM model selection.

    # TODO Section 5: SVM model selection.

    # TODO Section 6: Generate output

    raise NotImplementedError


class SvmRes:
    """Resent data resulting from SVM."""

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
    cp: NDArray[np.double], # Actually its an array and the type is dynamic
    k: str,
    params: NDArray[np.double],
) -> SvmRes:
    """
    Train a SVM model using MATLAB's 'fitcsvm' function.

    :param cp: Cross-validation splitting strategy from package/lib
    """
    raise NotImplementedError
