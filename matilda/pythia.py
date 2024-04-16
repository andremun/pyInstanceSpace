"""PYTHIA function for algorithm selection and performance evaluation using SVM."""

import numpy as np
from numpy.typing import NDArray
from scipy.stats import zscore
from pytictoc import TicToc
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
# from sklearn import SVC

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
    # No params in opt??
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
    
    if opts.pythia.use_lib_svm:
        print("  -> Using LIBSVM''s libraries.")

        if precalcparams:
            print("  -> Using pre-calculated hyper-parameters for the SVM.")
            params = opts.params
        else:
            print("  -> Search on a latin hyper-cube design will be used for parameter hyper-tunning.")
    else:
        print("  -> Using MATLAB''s SVM libraries.")

        if precalcparams:
            print("  -> Using pre-calculated hyper-parameters for the SVM.")
            params = opts.params
        else:
            print("    -> Bayesian Optimization will be used for parameter hyper-tunning.")

        print("-------------------------------------------------------------------------")
        
        if opts.pythia.use_weights:
            print("  -> PYTHIA is using cost-sensitive classification.")
            w = np.abs(y - np.nanmean(y))
            w [w == 0] = np.min(w [w != 0])
            w [np.isnan(w)] = np.max( w [~np.isnan(w)])
            # no need for w_aux?
        else:
            print("  -> PYTHIA is not using cost-sensitive classification.")
            w = np.ones((ninst, nalgos))

    print("-------------------------------------------------------------------------")     
    
    print(f"  -> Using a {opts.pythia.cv_folds}-fold stratified cross-validation experiment to evaluate the SVMs.")
    print("-------------------------------------------------------------------------")
    
    # TODO Section 3: Train SVM model for each algorithm & Evaluate performance.
    print('  -> Training has started. PYTHIA may take a while to complete...')

    t = TicToc()
    t.tic()

    for i in range(nalgos):
        t_inner = TicToc()
        t_inner.tic()

        state = np.random.get_state()
        np.random.seed(0)  # equivalent to MATLAB's rng('default') ?

        # REQUIRE: Test case for validation the result
        y_b = y_bin[:, i]
        cv = StratifiedKFold(n_splits = opts.pythia.cv_folds, shuffle = True, random_state = 0)
        #  c.split(np.zeros(group.shape), group)

        if opts.pythia.use_lib_svm:
            svm_res = fit_libsvm(z_norm, y_b, cv, kernel_fcn, params[i])
        else:
            svm_res = fit_mat_svm(z_norm, y_b, w[:, i], cv, kernel_fcn, params[i])

        np.random.set_state(state)

        # REQUIRE: Test case for validation the result
        aux = confusion_matrix(y_b, y_sub[:, i])

        if np.prod(aux.shape) != 4:
            caux = aux
            aux = np.zeros((2, 2))
    
            if np.all(y_b == 0):
                if np.all(y_sub[:, i] == 0):
                    aux[0, 0] = caux
                elif np.all(y_sub[:, i] == 1):
                    aux[1, 0] = caux

            elif np.all(y_b == 1):
                if np.all(y_sub[:, i] == 0):
                    aux[0, 1] = caux
                elif np.all(y_sub[:, i] == 1):
                    aux[1, 1] = caux

        cvcmat[:, i] = aux.flatten()
            
        models_left = nalgos - (i + 1)
        if models_left == 0:
            print(f"    -> PYTHIA has trained a model for {algo_labels[i]}, there are no models left to train.")
        elif models_left == 1:
            print(f"    -> PYTHIA has trained a model for {algo_labels[i]}, there is 1 model left to train.")
        else:
             print(f"    -> PYTHIA has trained a model for {algo_labels[i]}, there are {models_left} models left to train.")

        print(f"      -> Elapsed time: {t_inner.tocvalue():.2f}s")
    
    tn, fp, fn, tp = cvcmat[:, 0], cvcmat[:, 1], cvcmat[:, 2], cvcmat[:, 3]
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    accuracy = (tp + tn) / ninst

    print(f"Total elapsed time: {t.tocvalue():.2f}s")
    print("-------------------------------------------------------------------------")
    print("  -> PYTHIA has completed training the models.")
    print(f"  -> The average cross validated precision is: {np.round(100 * np.mean(precision), 1)}%")
    print(f"  -> The average cross validated accuracy is: {np.round(100 * np.mean(accuracy), 1)}%")
    print(f"      -> Elapsed time: {t.tocvalue():.2f}s")
    print("-------------------------------------------------------------------------")

    """We assume that the most precise SVM (as per CV-Precision) is the most reliable."""
    best, selection_0
    if nalgos > 1:
        best, selection_0 = np.max(y_hat * precision.T, axis=1), np.argmax(y_hat * precision.T, axis=1)
    else:
        best, selection_0 = y_hat, y_hat

    default = np.argmax(np.mean(y_bin, axis=0))
    selection_1 = selection_0.copy()
    selection_0[best <= 0] = 0
    selection_1[best <= 0] = default

    sel0 = selection_0[:, None] == np.arange(1, nalgos + 1)
    sel1 = selection_1[:, None] == np.arange(1, nalgos + 1)

    avgperf = np.nanmean(y)
    stdperf = np.nanstd(y)

    y_full = y.copy()
    y_svms = y.copy()

    y[~ sel0] = np.NaN
    y_full[~ sel1] = np.NaN
    y_svms[~ y_hat] = np.NaN

    pgood = np.mean(np.any(y_bin & sel1, axis=1))
    fb = np.sum(np.any(y_bin & ~sel0, axis=1))
    fg = np.sum(np.any(~ y_bin & sel0, axis=1))
    tg = np.sum(np.any(y_bin & sel0, axis=1))

    precisionsel = tg / (tg + fg)
    recallsel = tg / (tg + fb)

    # TODO Section 6: Generate output

    raise NotImplementedError


class SvmRes:
    """Resent data resulting from SVM."""

    svm: NDArray[np.double] # svm: SVC
    Ysub: NDArray[np.double]
    Psub: NDArray[np.double]
    Yhat: NDArray[np.double]
    Phat: NDArray[np.double]
    C: float
    g: float


def fit_libsvm(
    z: NDArray[np.double],
    y_bin: NDArray[np.double],
    n_folds: int,
    kernel: str,
    params: NDArray[np.double],
) -> SvmRes:
    """Train a SVM model using the LIBSVM library."""
    raise NotImplementedError


def fit_mat_svm(
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
