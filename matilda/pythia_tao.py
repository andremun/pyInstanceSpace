# Import all the necessary libraries
import numpy as np
import pandas as pd
import multiprocessing

from numpy.typing import NDArray
from pytictoc import TicToc

from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC

from matilda.data.model import AlgorithmSummary
from matilda.data.option import PythiaOptions


def pythia(
    z: NDArray[np.double],
    y: NDArray[np.double],
    y_bin: NDArray[np.double],
    y_best: NDArray[np.double],
    algo_labels: list[str],
    opts: PythiaOptions,
) -> list[AlgorithmSummary]:
    print(" -> Initializing PYTHIA.")

    # Initialise data with its structure that can be used in Pythia.py
    mu, sigma = np.mean(z, axis=0), np.std(z, ddof=1, axis=0)
    z_norm = (z - mu) / sigma
    ninst, nalgos = y_bin.shape
    cp, svm = [None] * nalgos, [None] * nalgos
    cvcmat = np.zeros((nalgos, 4))
    y_sub, y_hat = np.zeros_like(y_bin, dtype=bool), np.zeros_like(y_bin, dtype=bool)
    pr0_sub, pr0_hat = (
        np.zeros_like(y_bin, dtype=bool),
        np.zeros_like(y_bin, dtype=float),
    )
    box_const, k_scale = np.zeros(nalgos), np.zeros(nalgos)

    print("-------------------------------------------------------------------------")

    precalparams = (
        hasattr(opts, "params")
        and isinstance(opts.params, (list, np.ndarray))
        and np.array(opts.params).shape == (nalgos, 2)
    )
    params = np.full((nalgos, 2), np.nan)

    if opts.is_poly_krnl:
        kernel_fcn = "polynomial"
    else:
        if ninst > 1000:
            print(
                "  -> For datasets larger than 1K Instances, PYTHIA works better with a Polynomial kernel."
            )
            print(
                "  -> Consider changing the kernel if the results are unsatisfactory."
            )
            print(
                "-------------------------------------------------------------------------"
            )
        kernel_fcn = "gaussian"
    print(" => PYTHIA is using a " + kernel_fcn + " kernel")
    print("-------------------------------------------------------------------------")

    if opts.use_lib_svm:
        print(" -> Using LIBSVM's libraries")

        if precalparams:
            print(" -> Using pre-calculated hyper-parameters for the SVM.")
            params = opts.params
        else:
            print(
                " -> Search on a latin hyper-cube design will be used for parameter hyper-tunning."
            )
    else:
        print(" -> Using MATLAB's SVM libraries.")

        if precalparams:
            print(" -> Using pre-calculated hyper-parameters for the SVM.")
            params = opts.params
        else:
            print(" -> Bayesian Optimization will be used for parameter hyper-tunning.")

        print(
            "-------------------------------------------------------------------------"
        )

        if opts.use_weights:
            print(" -> PYTHIA is using cost-sensitive classification.")
            w = np.abs(y - np.nanmean(y))
            w[w == 0] = np.min(w[w != 0])
            w[np.isnan(w)] = np.max(w[~np.isnan(w)])
            w_aux = w
        else:
            print(" -> PYTHIA is not using cost-sensitive classification.")
            w = np.ones((ninst, nalgos))
    print("-------------------------------------------------------------------------")

    print(
        "  -> Using a "
        + opts.cv_folds
        + "-fold stratified cross-validation experiment to evaluate the SVMs."
    )
    print("-------------------------------------------------------------------------")
    print("  -> Training has started. PYTHIA may take a while to complete...")

    t = TicToc()
    t.tic()

    for i in range(nalgos):
        t_inner = TicToc()
        t_inner.tic()

        np.random.seed(0)

        y_b = y_bin[:, i]

        cp[i] = StratifiedKFold(n_splits=opts.cv_folds, shuffle=True, random_state=0)

        if opts.use_lib_svm:
            None  # fit_libsvm(z_norm, y_b, cp[i], kernel_fcn, params[i])
        else:
            svm, y_sub, p_sub, y_hat, p_hat, c, g = fit_mat_svm(
                z_norm,
                y_b,
                w_aux[:, i],
                cp[i],
                kernel_fcn,
                params[i],
            )

        aux = confusion_matrix(y_b, y_sub[i])
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
        print(
            "    -> PYTHIA has trained a model for "
            + algo_labels[i]
            + ", there are "
            + models_left
            + " models left to train."
        )
        print("    -> Elapsed time: " + t_inner.tocvalue() + "s")

    tn, fp, fn, tp = cvcmat[:, 0], cvcmat[:, 1], cvcmat[:, 2], cvcmat[:, 3]
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    accuracy = (tp + tn) / ninst

    print("Total elapsed time: " + t.tocvalue() + "s")
    print("-------------------------------------------------------------------------")
    print(" -> PYTHIA has completed training the models.")
    print(
        " -> The average cross validated precision is: "
        + np.round(100 * np.mean(precision), 1)
        + "%"
    )
    print(
        " -> The average cross validated accuracy is: "
        + np.round(100 * np.mean(accuracy), 1)
        + "%"
    )
    print("    -> Elapsed time: " + t.tocvalue() + "s")
    print("-------------------------------------------------------------------------")

    if nalgos > 1:
        best, selection_0 = (
            np.max(y_hat * precision.T, axis=1),
            np.argmax(y_hat * precision.T, axis=1),
        )
    else:
        best, selection_0 = y_hat, y_hat

    default = np.argmax(np.mean(y_bin, axis=0))
    selection_1 = selection_0.copy
    selection_0[best <= 0] = 0
    selection_1[best <= 0] = default

    sel0 = selection_0[:, None] == np.arrange(1, nalgos + 1)
    sel1 = selection_1[:, None] == np.arramge(1, nalgos + 1)
    avgperf = np.nanmean(y)
    stdperf = np.nanstd(y)
    y_full = y.copy()
    y_svms = y.copy()
    y[~sel0] = np.NaN
    y_full[~sel1] = np.NaN
    y_svms[~y_hat] = np.NaN

    pgood = np.mean(np.any(y_bin & sel1, axis=1))
    fb = np.sum(np.any(y_bin & ~sel0, axis=1))
    fg = np.sum(np.any(~y_bin & sel0, axis=1))
    tg = np.sum(np.any(y_bin & sel0, axis=1))

    precisionsel = tg / (tg + fg)
    recallsel = tg / (tg + fb)

    print("  -> PYTHIA is preparing the summary table.")
    summaries: list[AlgorithmSummary] = []

    for i, label in enumerate(algo_labels + ["Oracle", "Selector"]):
        summary = AlgorithmSummary(
            label,
            np.round(np.append(avgperf, [np.nanmean(y_best), np.nanmean(y_full)]), 3),
            np.round(np.append(stdperf, [np.nanstd(y_best), np.nanstd(y_full)]), 3),
            np.round(np.append(np.mean(y_bin, axis=0), [1, pgood]), 3),
            np.round(np.append(np.nanmean(y_svms), [np.nan, np.nanmean(y)]), 3),
            np.round(np.append(np.nanstd(y_svms), [np.nan, np.nanstd(y)]), 3),
            np.round(np.append(100 * accuracy, [np.nan, np.nan]), 1),
            np.round(np.append(100 * precision, [np.nan, precisionsel]), 1),
            np.round(100 * np.append(recall, [np.nan, recallsel]), 1),
            np.round(box_const, 3),
            np.round(k_scale, 3),
        )

        summaries.append(summary)

    print("  => PYTHIA has completed! Performance of the models:")
    print(" ")

    for result in summaries:
        print(result)

    return summaries


def fitmatsvm(
    z_norm: NDArray[np.double],
    y_bin: NDArray[np.double],
    w_aux: NDArray[np.double],
    cp: StratifiedKFold,  # not sure
    kernel_fcn: str,
    params: NDArray[np.double],
):
    # Scikit-learn lib need to ensure data contiguity
    z_norm = np.ascountiguousarray(z_norm)
    y_bin = np.ascountiguousarray(y_bin)
    w_aux = np.ascountiguousarray(w_aux)

    # Check if parallel processing is available
    n_workers = multiprocessing.cpu_count() if multiprocessing.cpu_count() > 1 else 1

    if np.any(np.isnan(params)):
        param_grid = {
            "C": np.logspace(-10, 4, num=15, base=2),
            "gamma": np.logspace(-10, 4, num=15, base=2),
        }

        svm_model = SVC(kernel=kernel_fcn, class_weight=None, random_state=0)
        grid_search = GridSearchCV(
            estimator=svm_model,
            params_grid=param_grid,
            cv=cp,
            n_jobs=n_workers,
            scoring="neg_log_loss",
            verbose=0,
        )
        grid_search.fit(z_norm, y_bin)

        best_svm = grid_search.best_estimator_
        c = grid_search.best_params_["C"]
        g = grid_search.best_estimator_["gamma"]

        y_sub = best_svm.predict(z_norm)
        p_sub = best_svm.predict_proba(z_norm)[:, 1]

        y_hat = y_sub
        p_hat = p_sub

    else:
        c = params[0]
        g = params[1]

        svm_model = SVC(C=c, gamma=g, kernel=kernel_fcn, probability=True)
        y_sub = np.zeros_like(y_bin)
        p_sub = np.zeros_like(y_bin, dtype=float)

        for train_index, test_index in cp.split(z_norm, y_bin):
            svm_model.fit(
                z_norm[train_index],
                y_bin[train_index],
                sample_weight=w_aux[train_index],
            )
            y_sub[test_index] = svm_model.predict(z_norm[test_index])
            p_sub[test_index] = svm_model.predict(z_norm[test_index])[:, 1]

        svm_model.fit(z_norm, y_bin, sample_weight=w_aux)
        y_hat = svm_model.predict(z_norm)
        p_hat = svm_model.predict(z_norm)[:, 1]

    return best_svm, y_sub, p_sub, y_hat, p_hat, C, g
