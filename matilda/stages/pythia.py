"""PYTHIA function for algorithm selection and performance evaluation using SVM."""

import time

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pytictoc import TicToc
from scipy import stats
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
)
from sklearn.svm import SVC

from matilda.data.model import PythiaDataChanged, PythiaOut
from matilda.data.options import PythiaOptions


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

    def __init__(self,
        x:NDArray[np.double],
        y:NDArray[np.double],
        y_bin:NDArray[np.bool_],
        y_best:NDArray[np.double],
        algo_labels:list[str],
        opts:PythiaOptions,
    ) -> None:
        """
        Initialize the Pythia object.

        Args:
        ----
            x: Feature matrix.
            y: Target variable vector.
            y_bin: Binary matrix indicating success/failure of algorithms.
            y_best: Vector containing the best performance of each instance.
            algo_labels: List of algorithm labels.
            opts: PythiaOptions object containing options.

        Returns:
        -------
            None

        """
        self.x = x,
        self.y = y,
        self.y_bin = y_bin
        self.y_best = y_best
        self.algo_labels = algo_labels
        self.opts = opts

        self.cp = [None] * len(algo_labels)
        self.rng = np.random.default_rng(seed=0)
        # self.PythiaOut = PythiaOut()
    @staticmethod
    def run(
        z: NDArray[np.double],
        y: NDArray[np.double],
        y_bin: NDArray[np.bool_],
        y_best: NDArray[np.double],  # noqa: ARG004
        algo_labels: list[str],  # noqa: ARG004
        opts: PythiaOptions,
    ) -> tuple[PythiaDataChanged, PythiaOut]:
        """
        PYTHIA function for algorithm selection and performance evaluation using SVM.

        Args:
        ----
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

        Returns:
        -------
            Summary of performance for each algorithm.

        """
        print("  -> Initializing PYTHIA.")
        z = stats.zscore(z, ddof=1)
        mu = np.mean(z, axis=0)
        sigma = np.std(z, ddof=1, axis=0)
        ninst, nalgos = y_bin.shape
        Yfull = y.copy()
        Ysvms = y.copy()
        # Step 2: Set elements to NaN based on conditions
        # y[~sel0] = np.nan       # Set elements to NaN where sel0 is False
        # Yfull[~sel1] = np.nan   # Set elements to NaN where sel1 is False
        # Ysvms[~out.Yhat] = np.nan
        pgood = np.mean(np.any(np.logical_and(y_bin, sel1), axis=1))
        params = np.full((nalgos, 2), np.nan)
        print("-------------------------------------------------------------------------")

        # Configure the SVM training process.
        # (Including kernel function selection, library usage, hyperparameter strategy,
        # and cost-sensitive classification.)
        # TODO: Check the precal params
        precalparams = (
            hasattr(opts, "params")
            and isinstance(opts.params, (list, np.ndarray))
            and np.array(opts.params).shape == (nalgos, 2)
        )
        kernel_fcn = "guassian"
        if opts.is_poly_krnl:
            kernel_fcn = "polynomial"
        elif ninst > 1000:  # noqa: PLR2004
            print(
                "  -> For datasets larger than 1K Instances, PYTHIA works better with a Polynomial kernel.",
            )
            print(
                "  -> Consider changing the kernel if the results are unsatisfactory.",
            )
            print(
                "-------------------------------------------------------------------------",
            )
        print(" => PYTHIA is using a " + kernel_fcn + " kernel")
        print("-------------------------------------------------------------------------")


        if precalparams:
            print(" -> Using pre-calculated hyper-parameters for the SVM.")
            params = opts.params
        else:
            print(
                " -> Search on a latin hyper-cube design will be used for parameter hyper-tunning.",
            )

        if precalparams:
            print(" -> Using pre-calculated hyper-parameters for the SVM.")
        else:
            print(" -> Bayesian Optimization will be used for parameter hyper-tunning.")

        print(
            "-------------------------------------------------------------------------",
        )
        if opts.use_lib_svm:
            print(" -> Using LIBSVM's libraries")
            if precalparams:
                print(" -> Using pre-calculated hyper-parameters for the SVM.")
                params = opts.params
        else:
            print(" -> Using MATLAB's libraries")
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
            + "-fold stratified cross-validation experiment to evaluate the SVMs.",
        )
        print("-------------------------------------------------------------------------")
        print("  -> Training has started. PYTHIA may take a while to complete...")

        # TODO Section 3: Train SVM model for each algorithm & Evaluate performance.
        start_time = time.time()
        for i in range(nalgos):
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
            PythiaOut.cp[i] = list(skf.split(z, y_bin[:, i]))
            t_inner = TicToc()
            t_inner.tic()
            if opts.use_lib_svm:
                svmres = fitlibsvm(z, y_bin[:, i],PythiaOut.cp[i], opts.cv_folds, kernel_fcn, params[i])
            else:
                #TODO Section 4: SVM model selection.
                svmres = fitmatsvm(z, y_bin[:, i], w[:, i], PythiaOut.cp[i], kernel_fcn, params[i])

        # Generate output
            aux = confusion_matrix(y_bin[:, i], svmres.Ysub[:, i])
            if aux.size != 4:
                caux = aux
                aux = np.zeros((2, 2), dtype=int)
                if np.all(y_bin[:, i] == 0):
                    if np.all(svmres.Ysub[:, i] == 0):
                        aux[0, 0] = caux
                    elif np.all(svmres.Ysub[:, i] == 1):
                        aux[1, 0] = caux
                elif np.all(y_bin[:, i] == 1):
                    if np.all(svmres.Ysub[:, i] == 0):
                        aux[0, 1] = caux
                    elif np.all(svmres.Ysub[:, i] == 1):
                        aux[1, 1] = caux
            svmres.cvcmat[i, :] = aux.flatten()

        if i == nalgos - 1:
            print(f"    -> PYTHIA has trained a model for '{algo_labels[i]}', there are no models left to train.")
        else:
            print(f"    -> PYTHIA has trained a model for '{algo_labels[i]}', there are {nalgos - i - 1} models left to train.")  # noqa: E501
        elapsed_time = time.time() - start_time
        print(f"      -> Elapsed time: {elapsed_time:.2f}s")
        # raise NotImplementedError
        # Assuming pythiaout has a method tocvalue() that returns elapsed time as a string

        print("Total elapsed time: " + pythiaout.tocvalue() + "s")
        print("-------------------------------------------------------------------------")
        print(" -> PYTHIA has completed training the models.")
        print(
            " -> The average cross validated precision is: "
            + str(np.round(100 * np.mean(PythiaOut.precision), 1))
            + "%",
        )
        print(
            " -> The average cross validated accuracy is: "
            + str(np.round(100 * np.mean(PythiaOut.accuracy), 1))
            + "%",
        )
        print("    -> Elapsed time: " + {time.time()-start_time} + "s")
        print("-------------------------------------------------------------------------")
        PythiaOut.summary = np.round([avgperf, np.nanmean(Ybest), np.nanmean(Yfull)], 3).tolist()
        PythiaOut.summary[1:, 2] = np.round([stdperf, np.nanstd(Ybest), np.nanstd(Yfull)], 3).tolist()
        PythiaOut.summary[1:, 3] = np.round([np.mean(Ybin), 1, pgood], 3).tolist()
        PythiaOut.summary[1:, 4] = np.round([np.nanmean(Ysvms), np.nan, np.nanmean(Y)], 3).tolist()
        PythiaOut.summary[1:, 5] = np.round([np.nanstd(Ysvms), np.nan, np.nanstd(Y)], 3).tolist()
        PythiaOut.summary[1:, 6] = np.round(100. * [PythiaOut.accuracy + [np.nan, np.nan]], 1).tolist()
        PythiaOut.summary[1:, 7] = np.round(100. * [PythiaOut.precision + [np.nan, precisionsel]], 1).tolist()
        PythiaOut.summary[1:, 8] = np.round(100. * [PythiaOut.recall + [np.nan, recallsel]], 1).tolist()
        PythiaOut.summary[1:nalgos+1, 9] = np.round(PythiaOut.boxcosnt, 3).tolist()
        PythiaOut.summary[1:nalgos+1, 10] = np.round(PythiaOut.kscale, 3).tolist()
    @staticmethod
    def fitlibsvm(
        z: NDArray[np.double],
        y_bin: NDArray[np.double],
        cp: NDArray[np.double],  # Actually its an array and the type is dynamic
        n_folds: int,
        kernel: str,
        params: NDArray[np.double],
    ) -> SvmRes:
        maxgrid = 4
        mingrid = -10
        nvals = 30
        ninst = z.shape[0]
        # Number of samples
        if(np.isnan(params).any()):
            lhs = stats.qmc.LatinHypercube(d=2, seed=rng)
            samples = lhs.random(30)
            paramgrid = 2 ** ((maxgrid - mingrid) * samples + mingrid)
            df = pd.DataFrame(paramgrid)
            df = df.sort_values(df.columns[0],ascending=True)
            print(df.dtypes)

            # df = df.sort_values(df.columns[0],ascending=True)
            paramgrid = df.to_numpy(dtype=np.float64)
        else:
            nvals = 1
            paramgrid = params

        y_bin = np.array(y_bin, dtype=float) + 1
        Ysub = np.zeros((ninst, nvals))
        Psub = np.zeros((ninst, nvals))

        kernel = "poly" if kernel == "polynomial" else "rbf"

        for i in range(n_folds):
            train_idx, test_idx = cp[i]
            idx = np.zeros(len(z),dtype=int)
            idx[train_idx] = 1

            z_train = z[idx == 1]
            y_train = y_bin[idx == 1]
            z_test = z[idx == 0]
            y_test = y_bin[idx == 0]

            # y_aux = np.zeros((z_test.shape[0], nvals))
            # p_aux = np.zeros((z_test.shape[0], nvals))

            paramgrid_dict = {
                "C": paramgrid[:, 0],
                "gamma": paramgrid[:, 1],
            }
            svc = SVC(probability=True,kernel=kernel)


            for train_idx, test_idx in cp:
                print(f"Max train_idx: {max(train_idx)}, Max test_idx: {max(test_idx)}")
                print(f"Length of Ztrain: {len(z_train)}, Length of Ytrain: {len(y_train)}")
                grid_search = GridSearchCV(estimator=svc,
                                        param_grid=paramgrid_dict,
                                        cv=cp,
                                        scoring="accuracy",
                                        n_jobs=1)
                grid_search.fit(z_train, y_train)
                best_idx = grid_search.best_index_
                best_model = grid_search.best_estimator_


                svm_res = SvmRes()
                svm_res.svm = best_model
                svm_res.Ysub = (Ysub[:, best_idx] == 2)
                svm_res.Psub = Psub[:, best_idx]
                svm_res.Yhat = best_model.predict(z_train)
                svm_res.Phat = best_model.predict_proba(z_train)
                svm_res.C = grid_search.best_params_["C"]
                svm_res.g = grid_search.best_params_["gamma"]

        return svm_res
    @staticmethod
    def fitmatsvm(
        z: NDArray[np.double],
        y_bin: NDArray[np.double],
        w: NDArray[np.double],
        cp: NDArray[np.double],  # Actually its an array and the type is dynamic
        k: str,
        params: NDArray[np.double],
    ) -> SvmRes:
        """Train a SVM model using MATLAB's 'fitcsvm' function."""
        raise NotImplementedError
# if __name__ == "__main__":
