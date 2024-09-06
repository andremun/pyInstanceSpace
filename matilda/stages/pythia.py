"""PYTHIA function for algorithm selection and performance evaluation using SVM."""

import time
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.svm import SVC
from skopt import BayesSearchCV
from skopt.space import Real

from matilda.data.model import PythiaDataChanged, PythiaOut
from matilda.data.options import PythiaOptions

script_dir = Path(__file__).parents[2] / "tests" / "test_data" / "pythia" / "input"


class SvmRes:
    def __init__(
        self,
        svm: SVC,  # TODO: Change it to proper type
        Ysub: NDArray[np.bool_],
        Psub: NDArray[np.double],
        Yhat: NDArray[np.bool_],
        Phat: NDArray[np.double],
        C: float,
        g: float,
        tn: int,
        fp: int,
        fn: int,
        tp: int,
        accuracy: float,
        precision: float,
        recall: float,
    ):
        self.svm = svm
        self.Ysub = Ysub
        self.Psub = Psub
        self.Yhat = Yhat
        self.Phat = Phat
        self.C = C
        self.g = g
        self.tn = tn
        self.fp = fp
        self.fn = fn
        self.tp = tp
        self.accuracy = accuracy
        self.precision = precision
        self.recall = recall

    def __str__(self) -> str:
        return (
            f"SvmRes(\n"
            f"  Ysub: {self.Ysub.flatten()},\n"
            f"  Psub: {self.Psub.flatten()},\n"
            f"  Yhat: {self.Yhat.flatten()},\n"
            f"  Phat: {self.Phat.flatten()},\n"
            f"  C: {self.C},\n"
            f"  g: {self.g}\n"
            f")"
        )


class Pythia:
    """See file docstring."""

    mu: list[float]
    sigma: list[float]
    cvcmat: NDArray[np.int32]
    y_sub: NDArray[np.bool_]  # = np.zeros((self.nalgos, self.x), dtype=bool)
    y_hat: NDArray[np.bool_]  # = np.zeros((self.nalgos, self.x),dtype=bool)
    pr0hat: NDArray[np.double]  # = np.zeros((self.nalgos, self.x),dtype=np.double)
    pr0sub: NDArray[np.double]  # = np.zeros((self.nalgos, self.x),dtype=np.double)
    box_consnt: list[float]
    k_scale: list[float]
    precision: list[float]
    recall: list[float]
    accuracy: list[float]
    selection0: NDArray[np.int32]
    selection1: NDArray[np.int32]  # Change it to proper type
    cp: StratifiedKFold  # Change it to proper type
    svm: any  # Change it to proper type
    summary: pd.DataFrame

    def __init__(
        self,
        x: NDArray[np.double],
        y: NDArray[np.double],
        y_bin: NDArray[np.bool_],
        y_best: NDArray[np.double],
        algo_labels: list[str],
        hyparams: NDArray[np.double] | None,
        opts: PythiaOptions,
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
        self.x = x
        self.y = y
        self.y_bin = y_bin
        self.y_best = y_best
        self.algo_labels = algo_labels
        self.opts = opts
        self.hyparams = hyparams
        self.nalgos = len(algo_labels)

        self.y_sub = np.zeros((self.x.shape[0], self.nalgos), dtype=bool)
        self.y_hat = np.zeros((self.x.shape[0], self.nalgos), dtype=bool)
        self.pr0sub = np.zeros((self.x.shape[0], self.nalgos), dtype=np.double)
        self.pr0hat = np.zeros((self.x.shape[0], self.nalgos), dtype=np.double)
        self.cp = []
        self.svm = None
        self.cvcmat = np.zeros((self.nalgos, 4), dtype=int)
        self.box_consnt = []
        self.k_scale = []
        self.accuracy = []
        self.precision = []
        self.recall = []
        self.rng = np.random.default_rng(seed=0)

    @staticmethod
    def run(
        z: NDArray[np.double],
        y: NDArray[np.double],
        y_bin: NDArray[np.bool_],
        y_best: NDArray[np.double],
        algo_labels: list[str],
        hyparams: NDArray[np.double] | None,
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
        pythia = Pythia(z, y, y_bin, y_best, algo_labels, hyparams, opts)
        print("  -> Initializing PYTHIA.")
        pythia.compute_sigma_mu(z)
        z = stats.zscore(z, ddof=1)

        ninst, nalgos = y_bin.shape

        kernel_fcn = "gaussian"
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
        print(
            "-------------------------------------------------------------------------",
        )

        params = pythia.check_hyparams(nalgos)
        # print(params)
        print(
            "-------------------------------------------------------------------------",
        )
        if opts.use_grid_search:
            print(" -> PYTHIA is using grid search for hyper-parameter optimization.")
        else:
            print(" -> PYTHIA is using Bayesian optimization for hyper-parameter optimization.")

        if opts.use_weights:
            print(" -> PYTHIA is using cost-sensitive classification.")
            w = np.abs(y - np.nanmean(y))
            w[w == 0] = np.min(w[w != 0])
            w[np.isnan(w)] = np.max(w[~np.isnan(w)])
            w_aux = w
        else:
            print(" -> PYTHIA is not using cost-sensitive classification.")
            w = np.ones((ninst, nalgos), dtype=int)
        print(
            "-------------------------------------------------------------------------",
        )

        print(
            "  -> Using a "
            + str(opts.cv_folds)
            + "-fold stratified cross-validation experiment to evaluate the SVMs.",
        )
        print(
            "-------------------------------------------------------------------------",
        )
        print("  -> Training has started. PYTHIA may take a while to complete...")

        # TODO Section 3: Train SVM model for each algorithm & Evaluate performance.
        overall_start_time = time.time()
        skf = StratifiedKFold(n_splits=opts.cv_folds, shuffle=True, random_state=0)
        for i in range(nalgos):
            algo_start_time = time.time()
            param_space = pythia.generate_params(opts.use_grid_search)

            res = pythia.fitmatsvm(z, y_bin[:, i], w[:, i], skf, kernel_fcn,param_space,opts.use_grid_search)
            pythia.record_perf(index=i, performance=res)
            # Generate output
            if i == nalgos - 1:
                print(
                    f"    -> PYTHIA has trained a model for '{algo_labels[i]}', there are no models left to train.",
                )
            else:
                print(
                    f"    -> PYTHIA has trained a model for '{algo_labels[i]}', there are {nalgos - i - 1} models left to train.",
                )
            print(f"      -> Elapsed time: {time.time() - algo_start_time:.2f}s")

        print(f"Total elapsed time:  {time.time() - overall_start_time:.2f}s")
        print(
            "-------------------------------------------------------------------------",
        )
        print(" -> PYTHIA has completed training the models.")
        pythia.display_avg_perf()

        pythia.determine_selections()

        print(
            "-------------------------------------------------------------------------",
        )

        """Generate a summary of the results."""
        pythia.generate_summary()

        return pythia.get_output()

    @staticmethod
    def fitmatsvm(
        z: NDArray[np.double],
        y_bin: NDArray[np.bool_],
        w: NDArray[np.double],
        skf: StratifiedKFold,  # Actually its an array and the type is dynamic
        k: str,
        param_space: dict| None,
        use_grid_search: bool,
    ) -> SvmRes:
        """Train a SVM model using MATLAB's 'fitcsvm' function."""
        if k == "gaussian":
            k = "rbf"
        elif k == "polynomial":
            k = "poly"
        elif k == "linear":
            k = "linear"
        else:
            raise ValueError(
                f"Unsupported kernel function: {k}. \
                                Supported kernels are 'gaussian', 'polynomial', and 'linear'.",
            )

        svm_model = SVC(
            kernel="rbf",
            random_state=0,
            probability=True,
        )
        if(use_grid_search):
            optimization = GridSearchCV(estimator=svm_model,
                                   cv = skf,
                                   param_grid=param_space,
                                   n_jobs=-1)
        else:
            optimization = BayesSearchCV(
                estimator=svm_model,
                n_iter=30,
                search_spaces=param_space,
                cv=skf,
                verbose=0,
                random_state=0,
            )
        optimization.fit(z, y_bin, sample_weight=w)
        best_svm = optimization.best_estimator_
        c = optimization.best_params_["C"]
        g = optimization.best_params_["gamma"]

        # Predictions on the training set
        y_sub = cross_val_predict(best_svm, z, y_bin, cv=skf, method="predict")
        p_sub = cross_val_predict(best_svm, z, y_bin, cv=skf, method="predict_proba")[:, 1]

        y_hat = best_svm.predict(z)
        p_hat = best_svm.predict_proba(z)[:, 1]

        cm = confusion_matrix(y_bin, y_sub)
        tn, fp, fn, tp = cm.ravel()

        accuracy = accuracy_score(y_bin, y_hat)
        precision = precision_score(y_bin, y_hat)
        recall = recall_score(y_bin, y_hat)

        svm_result = SvmRes(
            svm=best_svm,
            Yhat=y_hat,
            Ysub=y_sub,
            Psub=p_sub,
            Phat=p_hat,
            C=c,
            g=g,
            tn=tn,
            fp=fp,
            fn=fn,
            tp=tp,
            accuracy=accuracy,
            precision=precision,
            recall=recall,
        )
        return svm_result

    def display_avg_perf(self) -> None:
        print(
            " -> The average cross validated precision is: "
            + str(np.round(100 * np.mean(self.precision), 1))
            + "%",
        )

        print(
            " -> The average cross validated accuracy is: "
            + str(np.round(100 * np.mean(self.accuracy), 1))
            + "%",
        )
    def check_hyparams(self, nalgos: int) -> NDArray:
        """Check hyperparameters."""
        if (
            self.hyparams is None
            or self.hyparams.shape != (nalgos, 2)
            or self.hyparams.dtype != np.double
        ):
            print("-> Using pre-calculated hyper-parameters for the SVM.")
            return np.full((nalgos, 2), np.nan)
        print("-> Using pre-calculated hyper-parameters for the SVM.")
        return self.hyparams

    def record_perf(self, index: int, performance: SvmRes) -> None:
        # print(performance)
        """Record performance."""
        self.y_sub[:, [index]] = performance.Ysub.reshape(-1, 1)
        self.pr0sub[:, [index]] = performance.Psub.reshape(-1, 1)
        self.y_hat[:, [index]] = performance.Yhat.reshape(-1, 1)
        self.pr0hat[:, [index]] = performance.Phat.reshape(-1, 1)

        self.box_consnt.append(performance.C)
        self.k_scale.append(performance.g)
        self.cvcmat[index, :] = [
            performance.tn,
            performance.fp,
            performance.fn,
            performance.tp,
        ]
        self.accuracy.append(performance.accuracy)
        self.precision.append(performance.precision)
        self.recall.append(performance.recall)

    def determine_selections(self) -> None:
        """
        Determine the selections based on the predicted labels and precision.

        Returns
        -------
        None

        """
        default = 0
        if self.nalgos > 1:
            # Precision-weighted selection
            precision = np.array(self.precision)
            weighted_Yhat = self.y_hat.T * precision[:, np.newaxis]
            best = np.max(weighted_Yhat, axis=0)
            self.selection0 = np.argmax(weighted_Yhat, axis=0) + 1  # Algorithms indexed from 1
        else:
            best = self.y_hat.T
            self.selection0 = self.y_hat.T
            default = np.argmax(np.mean(self.y_bin, axis=0))
        self.selection1 = self.selection0
        self.selection0[best <= 0] = 0
        self.selection1[best <= 0] = default

    def get_output(
        self,
    ) -> tuple[PythiaDataChanged, PythiaOut]:
        """Generate output."""
        data_changed = PythiaDataChanged()
        pythia_output = PythiaOut(
            mu=self.mu,
            sigma=self.sigma,
            cp=self.cp,
            svm=[],
            cvcmat=self.cvcmat,
            y_sub=self.y_sub,
            y_hat=self.y_hat,
            pr0_sub=self.pr0sub,
            pr0_hat=self.pr0hat,
            box_consnt=self.box_consnt,
            k_scale=self.k_scale,
            precision=self.precision,
            recall=self.recall,
            accuracy=self.accuracy,
            selection0=self.selection0,
            selection1=self.selection1,
            summary=self.summary,
        )
        return (data_changed, pythia_output)

    def compute_sigma_mu(self, z: NDArray[np.double]) -> None:
        """Compute sigma and mu."""
        self.mu = np.mean(z, axis=0)
        self.sigma = np.std(z, ddof=1, axis=0)

    def generate_params(self,use_grid_search:bool) -> dict:
        """Generate parameters."""
        if use_grid_search:
            rng = np.random.default_rng(seed=0)
            maxgrid = 4
            mingrid = -10
            # Number of samples
            nvals = 30

            # Generate Latin Hypercube Samples
            lhs = stats.qmc.LatinHypercube(d=2, seed=rng)
            samples = lhs.random(nvals)
            C  = 2 ** ((maxgrid - mingrid) * samples[:,0] + mingrid)
            gamma = 2 ** ((maxgrid - mingrid) * samples[:,1] + mingrid)

            # Combine the two sets of samples into the parameter grid
            param_grid = {'C': list(C), 'gamma': list(gamma)}
            return param_grid
        return {
                "C": Real(2**-10, 2**4, prior="log-uniform"),
                "gamma": Real(2**-10, 2**4, prior="log-uniform"),
        }
    def generate_summary(self) -> None:
        """Generate a summary of the results."""
        print("  -> PYTHIA is preparing the summary table.")

        sel0 = self.selection0[:, np.newaxis] == np.arange(1, self.nalgos + 1)
        sel1 = self.selection1[:, np.newaxis] == np.arange(1, self.nalgos + 1)

        avgperf = np.round(np.nanmean(self.y, axis=0), 3)
        stdperf = np.round(np.nanstd(self.y, axis=0), 3)

        y_full = self.y.copy()
        y_svms = self.y.copy()

        self.y[~sel0] = np.nan
        y_full[~sel1] = np.nan
        y_svms[~self.y_hat] = np.nan

        pgood = np.mean(np.any(self.y_bin & sel1, axis=1))

        Ybin_flat = self.y_bin.flatten()
        sel0_flat = sel0.flatten()

        # Compute precision
        precisionsel = precision_score(Ybin_flat, sel0_flat)

        # Compute recall
        recallsel = recall_score(Ybin_flat, sel0_flat)

        # Compute confusion matrix
        # cm = confusion_matrix(Ybin_flat, sel0_flat)
        data = {
            "Algorithms": [*self.algo_labels,  "Oracle", "Selector"],
            "Avg_Perf_all_instances": np.round(np.append(avgperf,[np.nanmean(self.y_best), np.nanmean(y_full)]),3),
            "Std_Perf_all_instances": np.round(np.append(stdperf, [np.nanstd(self.y_best), np.nanstd(y_full)]),3
                                               ),
            "Probability_of_good": np.round(np.append(np.nanmean(self.y_bin,axis=0),[1,pgood]),3),
            "Avg_Perf_selected_instances": np.round(
                np.append(np.nanmean(y_svms,axis=0) ,[np.nan, np.nanmean(y_full)]),
            3),
            "Std_Perf_selected_instances": np.round(
                np.append(np.nanstd(y_svms,axis = 0),[np.nan, np.nanstd(y_full)]),
            3),
            "CV_model_accuracy": np.round(
                100 * np.append(self.accuracy,[np.nan, np.nan]),
            3),
            "CV_model_precision": np.round(
                100 * np.append(self.precision, [np.nan, precisionsel]),
            3),
            "CV_model_recall": np.round(
                100 * np.append(self.recall, [np.nan, recallsel]),
            3),
            "BoxConstraint": np.round(
                np.append(self.box_consnt,[np.nan, np.nan]),3),
            "KernelScale": np.round(
                np.append(self.k_scale, [np.nan, np.nan]),3),
        }

        df = pd.DataFrame(data).replace({np.nan: ""})
        self.summary = df
        print("  -> PYTHIA has completed! Performance of the models:")
        print(df)
        df.to_csv("output.csv", mode="w")


if __name__ == "__main__":
    csv_path_z = script_dir / "Z.csv"
    csv_path_y = script_dir / "y.csv"
    csv_path_hyparams = script_dir / "hyparams.csv"
    csv_path_ybin = script_dir / "ybin.csv"
    csv_path_ybest = script_dir / "ybest.csv"
    csv_path_algo = script_dir / "algolabels.csv"

    z_input = pd.read_csv(csv_path_z).values.astype(np.double)
    y_input = pd.read_csv(csv_path_y).values.astype(np.double)
    y_bin_input = pd.read_csv(csv_path_ybin).values.astype(np.bool_)
    y_best_input = pd.read_csv(csv_path_ybest).values.astype(np.double)
    hyparams_input = (
        pd.read_csv(csv_path_hyparams).values.astype(np.double)
        if csv_path_hyparams.exists()
        else None
    )

    algo_input = pd.read_csv(csv_path_algo, header=None).values.flatten().tolist()

    opts = PythiaOptions(
        cv_folds=5,
        use_grid_search=True,
        use_weights=False,
        is_poly_krnl=False,
    )

    data_change, pythia_output = Pythia.run(
        z_input,
        y_input,
        y_bin_input,
        y_best_input,
        algo_input,
        hyparams_input,
        opts,
    )
