"""PYTHIA function for algorithm selection and performance evaluation using SVM."""

import json
import sys
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
    """SVM result class."""

    def __init__(
        self,
        svm: SVC,
        ysub: NDArray[np.bool_],
        psub: NDArray[np.double],
        yhat: NDArray[np.bool_],
        phat: NDArray[np.double],
        c: float,
        g: float,
        tn: int,
        fp: int,
        fn: int,
        tp: int,
        accuracy: float,
        precision: float,
        recall: float,
    ) -> None:
        """Initialize the SVM result object."""
        self.svm = svm
        self.Ysub = ysub
        self.Psub = psub
        self.Yhat = yhat
        self.Phat = phat
        self.C = c
        self.g = g
        self.tn = tn
        self.fp = fp
        self.fn = fn
        self.tp = tp
        self.accuracy = accuracy
        self.precision = precision
        self.recall = recall

LARGE_NUM_INSTANCE:int = 1000
IF_PARAMS_FILE:int = 2
class Pythia:
    """See file docstring."""

    mu: list[float]
    sigma: list[float]
    cvcmat: NDArray[np.int32]
    y_sub: NDArray[np.bool_]
    y_hat: NDArray[np.bool_]
    pr0hat: NDArray[np.double]
    pr0sub: NDArray[np.double]
    box_consnt: list[float]
    k_scale: list[float]
    precision: list[float]
    recall: list[float]
    accuracy: list[float]
    selection0: NDArray[np.int32]
    selection1: NDArray[np.int32]
    cp: StratifiedKFold
    svm: SVC
    summary: pd.DataFrame

    def __init__(
        self,
        z: NDArray[np.double],
        y: NDArray[np.double],
        y_bin: NDArray[np.bool_],
        y_best: NDArray[np.double],
        algo_labels: list[str],
        opts: PythiaOptions,
    ) -> None:
        """
        Initialize the Pythia object.

        Args:
        ----
            z (NDArray[np.double]): Feature matrix.
            y (NDArray[np.double]): Target performance matrix.
            y_bin (NDArray[np.bool_]): Binary success/failure matrix.
            y_best (NDArray[np.double]): Best performance vector.
            algo_labels (list[str]): List of algorithm labels.
            opts (PythiaOptions): Dictionary containing options for the pythia.

        Returns:
        -------
            None

        """
        self.z = z
        self.y = y
        self.y_bin = y_bin
        self.y_best = y_best
        self.algo_labels = algo_labels
        self.opts = opts
        self.nalgos = len(algo_labels)

        self.y_sub = np.zeros((self.z.shape[0], self.nalgos), dtype=bool)
        self.y_hat = np.zeros((self.z.shape[0], self.nalgos), dtype=bool)
        self.pr0sub = np.zeros((self.z.shape[0], self.nalgos), dtype=np.double)
        self.pr0hat = np.zeros((self.z.shape[0], self.nalgos), dtype=np.double)

        self.preparams = self.check_precalcparams()
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
        opts: PythiaOptions,
    ) -> tuple[PythiaDataChanged, PythiaOut]:
        """
        Algorithm selection and performance evaluation using SVM.

        Args:
        ----
            z(NDArray[np.double]): Feature matrix (instances x features).
            y(NDArray[np.double]): Target variable vector (not used directly in this
                function, but part
            y_bin: Binary matrix indicating success/failure of
                algorithms.
            y_best: Vector containing the best performance of each
                instance.
            algo_labels: List of algorithm labels.
            opts: Dictionary of options.
        of the interface).

        Return:
        ------
        A tuple containing the processed feature matrix and
        PythiaOut

        """
        pythia = Pythia(z, y, y_bin, y_best, algo_labels, opts)

        print("  -> Initializing PYTHIA.")
        pythia.compute_sigma_mu(z)
        z = stats.zscore(z, ddof=1)

        ninst, nalgos = y_bin.shape

        if ninst > LARGE_NUM_INSTANCE and not opts.is_poly_krnl:
            print(
                "  -> For datasets larger than 1K Instances, " +
                "PYTHIA works better with a Polynomial kernel.",
            )
            print(
                "  -> Consider changing the kernel if the results are unsatisfactory.",
            )
            print(
                "-------------------------------------------------------------------------",
            )
        print(" => PYTHIA is using gaussian kernel")
        print(
            "-------------------------------------------------------------------------",
        )

        if opts.use_grid_search:
            print(" -> PYTHIA is using grid search for hyper-parameter optimization.")
        else:
            print(" -> PYTHIA is using Bayesian optimization"+
                  " for hyper-parameter optimization.")

        if opts.use_weights:
            print(" -> PYTHIA is using cost-sensitive classification.")
            w = np.abs(y - np.nanmean(y))
            w[w == 0] = np.min(w[w != 0])
            w[np.isnan(w)] = np.max(w[~np.isnan(w)])
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

        # Section 3: Train SVM model for each algorithm & Evaluate performance.
        overall_start_time = time.time()
        skf = StratifiedKFold(n_splits=opts.cv_folds, shuffle=True, random_state=0)
        for i in range(nalgos):
            algo_start_time = time.time()
            param_space = pythia.get_params(i)
            res = pythia.fitmatsvm(z, y_bin[:, i], w[:, i], skf,
                                   opts.is_poly_krnl,param_space,
                                   opts.use_grid_search)
            pythia.record_perf(index=i, performance=res)
            # Generate output
            if i == nalgos - 1:
                print(
                    f"    -> PYTHIA has trained a model for '{algo_labels[i]}',"
                    +" there are no models left to train.",
                )
            else:
                print(
                    f"    -> PYTHIA has trained a model for '{algo_labels[i]}'"+
                    f",there are {nalgos - i - 1} models left to train.",
                )
            print(f"      -> Elapsed time: {time.time() - algo_start_time:.2f}s")

        print(f"Total elapsed time:  {time.time() - overall_start_time:.2f}s")
        print(
            "--------------------------------------------------"+
            "-----------------------",
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
        skf: StratifiedKFold,
        is_poly_kernel: bool,
        param_space: dict| None,
        use_grid_search: bool,
    ) -> SvmRes:
        """Train a SVM model using MATLAB's 'fitcsvm' function."""
        kernel = "poly" if is_poly_kernel else "rbf"
        svm_model = SVC(
            kernel=kernel,
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
        p_sub = cross_val_predict(best_svm, z, y_bin, cv=skf,
                                  method="predict_proba")[:, 1]

        y_hat = best_svm.predict(z)
        p_hat = best_svm.predict_proba(z)[:, 1]

        cm = confusion_matrix(y_bin, y_sub)
        tn, fp, fn, tp = cm.ravel()

        accuracy = accuracy_score(y_bin, y_hat)
        precision = precision_score(y_bin, y_hat)
        recall = recall_score(y_bin, y_hat)

        return SvmRes(
            svm=best_svm,
            yhat=y_hat,
            ysub=y_sub,
            psub=p_sub,
            phat=p_hat,
            c=c,
            g=g,
            tn=tn,
            fp=fp,
            fn=fn,
            tp=tp,
            accuracy=accuracy,
            precision=precision,
            recall=recall,
        )

    def display_avg_perf(self) -> None:
        """Calculate overall performance."""
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

    def get_params(self,index:int) -> dict:
        """Check hyperparameters."""
        if self.preparams is not None:
            return self.preparams[index]
        return self.generate_params(self.opts.use_grid_search)
    def record_perf(self, index: int, performance: SvmRes) -> None:
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
            weighted_yhat = self.y_hat.T * precision[:, np.newaxis]
            best = np.max(weighted_yhat, axis=0)
            self.selection0 = np.argmax(weighted_yhat, axis=0) + 1
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

    def check_precalcparams(self) -> list| None:
        """Check pre-calculated hyper-parameters."""
        if len(sys.argv) == IF_PARAMS_FILE:
            try:
                with Path.open(sys.argv[1]) as file:
                    data = json.load(file)
            except json.JSONDecodeError as e:
                print(f"Error: Failed to decode JSON. {e}")
                return None
            if not isinstance(data, dict) or "C" not in data or "gamma" not in data:
                print("Invalid format as parameter file.")
                return None
            for key in ["C", "gamma"]:
                if not (isinstance(data[key], list) and
                        len(data[key]) == self.nalgos and
                        all(isinstance(i, int | float) for i in data[key])):
                    print(f"Error: length of {key} must match to number of algorithms.")
                    return None
            print("-> Using pre-calculated hyper-parameters for the SVM.")
            c_list = data.get("C", [])
            gamma_list = data.get("gamma", [])
            return [{"C": c, "gamma": g} for c, g in zip(c_list, gamma_list)]

        return None

    def generate_params(self,use_grid_search:bool) -> dict:
        """Generate parameters."""
        if use_grid_search:
            maxgrid = 4
            mingrid = -10
            # Number of samples
            nvals = 30

            # Generate Latin Hypercube Samples
            lhs = stats.qmc.LatinHypercube(d=2, seed=self.rng)
            samples = lhs.random(nvals)
            c  = 2 ** ((maxgrid - mingrid) * samples[:,0] + mingrid)
            gamma = 2 ** ((maxgrid - mingrid) * samples[:,1] + mingrid)

            # Combine the two sets of samples into the parameter grid
            return {"C": list(c), "gamma": list(gamma)}
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

        ybin_flat = self.y_bin.flatten()
        sel0_flat = sel0.flatten()

        # Compute precision
        precisionsel = precision_score(ybin_flat, sel0_flat)

        # Compute recall
        recallsel = recall_score(ybin_flat, sel0_flat)

        data = {
            "Algorithms": [*self.algo_labels,  "Oracle", "Selector"],
            "Avg_Perf_all_instances": np.round(
                np.append(avgperf,[np.nanmean(self.y_best),
                                   np.nanmean(y_full)]),3),
            "Std_Perf_all_instances": np.round(
                np.append(stdperf, [np.nanstd(self.y_best),
                                    np.nanstd(y_full)]),3),
            "Probability_of_good": np.round(
                np.append(np.nanmean(self.y_bin,axis=0),[1,pgood]),3),
            "Avg_Perf_selected_instances": np.round(
                np.append(np.nanmean(y_svms,axis=0) ,
                          [np.nan, np.nanmean(y_full)]),
            3),
            "Std_Perf_selected_instances": np.round(
                np.append(np.nanstd(y_svms,axis = 0),
                          [np.nan, np.nanstd(y_full)]),
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
        opts,
    )
