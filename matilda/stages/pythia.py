"""PYTHIA: Automated Algorithm Selection.

By training Support Vector Machines (SVMs) to predict the best-performing
algorithm for a given problem instance, using the coordinates of that
instance in a two-dimensional instance space. PYTHIA uses the trained models
generate overall summary of each algorithm performance and
recommend the best algorithm for a new problem instance.

Key steps for PYTHIA:
1. Normalize the instance space.
2. Train SVM models for each algorithm.
3. Evaluate the performance of the SVM models.
4. Generate a summary of the results.

This module is structured around the `PythiaStage` class

Dependencies:
- numpy
- pandas
- scipy
- sklearn
- skopt

Classes:
--------
- PythiaStage: The main class for the Pythia stage.

Functions:
----------
- pythia: The main function for the Pythia stage.
- _fitmatsvm: Train the SVM model with configurable options.
- _display_overall_perf: Output overall performance metrics.
- _compute_znorm: Compute normalized instance space.
- _check_precalcparams: Check pre-calculated hyper-parameters.
- _determine_selections: Determine the selections based on the precision metrics.
- _generate_params: Generate hyperparameters for the SVM models.
- _generate_summary: Generate a summary of the results.

"""

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import NamedTuple

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
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_predict,
)
from sklearn.svm import SVC
from skopt import BayesSearchCV
from skopt.space import Real

from matilda.data.options import ParallelOptions, PythiaOptions
from matilda.stages.stage import Stage

LARGE_NUM_INSTANCE: int = 1000
IF_PARAMS_FILE: int = 2


@dataclass(frozen=True)
class _SvmRes:
    """SVM result class."""

    svm: SVC
    Ysub: NDArray[np.bool_]
    Psub: NDArray[np.double]
    Yhat: NDArray[np.bool_]
    Phat: NDArray[np.double]
    c: float
    g: float


class PythiaInput(NamedTuple):
    """Inputs for the Pythia stage.

    Attributes
    ----------
    z : NDArray[np.double]
        The feature matrix.
    y : NDArray[np.double]
        The performance metrics.
    y_bin : NDArray[np.bool_]
        The binary labels.
    y_best : NDArray[np.double]
        The best performance metrics.
    algo_labels : list[str]
        The algorithm labels.
    pythia_options : PythiaOptions
        The options for the Pythia stage.
    parallel_options: ParallelOptions
        TODO: This.
    """

    z: NDArray[np.double]
    y_raw: NDArray[np.double]
    y_bin: NDArray[np.bool_]
    y_best: NDArray[np.double]
    algo_labels: list[str]
    pythia_options: PythiaOptions
    parallel_options: ParallelOptions


class PythiaOutput(NamedTuple):
    """Outputs from the Pythia stage.

    Attributes
    ----------
    mu : list[float]
    TODO: This.
    sigma : list[float]
    TODO: This.
    w : NDArray[np.double]
    TODO: This.
    cp : StratifiedKFold
    TODO: This.
    svm : list[SVC]
    TODO: This.
    cvcmat : NDArray[np.double]
    TODO: This.
    y_sub : NDArray[np.bool_]
    TODO: This.
    y_hat : NDArray[np.bool_]
    TODO: This.
    pr0_sub : NDArray[np.double]
    TODO: This.
    pr0_hat : NDArray[np.double]
    TODO: This.
    box_consnt : list[float]
    TODO: This.
    k_scale : list[float]
    TODO: This.
    accuracy : list[float]
    TODO: This.
    precision : list[float]
    TODO: This.
    recall : list[float]
    TODO: This.
    selection0 : NDArray[np.int_]
    TODO: This.
    selection1 : NDArray[np.int_]
    TODO: This.
    summary : pd.DataFrame
    TODO: This.
    """

    mu: list[float]
    sigma: list[float]
    w: NDArray[np.double]
    cp: StratifiedKFold
    svm: list[SVC]
    cvcmat: NDArray[np.double]
    y_sub: NDArray[np.bool_]
    y_hat: NDArray[np.bool_]
    pr0_sub: NDArray[np.double]
    pr0_hat: NDArray[np.double]
    box_consnt: list[float]
    k_scale: list[float]
    accuracy: list[float]
    precision: list[float]
    recall: list[float]
    selection0: NDArray[np.int_]
    selection1: NDArray[np.int_]
    pythia_summary: pd.DataFrame


class PythiaStage(Stage[PythiaInput, PythiaOutput]):
    """See file docstring."""

    def __init__(
        self,
        z: NDArray[np.double],
        y_raw: NDArray[np.double],
        y_bin: NDArray[np.bool_],
        y_best: NDArray[np.double],
        algo_labels: list[str],
    ) -> None:
        """See file docstring."""
        super().__init__()
        self.z = z
        self.y = y_raw
        self.y_bin = y_bin
        self.y_best = y_best
        self.algo_labels = algo_labels

    @staticmethod
    def _inputs() -> type[PythiaInput]:
        return PythiaInput

    @staticmethod
    def _outputs() -> type[PythiaOutput]:
        return PythiaOutput

    @staticmethod
    def _run(inputs: PythiaInput) -> PythiaOutput:
        return PythiaStage.pythia(
            inputs.z,
            inputs.y_raw,
            inputs.y_bin,
            inputs.y_best,
            inputs.algo_labels,
            inputs.pythia_options,
            inputs.parallel_options,
        )

    @staticmethod
    def pythia(
        z: NDArray[np.double],
        y: NDArray[np.double],
        y_bin: NDArray[np.bool_],
        y_best: NDArray[np.double],
        algo_labels: list[str],
        opts: PythiaOptions,
        parallel_options: ParallelOptions,
    ) -> PythiaOutput:
        """Run the Pythia stage.

        Parameters
        ----------
        z : NDArray[np.double]
            The feature matrix.
        y : NDArray[np.double]
            The performance metrics.
        y_bin : NDArray[np.bool_]
            The binary labels.
        y_best : NDArray[np.double]
            The best performance metrics.
        algo_labels : list[str]
            The algorithm labels.
        opts : PythiaOptions
            The options for the Pythia stage.
        parallel_options : ParallelOptions
            TODO: This.

        Returns
        -------
        PythiaOutput
            The output of the Pythia stage.
        """
        print(
            "=========================================================================",
        )
        print("-> Summoning PYTHIA to train the prediction models.")
        print(
            "=========================================================================",
        )
        print("  -> Initializing PYTHIA.")
        ninst, nalgos = y_bin.shape

        y_sub = np.zeros(y_bin.shape, dtype=bool)
        y_hat = np.zeros(y_bin.shape, dtype=bool)
        pr0sub = np.zeros(y_bin.shape, dtype=np.double)
        pr0hat = np.zeros(y_bin.shape, dtype=np.double)

        precalcparams = PythiaStage._check_precalcparams(opts.params, nalgos)
        cp = StratifiedKFold(n_splits=opts.cv_folds, shuffle=True, random_state=0)
        svm = []
        cvcmat = np.zeros((nalgos, 4), dtype=int)
        box_consnt = []
        k_scale = []
        accuracy_record = []
        precision_record = []
        recall_record = []

        w = np.ones((z.shape[0], nalgos), dtype=np.double)
        rng = np.random.default_rng(seed=0)
        (mu, sigma, z) = PythiaStage._compute_znorm(z)

        if ninst > LARGE_NUM_INSTANCE and not opts.is_poly_krnl:
            print(
                "  -> For datasets larger than 1K Instances, "
                + "PYTHIA works better with a Polynomial kernel.",
            )
            print(
                "  -> Consider changing the kernel if the results are unsatisfactory.",
            )
            print(
                "-------------------------------------------------------------------------",
            )

        if opts.is_poly_krnl:
            print(" => PYTHIA is using polynomial kernel")
        else:
            print(" => PYTHIA is using gaussian kernel")

        print(
            "-------------------------------------------------------------------------",
        )

        # Section 2: Configure hyperparameter optimization
        if opts.use_grid_search:
            print(" -> PYTHIA is using grid search for hyper-parameter optimization.")
        else:
            print(
                " -> PYTHIA is using Bayesian optimization"
                + " for hyper-parameter optimization.",
            )

        # Cost-sensitive classification
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
        overall_start_time = perf_counter()

        for i in range(nalgos):
            algo_start_time = perf_counter()
            param_space = (
                PythiaStage._generate_params(opts.use_grid_search, rng)
                if precalcparams is None
                else {"C": precalcparams[i][0], "gamma": precalcparams[i][1]}
            )
            res = PythiaStage._fitmatsvm(
                z=z,
                y_bin=y_bin[:, i],
                w=w[:, i].flatten(),
                skf=cp,
                is_poly_kernel=opts.is_poly_krnl,
                param_space=param_space,
                use_grid_search=opts.use_grid_search,
                parallel_options=parallel_options,
            )

            # Record performance metrics
            y_sub[:, [i]] = res.Ysub.reshape(-1, 1)
            pr0sub[:, [i]] = res.Psub.reshape(-1, 1)
            y_hat[:, [i]] = res.Yhat.reshape(-1, 1)
            pr0hat[:, [i]] = res.Phat.reshape(-1, 1)
            box_consnt.append(res.c)
            k_scale.append(res.g)
            svm.append(res.svm)

            cm = confusion_matrix(y_bin[:, i], res.Ysub)
            tn, fp, fn, tp = cm.ravel()

            accuracy = accuracy_score(y_bin[:, i], res.Yhat)
            precision = precision_score(y_bin[:, i], res.Yhat)
            recall = recall_score(y_bin[:, i], res.Yhat)

            cvcmat[i, :] = [tn, fp, fn, tp]
            accuracy_record.append(accuracy)
            precision_record.append(precision)
            recall_record.append(recall)

            if i == nalgos - 1:
                print(
                    f"    -> PYTHIA has trained a model for '{algo_labels[i]}',"
                    + " there are no models left to train.",
                )
            else:
                print(
                    f"    -> PYTHIA has trained a model for '{algo_labels[i]}'"
                    + f",there are {nalgos - i - 1} models left to train.",
                )
            print(f"      -> Elapsed time: {perf_counter() - algo_start_time:.2f}s")

        print(f"Total elapsed time:  {perf_counter() - overall_start_time:.2f}s")
        print(
            "-------------------------------------------------------------------------",
        )
        print(" -> PYTHIA has completed training the models.")
        PythiaStage._display_overall_perf(precision_record, accuracy_record)

        # Select the algorithm with the highest precision
        (selection0, selection1) = PythiaStage._determine_selections(
            nalgos,
            precision_record,
            y_hat,
            y_bin,
        )

        print(
            "-------------------------------------------------------------------------",
        )

        # Section4: Generate summary of the results
        summary = PythiaStage._generate_summary(
            nalgos,
            algo_labels,
            y,
            y_hat,
            y_bin,
            y_best,
            selection0,
            selection1,
            accuracy_record,
            precision_record,
            recall_record,
            box_consnt,
            k_scale,
        )

        return PythiaOutput(
            mu,
            sigma,
            w,
            cp,
            svm,
            cvcmat,
            y_sub,
            y_hat,
            pr0sub,
            pr0hat,
            box_consnt,
            k_scale,
            accuracy_record,
            precision_record,
            recall_record,
            selection0,
            selection1,
            summary,
        )

    @staticmethod
    def _fitmatsvm(
        z: NDArray[np.double],
        y_bin: NDArray[np.bool_],
        w: NDArray[np.double],
        skf: StratifiedKFold,
        is_poly_kernel: bool,
        param_space: dict[str, list[float]] | None,
        use_grid_search: bool,
        parallel_options: ParallelOptions,
    ) -> _SvmRes:
        """Train a SVM model based on configuration.

        Parameters
        ----------
        z : NDArray[np.double]
            The instance space.
        y_bin : NDArray[np.bool_]
            The binary labels.
        w : NDArray[np.double]
            The sample weights.
        skf : StratifiedKFold
            The stratified k-fold cross-validation object.
        is_poly_kernel : bool
            Whether to use a polynomial kernel.
        param_space : dict | None
            The hyperparameters for the SVM model.
        use_grid_search : bool
            Whether to use grid search for hyperparameter optimization.
        parallel_options : ParallelOptions
            TODO: This.

        Returns
        -------
        _SvmRes
        The SVM result object.
        """
        kernel = "poly" if is_poly_kernel else "rbf"
        svm_model = SVC(
            kernel=kernel,
            random_state=0,
            probability=True,
            degree=2,
            coef0=1,
        )
        if use_grid_search:
            # optimization = GridSearchCV(
            #     estimator=svm_model,
            #     cv=skf,
            #     param_grid=param_space,
            #     n_jobs=parallel_options.n_cores,
            # )
            optimization = RandomizedSearchCV(
                estimator=svm_model,
                n_iter=30,
                param_distributions=param_space,
                cv=skf,
                verbose=0,
                random_state=0,
                n_jobs=parallel_options.n_cores,
            )
        else:
            optimization = BayesSearchCV(
                estimator=svm_model,
                n_iter=30,
                search_spaces=param_space,
                cv=skf,
                verbose=0,
                random_state=0,
                n_jobs=parallel_options.n_cores,
            )
        optimization.fit(z, y_bin, sample_weight=w)
        best_svm = optimization.best_estimator_
        c = optimization.best_params_["C"]
        g = optimization.best_params_["gamma"]

        # Perform cross-validated predictions using the best SVM model
        y_sub = cross_val_predict(best_svm, z, y_bin, cv=skf, method="predict")
        p_sub = cross_val_predict(best_svm, z, y_bin, cv=skf, method="predict_proba")[
            :,
            1,
        ]
        # Predict the labels and probabilities for the entire dataset
        y_hat = best_svm.predict(z)
        p_hat = best_svm.predict_proba(z)[:, 1]

        return _SvmRes(
            svm=best_svm,
            Yhat=y_hat,
            Ysub=y_sub,
            Psub=p_sub,
            Phat=p_hat,
            c=c,
            g=g,
        )

    @staticmethod
    def _display_overall_perf(precision: list[float], accuracy: list[float]) -> None:
        """Calculate overall performance.

        Parameters
        ----------
        precision : list[float]
            The precision metrics.
        accuracy : list[float]
            The accuracy metrics.

        Returns
        -------
        None
        """
        print(
            " -> The average cross validated precision is: "
            + str(np.round(100 * np.mean(precision), 1))
            + "%",
        )

        print(
            " -> The average cross validated accuracy is: "
            + str(np.round(100 * np.mean(accuracy), 1))
            + "%",
        )

    @staticmethod
    def _compute_znorm(
        z: NDArray[np.double],
    ) -> tuple[list[float], list[float], NDArray[np.double]]:
        """Compute mormalized z, standard deviations and mean.

        Parameters
        ----------
        z : NDArray[np.double]
            The feature coordinates.

        Returns
        -------
        tuple[list[float], list[float], NDArray[np.double]]
        The mean, standard deviation and normalized feature coordinates.
        """
        z = stats.zscore(z, ddof=1)
        mu = np.mean(z, axis=0)
        sigma = np.std(z, ddof=1, axis=0)
        return (mu, sigma, z)

    @staticmethod
    def _check_precalcparams(
        params: NDArray[np.double] | None,
        nalgos: int,
    ) -> NDArray[np.double] | None:
        """Check pre-calculated hyper-parameters.

        Parameters
        ----------
        params : NDArray | None
            The pre-calculated hyper-parameters.
            nalgos : int
            The number of algorithms.
        nalgos : int
            The number of algorithms.

        Returns
        -------
        NDArray[np.double] | None
        The pre-calculated hyper-parameters or None.
        """
        if params is None:
            return None
        # Check if the shape of hyper-parameters is correct
        if params.shape != (nalgos, 2):
            print("-> Error: Incorrect number of hyper-parameters.")
            print("Hyper-parameters will be auto-generated.")
            return None
        print("-> Using pre-calculated hyper-parameters for the SVM.")
        return params

    @staticmethod
    def _determine_selections(
        nalgos: int,
        precision: list[float],
        y_hat: NDArray[np.bool_],
        y_bin: NDArray[np.bool_],
    ) -> tuple[NDArray[np.int_], NDArray[np.int_]]:
        """Determine the selections based on the predicted labels and precision.

        Parameters
        ----------
        nalgos : int
            The number of algorithms.
        precision : list[float]
            The precision metrics.
        y_hat : NDArray[np.bool_]
            The predicted labels.
        y_bin : NDArray[np.bool_]
            The binary labels.
        """
        # Stores the index of the column with the highest mean value.
        # Index starts from 0
        default = np.argmax(np.mean(y_bin, axis=0))
        """Selects the best-performing algorithm for each instance using
        precision-weighted predictions. If no algorithm is selected (i.e., all
        scores are non-positive), it defaults to the algorithm with the best
        average performance
        """
        if nalgos > 1:
            # Boardcast corresponding col of y_hat with precision
            precision_array = np.array(precision)
            weighted_yhat = y_hat * precision_array[np.newaxis, :]
            # Find the maximum value for each row in weighted_yhat
            best = np.max(weighted_yhat, axis=1)
            # Get the index of the maximum value in each row
            selection0 = np.argmax(weighted_yhat, axis=1)
        else:
            best = y_hat
            selection0 = y_hat.astype(np.int_)

        selection1 = np.copy(selection0)
        selection0[best <= 0] = 0
        selection1[best <= 0] = default
        return (selection0, selection1)

    @staticmethod
    def _generate_params(
        use_grid_search: bool,
        rng: np.random.Generator,
    ) -> dict[str, list[float]]:
        """Generate hyperparameters for the SVM models.

        Parameters
        ----------
        use_grid_search : bool
            Whether to use grid search for hyperparameter optimization.
        rng : np.random.Generator
            The random number generator.
        """
        if use_grid_search:
            maxgrid = 4
            mingrid = -10
            # Number of samples
            nvals = 30

            # Generate Latin Hypercube Samples
            lhs = stats.qmc.LatinHypercube(d=2, seed=rng)
            samples = lhs.random(nvals)
            c = 2 ** ((maxgrid - mingrid) * samples[:, 0] + mingrid)
            gamma = 2 ** ((maxgrid - mingrid) * samples[:, 1] + mingrid)
            print(c)
            print(gamma)
            # Combine the two sets of samples into the parameter grid
            return {"C": list(c), "gamma": list(gamma)}
        return {
            "C": Real(2**-10, 2**4, prior="log-uniform"),
            "gamma": Real(2**-10, 2**4, prior="log-uniform"),
        }

    @staticmethod
    def _generate_summary(
        nalgos: int,
        algo_labels: list[str],
        y: NDArray[np.double],
        y_hat: NDArray[np.bool_],
        y_bin: NDArray[np.bool_],
        y_best: NDArray[np.double],
        selection0: NDArray[np.int_],
        selection1: NDArray[np.int_],
        precision: list[float],
        accuracy: list[float],
        recall: list[float],
        box_consnt: list[float],
        k_scale: list[float],
    ) -> pd.DataFrame:
        """Generate a summary of the results.

        Parameters
        ----------
        nalgos : int
            The number of algorithms.
        algo_labels : list[str]
            The algorithm labels.
        y : NDArray[np.double]
            The performance metrics.
        y_hat : NDArray[np.bool_]
            The predicted labels.
        y_bin : NDArray[np.bool_]
            The binary labels.
        y_best : NDArray[np.double]
            The best performance metrics.
        selection0 : NDArray[np.integer]
            The selected algorithms.
        selection1 : NDArray[np.integer]
            Backup selected algorithm.
        precision : list[float]
            The precision metrics.
        accuracy : list[float]
            The accuracy metrics.
        recall : list[float]
            The recall metrics.
        box_consnt : list[float]
            The box constraints.
        k_scale : list[float]
            The kernel scales.
        """
        print("  -> PYTHIA is preparing the summary table.")

        # Obtain the corresponding selection matrix for the two selections.
        sel0 = selection0[:, np.newaxis] == np.arange(1, nalgos + 1)
        sel1 = selection1[:, np.newaxis] == np.arange(1, nalgos + 1)

        # Compute the average performance of the selected algorithms
        avgperf = np.round(np.nanmean(y, axis=0), 3)
        stdperf = np.round(np.nanstd(y, axis=0), 3)

        """This variable stores the full performance of the algorithms,
        but filtered based on selection1"""
        y_full = y.copy()

        # This variable stores the performance of the selected algorithms
        y_svms = y.copy()

        y[~sel0] = np.nan
        y_full[~sel1] = np.nan
        y_svms[~y_hat] = np.nan

        # Compute the probability of "good"
        pgood = np.mean(np.any(np.logical_and(y_bin, sel1), axis=1))

        ybin_flat = y_bin.flatten()
        sel0_flat = sel0.flatten()

        # Compute the precision of selected algorithms
        precisionsel = precision_score(ybin_flat, sel0_flat)

        # Compute the recall of selected algorithms
        recallsel = recall_score(ybin_flat, sel0_flat)

        # Prepare the data for the summary table
        data = {
            "Algorithms": [*algo_labels, "Oracle", "Selector"],
            "Avg_Perf_all_instances": np.round(
                np.append(avgperf, [np.nanmean(y_best), np.nanmean(y_full)]),
                3,
            ),
            "Std_Perf_all_instances": np.round(
                np.append(stdperf, [np.nanstd(y_best), np.nanstd(y_full)]),
                3,
            ),
            "Probability_of_good": np.round(
                np.append(np.nanmean(y_bin, axis=0), [1, pgood]),
                3,
            ),
            "Avg_Perf_selected_instances": np.round(
                np.append(
                    np.nanmean(y_svms, axis=0),
                    np.array([np.nan, np.nanmean(y_full)]),
                ),
                3,
            ),
            "Std_Perf_selected_instances": np.round(
                np.append(
                    np.nanstd(y_svms, axis=0),
                    np.array([np.nan, np.nanstd(y_full)]),
                ),
                3,
            ),
            "CV_model_accuracy": np.round(
                100 * np.append(accuracy, [np.nan, np.nan]),
                3,
            ),
            "CV_model_precision": np.round(
                100 * np.append(precision, [np.nan, precisionsel]),
                3,
            ),
            "CV_model_recall": np.round(
                100 * np.append(recall, [np.nan, recallsel]),
                3,
            ),
            "BoxConstraint": np.round(np.append(box_consnt, [np.nan, np.nan]), 3),
            "KernelScale": np.round(np.append(k_scale, [np.nan, np.nan]), 3),
        }

        df = pd.DataFrame(data).replace({np.nan: ""})
        print("  -> PYTHIA has completed! Performance of the models:")
        print(df)
        return df
if __name__ == "__main__":
    script_dir = Path(__file__).parent.parent.parent / "tests"
    print(script_dir)
    output_dir = script_dir / "test_data/pythia/output"

    csv_path_z_input = script_dir / "test_data/pythia/input/Z.csv"
    csv_path_y_input = script_dir / "test_data/pythia/input/y.csv"
    csv_path_algo_input = script_dir / "test_data/pythia/input/algolabels.csv"
    csv_path_y_best_input = script_dir / "test_data/pythia/input/ybest.csv"
    csv_path_y_bin_input = script_dir / "test_data/pythia/input/ybin.csv"
    z = np.genfromtxt(csv_path_z_input, delimiter=",")
    y = np.genfromtxt(csv_path_y_input, delimiter=",")
    algo = pd.read_csv(csv_path_algo_input, header=None).squeeze().tolist()
    y_best = np.genfromtxt(csv_path_y_best_input, delimiter=",")
    y_bin = np.genfromtxt(csv_path_y_bin_input, delimiter=",")
    opt = PythiaOptions(
    cv_folds=5,
    is_poly_krnl=False,
    use_weights=False,
    use_grid_search=True,
    params=None,
    )
    PythiaStage.pythia(z, y, y_bin, y_best, algo, opt, ParallelOptions.default())
