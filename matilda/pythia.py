"""PYTHIA function for algorithm selection and performance evaluation using SVM."""

import numpy as np
from numpy.typing import NDArray

from matilda.data.model import PythiaDataChanged, PythiaOut
from matilda.data.options import PythiaOptions
import numpy as np
import csv
import pandas as pd
from numpy.typing import NDArray
from scipy.stats import zscore,qmc
from pytictoc import TicToc
from scipy import optimize, stats
from sklearn.model_selection import GridSearchCV, StratifiedKFold, KFold, PredefinedSplit
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from pyDOE import lhs

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
        opts: PythiaOptions,  # noqa: ARG004
    ) -> tuple[PythiaDataChanged, PythiaOut]:
        """
        PYTHIA function for algorithm selection and performance evaluation using SVM.

        Args
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
        cv_folds = 5 #TODO get para from opts
        nan_array = np.array([[np.nan, np.nan]])
        mu, sigma = np.mean(z, axis=0), np.std(z, ddof=1, axis=0)  # noqa: E999
        z_norm = stats.zscore(z, ddof=1)
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
            and isinstance(opts.params, (list, np.ndarray))  # noqa: UP038
            and np.array(opts.params).shape == (nalgos, 2)
        )
        params = np.full((nalgos, 2), np.nan)

        if opts.is_poly_krnl:
            kernel_fcn = "polynomial"
        else:
            if ninst > 1000:  # noqa: PLR2004
                print(
                    "  -> For datasets larger than 1K Instances, PYTHIA works better with a Polynomial kernel.",
                )
                print(
                    "  -> Consider changing the kernel if the results are unsatisfactory.",
                )
                print(
                    "-------------------------------------------------------------------------",
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
                    " -> Search on a latin hyper-cube design will be used for parameter hyper-tunning.",
                )
        else:
            print(" -> Using MATLAB's SVM libraries.")

            if precalparams:
                print(" -> Using pre-calculated hyper-parameters for the SVM.")
                params = opts.params
            else:
                print(" -> Bayesian Optimization will be used for parameter hyper-tunning.")
            print(
                "-------------------------------------------------------------------------",
            )

            if opts.use_weights:
                print(" -> PYTHIA is using cost-sensitive classification.")
                w = np.abs(y - np.nanmean(y))
                w[w == 0] = np.min(w[w != 0])
                w[np.isnan(w)] = np.max(w[~np.isnan(w)])
            else:
                print(" -> PYTHIA is not using cost-sensitive classification.")
                w = np.ones((ninst, nalgos))
        print("-------------------------------------------------------------------------")

        print(
            "  -> Using a "
            + str(opts.cv_folds)
            + "-fold stratified cross-validation experiment to evaluate the SVMs.",
        )
        print("-------------------------------------------------------------------------")
        print("  -> Training has started. PYTHIA may take a while to complete...")

        # TODO Section 3: Train SVM model for each algorithm & Evaluate performance.
        # if(opts.use_lib_svm):
        [PythiaDataChanged,PythiaOut] = Pythia.fitlibsvm(z,y_bin,cv_folds,"rbf",params)
        return [PythiaDataChanged,PythiaOut]
        # TODO Section 4: SVM model selection.

        # TODO Section 5: SVM model selection.

        # TODO Section 6: Generate output


    @staticmethod
    def fitlibsvm(
        z: NDArray[np.double],
        y_bin: NDArray[np.double],
        n_folds: int,
        kernel: str,
        params: NDArray[np.double],
    ) -> tuple[PythiaDataChanged,PythiaOut]: # type: ignore
        """Train a SVM model using the LIBSVM library."""
        ninst = z.shape[0]
        maxgrid = 4
        mingrid = -10
        rng = np.random.default_rng(seed=0)

        # Number of samples
        nvals = 30

        # Generate Latin Hypercube Samples
        lhs = stats.qmc.LatinHypercube(d=2, seed=rng)
        samples = lhs.random(nvals)

        # Apply scaling and exponentiation
        paramgrid = 2 ** ((maxgrid - mingrid) * samples + mingrid)

        # Sort rows by the first column, and if equal, by the second column
        paramgrid = paramgrid[np.lexsort((paramgrid[:, 1], paramgrid[:, 0]))]
        print(paramgrid)

        Ybin = np.array(Ybin, dtype=float) + 1
        Ysub = np.zeros((ninst, nvals))
        Psub = np.zeros((ninst, nvals))

        for i in range(nalgos):
            rng = np.random.default_rng(seed=0)
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
            out['cp'][i] = list(skf.split(z_norm, Ybin[:, i]))

            if np.any(np.isnan(params)):
            # Generate Latin Hypercube Samples
                lhs = stats.qmc.LatinHypercube(d=2, seed=rng)
                samples = lhs.random(nvals)
                # Apply scaling and exponentiation
                paramgrid = 2 ** ((maxgrid - mingrid) * samples + mingrid)

                # Sort rows by the first column, and if equal, by the second column
                paramgrid = paramgrid[np.lexsort((paramgrid[:, 1], paramgrid[:, 0]))]
        return tuple[None,None]
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

    def get_out_put(self) -> tuple[PythiaDataChanged,PythiaOut]
    out = {
        'cp': [None] * nalgos,cparams
        'svm': [None] * nalgos,
        'cvcmat': np.zeros((nalgos, 4)),
        'Ysub': np.zeros_like(Ybin, dtype=bool),
        'Yhat': np.zeros_like(Ybin, dtype=bool),
        'Pr0sub': np.zeros_like(Ybin, dtype=float),
        'Pr0hat': np.zeros_like(Ybin, dtype=float),
        'boxcosnt': np.zeros(nalgos),
        'kscale': np.zeros(nalgos),
    }

if __name__ == "__main__":
    # csv_path_x = script_dir / "tmp_data/clustering/0-input_X.csv"
    # csv_path_y = script_dir / "tmp_data/clustering/0-input_Y.csv"
    # csv_path_ybin = script_dir / "tmp_data/clustering/0-input_Ybin.csv"
    # csv_path_feat_labels = script_dir / "tmp_data/clustering/0-input_featlabels.csv"

    # input_x = np.genfromtxt(csv_path_x, delimiter=",")
    # input_y = np.genfromtxt(csv_path_y, delimiter=",")
    # input_ybin = np.genfromtxt(csv_path_ybin, delimiter=",")
    # feat_labels = np.genfromtxt(csv_path_feat_labels, delimiter=",", dtype=str).tolist()

    # opts = SiftedOptions.default()

    # data_change, sifted_output = Sifted.run(input_x, input_y, input_ybin, feat_labels, opts)

    # np.savetxt(script_dir / "tmp_data/clustering_output/sifted_x.csv", data_change.x, delimiter=",")
