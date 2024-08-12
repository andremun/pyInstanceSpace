"""Provides functionality for feature selection and optimization in data analysis.

The SIFTED function performs the following steps:
  - Feature correlation analysis to reduce the dimensionality of the feature space.
  - Clustering of features based on their correlation to identify distinct groups.
  - Optimization using genetic algorithms (GA) or brute-force search to find the best
    feature combination that optimizes a specific cost function, typically related to
    model performance.
"""

from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy.stats import pearsonr
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from matilda.data.model import SiftedDataChanged, SiftedOut
from matilda.data.options import SiftedOptions

script_dir = Path(__file__).parent

class NotEnoughFeatureError(Exception):
    """Raised when there is not enough feature to continue feature selection."""

    def __init__(self, msg: str) -> None:
        """Initialise the NotEnoughFeatureError.

        Args:
            msg (str): error message to be displayed.
        """
        super().__init__(msg)


class Sifted:
    """See file docstring."""

    MIN_FEAT_REQUIRED: int = 3
    PVAL_THRESHOLD = 0.05

    x: NDArray[np.double]
    y: NDArray[np.double]
    y_bin: NDArray[np.bool_]
    opts: SiftedOptions

    rho: NDArray[np.double] | None
    pval: NDArray[np.double] | None
    selvars: NDArray[np.intc] | None
    clust: NDArray[np.bool_] | None
    ooberr: NDArray[np.double] | None

    def __init__(
        self,
        x: NDArray[np.double],
        y: NDArray[np.double],
        y_bin: NDArray[np.bool_],
        opts: SiftedOptions,
    ) -> None:
        """Initialize the Sifted stage.

        Args
        ----
            x (NDarray): The feature matrix (instances x features) to process.
            y (NDArray): The algorithm matrix (instances x algorithm performances).
            y_bin (NDArray): Binary labels for algorithm perfromance from prelim.
            opts (SiftedOptions): Sifted options.
        """
        self.x = x
        self.y = y
        self.y_bin = y_bin
        self.opts = opts

        self.rho = None
        self.pval = None
        self.selvars = None
        self.clust = None
        self.ooberr = None

        self.rng = np.random.default_rng(seed=0)

    @staticmethod
    def run(
        x: NDArray[np.double],
        y: NDArray[np.double],
        y_bin: NDArray[np.bool_],
        opts: SiftedOptions,
    ) -> tuple[SiftedDataChanged, SiftedOut]:
        """Process data matrices and options to produce a sifted dataset.

        Args
            x: The feature matrix (instances x features).
            y: The performance matrix (instances x algorithms).
            y_bin: The binary performance matrix
            opts: An instance of `SiftedOptions` containing processing
                parameters.

        Returns
        -------
            A tuple containing the processed feature matrix and
            SiftedOut
        """
        # TODO: Multiprocessing setup

        sifted = Sifted(x=x, y=y, y_bin=y_bin, opts=opts)

        nfeats = x.shape[1]

        if nfeats <= 1:
            raise NotEnoughFeatureError(
                "-> There is only 1 feature. Stopping space construction.",
            )

        if nfeats <= Sifted.MIN_FEAT_REQUIRED:
            print(
                "-> There are 3 or less features to do selection. \
                Skipping feature selection.",
            )
            sifted.selvars = np.arange(nfeats)
            return sifted.get_output()

        x_aux = sifted.select_features_by_performance()

        nfeats = x_aux.shape[1]

        if nfeats <= 1:
            raise NotEnoughFeatureError(
                "-> There is only 1 feature. Stopping space construction.",
            )

        if nfeats <= Sifted.MIN_FEAT_REQUIRED:
            print(
                "-> There are 3 or less features to do selection. \
                    Skipping correlation clustering selection.",
            )
            sifted.x = x_aux
            return sifted.get_output()

        if nfeats <= opts.k:
            print(
                "-> There are less features than clusters. \
                    Skipping correlation clustering selection.",
            )
            sifted.x =x_aux
            return sifted.get_output()

        sifted.select_features_by_clustering(x_aux)

        np.savetxt(script_dir / "tmp_data/clustering_output/Xaux.csv", x_aux, delimiter=",")

        return sifted.get_output()

    def select_features_by_performance(self) -> NDArray[np.double]:
        """Select features based on correlation with performance.

        Returns
        -------
            NDArray[np.double]: Auxillary matrix of x which contains features selected
                based on correlation with performance.
        """
        print("-> Selecting features based on correlation with performance.")

        self.rho, self.pval = self.compute_correlation()

        # Create a boolean mask where calculated pval exceeds threshold
        insignificant_pval = self.pval > Sifted.PVAL_THRESHOLD

        # Filter out insignificant correlations and take absolute values of correlations
        rho = self.rho
        rho[np.isnan(self.rho) | insignificant_pval] = 0
        rho = np.abs(rho)

        # Sort the correlations in descending order
        row = np.argsort(-rho, axis=0)
        sorted_rho = np.take_along_axis(rho, row, axis=0)

        np.savetxt(script_dir / "tmp_data/clustering_output/rho.csv", self.rho, delimiter=",")
        np.savetxt(script_dir / "tmp_data/clustering_output/rho_sorted.csv", sorted_rho, delimiter=",")

        nfeats = self.x.shape[1]
        selvars = np.zeros(nfeats, dtype=bool)

        # Always take the most correlated feature for each algorithm
        selvars[np.unique(row[0, :])] = True

        # Now take any feature that has correlation at least equal to opts.rho
        for i in range(nfeats):
            selvars[np.unique(row[i, rho[i, :] >= self.opts.rho])] = True

        # Get indices of selected features
        self.selvars = np.where(selvars)[0]

        return self.x[:,self.selvars]

    def select_features_by_clustering(self, x_aux: NDArray[np.double]) -> None:
        """Select features based on clustering.

        Args
        ----
            x_aux (NDArray[np.double]): feature matrix that contains values selected
                based on correlation with performance.
        """
        print("-> Selecting features based on correlation clustering.")
        self.evaluate_cluster(x_aux)

        kmeans = KMeans(
            n_clusters=self.opts.k,
            max_iter=self.opts.max_iter,
            n_init=self.opts.replicates,
            random_state=self.rng.integers(1000),
        )
        cluster_labels = kmeans.fit_predict(x_aux.T)

        # Create a boolean matrix where each column represents a cluster
        self.clust = np.zeros((x_aux.shape[1], self.opts.k), dtype=bool)
        for i in range(self.opts.k):
            self.clust[:, i] = (cluster_labels == i)

    def find_best_combination(self, x_aux: NDArray[np.double]) -> None:
        raise NotImplementedError

    def cost_fcn(
        self,
        comb: NDArray[np.double],  # not sure about the type
        x: NDArray[np.double],
        y_bin: NDArray[np.bool_],
        n_trees: int,
        n_workers: int,
    ) -> NDArray[np.double]:
        """Compute the cost function for a given combination of features or parameters.

        Args
        ----
            comb: Array representing the combination of parameters to
                evaluate.
            x: The feature matrix (instances x features).
            y_bin: The binary performance matrix indicating
                success/failure.
            n_trees: The number of trees to use in the model or method.
            n_workers: The number of parallel workers to use in
                computation.

        Returns
        -------
            An array representing the calculated cost for each
            combination.
        """
        # TODO: rewrite SIFTED logic in python
        raise NotImplementedError

    def evaluate_cluster(self, x_aux: NDArray[np.double]) -> None:
        """Evaluate cluster based on silhouette scores.

        Args
        ----
            x_aux (NDArray[np.double]): feature matrix that contains values selected
                based on correlation with performance.
        """
        min_clusters = 2
        max_clusters = x_aux.shape[1]

        silhouette_scores = {}

        for n in range(min_clusters, max_clusters):
            kmeans = KMeans(
                n_clusters=n,
                n_init="auto",
                random_state=self.rng.integers(1000),
            )
            cluster_labels = kmeans.fit_predict(x_aux.T)
            silhouette_scores[n] = silhouette_score(
                x_aux.T,
                cluster_labels,
                metric="correlation",
            )

        print(silhouette_scores)

    def compute_correlation(self) -> tuple[NDArray[np.double], NDArray[np.double]]:
        """Calculate the Pearson correlation coefficient for the dataset by row.

        Returns:
        -------
            tuple(NDArray[np.double], NDArray[np.double]: rho and p value calculated
                by Pearson correlation.
        """
        rows = self.x.shape[1]
        cols = self.y.shape[1]

        rho = np.zeros((rows, cols))
        pval = np.zeros((rows, cols))

        for i in range(rows):
            for j in range(cols):
                x_col = self.x[:, i]
                y_col = self.y[:, j]

                # Filter out NaN values of pairs
                valid_indices = ~np.isnan(x_col) & ~np.isnan(y_col)

                if np.any(valid_indices):
                    # Compute Pearson correlation for valid pairs
                    rho[i, j], pval[i, j] = pearsonr(
                        x_col[valid_indices],
                        y_col[valid_indices],
                    )
                else:
                    # Set value to Nan if there is no valid pairs
                    rho[i, j], pval[i,j] = np.nan, np.nan

        return (rho, pval)

    def get_output(self) -> tuple[SiftedDataChanged, SiftedOut]:
        """Generate outputs of Sifted stage.

        Returns
        -------
            tuple[SiftedDataChanged, SiftedOut]: SiftedDataChanged contains data that
                is changed during the Sifted stage and SiftedOut contains data generated
                from Sifted stage.
        """
        data_changed = SiftedDataChanged(x=self.x)
        output = SiftedOut(
            rho=self.rho,
            pval=self.pval,
            selvars=self.selvars,
            clust=self.clust,
            ooberr=self.ooberr,
        )
        return (data_changed, output)


if __name__ == "__main__":
    csv_path_x = script_dir / "tmp_data/clustering/0-input_X.csv"
    csv_path_y = script_dir / "tmp_data/clustering/0-input_Y.csv"
    csv_path_ybin = script_dir / "tmp_data/clustering/0-input_Ybin.csv"

    input_x = np.genfromtxt(csv_path_x, delimiter=",")
    input_y = np.genfromtxt(csv_path_y, delimiter=",")
    input_ybin = np.genfromtxt(csv_path_ybin, delimiter=",")

    opts = SiftedOptions.default()

    data_change, sifted_output = Sifted.run(input_x, input_y, input_ybin, opts)
