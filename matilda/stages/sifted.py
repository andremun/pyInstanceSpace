"""Provides functionality for feature selection and optimization in data analysis.

The SIFTED function performs the following steps:
  - Feature correlation analysis to reduce the dimensionality of the feature space.
  - Clustering of features based on their correlation to identify distinct groups.
  - Optimization using genetic algorithms (GA) or brute-force search to find the best
    feature combination that optimizes a specific cost function, typically related to
    model performance.
"""

import numpy as np
import pygad
from numpy.typing import NDArray
from scipy.stats import pearsonr
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier

from matilda.data.model import SiftedDataChanged, SiftedOut
from matilda.data.options import PilotOptions, SiftedOptions
from matilda.stages.pilot import Pilot


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
    PVAL_THRESHOLD: float = 0.05
    KFOLDS: int = 5
    K_NEIGHBORS: int = 3

    x: NDArray[np.double]
    y: NDArray[np.double]
    y_bin: NDArray[np.bool_]
    feat_labels: NDArray[np.str_]
    opts: SiftedOptions

    rho: NDArray[np.double]
    pval: NDArray[np.double]
    selvars: NDArray[np.intc]
    clust: NDArray[np.bool_]
    x_aux: NDArray[np.double]
    cv_partition: KFold
    ga_cache: dict[str, float]

    def __init__(
        self,
        x: NDArray[np.double],
        y: NDArray[np.double],
        y_bin: NDArray[np.bool_],
        feat_labels: list[str],
        opts: SiftedOptions,
    ) -> None:
        """Initialize the Sifted stage.

        Args
        ----
            x (NDarray): The feature matrix (instances x features) to process.
            y (NDArray): The algorithm matrix (instances x algorithm performances).
            y_bin (NDArray): Binary labels for algorithm perfromance from prelim.
            feat_labels(list[str]): List of feature labels.
            opts (SiftedOptions): Sifted options.
        """
        self.x = x
        self.y = y
        self.y_bin = y_bin
        self.feat_labels = np.array(feat_labels)
        self.opts = opts

        self.rng = np.random.default_rng(seed=0)

    @staticmethod
    def run(
        x: NDArray[np.double],
        y: NDArray[np.double],
        y_bin: NDArray[np.bool_],
        feat_labels: list[str],
        opts: SiftedOptions,
    ) -> tuple[SiftedDataChanged, SiftedOut]:
        """Process data matrices and options to produce a sifted dataset.

        Args
        ----
            x: The feature matrix (instances x features).
            y: The performance matrix (instances x algorithms).
            y_bin: The binary performance matrix
            feat_labels: A list of feature labels.
            opts: An instance of `SiftedOptions` containing processing
                parameters.

        Returns
        -------
            A tuple containing the processed feature matrix and
            SiftedOut
        """
        sifted = Sifted(x=x, y=y, y_bin=y_bin, feat_labels=feat_labels, opts=opts)

        nfeats = x.shape[1]

        if nfeats <= 1:
            raise NotEnoughFeatureError(
                "-> There is only 1 feature. Stopping space construction.",
            )

        if nfeats <= Sifted.MIN_FEAT_REQUIRED:
            print(
                "-> There are 3 or less features to do selection.",
                "Skipping feature selection.",
            )
            sifted.selvars = np.arange(nfeats)
            return sifted.get_output()

        sifted.select_features_by_performance()

        nfeats = sifted.x_aux.shape[1]

        if nfeats <= 1:
            raise NotEnoughFeatureError(
                "-> There is only 1 feature. Stopping space construction.",
            )

        if nfeats <= Sifted.MIN_FEAT_REQUIRED:
            print(
                "-> There are 3 or less features to do selection.",
                "Skipping correlation clustering selection.",
            )
            sifted.x = sifted.x_aux
            return sifted.get_output()

        if nfeats <= opts.k:
            print(
                "-> There are less features than clusters.",
                "Skipping correlation clustering selection.",
            )
            sifted.x = sifted.x_aux
            return sifted.get_output()

        sifted.select_features_by_clustering()
        sifted.find_best_combination()

        return sifted.get_output()

    def select_features_by_performance(self) -> NDArray[np.double]:
        """Select features based on correlation with performance."""
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
        # sorted_rho = np.take_along_axis(rho, row, axis=0)

        nfeats = self.x.shape[1]
        selvars = np.zeros(nfeats, dtype=bool)

        # Take the most correlated feature for each algorithm
        selvars[np.unique(row[0, :])] = True

        # Take any feature that has correlation at least equal to opts.rho
        for i in range(nfeats):
            selvars[np.unique(row[i, rho[i, :] >= self.opts.rho])] = True

        # Get indices of selected features
        self.selvars = np.where(selvars)[0]
        self.x_aux = self.x[:, self.selvars]

        return self.x_aux

    def select_features_by_clustering(self) -> NDArray[np.intc]:
        """Select features based on clustering."""
        print("-> Selecting features based on correlation clustering.")
        self.evaluate_cluster()

        kmeans = KMeans(
            n_clusters=self.opts.k,
            max_iter=self.opts.max_iter,
            n_init=self.opts.replicates,
            random_state=self.rng.integers(1000),
        )

        cluster_labels: NDArray[np.intc] = kmeans.fit_predict(self.x_aux.T)

        # Create a boolean matrix where each column represents a cluster
        self.clust = np.zeros((self.x_aux.shape[1], self.opts.k), dtype=bool)
        for i in range(self.opts.k):
            self.clust[:, i] = (cluster_labels == i)

        return cluster_labels

    def find_best_combination(self) -> None:
        """Find the best combination of features, using genetic algorithm."""
        self.cv_partition = KFold(
            n_splits=Sifted.KFOLDS,
            shuffle=True,
            random_state=self.rng.integers(1000),
        )

        ga_instance = pygad.GA(
            fitness_func=self.cost_fcn,
            num_generations=self.opts.num_generations,
            num_parents_mating=self.opts.num_parents_mating,
            sol_per_pop=self.opts.sol_per_pop,
            gene_type=int,
            num_genes=self.opts.k,
            parent_selection_type=self.opts.parent_selection_type,
            K_tournament=self.opts.k_tournament,
            keep_elitism=self.opts.keep_elitism,
            crossover_type=self.opts.crossover_type,
            crossover_probability=self.opts.cross_over_probability,
            mutation_type=self.opts.mutation_type,
            mutation_probability=self.opts.mutation_probability,
            # stop_criteria: saturate_5 or reach_0 (don't know if it is possible both)
            stop_criteria=self.opts.stop_criteria,
            random_seed=0,
            init_range_low=1,
            init_range_high=self.x_aux.shape[1],
            save_solutions=True,
        )

        ga_instance.run()

        best_solution, _, _ = ga_instance.best_solution()

        # Decode the chromosome
        decoder = np.zeros(self.x_aux.shape[1], dtype=bool)
        for i in range(self.opts.k):
            aux = np.where(self.clust[:, i])[0]
            selected = aux[int(best_solution[i]) % aux.size]
            decoder[selected] = True

        self.selvars = np.array(self.selvars)[decoder]
        self.x = self.x[:, self.selvars]

        print(
            f"-> Keeping {self.x.shape[1]} out of {self.x_aux.shape[1]}",
            "features (clustering).",
        )

    def cost_fcn(
        self,
        instance: pygad.GA,  # noqa: ARG002
        solutions: NDArray[np.intc],
        solution_idx: int,  # noqa: ARG002
    ) -> float:
        """Fitness function to evaluate the quality of solution in genetic algorithm.

        Args
        ----
            instance (pygad.GA): The instance of the genetic algorithm.
            solutions (NDArray[np.intc]): The array of integer values representing the
                solution to be evaluated.
            solution_idx (int): The index of the solution being evaluated.

        Returns
        -------
            float: The fitness score of the solution, representing the negative mean
                squared error of the k-NN classification.
        """
        idx = np.zeros(self.x.shape[1], dtype=bool)

        for i, value in enumerate(solutions):
            aux = np.where(self.clust[:, i])[0]
            selected = aux[value % aux.size]
            idx[selected] = True

        _, out = Pilot.run(
            self.x[:, idx],
            self.y,
            self.feat_labels[idx].tolist(),
            PilotOptions.default(),
        )

        z = out.z

        y = -np.inf
        for i in range(self.y.shape[1]):
            knn = KNeighborsClassifier(n_neighbors=Sifted.K_NEIGHBORS)
            scores = cross_val_score(
                knn, z, self.y_bin[:, i],
                cv=self.cv_partition,
                scoring="neg_mean_squared_error",
            )
            y = max(y, -scores.mean())

        return y


    def evaluate_cluster(self) -> NDArray[np.intc]:
        """Evaluate cluster based on silhouette scores.

        Returns
        -------
            dict[int, NDArray]: A dictionary containing the labels of the clusters
        """
        min_clusters = 2
        max_clusters = self.x_aux.shape[1]

        self.silhouette_scores: list[float] = []
        labels: dict[int, NDArray[np.intc]] = {}

        for n in range(min_clusters, max_clusters):
            kmeans = KMeans(
                n_clusters=n,
                n_init="auto",
                random_state=self.rng.integers(1000),
            )
            cluster_labels = kmeans.fit_predict(self.x_aux.T)
            labels[n] = cluster_labels
            self.silhouette_scores.append(
                silhouette_score(
                    self.x_aux.T,
                    cluster_labels,
                    metric="correlation",
                ),
            )

        # suggest k value that has highest silhoulette score if k is not the default
        # value and not the maximum nth cluster
        max_k_silhoulette_index = np.argmax(self.silhouette_scores)
        max_k_silhoulette = min_clusters + max_k_silhoulette_index

        if max_k_silhoulette not in (self.opts.k, max_clusters):
            print(
                f"    Suggested k value {max_k_silhoulette} with silhoulette score of",
                f"{self.silhouette_scores[max_k_silhoulette]:.4f}",
            )

        # matlab returning numOfObservation, inspected K value, criterion values,
        # and optimal K, but in python lets do k value first need to deal with, if
        # user choose optimal silhouette value, should change the output
        # check if silhoulette value is in bell shape, meaning increasing then
        # decreasing if max is not last, then can recommend max value, if last then how?
        return labels[self.opts.k]

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
            idx=self.selvars,
            silhouette_scores=self.silhouette_scores,
            clust=self.clust,
        )
        return (data_changed, output)
