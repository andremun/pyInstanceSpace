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
from matilda.data.options import ParallelOptions, PilotOptions, SiftedOptions
from matilda.stages.pilot import Pilot


class NotEnoughFeatureError(Exception):
    """Raised when there is not enough feature to continue feature selection."""

    def __init__(self, msg: str) -> None:
        """Initialise the NotEnoughFeatureError.

        Parameters
        ----------
        msg : str
            error message to be displayed.
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
    par_opts: ParallelOptions

    def __init__(
        self,
        x: NDArray[np.double],
        y: NDArray[np.double],
        y_bin: NDArray[np.bool_],
        feat_labels: list[str],
        opts: SiftedOptions,
        par_opts: ParallelOptions,
    ) -> None:
        """Initialize the Sifted stage.

        Parameters
        ----------
        x : NDArray[np.double]
            Feature matrix (instances x features) to process.
        y : NDArray[np.double]
            Algorithm matrix (instances x algorithm performances).
        y_bin : NDArray[np.bool_]
            Binary labels for algorithm performance.
        feat_labels : list[str]
            List of feature labels.
        opts : SiftedOptions
            Sifted options used for processing.
        par_opts : ParallelOptions
            An instance of ParallelOptions containing parallel processing parameters.
        """
        self.x = x
        self.y = y
        self.y_bin = y_bin
        self.feat_labels = np.array(feat_labels)
        self.opts = opts
        self.par_opts = par_opts

    @staticmethod
    def run(
        x: NDArray[np.double],
        y: NDArray[np.double],
        y_bin: NDArray[np.bool_],
        feat_labels: list[str],
        opts: SiftedOptions,
        par_opts: ParallelOptions,
    ) -> tuple[SiftedDataChanged, SiftedOut]:
        """Process data matrices and options to produce a sifted dataset.

        Parameters
        ----------
        x : NDArray[np.double]
            Feature matrix (instances x features).
        y : NDArray[np.double]
            Performance matrix (instances x algorithms).
        y_bin : NDArray[np.bool_]
            Binary performance matrix.
        feat_labels : list[str]
            List of feature labels.
        opts : SiftedOptions
            An instance of SiftedOptions containing processing parameters.
        par_opts : ParallelOptions
            An instance of ParallelOptions containing parallel processing parameters.

        Returns
        -------
        data_changed : SiftedDataChanged
            Processed feature matrix after feature selection.
        output : SiftedOut
            Output data from the Sifted stage including selected features and
            performance metrics.
        """
        sifted = Sifted(
            x=x,
            y=y,
            y_bin=y_bin,
            feat_labels=feat_labels,
            opts=opts,
            par_opts=par_opts,
        )

        nfeats = x.shape[1]
        idx = np.arange(nfeats)
        rng = np.random.default_rng(seed=0)

        if nfeats <= 1:
            raise NotEnoughFeatureError(
                "-> There is only 1 feature. Stopping space construction.",
            )

        if nfeats <= Sifted.MIN_FEAT_REQUIRED:
            print(
                "-> There are 3 or less features to do selection.",
                "Skipping feature selection.",
            )
            selvars = np.arange(nfeats)
            return sifted.generate_output(x=x, selvars=selvars, idx=idx)

        print("-> Selecting features based on correlation with performance.")

        x_aux, rho, pval, selvars = sifted.select_features_by_performance()

        nfeats = x_aux.shape[1]

        if nfeats <= 1:
            raise NotEnoughFeatureError(
                "-> There is only 1 feature. Stopping space construction.",
            )

        if nfeats <= Sifted.MIN_FEAT_REQUIRED:
            print(
                "-> There are 3 or less features to do selection.",
                "Skipping correlation clustering selection.",
            )
            return sifted.generate_output(
                x=x_aux,
                selvars=selvars,
                idx=idx,
                rho=rho,
                pval=pval,
            )

        if nfeats <= opts.k:
            print(
                "-> There are less features than clusters.",
                "Skipping correlation clustering selection.",
            )
            return sifted.generate_output(
                x=x_aux,
                selvars=selvars,
                idx=idx,
                rho=rho,
                pval=pval,
            )

        print("-> Selecting features based on correlation clustering.")

        silhouette_scores, _ = sifted.evaluate_cluster(x_aux, rng)

        clust, _ = sifted.select_features_by_clustering(x_aux, rng)

        x_aux, selvars = sifted.find_best_combination(x_aux, clust, selvars, rng)

        return sifted.generate_output(
            x=x_aux,
            selvars=selvars,
            idx=idx,
            rho=rho,
            pval=pval,
            silhouette_scores=silhouette_scores,
            clust=clust,
        )

    def select_features_by_performance(
        self,
    ) -> tuple[
        NDArray[np.double],
        NDArray[np.double],
        NDArray[np.double],
        NDArray[np.intc],
    ]:
        """Select features based on correlation with performance.

        Returns
        -------
        x_aux : NDArray[np.double]
            Filtered feature matrix after selection.
        rho : NDArray[np.double]
            Pearson correlation coefficients between features and performance.
        pval : NDArray[np.double]
            p-values for the correlation coefficients.
        selvars : NDArray[np.intc]
            Indices of the selected features.
        """
        rho, pval = self.compute_correlation(self.x, self.y)

        # Create a boolean mask where calculated pval exceeds threshold
        insignificant_pval = pval > Sifted.PVAL_THRESHOLD

        # Filter out insignificant correlations and take absolute values of correlations
        filtered_rho = rho
        filtered_rho[np.isnan(rho) | insignificant_pval] = 0
        filtered_rho = np.abs(filtered_rho)

        # Sort the correlations in descending order
        row = np.argsort(-filtered_rho, axis=0)
        # sorted_rho = np.take_along_axis(rho, row, axis=0)

        nfeats = self.x.shape[1]
        selvars = np.zeros(nfeats, dtype=bool)

        # Take the most correlated feature for each algorithm
        selvars[np.unique(row[0, :])] = True

        # Take any feature that has correlation at least equal to opts.rho
        for i in range(nfeats):
            selvars[np.unique(row[i, rho[i, :] >= self.opts.rho])] = True

        # Get indices of selected features
        selvars = np.where(selvars)[0]
        x_aux = self.x[:, selvars]

        return (x_aux, rho, pval, selvars)

    def select_features_by_clustering(
        self,
        x_aux: NDArray[np.double],
        rng: np.random.Generator,
    ) -> tuple[NDArray[np.bool_], NDArray[np.intc]]:
        """Select features based on clustering.

        Parameters
        ----------
        x_aux : NDArray[np.double]
            Auxiliary feature matrix used for clustering.
        rng : np.random.Generator
            Random number generator for reproducibility.

        Returns
        -------
        clust : NDArray[np.bool_]
            Boolean matrix where each column represents a cluster.
        cluster_labels : NDArray[np.intc]
            Labels for the clusters.
        """
        kmeans = KMeans(
            n_clusters=self.opts.k,
            max_iter=self.opts.max_iter,
            n_init=self.opts.replicates,
            random_state=rng.integers(1000),
        )

        cluster_labels: NDArray[np.intc] = kmeans.fit_predict(x_aux.T)

        # Create a boolean matrix where each column represents a cluster
        clust = np.zeros((x_aux.shape[1], self.opts.k), dtype=bool)
        for i in range(self.opts.k):
            clust[:, i] = cluster_labels == i

        return clust, cluster_labels

    @staticmethod
    def cost_fcn(
        instance: pygad.GA,
        solutions: NDArray[np.intc],
        _solution_idx: int,
    ) -> float:
        """Fitness function to evaluate the quality of solution in GA.

        Parameters
        ----------
        instance : pygad.GA
            The instance of the genetic algorithm.
        solutions : NDArray[np.intc]
            The array of integer values representing the solution to be evaluated.
        solution_idx : int
            The index of the solution being evaluated.

        Returns
        -------
        float
            The fitness score of the solution, representing the negative mean
            squared error of the k-NN classification.
        """
        idx = np.zeros(instance.selfx.shape[1], dtype=bool)

        for i, value in enumerate(solutions):
            aux = np.where(instance.clust[:, i])[0]
            selected = aux[value % aux.size]
            idx[selected] = True

        _, out = Pilot.run(
            instance.selfx[:, idx],
            instance.selfy,
            instance.selffeat_labels[idx].tolist(),
            PilotOptions.default(),
        )

        z = out.z

        y = -np.inf
        for i in range(instance.selfy.shape[1]):
            knn = KNeighborsClassifier(n_neighbors=Sifted.K_NEIGHBORS)
            scores: NDArray[np.double] = cross_val_score(
                knn,
                z,
                instance.selfy_bin[:, i],
                cv=instance.cv_partition,
                scoring="neg_mean_squared_error",
            )
            y = max(y, -scores.mean())

        return y

    def find_best_combination(
        self,
        x_aux: NDArray[np.double],
        clust: NDArray[np.bool_],
        selvars: NDArray[np.intc],
        rng: np.random.Generator,
    ) -> tuple[NDArray[np.double], NDArray[np.intc]]:
        """Find the best combination of features, using genetic algorithm.

        Parameters
        ----------
        x_aux : NDArray[np.double]
            Auxiliary feature matrix used in the optimization process.
        clust : NDArray[np.bool_]
            Boolean matrix where each column represents a cluster.
        selvars : NDArray[np.intc]
            Selected variables to be processed.
        rng : np.random.Generator
            Random number generator for reproducibility.

        Returns
        -------
        selected_x : NDArray[np.double]
            Matrix with selected features after optimization.
        decoded_selvars : NDArray[np.intc]
            Indices of the selected features.
        """
        cv_partition = KFold(
            n_splits=Sifted.KFOLDS,
            shuffle=True,
            random_state=rng.integers(1000),
        )

        ga_instance = pygad.GA(
            fitness_func=Sifted.cost_fcn,
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
            random_seed=int(rng.integers(1000)),
            init_range_low=1,
            init_range_high=x_aux.shape[1],
            save_solutions=True,
            parallel_processing=["process", self.par_opts.n_cores],
        )

        ga_instance.selfx = self.x
        ga_instance.selfy = self.y
        ga_instance.selfy_bin = self.y_bin
        ga_instance.cv_partition = cv_partition
        ga_instance.clust = clust
        ga_instance.selffeat_labels = self.feat_labels

        ga_instance.run()

        best_solution, best_solution_fitness, _ = ga_instance.best_solution()

        print(f"Cost value of the GA algorithm is:  {best_solution_fitness}")

        # Decode the chromosome
        decoder = np.zeros(x_aux.shape[1], dtype=bool)
        for i in range(self.opts.k):
            aux = np.where(clust[:, i])[0]
            selected = aux[int(best_solution[i]) % aux.size]
            decoder[selected] = True

        decoded_selvars = np.array(selvars)[decoder]
        selected_x = self.x[:, decoded_selvars]

        print(
            f"-> Keeping {selected_x.shape[1]} out of {x_aux.shape[1]}",
            "features (clustering).",
        )

        return selected_x, decoded_selvars

    def evaluate_cluster(
        self,
        x_aux: NDArray[np.double],
        rng: np.random.Generator,
    ) -> tuple[list[float], NDArray[np.intc]]:
        """Evaluate cluster based on silhouette scores.

        Parameters
        ----------
        x_aux : NDArray[np.double]
            Auxiliary feature matrix used for clustering.
        rng : np.random.Generator
            Random number generator for reproducibility.

        Returns
        -------
        silhouette_scores : list[float]
            Silhouette scores for different cluster configurations.
        cluster_labels : NDArray[np.intc]
            Labels for the selected cluster configuration.
        """
        min_clusters = 2
        max_clusters = x_aux.shape[1]

        silhouette_scores: list[float] = []
        labels: dict[int, NDArray[np.intc]] = {}

        for n in range(min_clusters, max_clusters):
            kmeans = KMeans(
                n_clusters=n,
                n_init="auto",
                random_state=rng.integers(1000),
            )
            cluster_labels = kmeans.fit_predict(x_aux.T)
            labels[n] = cluster_labels
            silhouette_scores.append(
                silhouette_score(
                    x_aux.T,
                    cluster_labels,
                    metric="correlation",
                ),
            )

        # suggest k value that has highest silhoulette score if k is not the default
        # value and not the maximum nth cluster
        max_k_silhoulette_index = np.argmax(silhouette_scores)
        max_k_silhoulette = min_clusters + max_k_silhoulette_index

        if max_k_silhoulette not in (self.opts.k, max_clusters):
            print(
                f"    Suggested k value {max_k_silhoulette} with silhoulette score of",
                f"{silhouette_scores[max_k_silhoulette]:.4f}",
            )

        # matlab returning numOfObservation, inspected K value, criterion values,
        # and optimal K, but in python lets do k value first need to deal with, if
        # user choose optimal silhouette value, should change the output
        # check if silhoulette value is in bell shape, meaning increasing then
        # decreasing if max is not last, then can recommend max value.
        return silhouette_scores, labels[self.opts.k]

    def compute_correlation(
        self,
        x: NDArray[np.double],
        y: NDArray[np.double],
    ) -> tuple[NDArray[np.double], NDArray[np.double]]:
        """Calculate the Pearson correlation coefficient for the dataset by row.

        Parameters
        ----------
        x : NDArray[np.double]
            Feature matrix for the correlation calculation.
        y : NDArray[np.double]
            Performance matrix for correlation.

        Returns
        -------
        rho : NDArray[np.double]
            Pearson correlation coefficients between features and performance.
        pval : NDArray[np.double]
            p-values for the correlation coefficients.
        """
        rows = x.shape[1]
        cols = y.shape[1]

        rho = np.zeros((rows, cols))
        pval = np.zeros((rows, cols))

        for i in range(rows):
            for j in range(cols):
                x_col = x[:, i]
                y_col = y[:, j]

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
                    rho[i, j], pval[i, j] = np.nan, np.nan

        return (rho, pval)

    def generate_output(
        self,
        x: NDArray[np.double],
        selvars: NDArray[np.intc],
        idx: NDArray[np.intc],
        rho: NDArray[np.double] | None = None,
        pval: NDArray[np.double] | None = None,
        silhouette_scores: list[float] | None = None,
        clust: NDArray[np.bool_] | None = None,
    ) -> tuple[SiftedDataChanged, SiftedOut]:
        """Generate outputs of Sifted stage.

        Parameters
        ----------
        x : NDArray[np.double]
            Processed feature matrix.
        selvars : NDArray[np.intc]
            Indices of the selected features.
        idx : NDArray[np.intc]
            Indices of all features.
        rho : NDArray[np.double], optional
            Pearson correlation coefficients.
        pval : NDArray[np.double], optional
            p-values for the correlation coefficients.
        silhouette_scores : list of float, optional
            Silhouette scores for clustering.
        clust : NDArray[np.bool_], optional
            Boolean matrix where each column represents a cluster.

        Returns
        -------
        data_changed : SiftedDataChanged
            Data that is changed during the Sifted stage.
        output : SiftedOut
            Output data generated from the Sifted stage, including selected features
            and performance metrics.
        """
        data_changed = SiftedDataChanged(x=x)
        output = SiftedOut(
            selvars=selvars,
            idx=idx[selvars],
            rho=rho,
            pval=pval,
            silhouette_scores=silhouette_scores,
            clust=clust,
        )
        return (data_changed, output)
