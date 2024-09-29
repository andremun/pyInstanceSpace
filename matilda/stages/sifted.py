"""
SIFTED Stage: Feature Selection and Optimization

This class provides functionality for feature selection, clustering, and optimization 
in data analysis using the SIFTED algorithm. The main steps include:

- Feature correlation analysis to identify the most relevant features based on their 
  correlation with performance metrics.
- Clustering of features to group similar ones and further reduce dimensionality.
- Optimization through genetic algorithms (GA) or brute-force search to select the 
  best feature combination, maximizing a model’s performance.

Attributes
----------
MIN_FEAT_REQUIRED : int
    Minimum number of features required for feature selection to proceed.
PVAL_THRESHOLD : float
    Threshold for statistical significance in feature-performance correlations.
KFOLDS : int
    Number of folds for cross-validation during feature selection.
K_NEIGHBORS : int
    Number of neighbors for k-NN used in the fitness function of the genetic algorithm.

Methods
-------
__init__(...)
    Initializes the SIFTED stage with the input feature and performance matrices, 
    binary performance labels, raw data, and other parameters.
_run(...)
    Executes the SIFTED algorithm to perform feature selection and optimization.
sifted(...)
    Main function that orchestrates the SIFTED algorithm for feature selection and 
    clustering, including optimization.
select_features_by_performance(...)
    Selects features based on their correlation with performance metrics.
select_features_by_clustering(...)
    Selects features based on clustering of feature correlations.
find_best_combination(...)
    Uses a genetic algorithm to find the best feature combination that maximizes 
    classification performance.
evaluate_cluster(...)
    Evaluates the clustering process using silhouette scores to suggest optimal 
    cluster numbers.
compute_correlation(...)
    Computes Pearson correlation coefficients between features and performance metrics.
"""

import numpy as np
import pygad
import pandas as pd
from numpy.typing import NDArray
from scipy.stats import pearsonr
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from matilda.stages.prelim_stage import DataDense
from matilda.data.options import PilotOptions, SiftedOptions, SelvarsOptions
from matilda.stages.pilot import Pilot
from matilda.stages.filter import Filter
from matilda.stages.stage import Stage


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

class Sifted(Stage):
    """Class for SIFTED stage"""
    MIN_FEAT_REQUIRED: int = 3
    PVAL_THRESHOLD: float = 0.05
    KFOLDS: int = 5
    K_NEIGHBORS: int = 3
    
    def __init__(
        self,
        x: NDArray[np.double],
        y: NDArray[np.double],
        y_bin: NDArray[np.bool_],
        x_raw: NDArray[np.double],
        y_raw: NDArray[np.double],
        beta: NDArray[np.bool_],
        num_good_algos: NDArray[np.double],
        y_best: NDArray[np.double],
        p: NDArray[np.double],
        inst_labels: pd.Series,
        s: set[str] | None,
        feat_labels: list[str],
        opts: SiftedOptions
    ) -> None:
        """
        Define the input variables for the stage.
        Args
        -----
        None

        Return
        -----
        x : NDArray[np.double]
            Feature matrix to be processed (instances x features).
        y : NDArray[np.double]
            Algorithm performance matrix (instances x algorithms).
        y_bin : NDArray[np.bool_]
            Binary matrix indicating good algorithm performance.
        feat_labels : list[str]
            List of feature labels.
        x_raw : NDArray[np.double]
            Raw feature matrix.
        y_raw : NDArray[np.double]
            Raw performance matrix.
        beta : NDArray[np.bool_]
            Binary selection for good features.
        num_good_algos : NDArray[np.double]
            Number of good algorithms for each feature.
        y_best : NDArray[np.double]
            Best algorithm performance for each instance.
        p : NDArray[np.double]
            Correlation p-values for features.
        inst_labels : pd.Series
            Instance labels for the dataset.
        """
        self.x = x
        self.y = y
        self.y_bin = y_bin
        self.feat_labels = np.array(feat_labels)
        self.x_raw = x_raw
        self.y_raw = y_raw
        self.beta = beta
        self.num_good_algos = num_good_algos
        self.y_best = y_best
        self.p = p
        self.inst_labels = inst_labels
        self.s = np.array(s)
        self.opts = opts


    @staticmethod
    def _inputs() -> list[tuple[str, type]]:
        """Use the method for determining the inputs for sifted.

        Args
        ----
            None

        Returns
        -------
            list[tuple[str, type]]
                List of inputs for the stage
        """
        return [
            ("x", NDArray[np.double]),
            ("y", NDArray[np.double]),
            ("y_bin", NDArray[np.bool_]),
            ("x_raw", NDArray[np.double]),
            ("y_raw", NDArray[np.double]),
            ("beta", NDArray[np.bool_]),
            ("num_good_algos", NDArray[np.double]),
            ("y_best", NDArray[np.double]),
            ("p", NDArray[np.double]),
            ("inst_labels", pd.Series),
            ("s", set[str] | None),
            ("feat_labels", list[str]),
            ("opts", SiftedOptions),
            ("opts_selvars", SelvarsOptions),
            ("data_dense", DataDense),
        ]

    @staticmethod
    def _outputs() -> list[tuple[str, type]]:
        """Use the method for determining the outputs for sifted.

        Args
        ----
            None

        Returns
        -------
            list[tuple[str, type]]
                List of outputs for the stage
        """
        return [
            ("x", NDArray[np.double]),
            ("y", NDArray[np.double]),
            ("y_bin", NDArray[np.bool_]),
            ("x_raw", NDArray[np.double]),
            ("y_raw", NDArray[np.double]),
            ("beta", NDArray[np.bool_]),
            ("num_good_algos", NDArray[np.double]),
            ("y_best", NDArray[np.double]),
            ("p", NDArray[np.double]),
            ("inst_labels", pd.Series),
            ("s", set[str] | None),
            ("feat_labels", list[str]),
            ("selvars", NDArray[np.intc]),
            ("idx", NDArray[np.intc]),
            ("rho", NDArray[np.double] | None),
            ("pval", NDArray[np.double] | None),
            ("silhouette_scores", list[float] | None),
            ("clust", NDArray[np.bool_] | None),
        ]


    def _run(self, opts_selvars: SelvarsOptions, data_dense: DataDense) -> tuple[
        NDArray[np.double],          # x
        NDArray[np.double],          # y
        NDArray[np.bool_],           # y_bin
        NDArray[np.double],          # x_raw
        NDArray[np.double],          # y_raw
        NDArray[np.bool_],           # beta
        NDArray[np.double],          # num_good_algos
        NDArray[np.double],          # y_best
        NDArray[np.double],          # p
        pd.Series,                   # inst_labels
        set[str] | None,             # s
        list[str],                   # feat_labels
        NDArray[np.intc],            # selvars
        NDArray[np.intc],            # idx
        NDArray[np.double] | None,   # rho
        NDArray[np.double] | None,   # pval
        list[float] | None,          # silhouette_scores
        NDArray[np.bool_] | None     # clust
    ]:
        """Execute the sifted stage of the pipeline using the provided options and data.

        Args
        -------
        opts : SiftedOptions
            Options for configuring the Sifted stage.
        opts_selvars : SelvarsOptions
            Selection variables options used for filtering.
        data_dense : Data
            Dense data representation used for processing.

        Return
        -------
        selvars
            NDArray[np.intc]
            Array of selected feature indices.
        idx
            NDArray[np.intc]
            Array of selected algorithm indices.
        rho
            NDArray[np.double] | None
            Coefficients or feature weights after the sifted stage.
        pval
            NDArray[np.double] | None
            Performance metrics after the sifted stage.
        silhouette_scores
            list[float] | None
            List of p-values for the sifted features.
        clust
            NDArray[np.bool_] | None
            Boolean array indicating whether features were selected or not.
        """
        return self.sifted(
            self,
            x = self.x,
            y = self.y,
            y_bin = self.y_bin,
            feat_labels = self.feat_labels,
            opts = self.opts,
            opts_selvars = opts_selvars,
            data_dense = data_dense,
            x_raw = self.x_raw,
            y_raw = self.y_raw,
            beta = self.beta,
            num_good_algos = self.num_good_algos,
            y_best = self.y_best,
            p = self.p,
            inst_labels = self.inst_labels,
            s = self.s
        )

    @staticmethod
    def sifted(
        self,
        x: NDArray[np.double],
        y: NDArray[np.double],
        y_bin: NDArray[np.bool_],
        x_raw: NDArray[np.double],
        y_raw: NDArray[np.double],
        beta: NDArray[np.bool_],
        num_good_algos: NDArray[np.double],
        y_best: NDArray[np.double],
        p: NDArray[np.double],
        inst_labels: pd.Series,
        s: set[str] | None,
        feat_labels: list[str],
        opts: SiftedOptions,
        opts_selvars: SelvarsOptions,
        data_dense: DataDense
    ) -> tuple[
        NDArray[np.double],          # x
        NDArray[np.double],          # y
        NDArray[np.bool_],           # y_bin
        NDArray[np.double],          # x_raw
        NDArray[np.double],          # y_raw
        NDArray[np.bool_],           # beta
        NDArray[np.double],          # num_good_algos
        NDArray[np.double],          # y_best
        NDArray[np.double],          # p
        pd.Series,                   # inst_labels
        set[str] | None,             # s
        list[str],                   # feat_labels
        NDArray[np.intc],            # selvars
        NDArray[np.intc],            # idx
        NDArray[np.double] | None,   # rho
        NDArray[np.double] | None,   # pval
        list[float] | None,          # silhouette_scores
        NDArray[np.bool_] | None     # clust
    ]:
        """See file docstring."""
        nfeats = x.shape[1]
        idx = np.arange(nfeats)
        rng = np.random.default_rng(seed=0)
        
        # Prepare for Filter
        bydensity = (
            opts_selvars != None and
            'density_flag' in opts_selvars.__dict__.keys() and
            opts_selvars.density_flag and
            'min_distance' in opts_selvars.__dict__.keys() and
            isinstance(opts_selvars.min_distance , float) and
            'selvars_type' in opts_selvars.__dict__.keys() and
            isinstance(opts_selvars.selvars_type , str) 
        )

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
            return Sifted.generate_output(x=x, selvars=selvars, idx=idx)

        print("-> Selecting features based on correlation with performance.")

        x_aux, rho, pval, selvars = self._select_features_by_performance()

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
            return [
                x_aux,
                y, 
                y_bin, 
                x_raw,
                y_raw,
                beta,
                num_good_algos,
                y_best,
                p, 
                inst_labels,
                s, 
                [feat_labels[i] for i in selvars],
                selvars,
                idx,
                rho,
                pval,
                None,
                None
            ]

        if nfeats <= opts.k:
            print(
                "-> There are less features than clusters.",
                "Skipping correlation clustering selection.",
            )
            return [
                x_aux,
                y, 
                y_bin, 
                x_raw,
                y_raw,
                beta,
                num_good_algos,
                y_best,
                p, 
                inst_labels,
                s, 
                [feat_labels[i] for i in selvars],
                selvars,
                idx,
                rho,
                pval,
                None,
                None
            ]

        print("-> Selecting features based on correlation clustering.")

        silhouette_scores, _ = self._evaluate_cluster(x_aux, rng)

        clust, _ = self._select_features_by_clustering(x_aux, rng)

        x_aux, selvars = self._find_best_combination(x_aux, clust, selvars, rng)
        
        print(f"Bydensity value is : {bydensity}")
        # Run filter for small experiment 
        if bydensity:
            subset_index, _, _, _ = Filter(
                data_dense.x[:, selvars],
                data_dense.y,
                data_dense.y_bin,
                opts_selvars
            )
            subset_index = ~subset_index
            if data_dense.s != None:
                s = data_dense.s[subset_index]
            return [
                data_dense.x[subset_index][:selvars],
                data_dense.y[subset_index][:],
                data_dense.y_bin[subset_index][:],
                data_dense.x_raw[subset_index][:],
                data_dense.y_raw[subset_index][:],
                data_dense.beta[subset_index],
                data_dense.num_good_algos[subset_index],
                data_dense.y_best[subset_index][:],
                data_dense.p[subset_index],
                data_dense.inst_labels[subset_index],
                s,
                [feat_labels[i] for i in selvars],
                selvars,
                idx,
                rho,
                pval,
                silhouette_scores,
                clust
            ]
        else:
            return [
                x_aux,
                y, 
                y_bin, 
                x_raw,
                y_raw,
                beta,
                num_good_algos,
                y_best,
                p, 
                inst_labels,
                s, 
                [feat_labels[i] for i in selvars],
                selvars,
                idx,
                rho,
                pval,
                silhouette_scores,
                clust
            ]
    
    def _select_features_by_performance(
        self,
    ) -> tuple[
        NDArray[np.double],
        NDArray[np.double],
        NDArray[np.double],
        NDArray[np.intc],
    ]:
        """
        Select features based on correlation with performance.

        Args
        -----
        None

        Return
        -----
        x_aux : NDArray[np.double]
            Filtered feature matrix after selection.
        rho : NDArray[np.double]
            Pearson correlation coefficients between features and performance.
        pval : NDArray[np.double]
            p-values for the correlation coefficients.
        selvars : NDArray[np.intc]
            Indices of the selected features.
        """
        rho, pval = self._compute_correlation(self.x, self.y)

        # Create a boolean mask where calculated pval exceeds threshold
        insignificant_pval = pval > self.PVAL_THRESHOLD

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

    def _select_features_by_clustering(
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

    def _find_best_combination(
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

        def cost_fcn(
            instance: pygad.GA,
            solutions: NDArray[np.intc],
            solution_idx: int,
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
            idx = np.zeros(self.x.shape[1], dtype=bool)

            for i, value in enumerate(solutions):
                aux = np.where(clust[:, i])[0]
                selected = aux[value % aux.size]
                idx[selected] = True

            _, _, _, _, _, z, _, _, _, _, _ = Pilot.pilot(
                self.x[:, idx],
                self.y,
                self.feat_labels[idx].tolist(),
                PilotOptions.default(),
            )

            y = -np.inf
            for i in range(self.y.shape[1]):
                knn = KNeighborsClassifier(n_neighbors=Sifted.K_NEIGHBORS)
                scores: NDArray[np.double] = cross_val_score(
                    knn,
                    z,
                    self.y_bin[:, i],
                    cv=cv_partition,
                    scoring="neg_mean_squared_error",
                )
                y = max(y, -scores.mean())

            return y

        ga_instance = pygad.GA(
            fitness_func=cost_fcn,
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
        )

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

    def _evaluate_cluster(
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

    def _compute_correlation(
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
