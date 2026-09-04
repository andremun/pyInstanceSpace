# SPDX-License-Identifier: LicenseRef-PolyForm-Noncommercial-1.0.0
# Copyright (c) 2024-2026 Mario Andrés Muñoz
"""SIFTED Stage: Feature Selection and Optimisation.

This module implements the SIFTED stage, which focuses on feature selection,
clustering, and optimization to enhance data analysis outcomes. The SIFTED
algorithm performs several key steps:

1. Conduct feature correlation analysis to identify the most relevant features
   based on their correlation with performance metrics.
2. Cluster features to group similar ones, further reducing dimensionality
   and improving selection efficiency.
3. Optimize feature combinations using genetic algorithms (GA) or brute-force
   search, aiming to maximize a model's performance.
4. Filter instances using a filtering method if the corresponding flag is raised

The SIFTED stage is encapsulated in the `SiftedStage` class, which manages
the entire feature selection process. It includes utility methods for
calculating correlations, evaluating clusters, and selecting optimal feature
combinations based on performance metrics.

Dependencies:
- numpy
- pandas
- pygad
- scipy
- scikit-learn

Classes
-------
SiftedStage :
    The primary class that implements the SIFTED stage, providing methods for
    feature selection, clustering, and optimization through various algorithms.

Functions
---------
sifted(x, y, y_bin, x_raw, y_raw, beta, num_good_algos, y_best, p,
       inst_labels, s, feat_labels, opts, opts_selvars,
       data_dense, parallel_options):
    The main function to execute the SIFTED stage for processing feature selection
    and performance optimization using provided datasets and configuration options.
"""

from typing import NamedTuple

import numpy as np
import pandas as pd
import pygad
from loguru import logger
from numpy.typing import NDArray
from scipy.stats import pearsonr
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier

from instancespace.data.model import DataDense
from instancespace.data.options import (
    GeneralOptions,
    ParallelOptions,
    PilotOptions,
    SelvarsOptions,
    SiftedOptions,
)
from instancespace.stages.pilot import PilotStage
from instancespace.stages.stage import Stage
from instancespace.utils.filter import do_filter


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


class SiftedInput(NamedTuple):
    """Inputs for the Sifted stage.

    Attributes
    ----------
    x : NDArray[np.double]
        Feature matrix to be processed (instances x features).
    y : NDArray[np.double]
        Algorithm performance matrix (instances x algorithms).
    y_bin : NDArray[np.bool_]
        Binary matrix indicating good algorithm performance per instance and algorithm.
    x_raw : NDArray[np.double]
        Unprocessed feature matrix (instances x features).
    y_raw : NDArray[np.double]
        Unprocessed performance matrix (instances x algorithms).
    beta : NDArray[np.bool_]
        Binary array indicating selected good features (features x 1).
    num_good_algos : NDArray[np.double]
        Array indicating the number of good algorithms for each feature.
    y_best : NDArray[np.double]
        Best observed algorithm performance per instance.
    p : NDArray[np.int_]
        Correlation p-values associated with features (features x 1).
    inst_labels : pd.Series
        Labels identifying instances in the dataset.
    feat_labels : list[str]
        List of labels for each feature.
    s : pd.Series, optional
        Selection series used during the processing stage, if any.
    sifted_options : SiftedOptions
        Configuration options for the Sifted stage.
    selvars_options : SelvarsOptions
        Configuration options for variable selection.
    data_dense : DataDense, optional
        Dense data representation, if applicable.
    parallel_options : ParallelOptions
        Configuration options for parallel processing.
    general_options : GeneralOptions
        General options (e.g. the RNG seed), not specific to any one stage.

    """

    x: NDArray[np.double]
    y: NDArray[np.double]
    y_bin: NDArray[np.bool_]
    x_raw: NDArray[np.double]
    y_raw: NDArray[np.double]
    beta: NDArray[np.bool_]
    num_good_algos: NDArray[np.double]
    y_best: NDArray[np.double]
    p: NDArray[np.int_]
    inst_labels: pd.Series  # type: ignore[type-arg]
    feat_labels: list[str]
    s: pd.Series | None  # type: ignore[type-arg]
    sifted_options: SiftedOptions
    selvars_options: SelvarsOptions
    data_dense: DataDense | None
    parallel_options: ParallelOptions
    general_options: GeneralOptions


class SiftedOutput(NamedTuple):
    """Outputs from the Sifted stage.

    Attributes
    ----------
    x : NDArray[np.double]
        Processed feature matrix (instances x features).
    y : NDArray[np.double]
        Processed algorithm performance matrix (instances x algorithms).
    y_bin : NDArray[np.bool_]
        Binary matrix indicating good algorithm performance for processed instances.
    x_raw : NDArray[np.double]
        Unprocessed feature matrix (instances x features).
    y_raw : NDArray[np.double]
        Unprocessed performance matrix (instances x algorithms).
    beta : NDArray[np.bool_]
        Binary array indicating selected good features.
    num_good_algos : NDArray[np.double]
        Number of good algorithms per feature after processing.
    y_best : NDArray[np.double]
        Best observed algorithm performance per instance after processing.
    p : NDArray[np.int_]
        Correlation p-values associated with features.
    inst_labels : pd.Series
        Labels for each instance in the dataset.
    s : pd.Series, optional
        Selection series used during processing, if available.
    feat_labels : list[str]
        List of feature labels.
    selvars : NDArray[np.intc]
        Array of indices for selected features after the sifted stage.
    rho : NDArray[np.double], optional
        Array of coefficients or weights for features after the sifted stage.
    pval : NDArray[np.double], optional
        Performance metric values for features after the sifted stage.
    silhouette_scores : list[float], optional
        List of silhouette scores for the sifted features.
    clust : NDArray[np.bool_], optional
        Boolean array indicating selected features (True if selected, False otherwise).

    """

    x: NDArray[np.double]
    y: NDArray[np.double]
    y_bin: NDArray[np.bool_]
    x_raw: NDArray[np.double]
    y_raw: NDArray[np.double]
    beta: NDArray[np.bool_]
    num_good_algos: NDArray[np.double]
    y_best: NDArray[np.double]
    p: NDArray[np.int_]
    inst_labels: pd.Series  # type: ignore[type-arg]
    s: pd.Series | None  # type: ignore[type-arg]
    feat_labels: list[str]
    selvars: NDArray[np.intc]
    rho: NDArray[np.double] | None
    pval: NDArray[np.double] | None
    silhouette_scores: list[float] | None
    clust: NDArray[np.bool_] | None


class SiftedStage(Stage[SiftedInput, SiftedOutput]):
    """Stage in the processing pipeline for feature selection and instance filtering.

    Attributes
    ----------
    MIN_FEAT_REQUIRED : int
        Minimum number of features required for selection (default is 3).
    KFOLDS : int
        Number of folds for cross-validation (default is 5).
    parallel_options : ParallelOptions
        Options for parallel processing.

    Methods
    -------
    __init__(x: NDArray[np.double], y: NDArray[np.double], y_bin: NDArray[np.bool_],
        x_raw: NDArray[np.double], y_raw: NDArray[np.double],
        beta: NDArray[np.bool_], num_good_algos: NDArray[np.double],
        y_best: NDArray[np.double], p: NDArray[np.int_], inst_labels: pd.Series,
        s: pd.Series | None, feat_labels: list[str], opts: SiftedOptions,
        parallel_options: ParallelOptions) -> None:
        Initializes the SiftedStage with provided parameters.

    _inputs() -> type[SiftedInput]:
        Returns the input type for the SIFTED stage.

    _outputs() -> type[SiftedOutput]:
        Returns the output type for the SIFTED stage.

    _compute_correlation(x: NDArray[np.double], y: NDArray[np.double]) ->
        tuple[NDArray[np.double], NDArray[np.double]]:
        Calculate the Pearson correlation coefficients between features and performance
        metrics.

    evaluate_cluster(x_aux: NDArray[np.double], rng: np.random.Generator) ->
        tuple[list[float], NDArray[np.intc]]:
        Evaluates clusters based on silhouette scores for different cluster
        configurations.

    _find_best_combination(x_aux: NDArray[np.double], clust: NDArray[np.bool_],
        selvars: NDArray[np.intc], rng: np.random.Generator)
        -> tuple[NDArray[np.double], NDArray[np.intc]]:
        Finds the best combination of features using a genetic algorithm.

    cost_fcn(instance: pygad.GA, solutions: NDArray[np.intc], _solution_idx: int)
        -> float:
        Fitness function to evaluate the quality of a solution in the genetic algorithm

    select_features_by_clustering(x_aux: NDArray[np.double],
        rng: np.random.Generator) -> tuple[NDArray[np.bool_], NDArray[np.intc]]:
        Selects features based on clustering results.

    select_features_by_performance() -> tuple[NDArray[np.double], NDArray[np.double]
        , NDArray[np.double], NDArray[np.intc]]:
        Selects features based on correlation with performance metrics.

    _sifted(opts_selvars: SelvarsOptions, data_dense: DataDense | None)
        -> SiftedOutput:
        Executes feature selection and optional instance filtering based on
        provided options.

    sifted(x: NDArray[np.double], y: NDArray[np.double], y_bin: NDArray[np.bool_],
        x_raw: NDArray[np.double], y_raw: NDArray[np.double],
        beta: NDArray[np.bool_], num_good_algos: NDArray[np.double],
        y_best: NDArray[np.double], p: NDArray[np.int_], inst_labels: pd.Series,
        s: pd.Series | None, feat_labels: list[str], opts: SiftedOptions,
        opts_selvars: SelvarsOptions, data_dense: DataDense | None,
        parallel_options: ParallelOptions) -> SiftedOutput:
        Executes the core functionality of the Sifted stage for processing feature
        selection.

    _run(inputs: SiftedInput) -> SiftedOutput:
        Executes the SIFTED stage of the pipeline using the provided inputs.
    """

    MIN_FEAT_REQUIRED: int = 3
    KFOLDS: int = 5
    # GA fitness function's internal PILOT call mirrors MATLAB's own hardcoded
    # costfcn constants (core/SIFTED.m) - analytic solve for speed (this runs
    # once per GA candidate, hundreds of times per SIFTED call) and ntries=5,
    # independent of whatever PilotOptions the outer pipeline actually uses.
    _GA_FITNESS_PILOT_ANALYTIC: bool = True
    _GA_FITNESS_PILOT_NTRIES: int = 5
    parallel_options: ParallelOptions

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
        p: NDArray[np.int_],
        inst_labels: pd.Series,  # type: ignore[type-arg]
        s: pd.Series | None,  # type: ignore[type-arg]
        feat_labels: list[str],
        opts: SiftedOptions,
        parallel_options: ParallelOptions,
        general_opts: GeneralOptions,
    ) -> None:
        """Define the input variables for the stage.

        Args
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
        p : NDArray[np.int_]
            Correlation p-values for features.
        inst_labels : pd.Series
            Instance labels for the dataset.
        parallel_options : ParallelOptions
            An instance of ParallelOptions containing parallel processing parameters.
        general_opts : GeneralOptions
            General options (e.g. the RNG seed), not specific to any one stage.
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
        self.s = s
        self.opts = opts
        self.parallel_options = parallel_options
        self.general_opts = general_opts

    def _log(self, msg: str) -> None:
        """Log a top-level, always-shown stage message."""
        logger.info(f"[SIFTED] {msg}")

    def _log_detail(self, msg: str) -> None:
        """Log per-trial/per-iteration detail, only shown when general.verbose."""
        if self.general_opts.verbose:
            logger.debug(f"[SIFTED] {msg}")

    @staticmethod
    def _inputs() -> type[SiftedInput]:
        return SiftedInput

    @staticmethod
    def _outputs() -> type[SiftedOutput]:
        return SiftedOutput

    @staticmethod
    def _run(inputs: SiftedInput) -> SiftedOutput:
        """Execute the SIFTED stage of the pipeline using the provided options and data.

        Parameters
        ----------
        inputs : SiftedInput
            Inputs for the sifted stage.

        Returns
        -------
        SiftedOutput
            Output of the sifted stage.
        """
        return SiftedStage.sifted(
            x=inputs.x,
            y=inputs.y,
            y_bin=inputs.y_bin,
            feat_labels=inputs.feat_labels,
            opts=inputs.sifted_options,
            opts_selvars=inputs.selvars_options,
            data_dense=inputs.data_dense,
            x_raw=inputs.x_raw,
            y_raw=inputs.y_raw,
            beta=inputs.beta,
            num_good_algos=inputs.num_good_algos,
            y_best=inputs.y_best,
            p=inputs.p,
            inst_labels=inputs.inst_labels,
            s=inputs.s,
            parallel_options=inputs.parallel_options,
            general_options=inputs.general_options,
        )

    def sift(
        self,
        opts_selvars: SelvarsOptions,
        data_dense: DataDense | None,
    ) -> SiftedOutput:
        """Public method to access the private _sifted method."""
        return self._sifted(opts_selvars, data_dense)

    @staticmethod
    def sifted(
        x: NDArray[np.double],
        y: NDArray[np.double],
        y_bin: NDArray[np.bool_],
        x_raw: NDArray[np.double],
        y_raw: NDArray[np.double],
        beta: NDArray[np.bool_],
        num_good_algos: NDArray[np.double],
        y_best: NDArray[np.double],
        p: NDArray[np.int_],
        inst_labels: pd.Series,  # type: ignore[type-arg]
        s: pd.Series | None,  # type: ignore[type-arg]
        feat_labels: list[str],
        opts: SiftedOptions,
        opts_selvars: SelvarsOptions,
        data_dense: DataDense | None,
        parallel_options: ParallelOptions,
        general_options: GeneralOptions,
    ) -> SiftedOutput:
        """Core Sifted stage to process feature selection and performance.

        Parameters
        ----------
        x : NDArray[np.double]
            Feature matrix (instances x features).
        y : NDArray[np.double]
            Performance matrix of algorithms (instances x algorithms).
        y_bin : NDArray[np.bool_]
            Binary matrix indicating instances with good algorithm performance.
        x_raw : NDArray[np.double]
            Unprocessed feature matrix.
        y_raw : NDArray[np.double]
            Unprocessed performance matrix.
        beta : NDArray[np.bool_]
            Binary array for selecting good features.
        num_good_algos : NDArray[np.double]
            Number of good algorithms per feature.
        y_best : NDArray[np.double]
            Best algorithm performance for each instance.
        p : NDArray[np.int_]
            Array of correlation p-values for features.
        inst_labels : pd.Series
            Labels for each instance in the dataset.
        s : pd.Series | None
            Series for optional selection during processing.
        feat_labels : list[str]
            List of feature labels.
        opts : SiftedOptions
            Configuration options for the Sifted stage.
        opts_selvars : SelvarsOptions
            Options for selecting variables within the stage.
        data_dense : DataDense | None
            Dense data representation, if available.
        parallel_options : ParallelOptions
            Options for parallel processing.
        general_options : GeneralOptions
            General options (e.g. the RNG seed), not specific to any one stage.

        Returns
        -------
        SiftedOutput
            Output of the Sifted stage, including selected feature and algorithm
            indices,correlation coefficients, performance metrics, silhouette scores,
            and feature selection flags.

        Notes
        -----
        This static method initialises a `SiftedStage` instance and calls its `_sifted`
        method, which uses the provided options and data to execute the Sifted stage
        pipeline and generate results.
        """
        sifted = SiftedStage(
            x,
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
            feat_labels,
            opts,
            parallel_options,
            general_options,
        )

        return sifted.sift(
            opts_selvars=opts_selvars,
            data_dense=data_dense,
        )

    def _sifted(
        self,
        opts_selvars: SelvarsOptions,
        data_dense: DataDense | None,
    ) -> SiftedOutput:
        """Execute feature selection and optional filtering for instance selection.

        This method first selects features based on correlation with performance metrics
        , then applies clustering for further feature selection. If the selected
        features exceed the minimum required, a genetic algorithm is run for optimal
        selection. When specific flags are enabled in `opts_selvars`, the method also
        filters instances based on density criteria.

        Parameters
        ----------
        opts_selvars : SelvarsOptions
            Configuration options for selecting variables, including density flag and
            type.
        data_dense : DataDense | None
            Dense data representation used for instance selection if filtering is
            enabled.

        Returns
        -------
        SiftedOutput
            Results of the Sifted stage, including selected features, correlation
            coefficients,performance metrics, and optionally filtered instances.

        Notes
        -----
        - Raises `NotEnoughFeatureError` if the number of features is insufficient for
            selection.
        - Performs initial feature selection based on correlation with algorithm
            performance.
        - Executes additional clustering for further selection if the number of features
            meets minimum requirements.
        - When the `density_flag` is active in `opts_selvars` and `data_dense` is
            provided, filters instances based on density and minimum distance
            constraints before generating the output.
        """
        x = self.x
        opts = self.opts

        nfeats = x.shape[1]
        rng = np.random.default_rng(seed=self.general_opts.seed)

        if not opts.flag:
            self._log("-> Feature selection is disabled. Using all features.")
            return self._finalize_output(
                x,
                np.arange(nfeats, dtype=np.intc),
                None,
                None,
                None,
                None,
                opts_selvars,
                data_dense,
                apply_density=False,
            )

        if nfeats <= 1:
            raise NotEnoughFeatureError(
                "-> There is only 1 feature. Stopping space construction.",
            )

        if nfeats <= SiftedStage.MIN_FEAT_REQUIRED:
            self._log(
                "-> There are 3 or less features to do selection. "
                "Skipping feature selection.",
            )
            return self._finalize_output(
                x,
                np.arange(nfeats, dtype=np.intc),
                None,
                None,
                None,
                None,
                opts_selvars,
                data_dense,
            )

        self._log("-> Selecting features based on correlation with performance.")

        x_aux, rho, pval, selvars = self.select_features_by_performance()

        nfeats = x_aux.shape[1]

        if nfeats <= 1:
            raise NotEnoughFeatureError(
                "-> There is only 1 feature. Stopping space construction.",
            )

        if nfeats <= SiftedStage.MIN_FEAT_REQUIRED:
            self._log(
                "-> There are 3 or less features to do selection. "
                "Skipping correlation clustering selection.",
            )
            return self._finalize_output(
                x_aux,
                selvars,
                rho,
                pval,
                None,
                None,
                opts_selvars,
                data_dense,
            )

        if nfeats <= opts.k:
            self._log(
                "-> There are less features than clusters. "
                "Skipping correlation clustering selection.",
            )
            return self._finalize_output(
                x_aux,
                selvars,
                rho,
                pval,
                None,
                None,
                opts_selvars,
                data_dense,
            )

        self._log("-> Selecting features based on correlation clustering.")

        silhouette_scores, _ = self.evaluate_cluster(x_aux, rng)

        clust, _ = self.select_features_by_clustering(x_aux, rng)

        x_aux, selvars = self._find_best_combination(x_aux, clust, selvars, rng)

        return self._finalize_output(
            x_aux,
            selvars,
            rho,
            pval,
            silhouette_scores,
            clust,
            opts_selvars,
            data_dense,
        )

    def _finalize_output(
        self,
        x: NDArray[np.double],
        selvars: NDArray[np.intc],
        rho: NDArray[np.double] | None,
        pval: NDArray[np.double] | None,
        silhouette_scores: list[float] | None,
        clust: NDArray[np.bool_] | None,
        opts_selvars: SelvarsOptions,
        data_dense: DataDense | None,
        *,
        apply_density: bool = True,
    ) -> SiftedOutput:
        """Build a SIFTED result and apply its optional post-selection filter."""
        by_density = apply_density and opts_selvars.density_flag
        self._log_detail(f"Bydensity value is : {by_density}")

        if by_density and data_dense is not None:
            subset_index, _, _, _ = do_filter(
                data_dense.x[:, selvars],
                data_dense.y,
                data_dense.y_bin,
                opts_selvars.selvars_type,
                opts_selvars.min_distance,
            )
            subset_index = ~subset_index
            dense_s = data_dense.s[subset_index] if data_dense.s is not None else None
            return SiftedOutput(
                data_dense.x[subset_index][:, selvars],
                data_dense.y[subset_index][:],
                data_dense.y_bin[subset_index][:],
                data_dense.x_raw[subset_index][:],
                data_dense.y_raw[subset_index][:],
                data_dense.beta[subset_index],
                data_dense.num_good_algos[subset_index],
                data_dense.y_best[subset_index][:],
                data_dense.p[subset_index],
                data_dense.inst_labels[subset_index],
                dense_s,
                self.feat_labels[selvars].tolist(),
                selvars,
                rho,
                pval,
                silhouette_scores,
                clust,
            )

        return SiftedOutput(
            x,
            self.y,
            self.y_bin,
            self.x_raw,
            self.y_raw,
            self.beta,
            self.num_good_algos,
            self.y_best,
            self.p,
            self.inst_labels,
            self.s,
            self.feat_labels[selvars].tolist(),
            selvars,
            rho,
            pval,
            silhouette_scores,
            clust,
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
        insignificant_pval = pval > self.opts.pval

        # Filter out insignificant correlations and take absolute values of
        # correlations. Not aliased to `rho` itself (unlike the previous
        # `filtered_rho = rho` here) so the correlation coefficients returned
        # in SiftedOutput.rho stay the genuine, unmodified values.
        filtered_rho = np.abs(rho)
        filtered_rho[np.isnan(rho) | insignificant_pval] = 0

        # Sort by descending absolute correlation - both the sorted values
        # and the feature indices that produced them. The threshold below
        # must compare against the *sorted* value at each rank, not the
        # unsorted, still-signed correlation of whichever feature happens to
        # share that row's original index (row[i, :] are rank-i feature
        # indices, so indexing rho by i directly compared the wrong feature's
        # correlation against opts.rho, and thresholding the signed rho
        # instead of its absolute value dropped legitimate strong negative
        # correlations too).
        sorted_rho = np.sort(filtered_rho, axis=0)[::-1, :]
        row = np.argsort(-filtered_rho, axis=0)

        nfeats = self.x.shape[1]
        selvars = np.zeros(nfeats, dtype=bool)

        # Take the most correlated feature for each algorithm
        selvars[np.unique(row[0, :])] = True

        # Take any feature that has correlation at least equal to opts.rho
        selvars[np.unique(row[sorted_rho >= self.opts.rho])] = True

        # Get indices of selected features
        selvars = np.where(selvars)[0]
        x_aux = self.x[:, selvars]

        return (x_aux, rho, pval, selvars)

    @staticmethod
    def _standardize_for_correlation_distance(
        x_aux: NDArray[np.double],
    ) -> NDArray[np.double]:
        """Z-score each feature (column of x_aux) across instances.

        MATLAB's `kmeans(Xaux', opts.K, 'Distance', 'correlation', ...)`
        (core/SIFTED.m) clusters features using correlation distance, not
        Euclidean - sklearn's `KMeans` has no metric parameter to match this
        directly (#300 issue 7). For two vectors `u`, `v` of length `n`
        z-scored with population (ddof=0) statistics, squared Euclidean
        distance `||u - v||^2 = 2n * (1 - corr(u, v))` - a positive
        monotonic transform of Pearson correlation distance. Clustering
        z-scored feature vectors with ordinary Euclidean k-means therefore
        assigns points to the same nearest centroid a correlation-distance
        k-means would, without a hand-rolled Lloyd's-algorithm
        implementation to independently verify.

        Returns
        -------
        NDArray[np.double]
            One row per feature (shape `(n_features, n_instances)`, i.e.
            already transposed - the orientation `KMeans.fit_predict`
            expects when clustering features rather than instances), each
            row zero-mean/unit-variance across instances. A constant
            feature (zero variance) is left zero-centred rather than
            divided by zero.
        """
        features = x_aux.T
        mean = features.mean(axis=1, keepdims=True)
        std = features.std(axis=1, keepdims=True)
        std[std == 0] = 1.0
        return np.asarray((features - mean) / std, dtype=np.double)

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

        standardized = SiftedStage._standardize_for_correlation_distance(x_aux)
        cluster_labels: NDArray[np.intc] = kmeans.fit_predict(standardized)

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
            The fitness score of the solution: the worst (lowest) per-
            algorithm k-NN classification accuracy. `pygad.GA` always
            *maximizes* whatever this returns, so maximizing the minimum
            per-algorithm accuracy directly is equivalent to MATLAB's
            `fitcknn`/`kfoldLoss`-based `costfcn` (core/SIFTED.m), which
            `ga()` *minimizes* as `max_i(loss_i)` - `-max_i(1 - acc_i) =
            min_i(acc_i) - 1`, the same optimum up to an additive constant
            - without needing a loss concept or a sign flip (#300 issue 5;
            #312, the sign-convention finding this formulation sidesteps
            rather than needing to resolve).
        """
        idx = np.zeros(instance.selfx.shape[1], dtype=bool)

        for i, value in enumerate(solutions):
            aux = np.where(instance.clust[:, i])[0]
            selected = aux[value % aux.size]
            idx[selected] = True

        # Cache fitness by feature-selection bitmask, matching MATLAB's own
        # costfcn (core/SIFTED.m: a persistent containers.Map keyed the same
        # way) - the GA re-visits the same combination many times across
        # generations. Scoped to this ga_instance (fresh dict per SIFTED
        # call, attached alongside selfx/selfy_bin/etc. below), so unlike
        # MATLAB's persistent variable there is no risk of stale results
        # leaking in from an unrelated prior SIFTED call. Under
        # parallel_processing, pygad pickles a separate ga_instance per
        # worker process, so each worker's cache is its own - matching
        # MATLAB's own per-worker persistent-state behaviour, not a gap
        # relative to it.
        cache_key = idx.tobytes()
        if cache_key in instance.cost_cache:
            cached: float = instance.cost_cache[cache_key]
            return cached

        out = PilotStage.pilot(
            instance.selfx[:, idx],
            instance.selfy,
            instance.selffeat_labels[idx].tolist(),
            PilotOptions.default(
                analytic=SiftedStage._GA_FITNESS_PILOT_ANALYTIC,
                n_tries=SiftedStage._GA_FITNESS_PILOT_NTRIES,
            ),
            instance.general_options,
            _do_output=False,
        )

        z = out.z

        # dims + 1 neighbours (D+1, generalised from 2D), matching MATLAB's
        # kneighbours = dims + 1 (core/SIFTED.m).
        kneighbours = instance.dims + 1
        # Track the worst (minimum) per-algorithm accuracy directly, since
        # pygad maximizes whatever this function returns - maximizing the
        # worst-case accuracy is the same search as MATLAB's ga() minimizing
        # the worst-case loss (see docstring), with no loss/sign-flip
        # arithmetic needed at all.
        y = np.inf
        for i in range(instance.selfy.shape[1]):
            knn = KNeighborsClassifier(n_neighbors=kneighbours)
            # Classification accuracy (#300 issue 5), not neg_mean_squared_error -
            # y_bin is a good/bad class label, not a regression target, matching
            # MATLAB's fitcknn/kfoldLoss classification loss.
            scores: NDArray[np.double] = cross_val_score(
                knn,
                z,
                instance.selfy_bin[:, i].astype(int),
                cv=instance.cv_partition,
                scoring="accuracy",
            )
            y = min(y, scores.mean())

        instance.cost_cache[cache_key] = y
        return y

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
            Matrix with selected features after optimization
        decoded_selvars : NDArray[np.intc]
            Indices of the selected features.
        """
        cv_partition = KFold(
            n_splits=SiftedStage.KFOLDS,
            shuffle=True,
            random_state=rng.integers(1000),
        )

        ga_instance = pygad.GA(
            fitness_func=SiftedStage.cost_fcn,
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
            parallel_processing=(
                [
                    "process",
                    self.parallel_options.n_cores,
                ]
                if self.parallel_options.flag
                else None
            ),
        )

        ga_instance.selfx = x_aux
        ga_instance.selfy = self.y
        ga_instance.selfy_bin = self.y_bin
        ga_instance.cv_partition = cv_partition
        ga_instance.clust = clust
        ga_instance.selffeat_labels = self.feat_labels[selvars]
        ga_instance.general_options = self.general_opts
        ga_instance.dims = self.opts.dims
        ga_instance.cost_cache = {}

        ga_instance.run()

        best_solution, best_solution_fitness, _ = ga_instance.best_solution()

        self._log(f"Cost value of the GA algorithm is:  {best_solution_fitness}")

        # Decode the chromosome
        decoder = np.zeros(x_aux.shape[1], dtype=bool)
        for i in range(self.opts.k):
            aux = np.where(clust[:, i])[0]
            selected = aux[int(best_solution[i]) % aux.size]
            decoder[selected] = True

        decoded_selvars = np.array(selvars)[decoder]
        selected_x = self.x[:, decoded_selvars]

        self._log(
            f"-> Keeping {selected_x.shape[1]} out of {x_aux.shape[1]}"
            " features (clustering).",
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
        min_clusters = 3
        max_clusters = x_aux.shape[1]

        silhouette_scores: list[float] = []
        labels: dict[int, NDArray[np.intc]] = {}

        # MATLAB's KList is 3:nfeats (inclusive of nfeats), but sklearn's
        # silhouette_score hard-errors when n_clusters == n_samples (nfeats
        # here, since features are the points being clustered) instead of
        # tolerating the singleton-cluster boundary the way MATLAB's
        # evalclusters does. Matching that upper bound exactly would require
        # a from-scratch silhouette implementation; kept exclusive for now.
        # Clustering itself now uses correlation distance too (#300 issue 7),
        # via the same standardize-then-Euclidean-k-means equivalence
        # `select_features_by_clustering` uses - previously this used plain
        # Euclidean k-means while silhouette_score below (unaffected by this
        # change) already used metric="correlation", an inconsistency this
        # closes.
        standardized = SiftedStage._standardize_for_correlation_distance(x_aux)
        for n in range(min_clusters, max_clusters):
            kmeans = KMeans(
                n_clusters=n,
                n_init="auto",
                random_state=rng.integers(1000),
            )
            cluster_labels = kmeans.fit_predict(standardized)
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
            self._log_detail(
                f"    Suggested k value {max_k_silhoulette} with silhoulette score of"
                f" {silhouette_scores[max_k_silhoulette_index]:.4f}",
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
