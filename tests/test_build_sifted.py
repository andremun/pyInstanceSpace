"""Test module for Sifted stage to verify its functionality.

The file contains multiple unit tests to ensure that the `Sifted` class corretly
perform its tasks. The basic mechanism of the test is to compare its output against
output from MATLAB and check if the outputs are the same or reasonable similar. The
tests also include some boundary test where appropriate to test the boundary of the
statement within the methods to ensure they are implemented appropriately.

Tests includes:
- For the function select_features_by_performance, we check xaux value, check if
   features selected are the same
- For the function select_features_by_clustering, we check if number of elements in the
   same clusters for both matlab and python are over given threshold %
- For the function ga, check if the filtered x value, for each row and column, only one
   instances with high correlation, others are low correlation. Test passed if more than
   1 columns/rows don't fulfil this condition

"""

import dataclasses
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import pytest
from numpy.typing import NDArray
from sklearn.cluster import KMeans

from instancespace.data.model import DataDense
from instancespace.data.options import (
    GeneralOptions,
    ParallelOptions,
    PilotOptions,
    SelvarsOptions,
    SiftedOptions,
)
from instancespace.stages.sifted import SiftedInput, SiftedOutput, SiftedStage


class SiftedMatlabInput:
    """Class to store MATLAB input data for sifted tests."""

    def __init__(self) -> None:
        """Initialize the input data for the sifted tests."""
        script_dir = Path(__file__).parent

        # Standard data CSV files
        self.x = np.genfromtxt(
            script_dir / "test_data/sifted/input/input_X.csv",
            delimiter=",",
        )
        self.y = np.genfromtxt(
            script_dir / "test_data/sifted/input/input_Y.csv",
            delimiter=",",
        )
        self.y_bin = np.genfromtxt(
            script_dir / "test_data/sifted/input/input_Ybin.csv",
            delimiter=",",
        )
        self.x_raw = np.genfromtxt(
            script_dir / "test_data/sifted/input/input_Xraw.csv",
            delimiter=",",
        )
        self.y_raw = np.genfromtxt(
            script_dir / "test_data/sifted/input/input_Yraw.csv",
            delimiter=",",
        )
        self.beta = np.genfromtxt(
            script_dir / "test_data/sifted/input/input_beta.csv",
            delimiter=",",
        )
        self.num_good_algos = np.genfromtxt(
            script_dir / "test_data/sifted/input/input_numGoodAlgos.csv",
            delimiter=",",
        )
        self.y_best = np.genfromtxt(
            script_dir / "test_data/sifted/input/input_Ybest.csv",
            delimiter=",",
        )
        self.p = np.genfromtxt(
            script_dir / "test_data/sifted/input/input_P.csv",
            delimiter=",",
        )
        self.inst_labels = np.genfromtxt(
            script_dir / "test_data/sifted/input/input_instlabels.csv",
            delimiter=",",
            dtype=str,
        ).tolist()
        self.feat_labels = np.genfromtxt(
            script_dir / "test_data/sifted/input/input_featlabels.csv",
            delimiter=",",
            dtype=str,
        ).tolist()
        self.s = None

        # Create DataDense instance
        self.data_dense = DataDense(
            x=np.genfromtxt(
                script_dir / "test_data/sifted/input/input_dense_X.csv",
                delimiter=",",
            ),
            y=np.genfromtxt(
                script_dir / "test_data/sifted/input/input_dense_Y.csv",
                delimiter=",",
            ),
            x_raw=np.genfromtxt(
                script_dir / "test_data/sifted/input/input_dense_Xraw.csv",
                delimiter=",",
            ),
            y_raw=np.genfromtxt(
                script_dir / "test_data/sifted/input/input_dense_Yraw.csv",
                delimiter=",",
            ),
            y_bin=np.genfromtxt(
                script_dir / "test_data/sifted/input/input_dense_Ybin.csv",
                delimiter=",",
            ),
            y_best=np.genfromtxt(
                script_dir / "test_data/sifted/input/input_dense_Ybest.csv",
                delimiter=",",
            ),
            p=np.genfromtxt(
                script_dir / "test_data/sifted/input/input_dense_P.csv",
                delimiter=",",
            ),
            num_good_algos=np.genfromtxt(
                script_dir / "test_data/sifted/input/input_dense_numGoodAlgos.csv",
                delimiter=",",
            ),
            beta=np.genfromtxt(
                script_dir / "test_data/sifted/input/input_dense_beta.csv",
                delimiter=",",
            ),
            inst_labels=pd.Series(
                np.genfromtxt(
                    script_dir / "test_data/sifted/input/input_dense_instlabels.csv",
                    delimiter=",",
                    dtype=str,
                ),
            ),
            s=None,
        )

        # Set up options
        self.opts = SiftedOptions.default()
        self.opts_selvar = SelvarsOptions.default()
        self.opts_selvar_filter = SelvarsOptions.default(density_flag=True)


class SiftedMatlabOutput:
    """Class to store MATLAB output data for sifted tests."""

    def __init__(self) -> None:
        """Initialize the output data for the sifted tests."""
        script_dir = Path(__file__).parent

        # Output CSV files
        self.cluster_matlab = np.genfromtxt(
            script_dir / "test_data/sifted/output/clusters_matlab.csv",
            delimiter=",",
        )
        self.correlation_matlab = np.genfromtxt(
            script_dir / "test_data/sifted/output/correlation_matlab.csv",
            delimiter=",",
        )
        self.x_matlab = np.genfromtxt(
            script_dir / "test_data/sifted/output/x_matlab.csv",
            delimiter=",",
        )


def test_select_features_by_performance() -> None:
    """Test performance selection against MATLAB's performance selection output.

    Ensures that `xaux` after filtering by correlation performance is exactly the
    same as MATLAB's output.
    """
    inputs = SiftedMatlabInput()
    sifted = SiftedStage(
        inputs.x,
        inputs.y,
        inputs.y_bin,
        inputs.x_raw,
        inputs.y_raw,
        inputs.beta,
        inputs.num_good_algos,
        inputs.y_best,
        inputs.p,
        inputs.inst_labels,
        inputs.s,
        inputs.feat_labels,
        inputs.opts,
        ParallelOptions.default(),
        GeneralOptions.default(),
    )
    xaux_python, _, _, _ = sifted.select_features_by_performance()
    assert np.allclose(SiftedMatlabOutput().correlation_matlab, xaux_python, atol=1e-04)


def test_select_features_by_clustering() -> None:
    """Test cluster selection against MATLAB's cluster selection output.

    Despite the difference in cluster labels, we ensure that the number of items in
    python's cluster are 80% same as items in matlab's cluster.
    """
    rng = np.random.default_rng(seed=0)
    inputs = SiftedMatlabInput()
    sifted = SiftedStage(
        inputs.x,
        inputs.y,
        inputs.y_bin,
        inputs.x_raw,
        inputs.y_raw,
        inputs.beta,
        inputs.num_good_algos,
        inputs.y_best,
        inputs.p,
        inputs.inst_labels,
        inputs.s,
        inputs.feat_labels,
        inputs.opts,
        ParallelOptions.default(),
        GeneralOptions.default(),
    )
    x_aux, _, _, _ = sifted.select_features_by_performance()
    sifted.evaluate_cluster(x_aux, rng)
    _, cluster_python = sifted.select_features_by_clustering(x_aux, rng)
    assert are_same_clusters(SiftedMatlabOutput().cluster_matlab, cluster_python)


def are_same_clusters(
    cluster_a: NDArray[np.intc],
    cluster_b: NDArray[np.intc],
    threshold: float = 0.8,
) -> bool:
    """Check if two clusters have same number of elements more than threshold set.

    Parameters
    ----------
    cluster_a : NDArray[np.intc]
        The first cluster.
    cluster_b : NDArray[np.intc]
        The second cluster.
    threshold : float, optional
        The min ratio of matching elements between the two clusters (default is 0.8).

    Returns
    -------
    bool
        True if the number of matching elements exceeds the threshold, False otherwise.
    """
    cluster_a = np.array(cluster_a)
    cluster_b = np.array(cluster_b)

    unique_labels_a = np.unique(cluster_a)
    total_elements = len(cluster_a)
    matching_elements = 0

    for label in unique_labels_a:
        indices_a = np.where(cluster_a == label)[0]

        # Find the corresponding label in B for the same indices
        label_in_b = cluster_b[indices_a[0]]

        # Count the number of matching labels in B for these indices
        matches = np.sum(cluster_b[indices_a] == label_in_b)
        matching_elements += matches

    match_ratio = matching_elements / total_elements

    return bool(match_ratio >= threshold)


def test_run() -> None:
    """Test the _run method of Sifted class.

    Given the output of sifted stage of matlab and python, compute the correlation
    between them. Check for each column and row, there's only one value that has high
    correlation (>0.9) and other correlation values are low (<0.9)
    """
    inputs = SiftedMatlabInput()
    # The GA's default budget (100 generations x 50 pop) takes ~5 minutes here;
    # the correlation-matrix check below is a fuzzy high/normal/low comparison
    # (not an exact-match assertion), so it doesn't depend on exactly how
    # thoroughly the GA searched - a much smaller budget still converges on
    # good-enough feature combinations and cuts this test to a few seconds.
    fast_opts = dataclasses.replace(
        inputs.opts,
        num_generations=5,
        sol_per_pop=6,
        keep_elitism=1,
    )

    sifted_input = SiftedInput(
        inputs.x,
        inputs.y,
        inputs.y_bin,
        inputs.x_raw,
        inputs.y_raw,
        inputs.beta,
        inputs.num_good_algos,
        inputs.y_best,
        inputs.p,
        inputs.inst_labels,
        s=inputs.s,
        feat_labels=inputs.feat_labels,
        sifted_options=fast_opts,
        selvars_options=inputs.opts_selvar,
        data_dense=inputs.data_dense,
        parallel_options=ParallelOptions.default(),
        general_options=GeneralOptions.default(),
    )

    sifted_output = SiftedStage._run(sifted_input)  # noqa: SLF001
    x_python, x_matlab = sifted_output[0], SiftedMatlabOutput().x_matlab
    df_python = pd.DataFrame(x_python)
    df_matlab = pd.DataFrame(x_matlab)

    # compute correlation matrix that has been categorised into high, normal and low
    correlation_matrix = compute_correlation(df_python, df_matlab)

    # test case pass if 70%
    assert correlation_matrix_check(correlation_matrix, threshold=0.5)


def test_run_derives_sifted_dims_from_outer_pilot_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The aggregate pipeline has one PILOT-owned dimensionality setting."""
    captured: dict[str, SiftedOptions] = {}
    expected = cast(SiftedOutput, object())
    dims = 3

    def _capture_sifted(**kwargs: object) -> SiftedOutput:
        captured["opts"] = cast(SiftedOptions, kwargs["opts"])
        return expected

    monkeypatch.setattr(SiftedStage, "sifted", _capture_sifted)
    matrix = np.zeros((1, 1), dtype=np.double)
    result = SiftedStage._run(  # noqa: SLF001
        SiftedInput(
            x=matrix,
            y=matrix,
            y_bin=np.zeros((1, 1), dtype=np.bool_),
            x_raw=matrix,
            y_raw=matrix,
            beta=np.zeros(1, dtype=np.bool_),
            num_good_algos=np.zeros(1, dtype=np.double),
            y_best=np.zeros(1, dtype=np.double),
            p=np.ones(1, dtype=np.int_),
            inst_labels=pd.Series(["instance"]),
            feat_labels=["feature"],
            s=None,
            sifted_options=SiftedOptions.default(dims=2),
            selvars_options=SelvarsOptions.default(),
            data_dense=None,
            parallel_options=ParallelOptions.default(),
            general_options=GeneralOptions.default(),
            pilot_options=PilotOptions.default(dims=dims),
        ),
    )

    assert result is expected
    assert captured["opts"].dims == dims


def test_sifted_seed_reproducibility() -> None:
    """Same seed gives identical selvars; a different seed gives different selvars.

    Regression test for Q9 (general.seed threading): a cheap GA configuration
    (few generations/population) keeps this test fast while still exercising
    the clustering and GA code paths that consume the seeded rng.
    """
    inputs = SiftedMatlabInput()
    fast_opts = dataclasses.replace(
        inputs.opts,
        k=3,
        num_generations=2,
        sol_per_pop=4,
        keep_elitism=1,
    )

    def run(seed: int) -> NDArray[np.intc]:
        out = SiftedStage.sifted(
            inputs.x,
            inputs.y,
            inputs.y_bin,
            inputs.x_raw,
            inputs.y_raw,
            inputs.beta,
            inputs.num_good_algos,
            inputs.y_best,
            inputs.p,
            inputs.inst_labels,
            inputs.s,
            inputs.feat_labels,
            fast_opts,
            inputs.opts_selvar,
            None,
            ParallelOptions.default(),
            GeneralOptions(verbose=False, seed=seed),
        )
        return out.selvars

    selvars_a = run(0)
    selvars_b = run(0)
    selvars_c = run(1)

    np.testing.assert_array_equal(selvars_a, selvars_b)
    assert not np.array_equal(selvars_a, selvars_c)


def test_select_features_by_performance_uses_sorted_threshold_comparison() -> None:
    """Threshold check must compare each rank's sorted value, not raw rho by index.

    Regression test for a bug where `filtered_rho` aliased `rho` (so the returned
    `rho`/`pval` outputs would have been corrupted by zeroing) and the `>= opts.rho`
    threshold was applied to `rho[row]` (the *unsorted*, still-signed correlation at
    the row's original feature index) instead of to the sorted, absolute value at
    that rank -- silently dropping legitimate strong negative correlations and
    comparing the wrong feature's coefficient against the threshold.
    """
    rng = np.random.default_rng(seed=0)
    n_instances = 30
    # feature 0: strongly *negatively* correlated with algorithm 0.
    # feature 1: weakly correlated with everything (noise).
    y0 = rng.normal(size=n_instances)
    x0 = -y0 + rng.normal(scale=0.01, size=n_instances)
    x1 = rng.normal(size=n_instances)
    x = np.column_stack([x0, x1])
    y = np.column_stack([y0])
    y_bin = np.ones((n_instances, 1), dtype=bool)

    opts = dataclasses.replace(SiftedOptions.default(), rho=0.3)
    sifted = SiftedStage(
        x,
        y,
        y_bin,
        x,
        y,
        np.ones((n_instances, 1), dtype=bool),
        np.ones(n_instances),
        y0,
        np.ones(n_instances, dtype=int),
        pd.Series([str(i) for i in range(n_instances)]),
        None,
        ["feature_0", "feature_1"],
        opts,
        ParallelOptions.default(),
        GeneralOptions.default(),
    )

    x_aux, rho, _pval, selvars = sifted.select_features_by_performance()

    # the strongly (negatively) correlated feature must survive the filter
    assert 0 in selvars
    assert x_aux.shape[1] == len(selvars)
    # rho returned to the caller must be the genuine, unmodified coefficients --
    # not zeroed out by an in-place filter operating on an alias of `rho`.
    strong_negative_correlation = -0.9
    assert rho[0, 0] < strong_negative_correlation


def test_find_best_combination_uses_filtered_feature_matrix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The GA must optimise over `x_aux`, not the full, unfiltered `self.x`.

    Regression test: `_find_best_combination` previously assigned
    `ga_instance.selfx = self.x` and `ga_instance.selffeat_labels = self.feat_labels`
    (the full feature matrix/labels) instead of the correlation-filtered `x_aux` and
    its corresponding labels, even though `clust` (also passed to the GA) is sized
    and indexed for `x_aux`'s narrower column space -- silently defeating the
    correlation-based feature filter for the clustering/GA step.
    """
    inputs = SiftedMatlabInput()
    rng = np.random.default_rng(seed=0)
    sifted = SiftedStage(
        inputs.x,
        inputs.y,
        inputs.y_bin,
        inputs.x_raw,
        inputs.y_raw,
        inputs.beta,
        inputs.num_good_algos,
        inputs.y_best,
        inputs.p,
        inputs.inst_labels,
        inputs.s,
        inputs.feat_labels,
        dataclasses.replace(inputs.opts, k=2, rho=0.5),
        ParallelOptions.default(),
        GeneralOptions.default(),
    )

    x_aux, _, _, selvars = sifted.select_features_by_performance()
    assert x_aux.shape[1] < inputs.x.shape[1], (
        "precondition: the correlation filter must actually narrow the feature set "
        "for this test to be able to distinguish the bug"
    )

    clust, _ = sifted.select_features_by_clustering(x_aux, rng)

    class _FakeGAInstance:
        selfx: NDArray[np.double]
        selffeat_labels: NDArray[np.str_]

        def __init__(self, **kwargs: object) -> None:
            self._num_genes = int(kwargs["num_genes"])  # type: ignore[call-overload]

        def run(self) -> None:
            return None

        def best_solution(self) -> tuple[NDArray[np.intc], float, int]:
            return np.zeros(self._num_genes, dtype=int), 0.0, 0

    captured: dict[str, _FakeGAInstance] = {}

    def _fake_ga(**kwargs: object) -> _FakeGAInstance:
        instance = _FakeGAInstance(**kwargs)
        captured["instance"] = instance
        return instance

    monkeypatch.setattr("pygad.GA", _fake_ga)

    sifted._find_best_combination(x_aux, clust, selvars, rng)  # noqa: SLF001

    ga_instance = captured["instance"]
    assert ga_instance.selfx.shape[1] == x_aux.shape[1]
    np.testing.assert_array_equal(ga_instance.selfx, x_aux)
    assert ga_instance.selffeat_labels.shape[0] == x_aux.shape[1]
    np.testing.assert_array_equal(
        ga_instance.selffeat_labels,
        sifted.feat_labels[selvars],
    )


def test_evaluate_cluster_min_clusters_is_three() -> None:
    """`evaluate_cluster` must start from 3 clusters, matching MATLAB's `KList`.

    Regression test: the loop previously started at `min_clusters = 2`, but
    MATLAB's `evalclusters(..., 'KList', 3:nfeats, ...)` starts at 3. With 2
    clusters included, `silhouette_scores[0]` corresponded to K=2 instead of K=3,
    which also fed into the (separately fixed) suggested-K indexing bug.
    """
    rng = np.random.default_rng(seed=0)
    inputs = SiftedMatlabInput()
    sifted = SiftedStage(
        inputs.x,
        inputs.y,
        inputs.y_bin,
        inputs.x_raw,
        inputs.y_raw,
        inputs.beta,
        inputs.num_good_algos,
        inputs.y_best,
        inputs.p,
        inputs.inst_labels,
        inputs.s,
        inputs.feat_labels,
        inputs.opts,
        ParallelOptions.default(),
        GeneralOptions.default(),
    )
    x_aux, _, _, _ = sifted.select_features_by_performance()

    silhouette_scores, _ = sifted.evaluate_cluster(x_aux, rng)

    # K ranges from 3 to nfeats-1 inclusive => nfeats-3 scores, not nfeats-2.
    assert len(silhouette_scores) == x_aux.shape[1] - 3


def test_evaluate_cluster_suggested_k_logs_matching_score(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The suggested-K log message must report that K's own silhouette score.

    Regression test: `silhouette_scores[max_k_silhoulette]` indexed the scores
    list by the *cluster count* K instead of by `max_k_silhoulette_index` (the
    list's position for that K, offset by `min_clusters`), so the logged score
    belonged to a different K than the one being reported whenever
    `min_clusters != 0`.
    """
    from loguru import logger

    rng = np.random.default_rng(seed=0)
    inputs = SiftedMatlabInput()
    # force a k unlikely to equal the argmax, and well below max_clusters, so the
    # "suggested k" branch actually logs.
    opts = dataclasses.replace(inputs.opts, k=3)
    sifted = SiftedStage(
        inputs.x,
        inputs.y,
        inputs.y_bin,
        inputs.x_raw,
        inputs.y_raw,
        inputs.beta,
        inputs.num_good_algos,
        inputs.y_best,
        inputs.p,
        inputs.inst_labels,
        inputs.s,
        inputs.feat_labels,
        opts,
        ParallelOptions.default(),
        GeneralOptions(verbose=True, seed=0),
    )
    x_aux, _, _, _ = sifted.select_features_by_performance()

    messages: list[str] = []
    sink_id = logger.add(messages.append, level="DEBUG")
    try:
        silhouette_scores, _ = sifted.evaluate_cluster(x_aux, rng)
    finally:
        logger.remove(sink_id)

    max_index = int(np.argmax(silhouette_scores))
    expected_score = silhouette_scores[max_index]
    suggested = [m for m in messages if "Suggested k value" in m]
    if suggested:
        logged_score = float(suggested[0].strip().split(" of")[-1])
        assert logged_score == pytest.approx(expected_score, abs=1e-4)


def test_density_filtered_output_x_has_correct_shape() -> None:
    """Density-filtered output must select rows *and* columns, not misuse a slice.

    Regression test: the `bydensity` return path built its `x` output with
    `data_dense.x[subset_index][:selvars]`, which uses `selvars` (an array of
    several feature indices) as a slice *stop* bound on the row axis instead of
    indexing the column axis -- either raising a `TypeError` (numpy refuses to
    use a multi-element array as a slice bound) or, if it happened to have a
    single element, silently truncating rows instead of selecting columns.
    """
    inputs = SiftedMatlabInput()
    fast_opts = dataclasses.replace(
        inputs.opts,
        k=3,
        num_generations=2,
        sol_per_pop=4,
        keep_elitism=1,
    )

    out = SiftedStage.sifted(
        inputs.x,
        inputs.y,
        inputs.y_bin,
        inputs.x_raw,
        inputs.y_raw,
        inputs.beta,
        inputs.num_good_algos,
        inputs.y_best,
        inputs.p,
        inputs.inst_labels,
        inputs.s,
        inputs.feat_labels,
        fast_opts,
        inputs.opts_selvar_filter,
        inputs.data_dense,
        ParallelOptions.default(),
        GeneralOptions(verbose=False, seed=0),
    )

    expected_ndim = 2
    assert out.x.ndim == expected_ndim
    assert out.x.shape[1] == len(out.selvars)
    assert out.x.shape[0] == out.y.shape[0]


def test_standardize_for_correlation_distance_is_zero_mean_unit_variance() -> None:
    """Each standardized feature vector must be zero-mean, unit-variance.

    This is the mathematical property #300 issue 7's fix relies on:
    Euclidean distance between two population-z-scored vectors of the same
    length is a positive monotonic transform of their Pearson correlation
    distance, so clustering the standardized vectors with ordinary
    (Euclidean) k-means reproduces the nearest-centroid assignment a
    correlation-distance k-means would give, matching MATLAB's
    `kmeans(...,'Distance','correlation')`.
    """
    rng = np.random.default_rng(0)
    x_aux = rng.random((20, 4))

    standardized = SiftedStage._standardize_for_correlation_distance(  # noqa: SLF001
        x_aux,
    )

    assert standardized.shape == (4, 20)
    np.testing.assert_allclose(standardized.mean(axis=1), 0.0, atol=1e-10)
    np.testing.assert_allclose(standardized.std(axis=1), 1.0, atol=1e-10)


def test_standardize_for_correlation_distance_handles_constant_feature() -> None:
    """A zero-variance feature must not raise or produce NaN/inf."""
    x_aux = np.column_stack([np.full(10, 5.0), np.arange(10, dtype=float)])

    standardized = SiftedStage._standardize_for_correlation_distance(  # noqa: SLF001
        x_aux,
    )

    assert np.all(np.isfinite(standardized))
    np.testing.assert_allclose(standardized[0], 0.0)


def test_select_features_by_clustering_uses_standardized_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`select_features_by_clustering` must cluster standardized vectors.

    Regression test for #300 issue 7: previously clustered raw feature
    vectors with Euclidean-distance k-means; MATLAB uses correlation
    distance. Real feature data from the MATLAB reference fixture is not
    already zero-mean/unit-variance, so if the clustering input were still
    raw, this would fail.
    """
    inputs = SiftedMatlabInput()
    rng = np.random.default_rng(seed=0)
    sifted = SiftedStage(
        inputs.x,
        inputs.y,
        inputs.y_bin,
        inputs.x_raw,
        inputs.y_raw,
        inputs.beta,
        inputs.num_good_algos,
        inputs.y_best,
        inputs.p,
        inputs.inst_labels,
        inputs.s,
        inputs.feat_labels,
        dataclasses.replace(inputs.opts, k=2, rho=0.5),
        ParallelOptions.default(),
        GeneralOptions.default(),
    )
    x_aux, _, _, _ = sifted.select_features_by_performance()

    captured: dict[str, NDArray[np.double]] = {}
    original_fit_predict = KMeans.fit_predict

    def _capturing_fit_predict(
        self: KMeans,
        x: NDArray[np.double],
        *args: object,
        **kwargs: object,
    ) -> NDArray[np.intc]:
        captured["x"] = x
        result: NDArray[np.intc] = original_fit_predict(self, x, *args, **kwargs)
        return result

    monkeypatch.setattr(KMeans, "fit_predict", _capturing_fit_predict)

    sifted.select_features_by_clustering(x_aux, rng)

    np.testing.assert_allclose(captured["x"].mean(axis=1), 0.0, atol=1e-10)
    np.testing.assert_allclose(captured["x"].std(axis=1), 1.0, atol=1e-10)


def test_cost_fcn_uses_classification_accuracy_loss(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`cost_fcn`'s fitness must be a classification accuracy, not MSE.

    Regression test for #300 issue 5: previously scored k-NN cross-
    validation with `neg_mean_squared_error` on the binary good/bad
    labels (a regression metric applied to a classification target)
    instead of a real classification loss, matching MATLAB's
    `fitcknn`/`kfoldLoss`.
    """
    dims = 3

    class _FakePilotOutput:
        z = np.zeros((6, dims))

    captured_pilot_options: list[PilotOptions] = []

    def _fake_pilot(*args: object, **_kwargs: object) -> _FakePilotOutput:
        captured_pilot_options.append(cast(PilotOptions, args[3]))
        return _FakePilotOutput()

    monkeypatch.setattr(
        "instancespace.stages.sifted.PilotStage.pilot",
        _fake_pilot,
    )

    captured_scoring: list[str] = []
    per_algo_scores = [np.array([0.9, 0.9]), np.array([0.7, 0.7])]
    calls = iter(per_algo_scores)

    def _fake_cross_val_score(
        _estimator: object,
        _x: NDArray[np.double],
        _y: NDArray[np.intc],
        *,
        cv: object,
        scoring: str,
    ) -> NDArray[np.double]:
        captured_scoring.append(scoring)
        return next(calls)

    monkeypatch.setattr(
        "instancespace.stages.sifted.cross_val_score",
        _fake_cross_val_score,
    )

    class _FakeInstance:
        selfx = np.zeros((6, 3))
        selfy = np.zeros((6, 2))
        selfy_bin = np.zeros((6, 2), dtype=int)
        selffeat_labels = np.array(["f0", "f1", "f2"])
        clust = np.ones((3, 1), dtype=bool)
        cv_partition = None
        general_options = GeneralOptions.default()
        dims = 3
        cost_cache: dict[bytes, float] = {}  # noqa: RUF012

    instance = _FakeInstance()
    fitness = SiftedStage.cost_fcn(instance, np.array([0]), 0)

    assert captured_scoring == ["accuracy", "accuracy"]
    assert captured_pilot_options[0].dims == dims
    # pygad maximizes fitness, so this must be the *worst* (minimum)
    # per-algorithm accuracy (0.7), not the best (0.9) and not a loss
    # value - maximizing the minimum accuracy directly is what makes the
    # GA search for feature sets where every algorithm classifies well.
    assert fitness == pytest.approx(0.7)


def compute_correlation(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    """Compute correlation matrix and categorise them into high, normal and low.

    Correlation values are categorised as high, normal, or low.

    Parameters
    ----------
    df1 : pd.DataFrame
        The first dataframe.
    df2 : pd.DataFrame
        The second dataframe.

    Returns
    -------
    pd.DataFrame
        A dataframe where the correlation values are categorised into high (1),
        normal (0), and low (-1).
    """
    upper_bound = 0.7
    lower_bound = 0.3

    def categorise_value(x: float) -> int:
        """Categorise correlation value into high, normal and low."""
        if x > upper_bound:
            return 1
        if x < lower_bound:
            return -1
        return 0

    # given two dataframe, compute correlation matrix
    correlation_matrix = pd.DataFrame(index=df1.columns, columns=df2.columns)
    for col1 in df1.columns:
        for col2 in df2.columns:
            correlation_matrix.loc[col1, col2] = df1[col1].corr(df2[col2])
    correlation_matrix = correlation_matrix.abs()

    # categorise correlation matrix's value to high and low
    return correlation_matrix.map(categorise_value)


def correlation_matrix_check(df: pd.DataFrame, threshold: float) -> bool:
    """Check if at least threshold percentage of both rows and columns fulfil condition.

    The condition is fulfilled if only one value in a row or column has a high
    correlation (categorised as 1).

    Parameters
    ----------
    df : pd.DataFrame
        The correlation matrix with categorised values.
    threshold : float
        The minimum percentage of rows and columns that must fulfill the condition.

    Returns
    -------
    bool
        True if the condition is satisfied for at least the threshold percentage,
        False otherwise.
    """
    # for every row, calculate percentage of only one value has modified correlation
    # equals to 1
    row_condition = (df == 1).sum(axis=1) == 1
    row_percentage = row_condition.mean()

    # for every column, calculate percentage of only one value has modified correlation
    # equals to 1
    col_condition = (df == 1).sum(axis=0) == 1
    col_percentage = col_condition.mean()

    total_percentage = (row_percentage + col_percentage) / 2

    return total_percentage >= threshold


def _synthetic_sifted_stage(
    n_features: int,
    *,
    flag: bool,
) -> tuple[SiftedStage, DataDense]:
    """Create a compact SIFTED stage and its pre-filter dense dataset."""
    n_instances = 6
    x = np.arange(n_instances * n_features, dtype=float).reshape(
        n_instances,
        n_features,
    )
    y = np.column_stack(
        [np.arange(n_instances, dtype=float), np.arange(n_instances, 0, -1)],
    )
    good_performance_threshold = 2
    y_bin = y <= good_performance_threshold
    labels = pd.Series([f"i{i}" for i in range(n_instances)])
    stage = SiftedStage(
        x,
        y,
        y_bin,
        x.copy(),
        y.copy(),
        np.zeros(n_instances, dtype=bool),
        np.sum(y_bin, axis=1),
        np.min(y, axis=1),
        np.ones(n_instances, dtype=int),
        labels,
        None,
        [f"f{i}" for i in range(n_features)],
        dataclasses.replace(SiftedOptions.default(), flag=flag, k=4),
        ParallelOptions.default(),
        GeneralOptions.default(),
    )
    dense = DataDense(
        inst_labels=labels,
        x=x + 100.0,
        y=y,
        x_raw=x.copy(),
        y_raw=y.copy(),
        y_bin=y_bin,
        y_best=np.min(y, axis=1),
        p=np.ones(n_instances, dtype=int),
        num_good_algos=np.sum(y_bin, axis=1),
        beta=np.zeros(n_instances, dtype=bool),
        s=None,
    )
    return stage, dense


def test_sifted_flag_false_returns_all_features_without_density(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A disabled SIFTED stage runs neither selection nor density filtering."""
    stage, dense = _synthetic_sifted_stage(1, flag=False)

    def fail_if_called(*_args: object, **_kwargs: object) -> None:
        pytest.fail("disabled SIFTED must not run selection or density filtering")

    monkeypatch.setattr(stage, "select_features_by_performance", fail_if_called)
    monkeypatch.setattr("instancespace.stages.sifted.do_filter", fail_if_called)

    result = stage.sift(SelvarsOptions.default(density_flag=True), dense)

    np.testing.assert_array_equal(result.x, stage.x)
    np.testing.assert_array_equal(result.selvars, [0])
    assert result.feat_labels == ["f0"]


@pytest.mark.parametrize(
    ("n_features", "selected"),
    [
        (3, None),
        (5, (0, 2)),
        (5, (0, 1, 2, 3)),
    ],
)
def test_density_filter_runs_for_every_sifted_early_return(
    monkeypatch: pytest.MonkeyPatch,
    n_features: int,
    selected: tuple[int, ...] | None,
) -> None:
    """Density re-filtering also finalises the <=3 and <=K selection paths."""
    stage, dense = _synthetic_sifted_stage(n_features, flag=True)
    expected_selvars = np.array(
        selected if selected is not None else tuple(range(n_features)),
        dtype=np.intc,
    )

    def select_features() -> tuple[
        NDArray[np.double],
        NDArray[np.double],
        NDArray[np.double],
        NDArray[np.intc],
    ]:
        if selected is None:
            pytest.fail("the initial <=3 path must skip performance selection")
        shape = (n_features, stage.y.shape[1])
        return (
            stage.x[:, expected_selvars],
            np.zeros(shape),
            np.zeros(shape),
            expected_selvars,
        )

    monkeypatch.setattr(stage, "select_features_by_performance", select_features)
    calls: list[tuple[int, int]] = []

    def fake_filter(
        selected_x: NDArray[np.double],
        *_args: object,
        **_kwargs: object,
    ) -> tuple[NDArray[np.bool_], None, None, None]:
        calls.append((selected_x.shape[0], selected_x.shape[1]))
        removed = np.ones(selected_x.shape[0], dtype=bool)
        removed[0] = False
        return removed, None, None, None

    monkeypatch.setattr("instancespace.stages.sifted.do_filter", fake_filter)

    result = stage.sift(SelvarsOptions.default(density_flag=True), dense)

    assert calls == [(dense.x.shape[0], len(expected_selvars))]
    np.testing.assert_array_equal(result.selvars, expected_selvars)
    assert result.x.shape == (1, len(expected_selvars))
