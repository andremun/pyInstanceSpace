"""Test module for Filter class to verify its functionality.

The file contains multiple unit tests to ensure that the `Filter` class correctly
perform its tasks. The basic mechanism of the test is to compare its output against
output from MATLAB and check if the outputs are the same or reasonable similar.

Tests include:
- Verifying ouput against MATLAB's output with 'Ftr' option type
- Verifying ouput against MATLAB's output with 'Ftr&AP' option type
- Verifying ouput against MATLAB's output with 'Ftr&AP&Good' option type
- Verifying ouput against MATLAB's output with 'Ftr&Good' option type
"""

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from loguru import logger
from numpy.typing import NDArray

from instancespace.data.options import SelvarsOptions
from instancespace.utils.filter import (
    MIN_KEPT_INSTANCES_FOR_UNIFORMITY,
    compute_uniformity,
    do_filter,
    filter_instance,
)

script_dir = Path(__file__).parent

csv_path_x = script_dir / "test_data/filter/input/input_X.csv"
csv_path_y = script_dir / "test_data/filter/input/input_Y.csv"
csv_path_y_bin = script_dir / "test_data/filter/input/input_Ybin.csv"

input_x = pd.read_csv(csv_path_x, header=None).to_numpy()
input_y = pd.read_csv(csv_path_y, header=None).to_numpy()
input_y_bin = pd.read_csv(csv_path_y_bin, header=None).to_numpy()


class TestFtr:
    """Test output with Ftr option type."""

    @pytest.fixture()
    def ftr_option(self) -> SelvarsOptions:
        """Fixture for creating selvars option with Ftr type.

        Returns
        -------
            SelvarsOption: A selvars option object with Ftr for type.
        """
        return SelvarsOptions(
            small_scale_flag=False,
            small_scale=0.5,
            file_idx_flag=False,
            file_idx="",
            selvars_type="Ftr",
            density_flag=True,
            min_distance=0.1,
            algos=list("abc"),
            feats=list("abc"),
        )

    def test_ftr_filter(self, ftr_option: SelvarsOptions) -> None:
        """Test output from filtering against MATLAB's output.

        Compare subset_index, is_dissimilar, is_visa obtained from filter_instance
        method against each corresponding output from MATLAB.

        Args
        ----
            ftr_options (SelvarsOptions): SelvarsOption with type equals to "Ftr"
        """
        csv_path_subset_index = (
            script_dir / "test_data/filter/output/ftr/subsetIndex.csv"
        )
        csv_path_is_dissimilar = (
            script_dir / "test_data/filter/output/ftr/isDissimilar.csv"
        )
        csv_path_is_visa = script_dir / "test_data/filter/output/ftr/isVISA.csv"

        subset_index_ml = pd.read_csv(
            csv_path_subset_index,
            header=None,
            dtype=bool,
        ).to_numpy()
        is_dissimilar_ml = pd.read_csv(
            csv_path_is_dissimilar,
            header=None,
            dtype=bool,
        ).to_numpy()
        is_visa_ml = pd.read_csv(csv_path_is_visa, header=None, dtype=bool).to_numpy()

        subset_index, is_dissimilar, is_visa, _ = do_filter(
            input_x,
            input_y,
            input_y_bin,
            ftr_option.selvars_type,
            ftr_option.min_distance,
        )

        assert np.all(subset_index == subset_index_ml[:, 0])
        assert np.all(is_dissimilar == is_dissimilar_ml[:, 0])
        assert np.all(is_visa == is_visa_ml[:, 0])

    def test_ftr_uniformity(self, ftr_option: SelvarsOptions) -> None:
        """Test output from comuting uniformity against MATLAB's output.

        Compare computed uniformity value from Filter against the uniformity value
        obtained from MATLAB.

        Args
        ----
            ftr_options (SelvarsOptions): SelvarsOption with type equals to "Ftr"
        """
        csv_path_uniformity = script_dir / "test_data/filter/output/ftr/uniformity.csv"

        uniformity_ml = (
            pd.read_csv(csv_path_uniformity, header=None, dtype=float).to_numpy().item()
        )

        subset_index, _, _ = filter_instance(
            input_x,
            input_y,
            input_y_bin,
            ftr_option.selvars_type,
            ftr_option.min_distance,
        )
        uniformity = compute_uniformity(
            input_x,
            subset_index,
        )

        assert np.allclose(uniformity, uniformity_ml)


class TestFtrAp:
    """Test output with Ftr&AP option type."""

    @pytest.fixture()
    def ftr_ap_option(self) -> SelvarsOptions:
        """Fixture for creating selvars option with Ftr&AP type.

        Returns
        -------
            SelvarsOption: A selvars option object with Ftr&AP for type.
        """
        return SelvarsOptions(
            small_scale_flag=False,
            small_scale=0.5,
            file_idx_flag=False,
            file_idx="",
            selvars_type="Ftr&AP",
            density_flag=True,
            min_distance=0.1,
            algos=list("abc"),
            feats=list("abc"),
        )

    def test_ftr_ap_filter(self, ftr_ap_option: SelvarsOptions) -> None:
        """Test output from filtering against MATLAB's output.

        Compare subset_index, is_dissimilar, is_visa obtained from filter_instance
        method against each corresponding output from MATLAB.

        Args
        ----
            ftr_ap_options (SelvarsOptions): SelvarsOption with type equals to "Ftr&AP"
        """
        csv_path_subset_index = (
            script_dir / "test_data/filter/output/ftr_ap/subsetIndex.csv"
        )
        csv_path_is_dissimilar = (
            script_dir / "test_data/filter/output/ftr_ap/isDissimilar.csv"
        )
        csv_path_is_visa = script_dir / "test_data/filter/output/ftr_ap/isVISA.csv"

        subset_index_ml = pd.read_csv(
            csv_path_subset_index,
            header=None,
            dtype=bool,
        ).to_numpy()
        is_dissimilar_ml = pd.read_csv(
            csv_path_is_dissimilar,
            header=None,
            dtype=bool,
        ).to_numpy()
        is_visa_ml = pd.read_csv(csv_path_is_visa, header=None, dtype=bool).to_numpy()

        subset_index, is_dissimilar, is_visa, _ = do_filter(
            input_x,
            input_y,
            input_y_bin,
            ftr_ap_option.selvars_type,
            ftr_ap_option.min_distance,
        )

        assert np.all(subset_index == subset_index_ml[:, 0])
        assert np.all(is_dissimilar == is_dissimilar_ml[:, 0])
        assert np.all(is_visa == is_visa_ml[:, 0])

    def test_ftr_ap_uniformity(self, ftr_ap_option: SelvarsOptions) -> None:
        """Test output from comuting uniformity against MATLAB's output.

        Compare computed uniformity value from Filter against the uniformity value
        obtained from MATLAB.

        Args
        ----
            ftr_ap_options (SelvarsOptions): SelvarsOption with type equals to "Ftr&AP"
        """
        csv_path_uniformity = (
            script_dir / "test_data/filter/output/ftr_ap/uniformity.csv"
        )

        uniformity_ml = (
            pd.read_csv(csv_path_uniformity, header=None, dtype=float).to_numpy().item()
        )

        subset_index, _, _ = filter_instance(
            input_x,
            input_y,
            input_y_bin,
            ftr_ap_option.selvars_type,
            ftr_ap_option.min_distance,
        )
        uniformity = compute_uniformity(input_x, subset_index)

        assert np.allclose(uniformity, uniformity_ml)


class TestFtrApGood:
    """Test output with Ftr&AP&Good option type."""

    @pytest.fixture()
    def ftr_ap_good_option(self) -> SelvarsOptions:
        """Fixture for creating selvars option with Ftr&AP&Good type.

        Returns
        -------
            SelvarsOption: A selvars option object with Ftr&AP&Good for type.
        """
        return SelvarsOptions(
            small_scale_flag=False,
            small_scale=0.5,
            file_idx_flag=False,
            file_idx="",
            selvars_type="Ftr&AP&Good",
            density_flag=True,
            min_distance=0.1,
            algos=list("abc"),
            feats=list("abc"),
        )

    def test_ftr_ap_good_filter(self, ftr_ap_good_option: SelvarsOptions) -> None:
        """Test output from filtering against MATLAB's output.

        Compare subset_index, is_dissimilar, is_visa obtained from filter_instance
        method against each corresponding output from MATLAB.

        Args
        ----
            ftr_options (SelvarsOptions): SelvarsOption with type equals to
                "Ftr&AP&Good"
        """
        csv_path_subset_index = (
            script_dir / "test_data/filter/output/ftr_ap_good/subsetIndex.csv"
        )
        csv_path_is_dissimilar = (
            script_dir / "test_data/filter/output/ftr_ap_good/isDissimilar.csv"
        )
        csv_path_is_visa = script_dir / "test_data/filter/output/ftr_ap_good/isVISA.csv"

        subset_index_ml = pd.read_csv(
            csv_path_subset_index,
            header=None,
            dtype=bool,
        ).to_numpy()
        is_dissimilar_ml = pd.read_csv(
            csv_path_is_dissimilar,
            header=None,
            dtype=bool,
        ).to_numpy()
        is_visa_ml = pd.read_csv(csv_path_is_visa, header=None, dtype=bool).to_numpy()

        subset_index, is_dissimilar, is_visa, _ = do_filter(
            input_x,
            input_y,
            input_y_bin,
            ftr_ap_good_option.selvars_type,
            ftr_ap_good_option.min_distance,
        )

        assert np.all(subset_index == subset_index_ml[:, 0])
        assert np.all(is_dissimilar == is_dissimilar_ml[:, 0])
        assert np.all(is_visa == is_visa_ml[:, 0])

    def test_ftr_ap_good_uniformity(self, ftr_ap_good_option: SelvarsOptions) -> None:
        """Test output from comuting uniformity against MATLAB's output.

        Compare computed uniformity value from Filter against the uniformity value
        obtained from MATLAB.

        Args
        ----
            ftr_options (SelvarsOptions): SelvarsOption with type equals to
                "Ftr&AP&Good"
        """
        csv_path_uniformity = (
            script_dir / "test_data/filter/output/ftr_ap_good/uniformity.csv"
        )

        uniformity_ml = (
            pd.read_csv(csv_path_uniformity, header=None, dtype=float).to_numpy().item()
        )

        subset_index, _, _ = filter_instance(
            input_x,
            input_y,
            input_y_bin,
            ftr_ap_good_option.selvars_type,
            ftr_ap_good_option.min_distance,
        )
        uniformity = compute_uniformity(input_x, subset_index)

        assert np.allclose(uniformity, uniformity_ml)


class TestFtrGood:
    """Test output with Ftr&Good option type."""

    @pytest.fixture()
    def ftr_good_option(self) -> SelvarsOptions:
        """Fixture for creating selvars option with Ftr&Good type.

        Returns
        -------
            SelvarsOption: A selvars option object with Ftr&Good for type.
        """
        return SelvarsOptions(
            small_scale_flag=False,
            small_scale=0.5,
            file_idx_flag=False,
            file_idx="",
            selvars_type="Ftr&Good",
            density_flag=True,
            min_distance=0.1,
            algos=list("abc"),
            feats=list("abc"),
        )

    def test_ftr_good_filter(self, ftr_good_option: SelvarsOptions) -> None:
        """Test output from filtering against MATLAB's output.

        Compare subset_index, is_dissimilar, is_visa obtained from filter_instance
        method against each corresponding output from MATLAB.

        Args
        ----
            ftr_options (SelvarsOptions): SelvarsOption with type equals to "Ftr&Good"
        """
        csv_path_subset_index = (
            script_dir / "test_data/filter/output/ftr_good/subsetIndex.csv"
        )
        csv_path_is_dissimilar = (
            script_dir / "test_data/filter/output/ftr_good/isDissimilar.csv"
        )
        csv_path_is_visa = script_dir / "test_data/filter/output/ftr_good/isVISA.csv"

        subset_index_ml = pd.read_csv(
            csv_path_subset_index,
            header=None,
            dtype=bool,
        ).to_numpy()
        is_dissimilar_ml = pd.read_csv(
            csv_path_is_dissimilar,
            header=None,
            dtype=bool,
        ).to_numpy()
        is_visa_ml = pd.read_csv(csv_path_is_visa, header=None, dtype=bool).to_numpy()

        subset_index, is_dissimilar, is_visa, _ = do_filter(
            input_x,
            input_y,
            input_y_bin,
            ftr_good_option.selvars_type,
            ftr_good_option.min_distance,
        )

        assert np.all(subset_index == subset_index_ml[:, 0])
        assert np.all(is_dissimilar == is_dissimilar_ml[:, 0])
        assert np.all(is_visa == is_visa_ml[:, 0])

    def test_ftr_good_uniformity(self, ftr_good_option: SelvarsOptions) -> None:
        """Test output from comuting uniformity against MATLAB's output.

        Compare computed uniformity value from Filter against the uniformity value
        obtained from MATLAB.

        Args
        ----
            ftr_options (SelvarsOptions): SelvarsOption with type equals to "Ftr&Good"
        """
        csv_path_uniformity = (
            script_dir / "test_data/filter/output/ftr_good/uniformity.csv"
        )

        uniformity_ml = (
            pd.read_csv(csv_path_uniformity, header=None, dtype=float).to_numpy().item()
        )

        subset_index, _, _ = filter_instance(
            input_x,
            input_y,
            input_y_bin,
            ftr_good_option.selvars_type,
            ftr_good_option.min_distance,
        )
        uniformity = compute_uniformity(input_x, subset_index)

        assert np.allclose(uniformity, uniformity_ml)


def _collect_warnings(
    fn: Callable[..., object],
    *args: object,
) -> list[str]:
    """Run fn(*args) and return the loguru WARNING-level messages it emitted."""
    messages: list[str] = []
    sink_id = logger.add(
        lambda msg: messages.append(msg.record["message"]),
        level="WARNING",
    )
    try:
        fn(*args)
    finally:
        logger.remove(sink_id)
    return messages


def test_uniformity_is_nan_with_fewer_than_two_kept_instances() -> None:
    """F12: fewer than 2 retained instances is degenerate, matching MATLAB's guard.

    `core/FILTER.m` returns NaN with an `ISA:FILTER:degenerateUniformity`
    warning rather than letting `std`/`mean` divide-by-zero into a
    meaningless value - verify Python does the same instead of raising or
    silently returning a bogus number.
    """
    x = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    # Keep only the first instance - subset_index marks the other two excluded.
    subset_index = np.array([False, True, True])

    messages = _collect_warnings(compute_uniformity, x, subset_index)

    assert np.isnan(compute_uniformity(x, subset_index))
    assert any("Uniformity is undefined" in m for m in messages)


def test_uniformity_is_nan_when_all_kept_instances_coincide() -> None:
    """F12: all-coincident retained instances (mean distance 0) is also degenerate."""
    x = np.array([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]])
    subset_index = np.array([False, False, False])

    messages = _collect_warnings(compute_uniformity, x, subset_index)

    assert np.isnan(compute_uniformity(x, subset_index))
    assert any("Uniformity is undefined" in m for m in messages)


def _brute_force_filter_instance(
    x: NDArray[np.double],
    y: NDArray[np.double],
    y_bin: NDArray[np.bool_],
    selvars_type: str,
    min_distance: float,
) -> tuple[NDArray[np.bool_], NDArray[np.bool_], NDArray[np.bool_]]:
    """Independent O(n^2) reference oracle for `filter_instance`.

    A plain double loop over every pair, computing each distance directly
    with `cdist` rather than a KD-tree - kept only in this test file as a
    ground-truth to check the KD-tree-based production code against on edge
    cases a KD-tree can behave subtly differently on (exact-boundary
    distances, coincident points), not as a second production
    implementation.
    """
    from scipy.spatial.distance import cdist as _cdist

    n_insts, n_algos = y.shape
    n_feats = x.shape[1]
    subset_index = np.zeros(n_insts, dtype=bool)
    is_dissimilar = np.ones(n_insts, dtype=bool)
    is_visa = np.zeros(n_insts, dtype=bool)
    gamma = np.sqrt(n_algos / n_feats) * min_distance

    for i in range(n_insts):
        if subset_index[i]:
            continue
        for j in range(i + 1, n_insts):
            if subset_index[j]:
                continue
            dx = _cdist([x[i, :]], [x[j, :]]).item()
            if dx > min_distance:
                continue
            dy = _cdist([y[i, :]], [y[j, :]]).item()
            db = np.all(np.logical_and(y_bin[i, :], y_bin[j, :]))
            is_dissimilar[j] = False
            if selvars_type == "Ftr":
                subset_index[j] = True
            elif selvars_type == "Ftr&AP":
                subset_index[j], is_visa[j] = (
                    (True, False) if dy <= gamma else (False, True)
                )
            elif selvars_type == "Ftr&Good":
                subset_index[j], is_visa[j] = (True, False) if db else (False, True)
            elif selvars_type == "Ftr&AP&Good":
                if db:
                    subset_index[j], is_visa[j] = (
                        (True, False) if dy <= gamma else (False, True)
                    )
                else:
                    is_visa[j] = True
    return subset_index, is_dissimilar, is_visa


def _brute_force_uniformity(
    x: NDArray[np.double],
    subset_index: NDArray[np.bool_],
) -> float:
    """Independent O(n^2) reference oracle for `compute_uniformity`."""
    from scipy.spatial.distance import pdist as _pdist
    from scipy.spatial.distance import squareform as _squareform

    x_kept = x[~subset_index, :]
    if x_kept.shape[0] < MIN_KEPT_INSTANCES_FOR_UNIFORMITY:
        return float("nan")
    d = _squareform(_pdist(x_kept))
    np.fill_diagonal(d, np.nan)
    nearest = np.nanmin(d, axis=0)
    if np.all(np.isnan(nearest)) or np.nanmean(nearest) == 0:
        return float("nan")
    return float(1 - (np.nanstd(nearest, ddof=1) / np.nanmean(nearest)))


@pytest.mark.parametrize(
    "selvars_type",
    ["Ftr", "Ftr&AP", "Ftr&Good", "Ftr&AP&Good"],
)
@pytest.mark.parametrize(
    ("name", "x", "y", "y_bin"),
    [
        (
            "coincident_points",
            np.array([[1.0, 1.0], [1.0, 1.0], [5.0, 5.0], [1.0, 1.0]]),
            np.array([[0.1, 0.2], [0.1, 0.2], [0.9, 0.8], [0.15, 0.25]]),
            np.array(
                [[True, False], [True, False], [False, True], [True, True]],
            ),
        ),
        (
            "exact_boundary_distance",
            np.array([[0.0, 0.0], [0.5, 0.0], [2.0, 2.0]]),
            np.array([[0.1, 0.2], [0.3, 0.4], [0.9, 0.8]]),
            np.array([[True, False], [True, False], [False, True]]),
        ),
        (
            "dense_cluster",
            np.array([[0.0, 0.0], [0.01, 0.0], [0.0, 0.01], [0.01, 0.01]]),
            np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]]),
            np.array(
                [
                    [True, False],
                    [True, True],
                    [False, True],
                    [True, False],
                ],
            ),
        ),
        (
            "no_neighbours_at_all",
            np.array([[0.0, 0.0], [100.0, 0.0], [0.0, 100.0]]),
            np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]),
            np.array([[True, False], [False, True], [True, True]]),
        ),
        (
            "very_small_n",
            np.array([[0.0, 0.0], [0.05, 0.0]]),
            np.array([[0.1, 0.2], [0.3, 0.4]]),
            np.array([[True, False], [True, True]]),
        ),
    ],
)
def test_kd_tree_matches_brute_force(
    name: str,
    x: NDArray[np.double],
    y: NDArray[np.double],
    y_bin: NDArray[np.bool_],
    selvars_type: str,
) -> None:
    """F12: the KD-tree rewrite must match the old O(n^2) algorithm exactly.

    Covers the edge cases a KD-tree can plausibly behave differently on:
    exact-coincident points, a pair exactly at the min_distance boundary, a
    dense cluster where every pair is within min_distance, instances with no
    neighbours at all, and n too small to matter either way.
    """
    min_distance = 0.5

    expected_subset, expected_dissimilar, expected_visa = _brute_force_filter_instance(
        x,
        y,
        y_bin,
        selvars_type,
        min_distance,
    )
    subset_index, is_dissimilar, is_visa = filter_instance(
        x,
        y,
        y_bin,
        selvars_type,
        min_distance,
    )

    assert np.array_equal(subset_index, expected_subset), name
    assert np.array_equal(is_dissimilar, expected_dissimilar), name
    assert np.array_equal(is_visa, expected_visa), name

    expected_uniformity = _brute_force_uniformity(x, expected_subset)
    uniformity = compute_uniformity(x, subset_index)
    if np.isnan(expected_uniformity):
        assert np.isnan(uniformity), name
    else:
        assert np.allclose(uniformity, expected_uniformity), name
