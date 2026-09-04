"""Test module for filter functionality post Prelim class to verify its functionality.

The file contains multiple unit tests to ensure that the `filter` function correctly
performs its tasks. The basic mechanism of the test is to compare its output against
output from MATLAB and check if the outputs are the same or reasonable similar.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from instancespace.data.options import GeneralOptions, PrelimOptions, SelvarsOptions
from instancespace.stages.prelim import PrelimStage

script_dir = Path(__file__).parent


def test_split_data() -> None:
    """
    Test case for the split data function by using matlab example.

    expected: No assertion errors.
    """
    # idx = np.genfromtxt(script_dir / "test_data/prelim/split/idx.txt", delimiter=",")

    x_before = np.genfromtxt(
        script_dir / "test_data/prelim/split/before/x_split.txt",
        delimiter=",",
    )
    y_before = np.genfromtxt(
        script_dir / "test_data/prelim/split/before/Y_split.txt",
        delimiter=",",
    )
    x_raw_before = np.genfromtxt(
        script_dir / "test_data/prelim/split/before/Xraw_split.txt",
        delimiter=",",
    )
    y_raw_before = np.genfromtxt(
        script_dir / "test_data/prelim/split/before/Yraw_split.txt",
        delimiter=",",
    )
    y_bin_before = np.genfromtxt(
        script_dir / "test_data/prelim/split/before/Ybin_split.txt",
        delimiter=",",
    )
    beta_before = np.genfromtxt(
        script_dir / "test_data/prelim/split/before/beta_split.txt",
        delimiter=",",
    )
    num_good_algos_before = np.genfromtxt(
        script_dir / "test_data/prelim/split/before/numGoodAlgos_split.txt",
        delimiter=",",
    )
    y_best_before = np.genfromtxt(
        script_dir / "test_data/prelim/split/before/Ybest_split.txt",
        delimiter=",",
    )
    p_before = np.genfromtxt(
        script_dir / "test_data/prelim/split/before/P_split.txt",
        delimiter=",",
    )
    inst_labels_before = pd.read_csv(
        script_dir / "test_data/prelim/split/before/instlabels_split.txt",
        header=None,
    ).loc[:, 0]

    inst_labels = inst_labels_before
    s_before = None

    prelim_opts = PrelimOptions(
        max_perf=True,
        abs_perf=True,
        epsilon=0.2,
        beta_threshold=0.55,
        bound=True,
        norm=True,
    )

    selvars_opts = SelvarsOptions(
        small_scale_flag=False,
        small_scale=0.50,
        file_idx_flag=False,
        file_idx="",
        feats=None,
        algos=None,
        selvars_type="Ftr&Good",
        density_flag=False,
        min_distance=0.1,
    )

    prelim = PrelimStage(
        x_before,
        y_before,
        x_raw_before,
        y_raw_before,
        s_before,
        inst_labels_before,
        prelim_opts,
        selvars_opts,
        GeneralOptions.default(),
    )

    (
        subset_index,
        x,
        y,
        x_raw,
        y_raw,
        y_bin,
        beta,
        num_good_algos,
        y_best,
        p,
        inst_labels,
        s,
        data_dense,
    ) = prelim._filter(
        inst_labels,
        x_before,
        y_before,
        y_bin_before,
        y_best_before,
        x_raw_before,
        y_raw_before,
        p_before,
        num_good_algos_before,
        beta_before,
        s_before,
        selvars_opts,
    )

    x_after = np.genfromtxt(
        script_dir / "test_data/prelim/split/after/x_split.txt",
        delimiter=",",
    )
    y_after = np.genfromtxt(
        script_dir / "test_data/prelim/split/after/Y_split.txt",
        delimiter=",",
    )
    x_raw_after = np.genfromtxt(
        script_dir / "test_data/prelim/split/after/Xraw_split.txt",
        delimiter=",",
    )
    y_raw_after = np.genfromtxt(
        script_dir / "test_data/prelim/split/after/Yraw_split.txt",
        delimiter=",",
    )
    y_bin_after = np.genfromtxt(
        script_dir / "test_data/prelim/split/after/Ybin_split.txt",
        delimiter=",",
    )
    beta_after = np.genfromtxt(
        script_dir / "test_data/prelim/split/after/beta_split.txt",
        delimiter=",",
    )
    num_good_algos_after = np.genfromtxt(
        script_dir / "test_data/prelim/split/after/numGoodAlgos_split.txt",
        delimiter=",",
    )
    y_best_after = np.genfromtxt(
        script_dir / "test_data/prelim/split/after/Ybest_split.txt",
        delimiter=",",
    )
    p_after = np.genfromtxt(
        script_dir / "test_data/prelim/split/after/P_split.txt",
        delimiter=",",
    )
    inst_labels_after = pd.read_csv(
        script_dir / "test_data/prelim/split/after/instlabels_split.txt",
        header=None,
    ).loc[:, 0]

    assert np.array_equal(x, x_after)
    assert np.array_equal(y, y_after)
    assert np.array_equal(x_raw, x_raw_after)
    assert np.array_equal(y_raw, y_raw_after)
    assert np.array_equal(y_bin, y_bin_after)
    assert np.array_equal(beta, beta_after)
    assert np.array_equal(num_good_algos, num_good_algos_after)
    assert np.array_equal(y_best, y_best_after)
    assert np.array_equal(p, p_after)
    assert np.array_equal(inst_labels, inst_labels_after)
    print("Split data tests passed!")


def test_split_fractional() -> None:
    """A fractional split returns exactly the rows selected by its mask."""
    # Create options for fractional split

    prelim_opts = PrelimOptions(
        max_perf=False,
        abs_perf=True,
        epsilon=0.20,
        beta_threshold=0.55,
        bound=True,
        norm=True,
    )

    selvars_opts = SelvarsOptions(
        small_scale_flag=True,
        small_scale=0.50,
        file_idx_flag=False,
        file_idx="",
        feats=None,
        algos=None,
        selvars_type="Ftr&Good",
        min_distance=0.1,
        density_flag=False,
    )

    x_before = np.genfromtxt(
        script_dir / "test_data/prelim/fractional/before/x_split.txt",
        delimiter=",",
    )
    y_before = np.genfromtxt(
        script_dir / "test_data/prelim/fractional/before/Y_split.txt",
        delimiter=",",
    )
    x_raw_before = np.genfromtxt(
        script_dir / "test_data/prelim/fractional/before/Xraw_split.txt",
        delimiter=",",
    )
    y_raw_before = np.genfromtxt(
        script_dir / "test_data/prelim/fractional/before/Yraw_split.txt",
        delimiter=",",
    )
    y_bin_before = np.genfromtxt(
        script_dir / "test_data/prelim/fractional/before/Ybin_split.txt",
        delimiter=",",
    )
    beta_before = np.genfromtxt(
        script_dir / "test_data/prelim/fractional/before/beta_split.txt",
        delimiter=",",
    )
    num_good_algos_before = np.genfromtxt(
        script_dir / "test_data/prelim/fractional/before/numGoodAlgos_split.txt",
        delimiter=",",
    )
    y_best_before = np.genfromtxt(
        script_dir / "test_data/prelim/fractional/before/Ybest_split.txt",
        delimiter=",",
    )
    p_before = np.genfromtxt(
        script_dir / "test_data/prelim/fractional/before/P_split.txt",
        delimiter=",",
    )
    inst_labels_before = pd.read_csv(
        script_dir / "test_data/prelim/fractional/before/instlabels_split.txt",
        header=None,
    ).loc[:, 0]

    s_before = None

    prelim = PrelimStage(
        x_before,
        y_before,
        x_raw_before,
        y_raw_before,
        s_before,
        inst_labels_before,
        prelim_opts,
        selvars_opts,
        GeneralOptions.default(),
    )

    (
        subset_index,
        x,
        y,
        x_raw,
        y_raw,
        y_bin,
        beta,
        num_good_algos,
        y_best,
        p,
        inst_labels,
        s,
        data_dense,
    ) = prelim._filter(
        inst_labels_before,
        x_before,
        y_before,
        y_bin_before,
        y_best_before,
        x_raw_before,
        y_raw_before,
        p_before,
        num_good_algos_before,
        beta_before,
        s_before,
        selvars_opts,
    )

    assert subset_index.dtype == np.bool_
    assert subset_index.shape == (x_before.shape[0],)
    assert np.count_nonzero(subset_index) == x_before.shape[0] // 2
    np.testing.assert_array_equal(x, x_before[subset_index])
    np.testing.assert_array_equal(y, y_before[subset_index])
    np.testing.assert_array_equal(x_raw, x_raw_before[subset_index])
    np.testing.assert_array_equal(y_raw, y_raw_before[subset_index])
    np.testing.assert_array_equal(y_bin, y_bin_before[subset_index])
    np.testing.assert_array_equal(beta, beta_before[subset_index])
    np.testing.assert_array_equal(
        num_good_algos,
        num_good_algos_before[subset_index],
    )
    np.testing.assert_array_equal(y_best, y_best_before[subset_index])
    np.testing.assert_array_equal(p, p_before[subset_index])
    pd.testing.assert_series_equal(inst_labels, inst_labels_before[subset_index])
    assert s is None
    assert data_dense is None


def test_split_fileindexed() -> None:
    """Test case for the split data function by using fileindexed option."""
    # Create options for fileindexed split

    prelim_opts = PrelimOptions(
        max_perf=False,
        abs_perf=True,
        epsilon=0.20,
        beta_threshold=0.55,
        bound=True,
        norm=True,
    )

    selvars_opts = SelvarsOptions(
        small_scale_flag=False,
        small_scale=0.50,
        file_idx_flag=True,
        file_idx="./tests/test_data/prelim/fileidx/fileidx.csv",
        feats=None,
        algos=None,
        selvars_type="Ftr&Good",
        min_distance=0.1,
        density_flag=False,
    )

    x_before = np.genfromtxt(
        script_dir / "test_data/prelim/fileidx/before/x_split.txt",
        delimiter=",",
    )
    y_before = np.genfromtxt(
        script_dir / "test_data/prelim/fileidx/before/Y_split.txt",
        delimiter=",",
    )
    x_raw_before = np.genfromtxt(
        script_dir / "test_data/prelim/fileidx/before/Xraw_split.txt",
        delimiter=",",
    )
    y_raw_before = np.genfromtxt(
        script_dir / "test_data/prelim/fileidx/before/Yraw_split.txt",
        delimiter=",",
    )
    y_bin_before = np.genfromtxt(
        script_dir / "test_data/prelim/fileidx/before/Ybin_split.txt",
        delimiter=",",
    )
    beta_before = np.genfromtxt(
        script_dir / "test_data/prelim/fileidx/before/beta_split.txt",
        delimiter=",",
    )
    num_good_algos_before = np.genfromtxt(
        script_dir / "test_data/prelim/fileidx/before/numGoodAlgos_split.txt",
        delimiter=",",
    )
    y_best_before = np.genfromtxt(
        script_dir / "test_data/prelim/fileidx/before/Ybest_split.txt",
        delimiter=",",
    )
    p_before = np.genfromtxt(
        script_dir / "test_data/prelim/fileidx/before/P_split.txt",
        delimiter=",",
    )
    inst_labels_before = pd.read_csv(
        script_dir / "test_data/prelim/fileidx/before/instlabels_split.txt",
        header=None,
    ).loc[:, 0]

    s_before = None

    prelim = PrelimStage(
        x_before,
        y_before,
        x_raw_before,
        y_raw_before,
        s_before,
        inst_labels_before,
        prelim_opts,
        selvars_opts,
        GeneralOptions.default(),
    )

    (
        subset_index,
        x,
        y,
        x_raw,
        y_raw,
        y_bin,
        beta,
        num_good_algos,
        y_best,
        p,
        inst_labels,
        s,
        data_dense,
    ) = prelim._filter(
        inst_labels_before,
        x_before,
        y_before,
        y_bin_before,
        y_best_before,
        x_raw_before,
        y_raw_before,
        p_before,
        num_good_algos_before,
        beta_before,
        s_before,
        selvars_opts,
    )

    x_after = np.genfromtxt(
        script_dir / "test_data/prelim/fileidx/after/x_split.txt",
        delimiter=",",
    )
    y_after = np.genfromtxt(
        script_dir / "test_data/prelim/fileidx/after/Y_split.txt",
        delimiter=",",
    )
    x_raw_after = np.genfromtxt(
        script_dir / "test_data/prelim/fileidx/after/Xraw_split.txt",
        delimiter=",",
    )
    y_raw_after = np.genfromtxt(
        script_dir / "test_data/prelim/fileidx/after/Yraw_split.txt",
        delimiter=",",
    )
    y_bin_after = np.genfromtxt(
        script_dir / "test_data/prelim/fileidx/after/Ybin_split.txt",
        delimiter=",",
    )
    beta_after = np.genfromtxt(
        script_dir / "test_data/prelim/fileidx/after/beta_split.txt",
        delimiter=",",
    )
    num_good_algos_after = np.genfromtxt(
        script_dir / "test_data/prelim/fileidx/after/numGoodAlgos_split.txt",
        delimiter=",",
    )
    y_best_after = np.genfromtxt(
        script_dir / "test_data/prelim/fileidx/after/Ybest_split.txt",
        delimiter=",",
    )
    p_after = np.genfromtxt(
        script_dir / "test_data/prelim/fileidx/after/P_split.txt",
        delimiter=",",
    )
    inst_labels_after = pd.read_csv(
        script_dir / "test_data/prelim/fileidx/after/instlabels_split.txt",
        header=None,
    ).loc[:, 0]

    assert np.array_equal(x, x_after)
    assert np.array_equal(y, y_after)
    assert np.array_equal(x_raw, x_raw_after)
    assert np.array_equal(y_raw, y_raw_after)
    assert np.array_equal(y_bin, y_bin_after)
    assert np.array_equal(beta, beta_after)
    assert np.array_equal(num_good_algos, num_good_algos_after)
    assert np.array_equal(y_best, y_best_after)
    assert np.array_equal(p, p_after)
    assert np.array_equal(inst_labels, inst_labels_after)
    print("Fileindexed tests passed!")


def _run_file_index_filter(
    index_path: Path,
    ninst: int = 4,
) -> tuple[np.ndarray, pd.Series]:  # type: ignore[type-arg]
    """Run PRELIM's file-indexed filter over a compact synthetic dataset."""
    x = np.arange(ninst * 2, dtype=float).reshape(ninst, 2)
    y = np.arange(ninst, dtype=float)[:, np.newaxis]
    y_bin = np.ones_like(y, dtype=bool)
    y_best = y[:, 0]
    p = np.ones(ninst, dtype=int)
    num_good_algos = np.ones(ninst)
    beta = np.zeros(ninst, dtype=bool)
    labels = pd.Series([f"i{i + 1}" for i in range(ninst)])
    options = SelvarsOptions.default(
        file_idx_flag=True,
        file_idx=str(index_path),
    )
    stage = PrelimStage(
        x,
        y,
        x.copy(),
        y.copy(),
        None,
        labels,
        PrelimOptions(False, True, 0.2, 0.55, False, False),
        options,
        GeneralOptions.default(),
    )
    result = stage._filter(
        labels,
        x,
        y,
        y_bin,
        y_best,
        x.copy(),
        y.copy(),
        p,
        num_good_algos,
        beta,
        None,
        options,
    )
    return result[0], result[10]


def test_file_indices_accept_first_and_last_matlab_indices(tmp_path: Path) -> None:
    """MATLAB's inclusive 1..ninst range maps once to Python positions."""
    index_path = tmp_path / "indices.csv"
    index_path.write_text("1\n4\n", encoding="utf-8")

    result = _run_file_index_filter(index_path)

    np.testing.assert_array_equal(result[0], [True, False, False, True])
    assert result[1].tolist() == ["i1", "i4"]


def test_file_indices_support_scalar_files(tmp_path: Path) -> None:
    """A one-entry subset file is treated as a one-element vector."""
    index_path = tmp_path / "scalar.csv"
    index_path.write_text("4\n", encoding="utf-8")

    result = _run_file_index_filter(index_path)

    np.testing.assert_array_equal(result[0], [False, False, False, True])
    assert result[1].tolist() == ["i4"]


@pytest.mark.parametrize("value", ["0", "-1", "5", "1.5"])
def test_file_indices_reject_invalid_values(tmp_path: Path, value: str) -> None:
    """Zero, negative, out-of-range, and fractional indices fail clearly."""
    index_path = tmp_path / "invalid.csv"
    index_path.write_text(f"{value}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Subset ind"):
        _run_file_index_filter(index_path)


def test_missing_file_index_path_fails_clearly(tmp_path: Path) -> None:
    """An enabled but missing subset file never silently selects all rows."""
    missing = tmp_path / "missing.csv"

    with pytest.raises(FileNotFoundError, match="Subset index file does not exist"):
        _run_file_index_filter(missing)


# Tao to complete this test
# def test_split_bydensity() -> None:
#     """Test case for the split data function by using bydensity option."""
