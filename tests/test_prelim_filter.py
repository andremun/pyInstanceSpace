"""Test module for filter functionality post Prelim class to verify its functionality.

The file contains multiple unit tests to ensure that the `filter` function correctly
performs its tasks. The basic mechanism of the test is to compare its output against
output from MATLAB and check if the outputs are the same or reasonable similar.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from instancespace.data.options import PrelimOptions, SelvarsOptions
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
    ) = prelim._filter(  # noqa: SLF001
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
    """Test case for the split data function by using fractional option."""
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

    data_dense = None

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
    ) = prelim._filter(  # noqa: SLF001
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
        script_dir / "test_data/prelim/fractional/after/x_split.txt",
        delimiter=",",
    )
    y_after = np.genfromtxt(
        script_dir / "test_data/prelim/fractional/after/Y_split.txt",
        delimiter=",",
    )
    x_raw_after = np.genfromtxt(
        script_dir / "test_data/prelim/fractional/after/Xraw_split.txt",
        delimiter=",",
    )
    y_raw_after = np.genfromtxt(
        script_dir / "test_data/prelim/fractional/after/Yraw_split.txt",
        delimiter=",",
    )
    y_bin_after = np.genfromtxt(
        script_dir / "test_data/prelim/fractional/after/Ybin_split.txt",
        delimiter=",",
    )
    beta_after = np.genfromtxt(
        script_dir / "test_data/prelim/fractional/after/beta_split.txt",
        delimiter=",",
    )
    num_good_algos_after = np.genfromtxt(
        script_dir / "test_data/prelim/fractional/after/numGoodAlgos_split.txt",
        delimiter=",",
    )
    y_best_after = np.genfromtxt(
        script_dir / "test_data/prelim/fractional/after/Ybest_split.txt",
        delimiter=",",
    )
    p_after = np.genfromtxt(
        script_dir / "test_data/prelim/fractional/after/P_split.txt",
        delimiter=",",
    )
    inst_labels_after = pd.read_csv(
        script_dir / "test_data/prelim/fractional/after/instlabels_split.txt",
        header=None,
    ).loc[:, 0]

    assert np.array_equal(x.shape, x_after.shape)
    assert np.array_equal(y.shape, y_after.shape)
    assert np.array_equal(x_raw.shape, x_raw_after.shape)
    assert np.array_equal(y_raw.shape, y_raw_after.shape)
    assert np.array_equal(y_bin.shape, y_bin_after.shape)
    assert np.array_equal(beta.shape, beta_after.shape)
    assert np.array_equal(num_good_algos.shape, num_good_algos_after.shape)
    assert np.array_equal(y_best.shape, y_best_after.shape)
    assert np.array_equal(p.shape, p_after.shape)
    assert np.array_equal(inst_labels.shape, inst_labels_after.shape)
    print("Fractional tests passed!")


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
    ) = prelim._filter(  # noqa: SLF001
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


# Tao to complete this test
# def test_split_bydensity() -> None:
#     """Test case for the split data function by using bydensity option."""
