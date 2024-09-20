from pathlib import Path

import numpy as np
import pandas as pd

from matilda.stages.prelim_stage import PrelimStage

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
    uniformity_before = None

    max_perf = False
    abs_perf = True
    epsilon = 0.20
    beta_threshold = 0.55
    bound = True
    norm = True

    small_scale_flag = False
    small_scale = 0.50
    file_idx_flag = False
    file_idx = ""
    selvars_type = "Ftr&Good"
    min_distance = 0.1
    density_flag = False
    feats = None
    algos = None
    data_dense_before = None

    prelim = PrelimStage(
        x_before,
        y_before,
        max_perf,
        abs_perf,
        epsilon,
        beta_threshold,
        bound,
        norm,
        small_scale_flag,
        small_scale,
        file_idx_flag,
        file_idx,
        feats,
        algos,
        selvars_type,
        density_flag,
        min_distance,
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
    ) = prelim.filter(
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
        data_dense_before,
        small_scale_flag,
        small_scale,
        file_idx_flag,
        file_idx,
        selvars_type,
        min_distance,
        density_flag,
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

    max_perf = False
    abs_perf = True
    epsilon = 0.20
    beta_threshold = 0.55
    bound = True
    norm = True

    small_scale_flag = True  # fractional
    small_scale = 0.50
    file_idx_flag = False
    file_idx = ""
    selvars_type = "Ftr&Good"
    min_distance = 0.1
    density_flag = False
    feats = None
    algos = None
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
    data_dense_before = None

    prelim = PrelimStage(
        x_before,
        y_before,
        max_perf,
        abs_perf,
        epsilon,
        beta_threshold,
        bound,
        norm,
        small_scale_flag,
        small_scale,
        file_idx_flag,
        file_idx,
        feats,
        algos,
        selvars_type,
        density_flag,
        min_distance,
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
    ) = prelim.filter(
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
        data_dense_before,
        small_scale_flag,
        small_scale,
        file_idx_flag,
        file_idx,
        selvars_type,
        min_distance,
        density_flag,
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

    max_perf = False
    abs_perf = True
    epsilon = (0.20,)
    beta_threshold = 0.55
    bound = True
    norm = True

    small_scale_flag = False
    small_scale = 0.50
    file_idx_flag = True
    file_idx = "./tests/test_data/prelim/fileidx/fileidx.csv"
    selvars_type = "Ftr&Good"
    min_distance = 0.1
    density_flag = False
    feats = []
    algos = []
    data_dense = None

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
    data_dense_before = None

    prelim = PrelimStage(
        x_before,
        y_before,
        max_perf,
        abs_perf,
        epsilon,
        beta_threshold,
        bound,
        norm,
        small_scale_flag,
        small_scale,
        file_idx_flag,
        file_idx,
        feats,
        algos,
        selvars_type,
        density_flag,
        min_distance,
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
    ) = prelim.filter(
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
        data_dense_before,
        small_scale_flag,
        small_scale,
        file_idx_flag,
        file_idx,
        selvars_type,
        min_distance,
        density_flag,
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
