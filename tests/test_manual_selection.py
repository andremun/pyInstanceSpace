"""
Contains test cases for the remove_instances_with_many_missing_values function.

These testing codes are tested by artificial data
(the data that I generated, rather than read from CSV)
and check against with the logic of original codes of BuildIS

"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from matilda.build import select_features_and_algorithms
from matilda.data.model import Data
from matilda.data.option import (
    AutoOptions,
    BoundOptions,
    CloisterOptions,
    NormOptions,
    Options,
    OutputOptions,
    ParallelOptions,
    PerformanceOptions,
    PilotOptions,
    PythiaOptions,
    SelvarsOptions,
    SiftedOptions,
    TraceOptions,
)

path_root = Path(__file__).parent
sys.path.append(str(path_root))


def create_dummy_opt(selvars: SelvarsOptions) -> Options:
    """Create a dummy model with the given data and selection variables."""
    return Options(
        parallel=ParallelOptions(flag=False, n_cores=1),
        perf=PerformanceOptions(max_perf=False, abs_perf=False,
                                epsilon=0.1, beta_threshold=0.5),
        auto=AutoOptions(preproc=False),
        bound=BoundOptions(flag=False),
        norm=NormOptions(flag=False),
        selvars=selvars,
        sifted=SiftedOptions(flag=False, rho=0.5, k=10,
                             n_trees=100, max_iter=100, replicates=10),
        pilot=PilotOptions(analytic=False, n_tries=10),
        cloister=CloisterOptions(p_val=0.05, c_thres=0.5),
        pythia=PythiaOptions(cv_folds=5, is_poly_krnl=False,
                             use_weights=False, use_lib_svm=False),
        trace=TraceOptions(use_sim=False, PI=0.95),
        outputs=OutputOptions(csv=False, web=False, png=False),
    )


def test_manual_selection() -> None:
    """
    The test case for select_features_and_algorithms.

    Main success scenario, no error
    """
    rng = np.random.default_rng()
    large_x = rng.random((100, 10))  # 100 rows, 10 features (columns)
    large_y = rng.random((100, 5))  # 100 rows, 5 features (columns)

    data = Data(
        inst_labels=pd.Series(),
        feat_labels=[f"feature{i}" for i in range(10)],
        algo_labels=[f"algo{i}" for i in range(5)],
        x=large_x,
        y=large_y,
        x_raw=np.array([], dtype=np.double),
        y_raw=np.array([], dtype=np.double),
        y_bin=np.array([], dtype=np.bool_),
        y_best=np.array([], dtype=np.double),
        p=np.array([], dtype=np.double),
        num_good_algos=np.array([], dtype=np.double),
        beta=np.array([], dtype=np.bool_),
        s=set(),
    )

    selvars = SelvarsOptions(
        feats=["feature1", "feature3", "feature5", "feature7", "feature9"],
        algos=["algo1", "algo3"],
        small_scale_flag=False,
        small_scale=0.1,
        file_idx_flag=False,
        file_idx="",
        type="",
        min_distance=0.0,
        density_flag=False,
    )

    opts = create_dummy_opt(selvars)

    out = select_features_and_algorithms(data, opts)

    assert out.feat_labels == ["feature1", "feature3", "feature5", "feature7",
                               "feature9"], "Feature selection failed"
    assert out.algo_labels == ["algo1", "algo3"], "Algorithm selection failed"

    # check the contents
    expected_x = large_x[:, [1, 3, 5, 7, 9]]
    expected_y = large_y[:, [1, 3]]
    np.testing.assert_array_equal(out.x, expected_x,
                                  err_msg="Feature data content mismatch")
    np.testing.assert_array_equal(out.y, expected_y,
                                  err_msg="Algorithm data content mismatch")


def test_manual_wrong_names() -> None:
    """
    The test case for select_features_and_algorithms.

    Main success scenario, no error
    """
    rng = np.random.default_rng()
    large_x = rng.random((100, 10))  # 100 rows, 10 features (columns)
    large_y = rng.random((100, 5))  # 100 rows, 5 features (columns)

    data = Data(
        inst_labels=pd.Series(),
        feat_labels=[f"feature{i}" for i in range(10)],
        algo_labels=[f"algo{i}" for i in range(5)],
        x=large_x,
        y=large_y,
        x_raw=np.array([], dtype=np.double),
        y_raw=np.array([], dtype=np.double),
        y_bin=np.array([], dtype=np.bool_),
        y_best=np.array([], dtype=np.double),
        p=np.array([], dtype=np.double),
        num_good_algos=np.array([], dtype=np.double),
        beta=np.array([], dtype=np.bool_),
        s=set(),
    )

    selvars = SelvarsOptions(
        feats=["feature1", "feature3", "feature5", "featu", "feature9"],
        algos=["al", "algo3"],
        small_scale_flag=False,
        small_scale=0.1,
        file_idx_flag=False,
        file_idx="",
        type="",
        min_distance=0.0,
        density_flag=False,
    )

    opts = create_dummy_opt(selvars)

    out = select_features_and_algorithms(data, opts)

    assert out.feat_labels == ["feature1", "feature3", "feature5",
                               "feature9"], "Feature selection failed"
    assert out.algo_labels == ["algo3"], "Algorithm selection failed"

    # check the contents
    expected_x = large_x[:, [1, 3, 5, 9]]
    expected_y = large_y[:, [3]]
    np.testing.assert_array_equal(out.x, expected_x,
                                  err_msg="Feature data content mismatch")
    np.testing.assert_array_equal(out.y, expected_y,
                                  err_msg="Algorithm data content mismatch")


def test_manual_empty_feats() -> None:
    """
    The test case for select_features_and_algorithms.

    Main success scenario, no error
    """
    rng = np.random.default_rng(33)
    large_x = rng.random((100, 10))  # 100 rows, 10 features (columns)
    large_y = rng.random((100, 5))  # 100 rows, 5 features (columns)

    data = Data(
        inst_labels=pd.Series(),
        feat_labels=[f"feature{i}" for i in range(10)],
        algo_labels=[f"algo{i}" for i in range(5)],
        x=large_x,
        y=large_y,
        x_raw=np.array([], dtype=np.double),
        y_raw=np.array([], dtype=np.double),
        y_bin=np.array([], dtype=np.bool_),
        y_best=np.array([], dtype=np.double),
        p=np.array([], dtype=np.double),
        num_good_algos=np.array([], dtype=np.double),
        beta=np.array([], dtype=np.bool_),
        s=set(),
    )

    selvars = SelvarsOptions(
        feats=[],
        algos=[],
        small_scale_flag=False,
        small_scale=0.1,
        file_idx_flag=False,
        file_idx="",
        type="",
        min_distance=0.0,
        density_flag=False,
    )

    opts = create_dummy_opt(selvars)

    out = select_features_and_algorithms(data, opts)

    assert out.feat_labels == [f"feature{i}" for i in range(10)], \
        "Feature selection failed"
    assert out.algo_labels == [f"algo{i}" for i in range(5)], \
        "Algorithm selection failed"

    # check the contents
    expected_x = large_x[:, :]
    expected_y = large_y[:, :]
    np.testing.assert_array_equal(out.x, expected_x,
                                  err_msg="Feature data content mismatch")
    np.testing.assert_array_equal(out.y, expected_y,
                                  err_msg="Algorithm data content mismatch")


if __name__ == "__main__":
    test_manual_selection()
    test_manual_wrong_names()
    test_manual_empty_feats()
