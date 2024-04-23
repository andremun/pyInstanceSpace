"""Test cases for the select_features_and_algorithms function."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from matilda.build import select_features_and_algorithms
from matilda.data.model import (
    CloisterOut,
    Data,
    FeatSel,
    Footprint,
    Model,
    PilotOut,
    PolyShape,
    PrelimOut,
    PythiaOut,
    SiftedOut,
    TraceOut,
)
from matilda.data.option import (
    AutoOptions,
    BoundOptions,
    CloisterOptions,
    NormOptions,
    Opts,
    OutputOptions,
    ParallelOptions,
    PerformanceOptions,
    PilotOptions,
    PythiaOptions,
    SelvarsOptions,
    SiftedOptions,
    TraceOptions,
)

path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))


def create_dummy_model(data: Data, selvars: SelvarsOptions) -> Model:
    """Create a dummy model with the given data and selection variables."""
    empty_feat_sel = FeatSel(idx=np.array([], dtype=np.intc))

    opts = Opts(
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

    empty_prelim = PrelimOut(
        med_val=np.array([], dtype=np.double),
        iq_range=np.array([], dtype=np.double),
        hi_bound=np.array([], dtype=np.double),
        lo_bound=np.array([], dtype=np.double),
        min_x=np.array([], dtype=np.double),
        lambda_x=np.array([], dtype=np.double),
        mu_x=np.array([], dtype=np.double),
        sigma_x=np.array([], dtype=np.double),
        min_y=np.array([], dtype=np.double),
        lambda_y=np.array([], dtype=np.double),
        sigma_y=np.array([], dtype=np.double),
    )
    empty_sifted = SiftedOut(flag=0, rho=np.double(0), k=0,
                             n_trees=0, max_lter=0, replicates=0)

    empty_pilot = PilotOut(
        X0=np.array([], dtype=np.double),
        alpha=np.array([], dtype=np.double),
        eoptim=np.array([], dtype=np.double),
        perf=np.array([], dtype=np.double),
        a=np.array([], dtype=np.double),
        z=np.array([], dtype=np.double),
        c=np.array([], dtype=np.double),
        b=np.array([], dtype=np.double),
        error=np.array([], dtype=np.double),
        r2=np.array([], dtype=np.double),
        summary=pd.DataFrame(),
    )
    empty_cloist = CloisterOut(Zedge=np.array([], dtype=np.double),
                               Zecorr=np.array([], dtype=np.double))
    empty_pythia = PythiaOut(
        mu=[],
        sigma=[],
        cp=None,
        svm=None,
        cvcmat=np.array([], dtype=np.double),
        y_sub=np.array([], dtype=np.bool_),
        y_hat=np.array([], dtype=np.bool_),
        pr0_sub=np.array([], dtype=np.double),
        pr0_hat=np.array([], dtype=np.double),
        box_consnt=[],
        k_scale=[],
        precision=[],
        recall=[],
        accuracy=[],
        selection0=np.array([], dtype=np.double),
        selection1=None,
        summary=pd.DataFrame(),
    )

    poly = PolyShape()
    dummy = Footprint(
        polygon=poly,
        area=float(0),
        elements=float(0),
        good_elements=float(0),
        density=float(0),
        purity=float(0),
    )
    empty_trace = TraceOut(space=dummy, good=[], best=[],
                           hard=dummy, summary=pd.DataFrame())

    return Model(
        data=data,
        data_dense=data,
        feat_sel=empty_feat_sel,
        prelim=empty_prelim,
        sifted=empty_sifted,
        pilot=empty_pilot,
        cloist=empty_cloist,
        pythia=empty_pythia,
        trace=empty_trace,
        opts=opts,
    )


def test_manual_selection() -> None:
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

    model = create_dummy_model(data, selvars)

    select_features_and_algorithms(model, model.opts)

    assert model.data.feat_labels == ["feature1", "feature3", "feature5", "feature7",
                                      "feature9"], "Feature selection failed"
    assert model.data.algo_labels == ["algo1", "algo3"], "Algorithm selection failed"

    # check the contents
    expected_x = large_x[:, [1, 3, 5, 7, 9]]
    expected_y = large_y[:, [1, 3]]
    np.testing.assert_array_equal(model.data.x, expected_x,
                                  err_msg="Feature data content mismatch")
    np.testing.assert_array_equal(model.data.y, expected_y,
                                  err_msg="Algorithm data content mismatch")


def test_manual_wrong_names() -> None:
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

    model = create_dummy_model(data, selvars)

    select_features_and_algorithms(model, model.opts)

    assert model.data.feat_labels == ["feature1", "feature3", "feature5",
                                      "feature9"], "Feature selection failed"
    assert model.data.algo_labels == ["algo3"], "Algorithm selection failed"

    # check the contents
    expected_x = large_x[:, [1, 3, 5, 9]]
    expected_y = large_y[:, [3]]
    np.testing.assert_array_equal(model.data.x, expected_x,
                                  err_msg="Feature data content mismatch")
    np.testing.assert_array_equal(model.data.y, expected_y,
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

    model = create_dummy_model(data, selvars)

    select_features_and_algorithms(model, model.opts)

    assert model.data.feat_labels == [f"feature{i}" for i in range(10)], \
        "Feature selection failed"
    assert model.data.algo_labels == [f"algo{i}" for i in range(5)], \
        "Algorithm selection failed"

    # check the contents
    expected_x = large_x[:, :]
    expected_y = large_y[:, :]
    np.testing.assert_array_equal(model.data.x, expected_x,
                                  err_msg="Feature data content mismatch")
    np.testing.assert_array_equal(model.data.y, expected_y,
                                  err_msg="Algorithm data content mismatch")


if __name__ == "__main__":
    test_manual_selection()
    test_manual_wrong_names()
    test_manual_empty_feats()
