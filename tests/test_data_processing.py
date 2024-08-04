"""Containing test cases for the data processing functions."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

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
from matilda.data.options import (
    AutoOptions,
    BoundOptions,
    CloisterOptions,
    InstanceSpaceOptions,
    NormOptions,
    OutputOptions,
    ParallelOptions,
    PerformanceOptions,
    PilotOptions,
    PythiaOptions,
    SelvarsOptions,
    SiftedOptions,
    TraceOptions,
)
from matilda.stages.preprocessing import Preprocessing

path_root = Path(__file__).parent
sys.path.append(str(path_root.parent))


def create_dummy_opt() -> InstanceSpaceOptions:
    """
    Create a dummy Options object with default values.

    Returns:
    -------
        Options: The dummy Options object.
    """
    return InstanceSpaceOptions(
        parallel=ParallelOptions(flag=False, n_cores=2),
        perf=PerformanceOptions(
            max_perf=False,
            abs_perf=True,
            epsilon=0.20,
            beta_threshold=0.55,
        ),
        auto=AutoOptions(preproc=True),
        bound=BoundOptions(flag=True),
        norm=NormOptions(flag=True),
        selvars=SelvarsOptions(
            small_scale_flag=False,
            small_scale=0.50,
            file_idx_flag=False,
            file_idx="",
            selvars_type="Ftr&Good",
            min_distance=0.1,
            density_flag=False,
            feats=None,
            algos=None,
        ),
        sifted=SiftedOptions(
            flag=True,
            rho=0.1,
            k=10,
            n_trees=50,
            max_iter=1000,
            replicates=100,
        ),
        pilot=PilotOptions(analytic=False, n_tries=5),
        cloister=CloisterOptions(p_val=0.05, c_thres=0.7),
        pythia=PythiaOptions(
            cv_folds=5,
            is_poly_krnl=False,
            use_weights=False,
            use_lib_svm=False,
        ),
        trace=TraceOptions(use_sim=True, pi=0.55),
        outputs=OutputOptions(csv=True, web=False, png=True),
    )


def create_dummy_model(data: Data, opts: InstanceSpaceOptions) -> Model:
    """
    Create a dummy Model object with default values.

    Returns:
    -------
        Model: The dummy Model object.
    """
    feat_sel = FeatSel(
        idx=np.array([], dtype=np.int64),
    )

    prelim = PrelimOut(
        med_val=np.array([], dtype=np.double),
        iq_range=np.array([], dtype=np.double),
        hi_bound=np.array([], dtype=np.double),
        lo_bound=np.array([], dtype=np.double),
        min_x=np.array([], dtype=np.double),
        lambda_x=np.array([], dtype=np.double),
        mu_x=np.array([], dtype=np.double),
        sigma_x=np.array([], dtype=np.double),
        min_y=0.0,
        lambda_y=np.array([], dtype=np.double),
        sigma_y=np.array([], dtype=np.double),
        mu_y=np.array([], dtype=np.double),
    )

    sifted = SiftedOut(
        flag=1,
        rho=np.double(0.1),
        k=10,
        n_trees=50,
        max_lter=1000,
        replicates=100,
    )

    pilot = PilotOut(
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

    cloist = CloisterOut(
        z_edge=np.array([], dtype=np.double),
        z_ecorr=np.array([], dtype=np.double),
    )

    pythia = PythiaOut(
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
    footprint = Footprint(
        polygon=PolyShape(),
        area=0.0,
        elements=0,
        good_elements=0,
        density=0.0,
        purity=0.0,
    )

    trace = TraceOut(
        space=footprint,
        good=[],
        best=[],
        hard=footprint,
        summary=pd.DataFrame(),
    )

    return Model(
        data=data,
        opts=opts,
        feat_sel=feat_sel,
        data_dense=data,
        prelim=prelim,
        sifted=sifted,
        pilot=pilot,
        cloist=cloist,
        pythia=pythia,
        trace=trace,
    )


def test_process_data() -> None:
    """
    Test case for the process_data function.

    expected: No assertion errors.
    input: data.X, data.Y, data.featlabels, data.algolabels, opts.perf,
        opts.bound.flag, opts.norm.flag
    values that need to be checked:
        model.data.x, model.data.y, opts.prelim, data.featlabels, data.algolabels
    """
    with Path.open(path_root / "process_Data/featlabels_before.txt") as file:
        line = file.readline()
        feat_labels_before = line.strip().split(",")
    with Path.open(path_root / "process_Data/featlabels_after.txt") as file:
        line = file.readline()
        feat_labels_after = line.strip().split(",")
    with Path.open(path_root / "process_Data/algolabels_before.txt") as file:
        line = file.readline()
        algo_labels_before = line.strip().split(",")
    with Path.open(path_root / "process_Data/algolabels_after.txt") as file:
        line = file.readline()
        algo_labels_after = line.strip().split(",")

    x_before = np.genfromtxt(path_root / "process_Data/X_before.csv", delimiter=",")
    y_before = np.genfromtxt(path_root / "process_Data/Y_before.csv", delimiter=",")
    x_after = np.genfromtxt(path_root / "process_Data/X_after.csv", delimiter=",")
    y_after = np.genfromtxt(path_root / "process_Data/Y_after.csv", delimiter=",")

    data = Data(
        inst_labels=pd.Series(),
        feat_labels=feat_labels_before,
        algo_labels=algo_labels_before,
        x=x_before,
        y=y_before,
        x_raw=np.array([], dtype=np.double),
        y_raw=np.array([], dtype=np.double),
        y_bin=np.array([], dtype=np.bool_),
        y_best=np.array([], dtype=np.double),
        p=np.array([], dtype=np.double),
        num_good_algos=np.array([], dtype=np.double),
        beta=np.array([], dtype=np.bool_),
        s=None,
        uniformity=None,
    )

    opts = create_dummy_opt()

    returned_data, prelim_opts = Preprocessing.process_data(data, opts)

    assert np.array_equal(returned_data.x, x_after)
    assert np.array_equal(returned_data.y, y_after)
    assert np.array_equal(returned_data.algo_labels, algo_labels_after)
    assert np.array_equal(returned_data.feat_labels, feat_labels_after)
    # Check if the prelim options are set correctly, the values taken from the
    # prelim_opts should be the same as before.
    assert np.array_equal(prelim_opts.max_perf, False)
    assert np.array_equal(prelim_opts.abs_perf, True)
    assert np.array_equal(prelim_opts.epsilon, 0.20)
    assert np.array_equal(prelim_opts.beta_threshold, 0.55)
    assert np.array_equal(prelim_opts.bound, True)
    assert np.array_equal(prelim_opts.norm, True)
    print("Process data tests passed!")


def test_remove_bad_instances_1() -> None:
    """
    Test case for testing if the function can remove_bad_instances.

    According to the matlab example, the function should not remove any instances.

    expected: No assertion errors.
    """
    # Read the data from the files.
    y_bin = np.genfromtxt(
        path_root / "Prelim_out/model-data-ybin.csv",
        delimiter=",",
    )
    y_bin = y_bin.astype(np.bool_)
    data = Data(
        inst_labels=pd.Series(),
        feat_labels=[],
        algo_labels=[],
        x=np.array([], dtype=np.double),
        y=np.array([], dtype=np.double),
        x_raw=np.array([], dtype=np.double),
        y_raw=np.array([], dtype=np.double),
        y_bin=y_bin,
        y_best=np.array([], dtype=np.double),
        p=np.array([], dtype=np.double),
        num_good_algos=np.array([], dtype=np.double),
        beta=np.array([], dtype=np.bool_),
        s=None,
        uniformity=None,
    )
    data = Preprocessing.remove_bad_instances(data)
    assert data.y_bin.shape == y_bin.shape
    print("Remove bad instances tests 1 (matlab example) passed!")


def test_remove_bad_instances_2() -> None:
    """
    In this test case, I make the y_bin array is all False, so all instances should be removed.

    expected: No assertion errors.
    """  # noqa: E501
    y_bin = np.genfromtxt(
        path_root / "Prelim_out/model-data-ybin.csv",
        delimiter=",",
    )
    y_bin = np.zeros_like(y_bin, dtype=np.bool_)

    x = np.genfromtxt(path_root / "process_Data/X_before.csv", delimiter=",")
    y = np.genfromtxt(path_root / "process_Data/Y_before.csv", delimiter=",")
    with Path.open(path_root / "process_Data/algolabels_after.txt") as file:
        line = file.readline()
        algo_labels = line.strip().split(",")

    data = Data(
        inst_labels=pd.Series(),
        feat_labels=[],
        algo_labels=algo_labels,
        x=x,
        y=y,
        x_raw=x,
        y_raw=y,
        y_bin=y_bin,
        y_best=np.array([], dtype=np.double),
        p=np.array([], dtype=np.double),
        num_good_algos=np.array([], dtype=np.double),
        beta=np.array([], dtype=np.bool_),
        s=None,
        uniformity=None,
    )

    error_msg = "'-> There are no ''good'' algorithms. Please verify\
    the binary performance measure. STOPPING!'"
    with pytest.raises(ValueError, match=error_msg) as e:
        data = Preprocessing.remove_bad_instances(data)
    assert "no ''good'' algorithms" in str(e.value), "Error message is not as expected."
    print("Remove bad instances tests 2 passed!")


def test_remove_bad_instances_3() -> None:
    """
    In this test case, there are some no good instances in the data should be removed.

    expected: No assertion errors.
    """
    y_bin = np.genfromtxt(
        path_root / "Prelim_out/model-data-ybin.csv",
        delimiter=",",
    )
    y_bin = np.zeros_like(y_bin, dtype=np.bool_)
    # make the first 3 algorithms good instances
    num_instances = 3
    y_bin[:, :num_instances] = True

    x = np.genfromtxt(path_root / "process_Data/X_before.csv", delimiter=",")
    y = np.genfromtxt(path_root / "process_Data/Y_before.csv", delimiter=",")
    with Path.open(path_root / "process_Data/algolabels_after.txt") as file:
        line = file.readline()
        algo_labels = line.strip().split(",")
    data = Data(
        inst_labels=pd.Series(),
        feat_labels=[],
        algo_labels=algo_labels,
        x=x,
        y=y,
        x_raw=x,
        y_raw=y,
        y_bin=y_bin,
        y_best=np.array([], dtype=np.double),
        p=np.array([], dtype=np.double),
        num_good_algos=np.array([], dtype=np.double),
        beta=np.array([], dtype=np.bool_),
        s=None,
        uniformity=None,
    )

    data = Preprocessing.remove_bad_instances(data)
    assert data.y_bin.shape[1] == num_instances
    print("Remove bad instances tests 3 passed!")


def test_split_data() -> None:
    """
    Test case for the split data function by using matlab example.

    expected: No assertion errors.
    """
    # idx = np.genfromtxt(path_root / "split/idx.txt", delimiter=",")

    x_before = np.genfromtxt(path_root / "split/before/x_split.txt", delimiter=",")
    y_before = np.genfromtxt(path_root / "split/before/Y_split.txt", delimiter=",")
    x_raw_before = np.genfromtxt(
        path_root / "split/before/Xraw_split.txt",
        delimiter=",",
    )
    y_raw_before = np.genfromtxt(
        path_root / "split/before/Yraw_split.txt",
        delimiter=",",
    )
    y_bin_before = np.genfromtxt(
        path_root / "split/before/Ybin_split.txt",
        delimiter=",",
    )
    beta_before = np.genfromtxt(
        path_root / "split/before/beta_split.txt",
        delimiter=",",
    )
    num_good_algos_before = np.genfromtxt(
        path_root / "split/before/numGoodAlgos_split.txt",
        delimiter=",",
    )
    y_best_before = np.genfromtxt(
        path_root / "split/before/Ybest_split.txt",
        delimiter=",",
    )
    p_before = np.genfromtxt(path_root / "split/before/P_split.txt", delimiter=",")
    inst_labels_before = pd.read_csv(
        path_root / "split/before/instlabels_split.txt",
        header=None,
    ).loc[:, 0]

    data = Data(
        inst_labels=inst_labels_before,
        feat_labels=[],
        algo_labels=[],
        x=x_before,
        y=y_before,
        x_raw=x_raw_before,
        y_raw=y_raw_before,
        y_bin=y_bin_before,
        y_best=y_best_before,
        p=p_before,
        num_good_algos=num_good_algos_before,
        beta=beta_before,
        s=None,
        uniformity=None,
    )

    opts = create_dummy_opt()
    model = create_dummy_model(data, opts)

    model = Preprocessing.split_data(data, opts, model)

    x_after = np.genfromtxt(path_root / "split/after/x_split.txt", delimiter=",")
    y_after = np.genfromtxt(path_root / "split/after/Y_split.txt", delimiter=",")
    x_raw_after = np.genfromtxt(
        path_root / "split/after/Xraw_split.txt",
        delimiter=",",
    )
    y_raw_after = np.genfromtxt(
        path_root / "split/after/Yraw_split.txt",
        delimiter=",",
    )
    y_bin_after = np.genfromtxt(
        path_root / "split/after/Ybin_split.txt",
        delimiter=",",
    )
    beta_after = np.genfromtxt(
        path_root / "split/after/beta_split.txt",
        delimiter=",",
    )
    num_good_algos_after = np.genfromtxt(
        path_root / "split/after/numGoodAlgos_split.txt",
        delimiter=",",
    )
    y_best_after = np.genfromtxt(
        path_root / "split/after/Ybest_split.txt",
        delimiter=",",
    )
    p_after = np.genfromtxt(path_root / "split/after/P_split.txt", delimiter=",")
    inst_labels_after = pd.read_csv(
        path_root / "split/after/instlabels_split.txt",
        header=None,
    ).loc[:, 0]

    assert np.array_equal(model.data.x, x_after)
    assert np.array_equal(model.data.y, y_after)
    assert np.array_equal(model.data.x_raw, x_raw_after)
    assert np.array_equal(model.data.y_raw, y_raw_after)
    assert np.array_equal(model.data.y_bin, y_bin_after)
    assert np.array_equal(model.data.beta, beta_after)
    assert np.array_equal(model.data.num_good_algos, num_good_algos_after)
    assert np.array_equal(model.data.y_best, y_best_after)
    assert np.array_equal(model.data.p, p_after)
    assert np.array_equal(model.data.inst_labels, inst_labels_after)
    print("Split data tests passed!")


def test_split_fractional() -> None:
    """Test case for the split data function by using fractional option."""
    # Create options for fractional split
    opts = InstanceSpaceOptions(
        parallel=ParallelOptions(flag=False, n_cores=2),
        perf=PerformanceOptions(
            max_perf=False,
            abs_perf=True,
            epsilon=0.20,
            beta_threshold=0.55,
        ),
        auto=AutoOptions(preproc=True),
        bound=BoundOptions(flag=True),
        norm=NormOptions(flag=True),
        selvars=SelvarsOptions(
            small_scale_flag=True,  # fractional
            small_scale=0.50,
            file_idx_flag=False,
            file_idx="",
            selvars_type="Ftr&Good",
            min_distance=0.1,
            density_flag=False,
            feats=None,
            algos=None,
        ),
        sifted=SiftedOptions(
            flag=True,
            rho=0.1,
            k=10,
            n_trees=50,
            max_iter=1000,
            replicates=100,
        ),
        pilot=PilotOptions(analytic=False, n_tries=5),
        cloister=CloisterOptions(p_val=0.05, c_thres=0.7),
        pythia=PythiaOptions(
            cv_folds=5,
            is_poly_krnl=False,
            use_weights=False,
            use_lib_svm=False,
        ),
        trace=TraceOptions(use_sim=True, pi=0.55),
        outputs=OutputOptions(csv=True, web=False, png=True),
    )

    x_before = np.genfromtxt(path_root / "fractional/before/x_split.txt", delimiter=",")
    y_before = np.genfromtxt(path_root / "fractional/before/Y_split.txt", delimiter=",")
    x_raw_before = np.genfromtxt(
        path_root / "fractional/before/Xraw_split.txt",
        delimiter=",",
    )
    y_raw_before = np.genfromtxt(
        path_root / "fractional/before/Yraw_split.txt",
        delimiter=",",
    )
    y_bin_before = np.genfromtxt(
        path_root / "fractional/before/Ybin_split.txt",
        delimiter=",",
    )
    beta_before = np.genfromtxt(
        path_root / "fractional/before/beta_split.txt",
        delimiter=",",
    )
    num_good_algos_before = np.genfromtxt(
        path_root / "fractional/before/numGoodAlgos_split.txt",
        delimiter=",",
    )
    y_best_before = np.genfromtxt(
        path_root / "fractional/before/Ybest_split.txt",
        delimiter=",",
    )
    p_before = np.genfromtxt(path_root / "fractional/before/P_split.txt", delimiter=",")
    inst_labels_before = pd.read_csv(
        path_root / "fractional/before/instlabels_split.txt",
        header=None,
    ).loc[:, 0]

    data = Data(
        inst_labels=inst_labels_before,
        feat_labels=[],
        algo_labels=[],
        x=x_before,
        y=y_before,
        x_raw=x_raw_before,
        y_raw=y_raw_before,
        y_bin=y_bin_before,
        y_best=y_best_before,
        p=p_before,
        num_good_algos=num_good_algos_before,
        beta=beta_before,
        s=None,
        uniformity=None,
    )

    model = create_dummy_model(data, opts)

    model = Preprocessing.split_data(data, opts, model)

    x_after = np.genfromtxt(path_root / "fractional/after/x_split.txt", delimiter=",")
    y_after = np.genfromtxt(path_root / "fractional/after/Y_split.txt", delimiter=",")
    x_raw_after = np.genfromtxt(
        path_root / "fractional/after/Xraw_split.txt",
        delimiter=",",
    )
    y_raw_after = np.genfromtxt(
        path_root / "fractional/after/Yraw_split.txt",
        delimiter=",",
    )
    y_bin_after = np.genfromtxt(
        path_root / "fractional/after/Ybin_split.txt",
        delimiter=",",
    )
    beta_after = np.genfromtxt(
        path_root / "fractional/after/beta_split.txt",
        delimiter=",",
    )
    num_good_algos_after = np.genfromtxt(
        path_root / "fractional/after/numGoodAlgos_split.txt",
        delimiter=",",
    )
    y_best_after = np.genfromtxt(
        path_root / "fractional/after/Ybest_split.txt",
        delimiter=",",
    )
    p_after = np.genfromtxt(path_root / "fractional/after/P_split.txt", delimiter=",")
    inst_labels_after = pd.read_csv(
        path_root / "fractional/after/instlabels_split.txt",
        header=None,
    ).loc[:, 0]

    assert np.array_equal(model.data.x.shape, x_after.shape)
    assert np.array_equal(model.data.y.shape, y_after.shape)
    assert np.array_equal(model.data.x_raw.shape, x_raw_after.shape)
    assert np.array_equal(model.data.y_raw.shape, y_raw_after.shape)
    assert np.array_equal(model.data.y_bin.shape, y_bin_after.shape)
    assert np.array_equal(model.data.beta.shape, beta_after.shape)
    assert np.array_equal(model.data.num_good_algos.shape, num_good_algos_after.shape)
    assert np.array_equal(model.data.y_best.shape, y_best_after.shape)
    assert np.array_equal(model.data.p.shape, p_after.shape)
    assert np.array_equal(model.data.inst_labels.shape, inst_labels_after.shape)
    print("Fractional tests passed!")


def test_split_fileindexed() -> None:
    """Test case for the split data function by using fileindexed option."""
    # Create options for fileindexed split
    opts = InstanceSpaceOptions(
        parallel=ParallelOptions(flag=False, n_cores=2),
        perf=PerformanceOptions(
            max_perf=False,
            abs_perf=True,
            epsilon=0.20,
            beta_threshold=0.55,
        ),
        auto=AutoOptions(preproc=True),
        bound=BoundOptions(flag=True),
        norm=NormOptions(flag=True),
        selvars=SelvarsOptions(
            small_scale_flag=False,
            small_scale=0.50,
            file_idx_flag=True,
            file_idx="./fileidx/fileidx.csv",
            selvars_type="Ftr&Good",
            min_distance=0.1,
            density_flag=False,
            feats=[],
            algos=[],
        ),
        sifted=SiftedOptions(
            flag=True,
            rho=0.1,
            k=10,
            n_trees=50,
            max_iter=1000,
            replicates=100,
        ),
        pilot=PilotOptions(analytic=False, n_tries=5),
        cloister=CloisterOptions(p_val=0.05, c_thres=0.7),
        pythia=PythiaOptions(
            cv_folds=5,
            is_poly_krnl=False,
            use_weights=False,
            use_lib_svm=False,
        ),
        trace=TraceOptions(use_sim=True, pi=0.55),
        outputs=OutputOptions(csv=True, web=False, png=True),
    )

    x_before = np.genfromtxt(path_root / "fileidx/before/x_split.txt", delimiter=",")
    y_before = np.genfromtxt(path_root / "fileidx/before/Y_split.txt", delimiter=",")
    x_raw_before = np.genfromtxt(
        path_root / "fileidx/before/Xraw_split.txt",
        delimiter=",",
    )
    y_raw_before = np.genfromtxt(
        path_root / "fileidx/before/Yraw_split.txt",
        delimiter=",",
    )
    y_bin_before = np.genfromtxt(
        path_root / "fileidx/before/Ybin_split.txt",
        delimiter=",",
    )
    beta_before = np.genfromtxt(
        path_root / "fileidx/before/beta_split.txt",
        delimiter=",",
    )
    num_good_algos_before = np.genfromtxt(
        path_root / "fileidx/before/numGoodAlgos_split.txt",
        delimiter=",",
    )
    y_best_before = np.genfromtxt(
        path_root / "fileidx/before/Ybest_split.txt",
        delimiter=",",
    )
    p_before = np.genfromtxt(path_root / "fileidx/before/P_split.txt", delimiter=",")
    inst_labels_before = pd.read_csv(
        path_root / "fileidx/before/instlabels_split.txt",
        header=None,
    ).loc[:, 0]

    data = Data(
        inst_labels=inst_labels_before,
        feat_labels=[],
        algo_labels=[],
        x=x_before,
        y=y_before,
        x_raw=x_raw_before,
        y_raw=y_raw_before,
        y_bin=y_bin_before,
        y_best=y_best_before,
        p=p_before,
        num_good_algos=num_good_algos_before,
        beta=beta_before,
        s=None,
        uniformity=None,
    )

    model = create_dummy_model(data, opts)
    model = Preprocessing.split_data(data, opts, model)

    x_after = np.genfromtxt(path_root / "fileidx/after/x_split.txt", delimiter=",")
    y_after = np.genfromtxt(path_root / "fileidx/after/Y_split.txt", delimiter=",")
    x_raw_after = np.genfromtxt(
        path_root / "fileidx/after/Xraw_split.txt",
        delimiter=",",
    )
    y_raw_after = np.genfromtxt(
        path_root / "fileidx/after/Yraw_split.txt",
        delimiter=",",
    )
    y_bin_after = np.genfromtxt(
        path_root / "fileidx/after/Ybin_split.txt",
        delimiter=",",
    )
    beta_after = np.genfromtxt(
        path_root / "fileidx/after/beta_split.txt",
        delimiter=",",
    )
    num_good_algos_after = np.genfromtxt(
        path_root / "fileidx/after/numGoodAlgos_split.txt",
        delimiter=",",
    )
    y_best_after = np.genfromtxt(
        path_root / "fileidx/after/Ybest_split.txt",
        delimiter=",",
    )
    p_after = np.genfromtxt(path_root / "fileidx/after/P_split.txt", delimiter=",")
    inst_labels_after = pd.read_csv(
        path_root / "fileidx/after/instlabels_split.txt",
        header=None,
    ).loc[:, 0]

    assert np.array_equal(model.data.x, x_after)
    assert np.array_equal(model.data.y, y_after)
    assert np.array_equal(model.data.x_raw, x_raw_after)
    assert np.array_equal(model.data.y_raw, y_raw_after)
    assert np.array_equal(model.data.y_bin, y_bin_after)
    assert np.array_equal(model.data.beta, beta_after)
    assert np.array_equal(model.data.num_good_algos, num_good_algos_after)
    assert np.array_equal(model.data.y_best, y_best_after)
    assert np.array_equal(model.data.p, p_after)
    assert np.array_equal(model.data.inst_labels, inst_labels_after)
    print("Fileindexed tests passed!")


# def test_split_bydensity() -> None:
#     """Test case for the split data function by using bydensity option."""
#    Since except Filter, there is only one line of the code(negation) that need to be
#   tested, and the rest part has been tested in other cases therefore I choose not to
#   test this case.
