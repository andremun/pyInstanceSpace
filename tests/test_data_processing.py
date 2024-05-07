import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from matilda.data.model import Data, Model
from matilda.data.option import (
    AutoOptions,
    BoundOptions,
    CloisterOptions,
    GeneralOptions,
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
from matilda.stages.build import process_data, remove_bad_instances, split_data

path_root = Path(__file__).parent
sys.path.append(str(path_root.parent))


def create_dummy_opt() -> Options:
    return Options(
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
            type="Ftr&Good",
            min_distance=0.1,
            density_flag=False,
            feats=pd.DataFrame(),
            algos=pd.DataFrame(),
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
        trace=TraceOptions(use_sim=True, PI=0.55),
        outputs=OutputOptions(csv=True, web=False, png=True),
        general=GeneralOptions(beta_threshold=0.55),
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
    with open(path_root / "process_Data/featlabels_before.txt") as file:
        line = file.readline()
        feat_labels_before = line.strip().split(",")
    with open(path_root / "process_Data/featlabels_after.txt") as file:
        line = file.readline()
        feat_labels_after = line.strip().split(",")
    with open(path_root / "process_Data/algolabels_before.txt") as file:
        line = file.readline()
        algo_labels_before = line.strip().split(",")
    with open(path_root / "process_Data/algolabels_after.txt") as file:
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
    )

    opts = create_dummy_opt()
    model = Model(
        data=data,
        opts=opts,
        feat_sel=None,
        data_dense=None,
        prelim=None,
        sifted=None,
        pilot=None,
        cloist=None,
        pythia=None,
        trace=None,
    )

    prelim_opts = process_data(model)

    assert np.array_equal(model.data.x, x_after)
    assert np.array_equal(model.data.y, y_after)
    assert np.array_equal(algo_labels_after, model.data.algo_labels)
    assert np.array_equal(feat_labels_after, model.data.feat_labels)
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
        feat_labels=None,
        algo_labels=None,
        x=None,
        y=None,
        x_raw=None,
        y_raw=None,
        y_bin=y_bin,
        y_best=None,
        p=None,
        num_good_algos=None,
        beta=None,
        s=None,
    )
    opts = create_dummy_opt()
    model = Model(
        data=data,
        opts=opts,
        feat_sel=None,
        data_dense=None,
        prelim=None,
        sifted=None,
        pilot=None,
        cloist=None,
        pythia=None,
        trace=None,
    )
    idx = remove_bad_instances(model)
    assert model.data.y_bin.shape == y_bin.shape
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
    with open(path_root / "process_Data/algolabels_after.txt") as file:
        line = file.readline()
        algo_labels = line.strip().split(",")

    data = Data(
        inst_labels=pd.Series(),
        feat_labels=None,
        algo_labels=algo_labels,
        x=x,
        y=y,
        x_raw=x,
        y_raw=y,
        y_bin=y_bin,
        y_best=None,
        p=None,
        num_good_algos=None,
        beta=None,
        s=None,
    )

    opts = create_dummy_opt()
    model = Model(
        data=data,
        opts=opts,
        feat_sel=None,
        data_dense=None,
        prelim=None,
        sifted=None,
        pilot=None,
        cloist=None,
        pythia=None,
        trace=None,
    )
    with pytest.raises(Exception) as e:
        idx = remove_bad_instances(model)
    assert "no ''good'' algorithms" in str(e.value), "Error message is not as expected."
    print("Remove bad instances tests 2 passed!")


def test_remove_bad_instances_3() -> None:
    """
    In this test case, there are some no good instances in the data, so they should be removed.

    expected: No assertion errors.
    """
    y_bin = np.genfromtxt(
        path_root / "Prelim_out/model-data-ybin.csv",
        delimiter=",",
    )
    y_bin = np.zeros_like(y_bin, dtype=np.bool_)
    # make the first 3 algorithms good instances
    NUM_INSTANCES = 3
    y_bin[:, :NUM_INSTANCES] = True

    x = np.genfromtxt(path_root / "process_Data/X_before.csv", delimiter=",")
    y = np.genfromtxt(path_root / "process_Data/Y_before.csv", delimiter=",")
    with open(path_root / "process_Data/algolabels_after.txt") as file:
        line = file.readline()
        algo_labels = line.strip().split(",")
    data = Data(
        inst_labels=pd.Series(),
        feat_labels=None,
        algo_labels=algo_labels,
        x=x,
        y=y,
        x_raw=x,
        y_raw=y,
        y_bin=y_bin,
        y_best=None,
        p=None,
        num_good_algos=None,
        beta=None,
        s=None,
    )

    opts = create_dummy_opt()
    model = Model(
        data=data,
        opts=opts,
        feat_sel=None,
        data_dense=None,
        prelim=None,
        sifted=None,
        pilot=None,
        cloist=None,
        pythia=None,
        trace=None,
    )
    idx = remove_bad_instances(model)
    assert model.data.y_bin.shape[1] == NUM_INSTANCES
    print("Remove bad instances tests 3 passed!")


def test_split_data() -> None:
    """
    Test case for the split data function by using matlab example.

    expected: No assertion errors.
    """
    idx = np.genfromtxt(path_root / "split/idx.txt", delimiter=",")

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
        feat_labels=None,
        algo_labels=None,
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
    )

    opts = create_dummy_opt()
    model = Model(
        data=data,
        opts=opts,
        feat_sel=None,
        data_dense=None,
        prelim=None,
        sifted=None,
        pilot=None,
        cloist=None,
        pythia=None,
        trace=None,
    )
    split_data(idx, model)

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
    opts = Options(
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
            type="Ftr&Good",
            min_distance=0.1,
            density_flag=False,
            feats=pd.DataFrame(),
            algos=pd.DataFrame(),
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
        trace=TraceOptions(use_sim=True, PI=0.55),
        outputs=OutputOptions(csv=True, web=False, png=True),
        general=GeneralOptions(beta_threshold=0.55),
    )

    idx = np.genfromtxt(path_root / "fractional/idx.txt", delimiter=",")

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
        feat_labels=None,
        algo_labels=None,
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
    )

    model = Model(
        data=data,
        opts=opts,
        feat_sel=None,
        data_dense=None,
        prelim=None,
        sifted=None,
        pilot=None,
        cloist=None,
        pythia=None,
        trace=None,
    )

    split_data(idx, model)

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
    opts = Options(
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
            type="Ftr&Good",
            min_distance=0.1,
            density_flag=False,
            feats=pd.DataFrame(),
            algos=pd.DataFrame(),
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
        trace=TraceOptions(use_sim=True, PI=0.55),
        outputs=OutputOptions(csv=True, web=False, png=True),
        general=GeneralOptions(beta_threshold=0.55),
    )

    idx = np.genfromtxt(path_root / "fileidx/idx.txt", delimiter=",")

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
        feat_labels=None,
        algo_labels=None,
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
    )

    model = Model(
        data=data,
        opts=opts,
        feat_sel=None,
        data_dense=None,
        prelim=None,
        sifted=None,
        pilot=None,
        cloist=None,
        pythia=None,
        trace=None,
    )

    split_data(idx, model)

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

if __name__ == "__main__":
    test_process_data()
    test_remove_bad_instances_1()
    test_remove_bad_instances_2()
    test_remove_bad_instances_3()
    test_split_data()
    test_split_fractional()
    # test_split_fileindexed()
