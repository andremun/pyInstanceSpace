import sys
from pathlib import Path

import numpy as np
import pandas as pd

from matilda.build import process_data
from matilda.data.model import Data, Model
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
    PrelimOptions,
    PythiaOptions,
    SelvarsOptions,
    SiftedOptions,
    TraceOptions,
)

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
        prelim=PrelimOptions(
            max_perf=False,
            abs_perf=True,
            epsilon=0.20,
            beta_threshold=0.55,
            bound=True,
            norm=True,
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

    process_data(model)

    assert np.array_equal(model.data.x, x_after)
    assert np.array_equal(model.data.y, y_after)
    assert np.array_equal(algo_labels_after, model.data.algo_labels)
    assert np.array_equal(feat_labels_after, model.data.feat_labels)
    # the representation of PrelimOptions is not implemented yet


if __name__ == "__main__":
    test_process_data()
