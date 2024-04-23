"""
Contains test cases for the remove_instances_with_many_missing_values function.

The basic mechanism of the test is to compare its output against
the expected output from the specification of original BuildIS code,
and check if the outputs are exactly same with the exception.

"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from matilda.build import remove_instances_with_many_missing_values
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


def create_default_model(data: Data) -> Model:
    """
    Create a default model with the given data.

    All the value that assigned to the model are trivial, except the parameter: data
    """
    opts = Opts(
        parallel=ParallelOptions(flag=False, n_cores=1),
        perf=PerformanceOptions(
            max_perf=False, abs_perf=False, epsilon=0.1, beta_threshold=0.5,
        ),
        auto=AutoOptions(preproc=False),
        bound=BoundOptions(flag=False),
        norm=NormOptions(flag=False),
        selvars=SelvarsOptions(
            small_scale_flag=False,
            small_scale=0.1,
            file_idx_flag=False,
            file_idx="",
            feats=[],
            algos=[],
            type="",
            min_distance=0.0,
            density_flag=False,
        ),
        sifted=SiftedOptions(
            flag=False, rho=0.5, k=10, n_trees=100, max_iter=100, replicates=10,
        ),
        pilot=PilotOptions(analytic=False, n_tries=10),
        cloister=CloisterOptions(p_val=0.05, c_thres=0.5),
        pythia=PythiaOptions(
            cv_folds=5, is_poly_krnl=False, use_weights=False, use_lib_svm=False,
        ),
        trace=TraceOptions(use_sim=False, PI=0.95),
        outputs=OutputOptions(csv=False, web=False, png=False),
    )

    empty_feat_sel = FeatSel(idx=np.array([], dtype=np.intc))
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
    empty_sifted = SiftedOut(flag=0, rho=np.double(0),
                             k=0, n_trees=0, max_lter=0, replicates=0)
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
    empty_cloist = CloisterOut(
        Zedge=np.array([], dtype=np.double), Zecorr=np.array([], dtype=np.double),
    )
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


def test_remove_instances_with_two_row_missing() -> None:
    """Test remove_instances_with_many_missing_values function with two rows missing."""
    rng = np.random.default_rng(33)

    # Create sample data with missing values (10 rows)
    large_x = rng.random((10, 10))
    large_x[0, :] = np.nan  # First row all NaN
    large_x[1, :5] = np.nan  # Second row first 5 columns NaN
    large_x[:, 0] = np.nan  # First column all NaN (> 20% missing)

    large_y = rng.random((10, 5))
    large_y[1, :] = np.nan  # second row all NaN

    data = Data(
        inst_labels=pd.Series(["inst" + str(i) for i in range(10)]),
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

    model = create_default_model(data)

    remove_instances_with_many_missing_values(model)

    expected_rows = 8

    # Check instances removal
    assert (
            model.data.x.shape[0] == expected_rows
    ), "Instances with all NaN values not removed correctly"
    assert (
            model.data.y.shape[0] == expected_rows
    ), "Instances with all NaN values not removed correctly"
    assert (
            model.data.inst_labels.shape[0] == expected_rows
    ), "Instance labels not updated after removal"

    expected_x_columns = 10
    expected_y_columns = 5

    # Check feature dimensions are unchanged
    assert model.data.x.shape[1] == expected_x_columns, "x dimensions should not change"
    assert model.data.y.shape[1] == expected_y_columns, "y dimensions should not change"

    # Check inst_labels content
    assert model.data.inst_labels.tolist() == [
        "inst" + str(i) for i in range(2, 10)
    ], "inst_labels content not right"

    # not sure how to test idx,


def test_remove_instances_with_3_row_missing() -> None:
    """Test remove_instances_with_many_missing_values function with three rows missing."""
    rng = np.random.default_rng(33)

    # Create sample data with missing values (10 rows)
    large_x = rng.random((10, 10))
    large_x[2, :] = np.nan  # third row all NaN
    large_x[1, :5] = np.nan  # Second row first 5 columns NaN

    large_y = rng.random((10, 5))
    large_y[4, :] = np.nan  # fifth row all NaN
    large_y[3, :] = np.nan  # forth row all NaN

    data = Data(
        inst_labels=pd.Series(["inst" + str(i) for i in range(10)]),
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

    model = create_default_model(data)

    remove_instances_with_many_missing_values(model)

    expected_rows = 7

    # Check instances removal
    assert (
            model.data.x.shape[0] == expected_rows
    ), "Instances with all NaN values not removed correctly"
    assert (
            model.data.y.shape[0] == expected_rows
    ), "Instances with all NaN values not removed correctly"
    assert (
            model.data.inst_labels.shape[0] == expected_rows
    ), "Instance labels not updated after removal"

    expected_x_columns = 10
    expected_y_columns = 5

    # Check feature dimensions are unchanged
    assert model.data.x.shape[1] == expected_x_columns, "x dimensions should not change"
    assert model.data.y.shape[1] == expected_y_columns, "y dimensions should not change"

    # Check inst_labels content
    assert model.data.inst_labels.tolist() == [
        "inst0",
        "inst1",
        "inst5",
        "inst6",
        "inst7",
        "inst8",
        "inst9",
    ], "inst_labels content not right"

    # not sure how to test idx,


def test_remove_instances_keep_same() -> None:
    """
    Test remove_instances_with_many_missing_values function with no rows missing.
    """
    rng = np.random.default_rng(33)

    # Create sample data with missing values (10 rows)
    large_x = rng.random((10, 10))

    large_x[1, :5] = np.nan  # Second row first 5 columns NaN

    large_x[4, :5] = np.nan  # 5th row first 5 columns NaN

    large_y = rng.random((10, 5))
    large_y[6, :2] = np.nan  # 7th row first 2columns NaN

    data = Data(
        inst_labels=pd.Series(["inst" + str(i) for i in range(10)]),
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

    model = create_default_model(data)

    remove_instances_with_many_missing_values(model)

    expected_rows = 10

    # Check instances removal
    assert (
            model.data.x.shape[0] == expected_rows
    ), "Instances with all NaN values not removed correctly"
    assert (
            model.data.y.shape[0] == expected_rows
    ), "Instances with all NaN values not removed correctly"
    assert (
            model.data.inst_labels.shape[0] == expected_rows
    ), "Instance labels not updated after removal"

    expected_x_columns = 10
    expected_y_columns = 5

    # Check feature dimensions are unchanged
    assert model.data.x.shape[1] == expected_x_columns, "x dimensions should not change"
    assert model.data.y.shape[1] == expected_y_columns, "y dimensions should not change"

    # Check inst_labels content
    assert model.data.inst_labels.tolist() == [
        "inst" + str(i) for i in range(0, 10)
    ], "inst_labels content not right"

    # not sure how to test idx,


if __name__ == "__main__":
    test_remove_instances_with_two_row_missing()
    test_remove_instances_with_3_row_missing()
    test_remove_instances_keep_same()
