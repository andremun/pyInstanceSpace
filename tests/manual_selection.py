

from matilda.build import select_features_and_algorithms
from matilda.data.model import Data, Model
from matilda.data.option import Opts, SelvarsOptions

import sys
from pathlib import Path

path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))


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

    opts = Opts(
        selvars=SelvarsOptions(
            feats=["feature1", "feature3", "feature5", "feature7", "feature9"],
            algos=["algo1", "algo3"],
            small_scale_flag=False,
            small_scale=0.1,
            file_idx_flag=False,
            file_idx="",
            type="",
            min_distance=0.0,
            density_flag=False,
        ),
        parallel=None,
        perf=None,
        auto=None,
        bound=None,
        norm=None,
        sifted=None,
        pilot=None,
        cloister=None,
        pythia=None,
        trace=None,
        outputs=None,
    )

    empty_feat_sel = None
    empty_prelim = None
    empty_sifted = None
    empty_pilot = None
    empty_cloist = None
    empty_pythia = None
    empty_trace = None

    model = Model(data=data, data_dense=data, feat_sel=empty_feat_sel,
                  prelim=empty_prelim, sifted=empty_sifted, pilot=empty_pilot,
                  cloist=empty_cloist, pythia=empty_pythia,
                  trace=empty_trace, opts=opts)

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

