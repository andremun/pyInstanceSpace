"""Test module for serialisers."""

import os
import shutil
import warnings
import zipfile
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import joblib
import matplotlib as mpl
import numpy as np
import pandas as pd
import pytest

mpl.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.path import Path as MatplotlibPath
from numpy.typing import NDArray
from scipy.io import loadmat
from shapely.geometry import MultiPolygon, Polygon

from instancespace import _serialisers as serialisers
from instancespace.data.model import (
    CloisterOut,
    Data,
    DataDense,
    FeatSel,
    Footprint,
    PilotOut,
    PrelimOut,
    PythiaOut,
    SiftedOut,
    TraceOut,
)
from instancespace.data.options import (
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
from instancespace.model import Model
from instancespace.stages.pilot_viewpoint import PilotViewpointResult

script_dir = Path(__file__).parent

# Clear the output before running the test
for directory in ["csv", "web", "png"]:
    output_directory = script_dir / "test_data/serialisers/actual_output" / directory
    for file in os.listdir(output_directory):
        if ".gitignore" not in file:
            Path(output_directory / file).unlink()


@dataclass
class _MatlabResults:
    workspace_data: dict  # type: ignore
    s_data: dict  # type: ignore
    clean_trace: dict  # type: ignore

    def __init__(self) -> None:
        self.workspace_data = loadmat(
            script_dir / "test_data/serialisers/input/workspace.mat",
            simplify_cells=True,
            chars_as_strings=True,
        )

        self.s_data = loadmat(
            script_dir / "test_data/serialisers/input/S.mat",
            chars_as_strings=True,
            simplify_cells=True,
        )

        self.clean_trace = loadmat(
            script_dir / "test_data/serialisers/input/clean_trace.mat",
            chars_as_strings=True,
            simplify_cells=True,
        )["clean_trace"]

    def get_model(self) -> Model:
        opts = self.workspace_data["model"]["opts"]
        # MATLAB's .mat serialisation stores a `logical` field as a numeric
        # 0/1 double whenever it wasn't set via an explicit true/false
        # literal - loadmat then hands back a Python/numpy number, not a
        # bool. bool(...) here mirrors what a real options.json (JSON
        # true/false -> Python bool) already gives for free, and what
        # MATLAB's own logical fields are semantically meant to be.
        parallel_options = ParallelOptions(
            flag=bool(opts["parallel"]["flag"]),
            n_cores=opts["parallel"]["ncores"],
        )
        performance_options = PerformanceOptions(
            max_perf=bool(opts["perf"]["MaxPerf"]),
            abs_perf=bool(opts["perf"]["AbsPerf"]),
            epsilon=opts["perf"]["epsilon"],
            beta_threshold=opts["perf"]["betaThreshold"],
        )
        auto_options = AutoOptions(preproc=bool(opts["auto"]["preproc"]))
        bound_options = BoundOptions(flag=bool(opts["bound"]["flag"]))
        norm_options = NormOptions(flag=bool(opts["norm"]["flag"]))
        selvars_options = SelvarsOptions(
            small_scale_flag=bool(opts["selvars"]["smallscaleflag"]),
            small_scale=opts["selvars"]["smallscale"],
            file_idx_flag=bool(opts["selvars"]["fileidxflag"]),
            file_idx="",
            feats=None,
            algos=None,
            # workspace.mat stores this as the literal typo 'Ftr&&Good' -
            # not a value any real MATLAB build (or this repo) has ever
            # recognised (core/FILTER.m's validTypes has no double-'&'
            # variant); irrelevant to what these serialiser tests actually
            # check, so corrected here rather than touching the binary fixture.
            selvars_type=opts["selvars"]["type"].replace("&&", "&"),
            min_distance=opts["selvars"]["mindistance"],
            density_flag=bool(opts["selvars"]["densityflag"]),
        )
        sifted_options = SiftedOptions.default(
            flag=bool(opts["sifted"]["flag"]),
            rho=opts["sifted"]["rho"],
            k=opts["sifted"]["K"],
            n_trees=opts["sifted"]["NTREES"],
            max_iter=opts["sifted"]["MaxIter"],
            replicates=opts["sifted"]["Replicates"],
        )
        pilot_options = PilotOptions.default(
            analytic=bool(opts["pilot"]["analytic"]),
            n_tries=opts["pilot"]["ntries"],
        )
        cloister_options = CloisterOptions(
            p_val=opts["cloister"]["pval"],
            c_thres=opts["cloister"]["cthres"],
        )
        pythia_options = PythiaOptions.default(
            cv_folds=opts["pythia"]["cvfolds"],
            is_poly_krnl=bool(opts["pythia"]["ispolykrnl"]),
            use_weights=bool(opts["pythia"]["useweights"]),
            # use_lib_svm=opts["pythia"]["uselibsvm"],
        )
        trace_options = TraceOptions.default(
            use_sim=bool(opts["trace"]["usesim"]),
            purity=opts["trace"]["PI"],
        )
        output_options = OutputOptions(
            csv=bool(opts["outputs"]["csv"]),
            web=bool(opts["outputs"]["web"]),
            png=bool(opts["outputs"]["png"]),
        )

        options = InstanceSpaceOptions(
            parallel=parallel_options,
            perf=performance_options,
            auto=auto_options,
            bound=bound_options,
            norm=norm_options,
            selvars=selvars_options,
            sifted=sifted_options,
            pilot=pilot_options,
            cloister=cloister_options,
            pythia=pythia_options,
            trace=trace_options,
            outputs=output_options,
        )

        data = Data(
            inst_labels=self.workspace_data["model"]["data"]["instlabels"],
            feat_labels=self.workspace_data["model"]["data"]["featlabels"],
            algo_labels=self.workspace_data["model"]["data"]["algolabels"],
            x=self.workspace_data["model"]["data"]["X"],
            y=self.workspace_data["model"]["data"]["Y"],
            x_raw=self.workspace_data["model"]["data"]["Xraw"],
            y_raw=self.workspace_data["model"]["data"]["Yraw"],
            y_bin=self.workspace_data["model"]["data"]["Ybin"],
            y_best=self.workspace_data["model"]["data"]["Ybest"],
            p=self.workspace_data["model"]["data"]["P"],
            num_good_algos=self.workspace_data["model"]["data"]["numGoodAlgos"],
            beta=self.workspace_data["model"]["data"]["beta"],
            s=self.s_data["S_cell"],
            # uniformity=None,
        )

        prelim_out = PrelimOut(
            med_val=self.workspace_data["model"]["prelim"]["medval"],
            iq_range=self.workspace_data["model"]["prelim"]["iqrange"],
            hi_bound=self.workspace_data["model"]["prelim"]["hibound"],
            lo_bound=self.workspace_data["model"]["prelim"]["lobound"],
            min_x=self.workspace_data["model"]["prelim"]["minX"],
            lambda_x=self.workspace_data["model"]["prelim"]["lambdaX"],
            mu_x=self.workspace_data["model"]["prelim"]["muX"],
            sigma_x=self.workspace_data["model"]["prelim"]["sigmaY"],
            min_y=self.workspace_data["model"]["prelim"]["minY"],
            lambda_y=self.workspace_data["model"]["prelim"]["lambdaY"],
            mu_y=self.workspace_data["model"]["prelim"]["muY"],
            sigma_y=self.workspace_data["model"]["prelim"]["sigmaY"],
        )

        sifted_out = SiftedOut(
            rho=self.workspace_data["model"]["sifted"]["rho"],
            # MATLAB indexes by 1
            selvars=self.workspace_data["model"]["sifted"]["selvars"] - 1,
            pval=None,  # self.workspace_data["model"]["sifted"]["pval"],
            silhouette_scores=None,  # self.workspace_data["model"]["sifted"][
            #    "silhouette_scores"
            # ],
            clust=None,  # self.workspace_data["model"]["sifted"]["clust"],
        )

        cloister_out = CloisterOut(
            z_edge=self.workspace_data["model"]["cloist"]["Zedge"],
            z_ecorr=self.workspace_data["model"]["cloist"]["Zecorr"],
        )

        def matlab_array_to_dataframe(arr: NDArray[Any]) -> pd.DataFrame:
            summary = arr.tolist()
            headers = summary[0]
            headers[0] = "Row"
            data = summary[1:]
            return pd.DataFrame(data, columns=headers)

        pilot_out = PilotOut(
            X0=self.workspace_data["model"]["pilot"]["X0"],
            alpha=self.workspace_data["model"]["pilot"]["alpha"],
            eoptim=self.workspace_data["model"]["pilot"]["eoptim"],
            perf=self.workspace_data["model"]["pilot"]["perf"],
            a=self.workspace_data["model"]["pilot"]["A"],
            z=self.workspace_data["model"]["pilot"]["Z"],
            c=self.workspace_data["model"]["pilot"]["C"],
            b=self.workspace_data["model"]["pilot"]["B"],
            error=self.workspace_data["model"]["pilot"]["error"],
            r2=self.workspace_data["model"]["pilot"]["R2"],
            summary=matlab_array_to_dataframe(
                self.workspace_data["model"]["pilot"]["summary"],
            ),
        )

        def translate_footprint(in_from_matlab: dict[str, Any]) -> Footprint:
            if len(in_from_matlab["polygon"]):
                vertices = in_from_matlab["polygon"]["Vertices"]
            else:
                vertices = []

            polygon_ndarray: NDArray[np.double] = vertices
            polygon = Polygon(polygon_ndarray)

            return Footprint(
                polygon=polygon,
                area=in_from_matlab["area"],
                elements=in_from_matlab["elements"],
                good_elements=in_from_matlab["goodElements"],
                density=in_from_matlab["density"],
                purity=in_from_matlab["purity"],
            )

        trace_out = TraceOut(
            # TODO: This will need to be translated to our footprint struct
            space=translate_footprint(self.clean_trace["space"]),
            good=[translate_footprint(i) for i in self.clean_trace["good"]],
            best=[translate_footprint(i) for i in self.clean_trace["best"]],
            # TODO: This will need to be translated to our footprint struct
            hard=translate_footprint(self.clean_trace["hard"]),
            summary=matlab_array_to_dataframe(
                self.workspace_data["model"]["trace"]["summary"],
            ),
        )

        summary = self.workspace_data["model"]["pythia"]["summary"]
        for i in range(summary.shape[0]):
            for j in range(summary.shape[1]):
                if type(summary[i, j]) is np.ndarray:
                    summary[i, j] = None

        pythia_out = PythiaOut(
            mu=self.workspace_data["model"]["pythia"]["mu"],
            sigma=self.workspace_data["model"]["pythia"]["sigma"],
            cp=self.workspace_data["model"]["pythia"]["cp"],
            svm=self.workspace_data["model"]["pythia"]["svm"],
            cvcmat=self.workspace_data["model"]["pythia"]["cvcmat"],
            y_sub=self.workspace_data["model"]["pythia"]["Ysub"],
            y_hat=self.workspace_data["model"]["pythia"]["Yhat"],
            pr0_sub=self.workspace_data["model"]["pythia"]["Pr0sub"],
            pr0_hat=self.workspace_data["model"]["pythia"]["Pr0hat"],
            box_consnt=self.workspace_data["model"]["pythia"]["boxcosnt"],
            k_scale=self.workspace_data["model"]["pythia"]["kscale"],
            precision=self.workspace_data["model"]["pythia"]["precision"],
            recall=self.workspace_data["model"]["pythia"]["recall"],
            accuracy=self.workspace_data["model"]["pythia"]["accuracy"],
            selection0=self.workspace_data["model"]["pythia"]["selection0"],
            selection1=self.workspace_data["model"]["pythia"]["selection1"],
            summary=matlab_array_to_dataframe(summary),
        )

        feat_sel = FeatSel(
            idx=self.workspace_data["model"]["featsel"]["idx"] - 1,
        )

        return Model(
            data=data,
            data_dense=cast(DataDense, data),
            feat_sel=feat_sel,
            prelim=prelim_out,
            sifted=sifted_out,
            pilot=pilot_out,
            cloister=cloister_out,
            pythia=pythia_out,
            trace=trace_out,
            opts=options,
        )


def _three_dimensional_model(model: Model) -> Model:
    """Extend the MATLAB 2D fixture without inventing 3D TRACE geometry."""
    third_coordinate = np.linspace(-1.0, 1.0, model.pilot.z.shape[0])
    projection = np.column_stack((model.pilot.z, third_coordinate))
    projection_matrix = np.vstack(
        (model.pilot.a, np.linspace(0.1, 1.0, model.pilot.a.shape[1])),
    )
    response_coefficients = np.vstack(
        (model.pilot.c, np.zeros((1, model.pilot.c.shape[1]))),
    )
    inverse_projection = np.column_stack(
        (model.pilot.b, np.zeros(model.pilot.b.shape[0])),
    )
    projection_summary = pd.concat(
        (model.pilot.summary, model.pilot.summary.iloc[[-1]].copy()),
        ignore_index=True,
    )
    projection_summary.iloc[-1, 0] = "Z_{3}"
    viewpoint = PilotViewpointResult(
        groups=((0,),),
        a=(np.eye(2, 3, dtype=np.float64),),
        azimuth=(0.0,),
        elevation=(np.pi / 2,),
    )

    edge_coordinate = np.linspace(-2.0, 2.0, model.cloister.z_edge.shape[0])
    pruned_coordinate = np.linspace(
        -1.5,
        1.5,
        model.cloister.z_ecorr.shape[0],
    )
    return replace(
        model,
        pilot=replace(
            model.pilot,
            a=projection_matrix,
            z=projection,
            c=response_coefficients,
            b=inverse_projection,
            summary=projection_summary,
            viewpoint=viewpoint,
        ),
        cloister=replace(
            model.cloister,
            z_edge=np.column_stack((model.cloister.z_edge, edge_coordinate)),
            z_ecorr=np.column_stack((model.cloister.z_ecorr, pruned_coordinate)),
        ),
    )


def test_save_to_csv() -> None:
    """Test saving information from a completed instance space to CSVs."""
    model = _MatlabResults().get_model()

    model.save_to_csv(script_dir / "test_data/serialisers/actual_output/csv")

    test_data_dir = script_dir / "test_data/serialisers"

    for csv_file in os.listdir(
        test_data_dir / "expected_output/csv",
    ):
        expected_file_path = test_data_dir / "expected_output/csv" / csv_file
        actual_file_path = test_data_dir / "actual_output/csv" / csv_file

        # Expected file isn't a directory, and actual file exists
        assert Path.is_file(expected_file_path)
        assert Path.is_file(actual_file_path)
        print("----------CSV File:", csv_file)
        expected_data = pd.read_csv(expected_file_path)
        actual_data = pd.read_csv(actual_file_path)

        if csv_file.endswith(("_best.csv", "_good.csv")):
            assert list(actual_data.columns) == [
                "Row",
                "Part",
                "Ring",
                "Vertex",
                "z_1",
                "z_2",
            ]
            np.testing.assert_array_equal(
                actual_data["Row"],
                np.arange(1, len(actual_data) + 1),
            )
            np.testing.assert_array_equal(actual_data["Part"], 1)
            np.testing.assert_array_equal(actual_data["Ring"], "exterior")
            np.testing.assert_array_equal(
                actual_data["Vertex"],
                np.arange(1, len(actual_data) + 1),
            )
            pd.testing.assert_frame_equal(
                expected_data[["z_1", "z_2"]],
                actual_data[["z_1", "z_2"]],
            )
        else:
            pd.testing.assert_frame_equal(expected_data, actual_data)


def test_3d_csv_export_uses_three_coordinates_and_omits_footprints(
    tmp_path: Path,
) -> None:
    """A 3D projection keeps numeric outputs but never projects Shapely polygons."""
    model = _three_dimensional_model(_MatlabResults().get_model())

    model.save_to_csv(tmp_path)

    expected_columns = ["Row", "z_1", "z_2", "z_3"]
    coordinates = pd.read_csv(tmp_path / "coordinates.csv")
    bounds = pd.read_csv(tmp_path / "bounds.csv")
    pruned_bounds = pd.read_csv(tmp_path / "bounds_prunned.csv")
    assert list(coordinates.columns) == expected_columns
    assert list(bounds.columns) == expected_columns
    assert list(pruned_bounds.columns) == expected_columns
    np.testing.assert_allclose(coordinates["z_3"], model.pilot.z[:, 2])
    np.testing.assert_allclose(bounds["z_3"], model.cloister.z_edge[:, 2])
    np.testing.assert_allclose(
        pruned_bounds["z_3"],
        model.cloister.z_ecorr[:, 2],
    )

    assert not list(tmp_path.glob("footprint_*_best.csv"))
    assert not list(tmp_path.glob("footprint_*_good.csv"))
    assert (tmp_path / "footprint_performance.csv").is_file()
    assert (tmp_path / "projection_matrix.csv").is_file()
    assert (
        len(pd.read_csv(tmp_path / "projection_matrix.csv")) == model.pilot.z.shape[1]
    )


def test_save_for_web() -> None:
    """Test saving information for export to the web frontend."""
    model = _MatlabResults().get_model()

    model.save_for_web(script_dir / "test_data/serialisers/actual_output/web")

    test_data_dir = script_dir / "test_data/serialisers"

    for csv_file in os.listdir(
        test_data_dir / "expected_output/web",
    ):
        expected_file_path = test_data_dir / "expected_output/web" / csv_file
        actual_file_path = test_data_dir / "actual_output/web" / csv_file

        # Expected file isn't a directory, and actual file exists
        assert Path.is_file(expected_file_path)
        assert Path.is_file(actual_file_path)

        expected_data = pd.read_csv(expected_file_path)
        actual_data = pd.read_csv(actual_file_path)

        if csv_file in [
            "good_algos_color.csv",
            "algorithm_process_single_color.csv",
            "feature_raw_color.csv",
            "feature_process_color.csv",
            "algorithm_raw_single_color.csv",
        ]:
            # There seems to be a rounding error in either python or MATLAB, so
            # allow an error of 1 for colours
            pd.testing.assert_frame_equal(expected_data, actual_data, rtol=0, atol=1)
        elif csv_file in ["color_table.csv"]:
            # We are using a different colormap, because the matlab one is proprietary
            pass
        else:
            pd.testing.assert_frame_equal(expected_data, actual_data)


def test_save_graphs() -> None:
    """Test saving graphs from a completed instance space."""
    model = _MatlabResults().get_model()

    model.save_graphs(script_dir / "test_data/serialisers/actual_output/png")

    test_data_dir = script_dir / "test_data/serialisers"

    for csv_file in os.listdir(
        test_data_dir / "expected_output/png",
    ):
        expected_file_path = test_data_dir / "expected_output/png" / csv_file
        actual_file_path = test_data_dir / "actual_output/png" / csv_file

        # Expected file isn't a directory, and actual file exists
        assert Path.is_file(expected_file_path)
        assert Path.is_file(actual_file_path)

        # We can't test the images, so we must check visually that they are consistant


def test_graph_portfolio_boundaries_use_zero_based_internal_indices(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Graph orchestration converts only Data.p and leaves PYTHIA unchanged."""
    data = cast(
        Data,
        SimpleNamespace(
            x=np.array([[0.0], [1.0]], dtype=np.double),
            y=np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.double),
            y_raw=np.array([[1.0, 2.0], [2.0, 1.0]], dtype=np.double),
            y_bin=np.array([[False, True], [True, False]], dtype=np.bool_),
            feat_labels=["feature"],
            algo_labels=["first", "last"],
            num_good_algos=np.ones(2, dtype=np.double),
            p=np.array([1, 2], dtype=np.int_),
            beta=np.array([False, True], dtype=np.bool_),
            s=None,
        ),
    )
    pythia = cast(
        PythiaOut,
        SimpleNamespace(
            y_hat=np.array([[True, False], [False, False]], dtype=np.bool_),
            selection0=np.array([0, -1], dtype=np.int_),
        ),
    )
    pilot = cast(
        PilotOut,
        SimpleNamespace(z=np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.double)),
    )
    empty = Footprint(None, 0, 0, 0, 0, 0)
    trace = cast(
        TraceOut,
        SimpleNamespace(good=[empty, empty], best=[empty, empty]),
    )
    selection_calls: list[NDArray[np.int_]] = []
    footprint_calls: list[NDArray[np.int_]] = []

    def no_draw(*args: object, **kwargs: object) -> None:
        del args, kwargs

    def capture_selections(*args: object, **kwargs: object) -> None:
        del kwargs
        selection_calls.append(np.asarray(args[1], dtype=np.int_).copy())

    def capture_footprint(*args: object, **kwargs: object) -> None:
        del kwargs
        footprint_calls.append(np.asarray(args[2], dtype=np.int_).copy())

    monkeypatch.setattr(serialisers, "_draw_scatter", no_draw)
    monkeypatch.setattr(serialisers, "_draw_binary_performance", no_draw)
    monkeypatch.setattr(serialisers, "_draw_good_bad_footprint", no_draw)
    monkeypatch.setattr(serialisers, "_draw_sources", no_draw)
    monkeypatch.setattr(serialisers, "_draw_portfolio_selections", capture_selections)
    monkeypatch.setattr(serialisers, "_draw_portfolio_footprint", capture_footprint)

    experimental_options = cast(
        InstanceSpaceOptions,
        SimpleNamespace(trace=SimpleNamespace(use_sim=False)),
    )
    simulated_options = cast(
        InstanceSpaceOptions,
        SimpleNamespace(trace=SimpleNamespace(use_sim=True)),
    )
    serialisers.save_instance_space_graphs(
        tmp_path,
        data,
        experimental_options,
        pythia,
        pilot,
        trace,
    )
    serialisers.save_instance_space_graphs(
        tmp_path,
        data,
        simulated_options,
        pythia,
        pilot,
        trace,
    )

    np.testing.assert_array_equal(selection_calls[0], [0, 1])
    np.testing.assert_array_equal(selection_calls[1], [0, -1])
    np.testing.assert_array_equal(selection_calls[2], [0, 1])
    np.testing.assert_array_equal(selection_calls[3], [0, -1])
    np.testing.assert_array_equal(footprint_calls[0], [0, 1])
    np.testing.assert_array_equal(footprint_calls[1], [0, -1])


def test_draw_portfolio_selections_labels_every_internal_selection(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Only -1 is None; algorithm zero and the last algorithm keep their labels."""
    scatter_calls: list[tuple[str, NDArray[np.double]]] = []

    def capture_scatter(
        _self: Axes,
        x: NDArray[np.double],
        _y: NDArray[np.double],
        **kwargs: object,
    ) -> None:
        scatter_calls.append((str(kwargs["label"]), np.asarray(x).copy()))

    def no_legend(_self: Axes) -> None:
        pass

    monkeypatch.setattr("matplotlib.axes.Axes.scatter", capture_scatter)
    monkeypatch.setattr("matplotlib.axes.Axes.legend", no_legend)
    z = np.array(
        [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]],
        dtype=np.double,
    )

    serialisers._draw_portfolio_selections(  # noqa: SLF001
        z,
        np.array([-1, 0, 2, 1], dtype=np.int_),
        np.array(["first_algo", "middle_algo", "last_algo"]),
        "Portfolio",
        tmp_path / "portfolio.png",
    )

    assert [label for label, _ in scatter_calls] == [
        "None",
        "first algo",
        "middle algo",
        "last algo",
    ]
    assert [points.tolist() for _, points in scatter_calls] == [
        [0.0],
        [1.0],
        [3.0],
        [2.0],
    ]


def test_draw_portfolio_footprint_matches_each_algorithm_to_its_footprint(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Portfolio polygons use best[i] for the matching zero-based selection i."""
    best = [Footprint(None, i, 0, 0, 0, 0) for i in range(3)]
    drawn: list[Footprint] = []

    def no_scatter(
        _self: Axes,
        _x: NDArray[np.double],
        _y: NDArray[np.double],
        **_kwargs: object,
    ) -> None:
        pass

    def no_legend(_self: Axes) -> None:
        pass

    def capture_footprint(
        _ax: Axes,
        footprint: Footprint,
        _colour: tuple[float, float, float, float],
        _alpha: float,
    ) -> None:
        drawn.append(footprint)

    monkeypatch.setattr("matplotlib.axes.Axes.scatter", no_scatter)
    monkeypatch.setattr("matplotlib.axes.Axes.legend", no_legend)
    monkeypatch.setattr(serialisers, "_draw_footprint", capture_footprint)

    serialisers._draw_portfolio_footprint(  # noqa: SLF001
        np.array(
            [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]],
            dtype=np.double,
        ),
        best,
        np.array([-1, 0, 2, 1], dtype=np.int_),
        np.array(["first", "middle", "last"]),
        tmp_path / "footprints.png",
    )

    assert drawn == best


def test_save_mat() -> None:
    """Test saving a mat file of the output directory."""
    model = _MatlabResults().get_model()
    model.save_to_mat(script_dir / "test_data/serialisers/actual_output/mat")
    actual_output = loadmat(
        script_dir / "test_data/serialisers/actual_output/mat/model.mat",
        chars_as_strings=True,
        simplify_cells=True,
    )["data"]["algolabels"]
    print(actual_output)
    assert np.array_equal(model.data.algo_labels, actual_output)


def test_save_zip() -> None:
    """Test saving a zip file of the output directory."""
    model = _MatlabResults().get_model()
    # Clear the output before running the test
    clean_dir(script_dir / "test_data/serialisers/actual_output/png")
    clean_dir(script_dir / "test_data/serialisers/actual_output/csv")
    clean_dir(script_dir / "test_data/serialisers/actual_output/web")
    clean_dir(script_dir / "test_data/serialisers/actual_output/mat")

    # Save the data to the output directory
    model.save_graphs(script_dir / "test_data/serialisers/actual_output/png")
    model.save_to_csv(script_dir / "test_data/serialisers/actual_output/csv")
    model.save_for_web(script_dir / "test_data/serialisers/actual_output/web")
    model.save_to_mat(script_dir / "test_data/serialisers/actual_output/mat")

    # Copy metadata and options from input folder into expected output folder
    shutil.copy(
        script_dir / "test_data/serialisers/input/metadata.csv",
        script_dir / "test_data/serialisers/actual_output/csv/metadata.csv",
    )
    zip_filename = "output.zip"
    model.save_zip(zip_filename, script_dir / "test_data/serialisers/actual_output")
    """Require the following files to be in the zip for dashboard"""
    required_files = [
        "coordinates.csv",
        "metadata.csv",
        "svm_table.csv",
        "bounds_prunned.csv",
        "feature_process.csv",
        "feature_raw.csv",
        "algorithm_raw.csv",
        "algorithm_process.csv",
        "algorithm_svm.csv",
        "portfolio_svm.csv",
        "model.mat",
    ]
    with zipfile.ZipFile(
        script_dir / "test_data/serialisers/actual_output" / zip_filename,
        "r",
    ) as zf:
        file_list = [Path(f).name for f in zf.namelist()]
        assert all(
            item in file_list for item in required_files
        ), f"Missing files: {set(required_files) - set(file_list)}"

        assert "output/csv/coordinates.csv" in zf.namelist()
        assert "output/mat/model.mat" in zf.namelist()


def test_csv_export_is_read_only_and_idempotent(tmp_path: Path) -> None:
    """Repeated CSV exports must not change any model-owned state."""
    model = _MatlabResults().get_model()
    model_hash_before = joblib.hash(model)
    trace_before = model.trace.summary.copy(deep=True)
    pilot_summary = model.pilot.summary
    assert pilot_summary is not None
    pilot_before = pilot_summary.copy(deep=True)
    pythia_before = model.pythia.summary.copy(deep=True)

    with warnings.catch_warnings():
        warnings.simplefilter("error", pd.errors.SettingWithCopyWarning)
        model.save_to_csv(tmp_path)
    first_export = {path.name: path.read_bytes() for path in tmp_path.iterdir()}

    model.save_to_csv(tmp_path)
    second_export = {path.name: path.read_bytes() for path in tmp_path.iterdir()}

    pd.testing.assert_frame_equal(model.trace.summary, trace_before)
    pd.testing.assert_frame_equal(pilot_summary, pilot_before)
    pd.testing.assert_frame_equal(model.pythia.summary, pythia_before)
    assert joblib.hash(model) == model_hash_before
    assert second_export == first_export


def test_footprint_csv_v2_preserves_parts_and_holes() -> None:
    """The footprint table must keep each component and interior ring separate."""
    first = Polygon(
        [(0, 0), (4, 0), (4, 4), (0, 4)],
        holes=[[(1, 1), (1, 2), (2, 2), (2, 1)]],
    )
    second = Polygon([(10, 10), (12, 10), (12, 12), (10, 12)])

    frame = serialisers._footprint_boundary_frame(  # noqa: SLF001
        MultiPolygon([first, second]),
    )

    assert list(frame.columns) == ["Row", "Part", "Ring", "Vertex", "z_1", "z_2"]
    assert frame.groupby(["Part", "Ring"], sort=False).size().to_dict() == {
        (1, "exterior"): 4,
        (1, "hole_1"): 4,
        (2, "exterior"): 4,
    }
    np.testing.assert_array_equal(frame["Row"], np.arange(1, 13))
    assert not np.any(
        np.logical_and(
            frame["z_1"] == frame["z_1"].shift(),
            frame["z_2"] == frame["z_2"].shift(),
        ),
    )


def test_compound_footprint_paths_keep_holes_and_components() -> None:
    """Each polygon part must have one path, and each hole must start a subpath."""
    first = Polygon(
        [(0, 0), (4, 0), (4, 4), (0, 4)],
        holes=[[(1, 1), (1, 2), (2, 2), (2, 1)]],
    )
    second = Polygon([(10, 10), (12, 10), (12, 12), (10, 12)])
    footprint = Footprint(MultiPolygon([first, second]), 0, 0, 0, 0, 0)
    expected_parts = 2
    fig, ax = plt.subplots()
    try:
        serialisers._draw_footprint(  # noqa: SLF001
            ax,
            footprint,
            (0.0, 0.0, 1.0, 1.0),
            0.3,
        )
        assert len(ax.patches) == expected_parts
        move_counts = [
            int(np.count_nonzero(patch.get_path().codes == MatplotlibPath.MOVETO))
            for patch in ax.patches
        ]
        assert move_counts == [2, 1]
    finally:
        plt.close(fig)


def test_portable_stems_are_safe_unique_and_deterministic() -> None:
    """Unsafe and colliding labels must map to unique portable stems."""
    labels = ["../../same", "..\\..\\same", "CON", "con", "", "A:B", "A?B"]

    stems = serialisers._portable_stems(labels, "algorithm")  # noqa: SLF001

    assert stems == serialisers._portable_stems(labels, "algorithm")  # noqa: SLF001
    assert len({stem.casefold() for stem in stems}) == len(stems)
    assert all("/" not in stem and "\\" not in stem for stem in stems)
    assert all(stem not in {"", ".", ".."} for stem in stems)
    assert all(stem.split(".", maxsplit=1)[0].upper() != "CON" for stem in stems)


@pytest.mark.filterwarnings("error::RuntimeWarning")
@pytest.mark.filterwarnings("error::UserWarning")
def test_scaling_and_scatter_handle_constant_and_missing_data(tmp_path: Path) -> None:
    """Constant and missing values must scale and plot without warnings."""
    values = np.array([[5.0, np.nan, 1.0], [5.0, np.nan, 3.0]])

    scaled = serialisers._minmax_scale(values, axis=0)  # noqa: SLF001
    np.testing.assert_allclose(scaled[:, [0, 2]], [[0.0, 0.0], [0.0, 1.0]])
    assert np.all(np.isnan(scaled[:, 1]))
    colours = serialisers._colour_scale(values)  # noqa: SLF001
    np.testing.assert_allclose(colours[:, [0, 2]], [[0, 0], [0, 255]])
    assert np.all(np.isnan(colours[:, 1]))
    serialisers._write_colour_array_to_csv(  # noqa: SLF001
        colours,
        pd.Series(["constant", "missing", "range"]),
        pd.Series(["row_1", "row_2"]),
        tmp_path / "colours.csv",
    )
    written_colours = pd.read_csv(tmp_path / "colours.csv")
    assert np.all(np.isnan(written_colours["missing"]))
    serialisers._draw_scatter(  # noqa: SLF001
        np.array([[0.0, 0.0], [1.0, 1.0]]),
        np.array([np.nan, np.nan]),
        "Missing",
        tmp_path / "missing.png",
    )
    assert (tmp_path / "missing.png").is_file()


def test_graph_scaling_matches_matlab_axis_contract(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Individual performance uses columns, and global performance uses one range."""
    data = cast(
        Data,
        SimpleNamespace(
            x=np.array([[4.0], [4.0]]),
            y=np.ones((2, 2)),
            y_raw=np.array([[0.0, 9.0], [9.0, 99.0]]),
            y_bin=np.zeros((2, 2), dtype=np.bool_),
            feat_labels=["feature"],
            algo_labels=["first", "second"],
            num_good_algos=np.zeros(2),
            p=np.ones(2, dtype=np.int_),
            beta=np.zeros(2, dtype=np.bool_),
            s=None,
        ),
    )
    pythia = cast(
        PythiaOut,
        SimpleNamespace(
            y_hat=np.zeros((2, 2), dtype=np.bool_),
            selection0=np.full(2, -1, dtype=np.int_),
        ),
    )
    pilot = cast(PilotOut, SimpleNamespace(z=np.array([[0.0, 0.0], [1.0, 1.0]])))
    empty = Footprint(None, 0, 0, 0, 0, 0)
    trace = cast(TraceOut, SimpleNamespace(good=[empty, empty], best=[empty, empty]))
    options = cast(
        InstanceSpaceOptions,
        SimpleNamespace(trace=SimpleNamespace(use_sim=True)),
    )
    scatter_values: dict[str, NDArray[np.double]] = {}

    def capture_scatter(
        _z: NDArray[np.double],
        values: NDArray[np.double],
        _title: str,
        output: Path,
    ) -> None:
        scatter_values[output.name] = np.asarray(values).copy()

    def no_draw(*_args: object, **_kwargs: object) -> None:
        pass

    monkeypatch.setattr(serialisers, "_draw_scatter", capture_scatter)
    monkeypatch.setattr(serialisers, "_draw_binary_performance", no_draw)
    monkeypatch.setattr(serialisers, "_draw_good_bad_footprint", no_draw)
    monkeypatch.setattr(serialisers, "_draw_portfolio_selections", no_draw)
    monkeypatch.setattr(serialisers, "_draw_portfolio_footprint", no_draw)

    serialisers.save_instance_space_graphs(
        tmp_path,
        data,
        options,
        pythia,
        pilot,
        trace,
    )

    np.testing.assert_allclose(
        scatter_values["distribution_performance_individual_normalized_first.png"],
        [0.0, 1.0],
    )
    np.testing.assert_allclose(
        scatter_values["distribution_performance_individual_normalized_second.png"],
        [0.0, 1.0],
    )
    np.testing.assert_allclose(
        scatter_values["distribution_performance_global_normalized_first.png"],
        [0.0, 0.5],
    )
    np.testing.assert_allclose(
        scatter_values["distribution_performance_global_normalized_second.png"],
        [0.5, 1.0],
    )


def test_save_errors_include_the_operation_and_target(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """CSV, image, and MAT failures must raise contextual errors."""
    csv_target = tmp_path / "table.csv"

    def fail_csv(*_args: object, **_kwargs: object) -> None:
        raise PermissionError("denied")

    monkeypatch.setattr(pd.DataFrame, "to_csv", fail_csv)
    with pytest.raises(serialisers.SerializationError, match="table.csv") as csv_error:
        serialisers._write_dataframe_to_csv(  # noqa: SLF001
            pd.DataFrame({"a": [1]}),
            csv_target,
        )
    assert isinstance(csv_error.value.__cause__, PermissionError)

    fig = plt.figure()
    try:
        monkeypatch.setattr(fig, "savefig", fail_csv)
        with pytest.raises(serialisers.SerializationError, match="plot.png"):
            serialisers._save_figure(fig, tmp_path / "plot.png")  # noqa: SLF001
    finally:
        plt.close(fig)

    monkeypatch.setattr(serialisers, "savemat", fail_csv)
    data = cast(Data, SimpleNamespace(algo_labels=["algorithm"]))
    with pytest.raises(serialisers.SerializationError, match="model.mat"):
        serialisers.save_instance_space_output_mat(tmp_path, data)


def test_zip_preserves_relative_paths_and_unique_members(tmp_path: Path) -> None:
    """Duplicate basenames in separate folders must stay separate in the archive."""
    model = object.__new__(Model)
    (tmp_path / "csv").mkdir()
    (tmp_path / "web").mkdir()
    (tmp_path / "csv" / "summary.txt").write_text("csv", encoding="utf-8")
    (tmp_path / "web" / "summary.txt").write_text("web", encoding="utf-8")

    model.save_zip("bundle.zip", tmp_path)
    model.save_zip("bundle.zip", tmp_path)

    with zipfile.ZipFile(tmp_path / "bundle.zip") as archive:
        names = archive.namelist()
        assert names == ["output/csv/summary.txt", "output/web/summary.txt"]
        assert len(names) == len(set(names))
        assert archive.read("output/csv/summary.txt") == b"csv"
        assert archive.read("output/web/summary.txt") == b"web"


@pytest.mark.parametrize(
    "archive_name",
    [
        "../escape.zip",
        "nested/file.zip",
        "bad\\file.zip",
        "CON.zip",
        "trailing.zip.",
    ],
)
def test_zip_rejects_unsafe_archive_names(
    tmp_path: Path,
    archive_name: str,
) -> None:
    """An archive name must not contain a path."""
    model = object.__new__(Model)

    with pytest.raises(ValueError, match="safe filename"):
        model.save_zip(archive_name, tmp_path)


def test_zip_rejects_symlink_inputs(tmp_path: Path) -> None:
    """Archives must not follow symlinks inside the output tree."""
    model = object.__new__(Model)
    target = tmp_path / "target.txt"
    target.write_text("data", encoding="utf-8")
    (tmp_path / "linked.txt").symlink_to(target)

    with pytest.raises(ValueError, match="ZIP input must not be a symlink"):
        model.save_zip("bundle.zip", tmp_path)


def test_zip_write_errors_include_target(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Archive creation failures must identify the failed target."""
    model = object.__new__(Model)

    def fail_zip(*_args: object, **_kwargs: object) -> None:
        raise PermissionError("denied")

    monkeypatch.setattr(zipfile, "ZipFile", fail_zip)
    with pytest.raises(serialisers.SerializationError, match="bundle.zip") as error:
        model.save_zip("bundle.zip", tmp_path)
    assert isinstance(error.value.__cause__, PermissionError)


def clean_dir(path: Path) -> None:
    """Remove all files in a directory."""
    ignored_files = [".gitignore"]

    for file in os.listdir(path):
        if file in ignored_files:
            continue
        Path.unlink(path / file)
