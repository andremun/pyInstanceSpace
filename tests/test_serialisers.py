"""Test module for serialisers."""

import os
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.io import loadmat
from shapely.geometry import Polygon

from matilda.data.model import (
    CloisterOut,
    Data,
    FeatSel,
    Footprint,
    PilotOut,
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
from matilda.model import Model

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
        parallel_options = ParallelOptions(
            flag=opts["parallel"]["flag"],
            n_cores=opts["parallel"]["ncores"],
        )
        performance_options = PerformanceOptions(
            max_perf=opts["perf"]["MaxPerf"],
            abs_perf=opts["perf"]["AbsPerf"],
            epsilon=opts["perf"]["epsilon"],
            beta_threshold=opts["perf"]["betaThreshold"],
        )
        auto_options = AutoOptions(preproc=opts["auto"]["preproc"])
        bound_options = BoundOptions(flag=opts["bound"]["flag"])
        norm_options = NormOptions(flag=opts["norm"]["flag"])
        selvars_options = SelvarsOptions(
            small_scale_flag=opts["selvars"]["smallscaleflag"],
            small_scale=opts["selvars"]["smallscale"],
            file_idx_flag=opts["selvars"]["fileidxflag"],
            file_idx=opts["selvars"]["fileidx"],
            feats=None,
            algos=None,
            selvars_type=opts["selvars"]["type"],
            min_distance=opts["selvars"]["mindistance"],
            density_flag=opts["selvars"]["densityflag"],
        )
        sifted_options = SiftedOptions.default(
            flag=opts["sifted"]["flag"],
            rho=opts["sifted"]["rho"],
            k=opts["sifted"]["K"],
            n_trees=opts["sifted"]["NTREES"],
            max_iter=opts["sifted"]["MaxIter"],
            replicates=opts["sifted"]["Replicates"],
        )
        pilot_options = PilotOptions.default(
            analytic=opts["pilot"]["analytic"],
            n_tries=opts["pilot"]["ntries"],
        )
        cloister_options = CloisterOptions(
            p_val=opts["cloister"]["pval"],
            c_thres=opts["cloister"]["cthres"],
        )
        pythia_options = PythiaOptions.default(
            cv_folds=opts["pythia"]["cvfolds"],
            is_poly_krnl=opts["pythia"]["ispolykrnl"],
            use_weights=opts["pythia"]["useweights"],
            # use_lib_svm=opts["pythia"]["uselibsvm"],
        )
        trace_options = TraceOptions.default(
            use_sim=opts["trace"]["usesim"],
            purity=opts["trace"]["PI"],
        )
        output_options = OutputOptions(
            csv=opts["outputs"]["csv"],
            web=opts["outputs"]["web"],
            png=opts["outputs"]["png"],
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
            idx=self.workspace_data["model"]["featsel"]["idx"] - 1,
            selvars=self.workspace_data["model"]["sifted"]["selvars"],
            pval=None,  # self.workspace_data["model"]["sifted"]["pval"],
            silhouette_scores=None,  # self.workspace_data["model"]["sifted"][
            #    "silhouette_scores"
            # ],
            clust=None,  # self.workspace_data["model"]["sifted"]["clust"],
        )

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
            summary=self.workspace_data["model"]["pilot"]["summary"],
        )

        cloister_out = CloisterOut(
            z_edge=self.workspace_data["model"]["cloist"]["Zedge"],
            z_ecorr=self.workspace_data["model"]["cloist"]["Zecorr"],
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
            summary=self.workspace_data["model"]["trace"]["summary"],
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
            summary=self.workspace_data["model"]["pythia"]["summary"],
        )

        feat_sel = FeatSel(
            idx=self.workspace_data["model"]["featsel"]["idx"] - 1,
        )

        return Model(
            data=data,
            data_dense=data,
            feat_sel=feat_sel,
            prelim=prelim_out,
            sifted=sifted_out,
            pilot=pilot_out,
            cloister=cloister_out,
            pythia=pythia_out,
            trace=trace_out,
            opts=options,
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

        expected_data = pd.read_csv(expected_file_path)
        actual_data = pd.read_csv(actual_file_path)

        pd.testing.assert_frame_equal(expected_data, actual_data)


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


def test_save_mat() -> None:
    """Test saving a mat file of the output directory."""
    model = _MatlabResults().get_model()
    model.save_to_mat(script_dir / "test_data/serialisers/actual_output/mat")
    actual_output = loadmat(
        script_dir / "test_data/serialisers/actual_output/mat/model.mat",
        chars_as_strings=True,
        simplify_cells=True,
    )["data"]["algo_labels"]
    print(actual_output)
    assert np.array_equal(model.data.algo_labels, actual_output)


def test_save_zip() -> None:
    """Test saving a zip file of the output directory."""
    model = _MatlabResults().get_model()

    model.save_zip(script_dir / "test_data/serialisers/actual_output")
    """Require the following files to be in the zip for dashboard"""
    required_files = [
        "coordinates.csv",
        # "metadata.csv",
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
    with zipfile.ZipFile(script_dir.parent / "output.zip", "r") as zf:
        file_list = [Path(f).name for f in zf.namelist()]
        assert all(
            item in file_list for item in required_files
        ), f"Missing files: {set(required_files) - set(file_list)}"
