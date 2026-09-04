"""
Testcases for the integration of preprocessing and PRELIM stage of the Matilda project.

Specifically, it tests the `run` method of the `Preprocessing` and 'PRELIM' class to
ensure that the output matches the expected results of MATLAB results
"""

from pathlib import Path

import numpy as np
import pandas as pd

from instancespace.data.options import PrelimOptions
from instancespace.instance_space import instance_space_from_files
from instancespace.stages.prelim import PrelimInput, PrelimStage
from instancespace.stages.preprocessing import PreprocessingInput, PreprocessingStage


def test_integrated_prepro_n_prelim() -> None:
    """
    Test the preprocessing and PRELIM run method.

    This test verifies that the output of the integarted run method matches
    the expected results of corresponding MATLAB codes
    """
    script_dir = Path(__file__).resolve().parent
    metadata_path = script_dir / "test_data/preprocessing/metadata.csv"
    option_path = script_dir / "test_data/preprocessing/options.json"
    instance_space = instance_space_from_files(metadata_path, option_path)
    assert instance_space is not None

    preprocessing_input = PreprocessingInput(
        feature_names=instance_space.metadata.feature_names,
        algorithm_names=instance_space.metadata.algorithm_names,
        instance_labels=instance_space.metadata.instance_labels,
        instance_sources=instance_space.metadata.instance_sources,
        features=instance_space.metadata.features,
        algorithms=instance_space.metadata.algorithms,
        selvars_options=instance_space.options.selvars,
    )

    run_method = getattr(PreprocessingStage, "_run")
    pre_output = run_method(preprocessing_input)

    # Load prelim test data
    csv_output_prelim_p_run = script_dir / "test_data/prelim/run/output/output_P.csv"
    csv_output_prelim_ybest_run = (
        script_dir / "test_data/prelim/run/output/output_Ybest.csv"
    )
    csv_output_prelim_ybin_run = (
        script_dir / "test_data/prelim/run/output/output_Ybin.csv"
    )
    csv_output_prelim_x_run = script_dir / "test_data/prelim/run/output/output_X.csv"
    csv_output_prelim_y_run = script_dir / "test_data/prelim/run/output/output_Y.csv"
    csv_output_prelim_xraw_run = (
        script_dir / "test_data/prelim/run/output/output_Xraw.csv"
    )
    csv_output_prelim_yraw_run = (
        script_dir / "test_data/prelim/run/output/output_Yraw.csv"
    )
    csv_output_prelim_beta_run = (
        script_dir / "test_data/prelim/run/output/output_beta.csv"
    )
    csv_output_prelim_instlabels_run = (
        script_dir / "test_data/prelim/run/output/output_instlabels.csv"
    )
    csv_output_prelim_num_good_algos_run = (
        script_dir / "test_data/prelim/run/output/output_numGoodAlgos.csv"
    )

    p_output_run = (
        pd.read_csv(csv_output_prelim_p_run, sep=",", header=None).iloc[:, 0].values
    )
    ybest_output_run = (
        pd.read_csv(csv_output_prelim_ybest_run, sep=",", header=None).iloc[:, 0].values
    )
    ybin_output_run = pd.read_csv(csv_output_prelim_ybin_run, sep=",", header=None)
    x_output_run = pd.read_csv(csv_output_prelim_x_run, header=None).to_numpy()
    y_output_run = pd.read_csv(csv_output_prelim_y_run, header=None).to_numpy()
    xraw_output_run = pd.read_csv(csv_output_prelim_xraw_run, header=None).to_numpy()
    yraw_output_run = pd.read_csv(csv_output_prelim_yraw_run, header=None).to_numpy()
    beta_output_run = (
        pd.read_csv(csv_output_prelim_beta_run, header=None).iloc[:, 0].values
    )
    instlabels_output_run = (
        pd.read_csv(csv_output_prelim_instlabels_run, header=None).iloc[:, 0].values
    )
    num_good_algos_output_run = (
        pd.read_csv(csv_output_prelim_num_good_algos_run, header=None).iloc[:, 0].values
    )

    prelim_opts = PrelimOptions(
        max_perf=instance_space.options.perf.max_perf,
        abs_perf=instance_space.options.perf.abs_perf,
        epsilon=instance_space.options.perf.epsilon,
        beta_threshold=instance_space.options.perf.beta_threshold,
        bound=instance_space.options.bound.flag,
        norm=instance_space.options.norm.flag,
    )

    prelim_input = PrelimInput(
        x=pre_output.x,  # Use x from preprocessed data
        y=pre_output.y,  # Use y from preprocessed data
        x_raw=pre_output.x_raw,  # Raw data
        y_raw=pre_output.y_raw,  # Raw data
        s=pre_output.s,
        inst_labels=pre_output.inst_labels,
        prelim_options=prelim_opts,
        selvars_options=instance_space.options.selvars,
        general_options=instance_space.options.general,
    )

    # Execute prelim stage's private run method
    prelim_run_method = getattr(PrelimStage, "_run")
    prelim_output = prelim_run_method(prelim_input)

    # Validate Prelim output against expected output data
    assert np.allclose(prelim_output.x, x_output_run)
    assert np.allclose(prelim_output.y, y_output_run)
    assert np.allclose(prelim_output.y_bin, ybin_output_run)
    assert np.allclose(
        np.array(prelim_output.y_best).flatten(),
        np.array(ybest_output_run, dtype=np.float64),
    )
    expected_p = np.asarray(p_output_run, dtype=np.int_)
    finite_y = np.where(np.isnan(pre_output.y_raw), np.inf, pre_output.y_raw)
    tied = np.sum(finite_y == np.min(finite_y, axis=1)[:, None], axis=1) > 1
    np.testing.assert_array_equal(prelim_output.p[~tied], expected_p[~tied])
    rows = np.arange(pre_output.y_raw.shape[0])
    assert np.all(
        finite_y[rows, prelim_output.p - 1] == np.min(finite_y, axis=1),
    )
    assert np.allclose(prelim_output.x_raw, xraw_output_run)
    assert np.allclose(prelim_output.y_raw, yraw_output_run)
    assert np.array_equal(
        np.asarray(prelim_output.beta, dtype=bool),
        beta_output_run.astype(bool),
    )
    assert np.allclose(
        np.array(prelim_output.num_good_algos).flatten(),
        np.array(num_good_algos_output_run, dtype=np.float64),
    )
    assert prelim_output.instlabels is not None
    assert list(prelim_output.instlabels) == list(instlabels_output_run)
