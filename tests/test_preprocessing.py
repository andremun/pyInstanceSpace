"""
Testcases for the preprocessing stage of the Matilda project.

Specifically, it tests the `run` method of the `Preprocessing` class to ensure that
the output matches the expected results stored in `X.csv` and `Y.csv`.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from matilda.instance_space import instance_space_from_files
from matilda.stages.preprocessing import PreprocessingStage


def test_run_method() -> None:
    """
    Test the preprocessing run method.

    This test verifies that the output of the Preprocessing run method matches
    the expected results stored in the X.csv and Y.csv files.
    """
    script_dir = Path(__file__).resolve().parent
    metadata_path = script_dir / "test_data/preprocessing/metadata.csv"
    option_path = script_dir / "test_data/preprocessing/options.json"
    instance_space = instance_space_from_files(metadata_path, option_path)
    assert instance_space is not None

    preprocessing_stage = PreprocessingStage(
        instance_space.metadata.feature_names,
        instance_space.metadata.algorithm_names,
        instance_space.metadata.instance_labels,
        instance_space.metadata.instance_sources,
        instance_space.metadata.features,
        instance_space.metadata.algorithms,
        instance_space.options.selvars,
    )

    run_method = getattr(preprocessing_stage, "_run")
    (
        updated_inst_labels,
        updated_feat_labels,
        new_algo_labels,
        updated_x,
        updated_y,
        updated_s,
    ) = run_method(instance_space.options.selvars)

    df_x = pd.read_csv(script_dir / "test_data/preprocessing/X.csv", header=None)
    df_y = pd.read_csv(script_dir / "test_data/preprocessing/Y.csv", header=None)

    assert np.array_equal(
        updated_x,
        df_x,
    ), "The data arrays X and Y are not equal."

    assert np.allclose(
        updated_x,
        df_x,
    ), "The data arrays X and Y are not approximately equal."

    assert np.array_equal(
        updated_y,
        df_y,
    ), "The data arrays X and Y are not equal."

    assert np.allclose(
        updated_y,
        df_y,
    ), "The data arrays X and Y are not approximately equal."


"""
Testcases for the preprocessing stage of the Matilda project.

Specifically, it tests the `run` method of the `Preprocessing` class to ensure that
the output matches the expected results stored in `X.csv` and `Y.csv`.
"""
