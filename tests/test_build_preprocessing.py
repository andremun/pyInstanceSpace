"""
Testcases for the preprocessing stage of the Matilda project.

Specifically, it tests the `run` method of the `Preprocessing` class to ensure that
the output matches the expected results stored in `X.csv` and `Y.csv`.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from instancespace.data.options import PrelimOptions, SelvarsOptions
from instancespace.instance_space import instance_space_from_files
from instancespace.stages.preprocessing import PreprocessingInput, PreprocessingStage


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

    df_x = pd.read_csv(script_dir / "test_data/preprocessing/X.csv", header=None)
    df_y = pd.read_csv(script_dir / "test_data/preprocessing/Y.csv", header=None)

    assert np.array_equal(
        pre_output.x,
        df_x,
    ), "The data arrays X and Y are not equal."

    assert np.allclose(
        pre_output.x,
        df_x,
    ), "The data arrays X and Y are not approximately equal."

    assert np.array_equal(
        pre_output.y,
        df_y,
    ), "The data arrays X and Y are not equal."

    assert np.allclose(
        pre_output.y,
        df_y,
    ), "The data arrays X and Y are not approximately equal."


def test_composed_nan_threshold_controls_feature_washing() -> None:
    """PREPROCESSING consumes the configured MATLAB NaN-fraction threshold."""
    features = np.array(
        [
            [np.nan, 1.0, 2.0, 3.0],
            [1.0, 2.0, 3.0, 4.0],
            [2.0, 3.0, 4.0, 5.0],
            [3.0, 4.0, 5.0, 6.0],
        ],
    )
    algorithms = np.arange(4, dtype=np.double).reshape(-1, 1)
    feature_names = ["f1", "f2", "f3", "f4"]
    instance_labels = pd.Series(["i1", "i2", "i3", "i4"])
    legacy_input = PreprocessingInput(
        feature_names,
        ["a1"],
        instance_labels,
        None,
        features,
        algorithms,
        SelvarsOptions.default(),
    )
    assert legacy_input.prelim_options == PrelimOptions.default()

    configured_input = legacy_input._replace(
        prelim_options=PrelimOptions(
            False,
            True,
            0.2,
            0.55,
            True,
            True,
            nan_threshold=0.30,
        ),
    )

    output = PreprocessingStage._run(configured_input)

    assert output.feat_labels == feature_names
    assert output.x.shape[1] == len(feature_names)


"""
Testcases for the preprocessing stage of the Matilda project.

Specifically, it tests the `run` method of the `Preprocessing` class to ensure that
the output matches the expected results stored in `X.csv` and `Y.csv`.
"""
