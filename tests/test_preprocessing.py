"""
Testcases for the preprocessing stage of the Matilda project.

Specifically, it tests the `run` method of the `Preprocessing` class to ensure that
the output matches the expected results stored in `X.csv` and `Y.csv`.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from matilda.instance_space import instance_space_from_files
from matilda.stages.preprocessing import Preprocessing


def test_run_method() -> None:
    """
    Test the preprocessing run method.

    This test verifies that the output of the Preprocessing run method matches
    the expected results stored in the X.csv and Y.csv files.
    """
    script_dir = Path(__file__).resolve().parent
    metadata_path = script_dir / "test_integration/metadata.csv"
    option_path = script_dir / "test_integration/options.json"
    instance_space = instance_space_from_files(metadata_path, option_path)
    assert instance_space is not None

    pre_data_changed, preprocess_out = Preprocessing.run(
        instance_space.metadata,
        instance_space.options,
    )

    df_x = pd.read_csv(script_dir / "test_integration/X.csv", header=None)
    df_y = pd.read_csv(script_dir / "test_integration/Y.csv", header=None)

    assert np.array_equal(
        pre_data_changed.x,
        df_x,
    ), "The data arrays X and Y are not equal."

    assert np.allclose(
        pre_data_changed.x,
        df_x,
    ), "The data arrays X and Y are not approximately equal."

    assert np.array_equal(
        pre_data_changed.y,
        df_y,
    ), "The data arrays X and Y are not equal."

    assert np.allclose(
        pre_data_changed.y,
        df_y,
    ), "The data arrays X and Y are not approximately equal."
