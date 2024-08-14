from pathlib import Path

import numpy as np
import pandas as pd

from matilda.instance_space import instance_space_from_files
from matilda.stages.preprocessing import Preprocessing


def test_run_method() -> None:
    script_dir = Path(__file__).resolve().parent
    metadata_path = script_dir / "test_integration/metadata.csv"
    option_path = script_dir / "test_integration/options.json"
    InstanceSpace = instance_space_from_files(metadata_path, option_path)

    pre_data_changed, preprocess_out = Preprocessing.run(
        InstanceSpace.metadata,
        InstanceSpace.options,
    )

    df_X = pd.read_csv(script_dir / "test_integration/X.csv", header=None)
    df_Y = pd.read_csv(script_dir / "test_integration/Y.csv", header=None)

    assert np.array_equal(
        pre_data_changed.x,
        df_X,
    ), "The data arrays X and Y are not equal."
    # 或者使用 np.allclose 进行比较（如果数据有浮点数精度误差）
    assert np.allclose(
        pre_data_changed.x,
        df_X,
    ), "The data arrays X and Y are not approximately equal."

    assert np.array_equal(
        pre_data_changed.y,
        df_Y,
    ), "The data arrays X and Y are not equal."

    # 或者使用 np.allclose 进行比较（如果数据有浮点数精度误差）
    assert np.allclose(
        pre_data_changed.y,
        df_Y,
    ), "The data arrays X and Y are not approximately equal."
