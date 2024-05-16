"""Test module for Filter class to verify its functionality.

The file contains multiple unit tests to ensure that the `Filter` class correctly
perform its tasks. The basic mechanism of the test is to compare its output against
output from MATLAB and check if the outputs are the same or reasonable similar.

Tests include:
- Verifying ouput against MATLAB's output with 'Ftr' option type
- Verifying ouput against MATLAB's output with 'Ftr&AP' option type
- Verifying ouput against MATLAB's output with 'Ftr&AP&Good' option type
- Verifying ouput against MATLAB's output with 'Ftr&Good' option type
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from matilda.data.option import SelvarsOptions
from matilda.stages.filter import Filter

script_dir = Path(__file__).parent

csv_path_x = script_dir / "test_data/filter/input/input_X.csv"
csv_path_y = script_dir / "test_data/filter/input/input_Y.csv"
csv_path_y_bin = script_dir / "test_data/filter/input/input_Ybin.csv"

input_x = pd.read_csv(csv_path_x, header=None).to_numpy()
input_y = pd.read_csv(csv_path_y, header=None).to_numpy()
input_y_bin = pd.read_csv(csv_path_y_bin, header=None).to_numpy()
