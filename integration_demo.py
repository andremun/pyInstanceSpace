"""Testing script for integration."""

import sys
from pathlib import Path

from matilda import InstanceSpace
from matilda.data import metadata, options

script_dir = Path(__file__).parent / "tests"
# script_dir / "test_data/serialisers/actual_output" / directory

metadata_path = script_dir / "test_integration" / "metadata.csv"
options_path = script_dir / "test_integration" / "options.json"

metadata_object = metadata.from_csv_file(metadata_path)
options_object = options.from_json_file(options_path)

if metadata_object is None or options_object is None:
    print("ERR: File reading failed!")
    sys.exit()

instance_space = InstanceSpace(metadata_object, options_object)
