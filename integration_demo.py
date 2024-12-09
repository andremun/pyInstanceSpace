"""Testing script for integration."""

import sys
from pathlib import Path

from instancespace import InstanceSpace
from instancespace.data import metadata, options
from instancespace.stages.cloister import CloisterStage
from instancespace.stages.pilot import PilotStage
from instancespace.stages.prelim import PrelimStage
from instancespace.stages.preprocessing import PreprocessingStage
from instancespace.stages.pythia import PythiaStage
from instancespace.stages.sifted import SiftedStage
from instancespace.stages.trace import TraceStage

script_dir = Path(__file__).parent / "tests" / "test_data" / "demo"
# script_dir / "test_data/serialisers/actual_output" / directory

metadata_path = script_dir / "metadata.csv"
options_path = script_dir / "options.json"

metadata_object = metadata.from_csv_file(metadata_path)
options_object = options.from_json_file(options_path)

if metadata_object is None or options_object is None:
    print("ERR: File reading failed!")
    sys.exit()

instance_space = InstanceSpace(
    metadata_object,
    options_object,
    stages=[
        PreprocessingStage,
        PrelimStage,
        SiftedStage,
        PilotStage,
        PythiaStage,
        CloisterStage,
        TraceStage,
    ],
)

print(instance_space._runner._stage_order)  # noqa: SLF001

instance_space.build()
