"""Testing script for integration."""

import sys
from pathlib import Path
from time import perf_counter, time

import pandas as pd

from matilda import InstanceSpace
from matilda.data import metadata, options
from matilda.stages.cloister import CloisterStage
from matilda.stages.pilot import PilotStage
from matilda.stages.prelim import PrelimStage
from matilda.stages.preprocessing import PreprocessingStage
from matilda.stages.pythia import PythiaStage
from matilda.stages.sifted import SiftedStage
from matilda.stages.trace import TraceStage

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
start = perf_counter()
# instance_space.run_stage(PreprocessingStage,
#                     feature_names=metadata_object.feature_names,
#                     algorithm_names=metadata_object.algorithm_names,
#                     instance_labels=metadata_object.instance_labels,
#                     instance_sources=metadata_object.instance_sources,
#                     features=metadata_object.features,
#                     algorithms=metadata_object.algorithms,
#                     selvars_options=options_object.selvars)
print(perf_counter() - start)
print(instance_space._runner.run_iter)  # noqa: SLF001

summary = list()#pd.DataFrame(columns=['Iteration', 'Time'])
for i in range(1, 10):
    start = perf_counter()
    instance_space.build()
    elapsed = perf_counter() - start
    print(f"Elapsed time: {elapsed}")
    summary.append(elapsed)
print(elapsed)