"""Example of a plugin."""

import sys
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from instancespace import InstanceSpace
from instancespace.data import metadata, options
from instancespace.stages.cloister import CloisterStage
from instancespace.stages.pilot import PilotStage
from instancespace.stages.prelim import PrelimStage
from instancespace.stages.preprocessing import PreprocessingStage
from instancespace.stages.pythia import PythiaStage
from instancespace.stages.sifted import SiftedStage
from instancespace.stages.stage import Stage
from instancespace.stages.trace import TraceStage


class ExamplePluginInput(NamedTuple):  # noqa: D101
    accuracy: list[float]
    precision: list[float]
    recall: list[float]
    selection0: NDArray[np.int_]
    selection1: NDArray[np.int_]
    pythia_summary: pd.DataFrame


class ExamplePluginOutput(NamedTuple):  # noqa: D101
    # Output can't be empty
    blank: str


class ExamplePlugin(Stage[ExamplePluginInput, ExamplePluginOutput]):  # noqa: D101
    @staticmethod
    def _inputs() -> type[NamedTuple]:
        return ExamplePluginInput

    @staticmethod
    def _outputs() -> type[NamedTuple]:
        return ExamplePluginOutput

    @staticmethod
    def _run(inputs: ExamplePluginInput) -> ExamplePluginOutput:
        print("Running example plugin")

        if inputs.accuracy is not None:
            print("accuracy:")
            print(inputs.accuracy)

        if inputs.precision is not None:
            print("precision:")
            print(inputs.precision)

        if inputs.recall is not None:
            print("recall:")
            print(inputs.recall)

        if inputs.selection0 is not None:
            print("selection0:")
            print(inputs.selection0)

        if inputs.selection1 is not None:
            print("selection1:")
            print(inputs.selection1)

        if inputs.pythia_summary is not None:
            print("pythia_summary:")
            print(inputs.pythia_summary)

        return ExamplePluginOutput(blank="")


script_dir = Path(__file__).parent / "tests" / "test_data" / "demo"

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
        ExamplePlugin,
    ],
)

print(instance_space._runner._stage_order)  # noqa: SLF001

instance_space.build()
