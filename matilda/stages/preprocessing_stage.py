"""Process the input data before running the main analysis."""
import stage
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from matilda.data.metadata import Metadata
from matilda.data.model import (
    Data,
    PreprocessingDataChanged,
    PreprocessingOut,
)
from matilda.data.options import InstanceSpaceOptions, PrelimOptions
from matilda.stages.filter import Filter

class preprocessingStage(stage):

    def __init__(self, data: Data, options: InstanceSpaceOptions) -> None:
        self.data = data
        self.options = options

    @staticmethod
    def _inputs() -> list[tuple[str, type]]:
        return [
            ["data", Data],
            ["options", InstanceSpaceOptions],
        ]

    @staticmethod
    def _outputs() -> list[tuple[str, type]]:
        return [
            ["data", PreprocessingDataChanged],
        ]

    def _run(self, options: InstanceSpaceOptions) -> PreprocessingOut:
        # All the code including the code in the buildIS should be here
        raise NotImplementedError
    
    @staticmethod
    def preprocessing(metadata: Metadata, options: InstanceSpaceOptions) -> Data:
        # All the code for preprocessing should be here
        raise NotImplementedError