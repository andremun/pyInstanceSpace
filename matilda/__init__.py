"""
Contains modules for instance space analysis.

The module consists of various algorithms to perform instance space analysis.
- preprocessing: Prepare data to be used by stages.
- prelim: Performing preliminary data processing.
- sifted: Perform feature selection and optimization in data analysis.
- pilot: Obtaining a two-dimensional projection.
- pythia: Perform algorithm selection and performance evaluation using SVM.
- cloister: Perform correlation analysis to estimate a boundary for the space.
- trace: Calculating the algorithm footprints.

Perform instance space analysis on given dataset and configuration.

Construct an instance space from data and configuration files located in a specified
directory. The instance space is represented as a Model object, which encapsulates the
analytical results and metadata of the instance space analysis.
"""

from . import data, instance_space, stages
from .data import metadata, options
from .data.metadata import Metadata
from .data.options import InstanceSpaceOptions
from .instance_space import InstanceSpace
from .model import Model

__all__ = [
    "InstanceSpace",
    "InstanceSpaceOptions",
    "Metadata",
    "Model",
    "options",
    "metadata",
    "data",
    "stages",
    "instance_space",
    "stage_builder",
    "stage_runner",
]
