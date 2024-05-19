"""Contains modules for instance space analysis.

The module consists of various algorithms to perform instance space analysis.
    - build: Perform instance space analysis on given dataset and configuration.
    - prelim: Performing preliminary data processing.
    - sifted: Perform feature selection and optimization in data analysis.
    - pythia: Perform algorithm selection and performance evaluation using SVM.
    - cloister: Perform correlation analysis to estimate a boundary for the space.
    - pilot: Obtaining a two-dimensional projection.
    - trace: Calculating the algorithm footprints.
    - example: IPython Notebook to run analysis on local machine

Perform instance space analysis on given dataset and configuration.

Construct an instance space from data and configuration files located in a specified
directory. The instance space is represented as a Model object, which encapsulates the
analytical results and metadata of the instance space analysis.

The main function in this module, `build_instance_space`, reads the necessary
data from the provided directory, performs instance space analysis, and then
constructs a Model object that represents this analysis. This Model object can
then be used for further analysis, visualization, or processing within the
larger framework of the Matilda data analysis suite.

Functions:
    build_instance_space(rootdir: str) -> Model:
        Construct and return a Model object after performing instance space analysis
        on the data and configurations found in the specified root directory.

Example usage:
    python your_module_name.py /path/to/data
"""

from . import data, instance_space, stages
from .data.metadata import Metadata
from .data.option import Options
from .instance_space import InstanceSpace

__all__ = ["InstanceSpace", "Options", "Metadata", "data", "stages", "instance_space"]

