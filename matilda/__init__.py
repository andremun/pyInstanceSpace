"""
Contains modules for instance space analysis.

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

from __future__ import annotations

from collections import defaultdict
from enum import Enum
from typing import Self

from matilda.data.metadata import Metadata
from matilda.data.model import Model
from matilda.data.option import Options
from matilda.prelim import Prelim


class _Stage(Enum):
    PRELIM = "prelim"
    SIFTED = "sifted"
    PILOT = "pilot"
    CLOISTER = "cloister"
    TRACE = "trace"
    PYTHIA = "pythia"

class StageError(Exception):
    """
    Prerequisite stages haven't been ran.

    An error raised when a user attempts to run a stage without first running any
    prerequisite stages.
    """

    pass

class InstanceSpace:
    """TODO: Describe what an instance space IS."""

    _stages: dict[_Stage, bool]
    _metadata: Metadata
    _options: Options

    _model: Model | None = None


    @staticmethod
    def new(metadata: Metadata, options: Options) -> InstanceSpace:
        """
        Create a new InstanceSpace object.

        Args:
        ----
            metadata (Metadata): _description_
            options (Options): _description_

        Returns:
        -------
            InstanceSpace: _description_

        """
        new = InstanceSpace()

        # Assigning to private member ok in constructor
        new._stages = defaultdict(lambda: False)  # noqa: SLF001
        new._metadata = metadata  # noqa: SLF001
        new._options = options  # noqa: SLF001

        return new


    def build(self: Self) -> Model:
        """
        Construct and return a Model object after instance space analysis.

        This runs all stages.

        Returns
        -------
            model: A Model object representing the built instance space.

        """
        raise NotImplementedError


    def prelim(self) -> None:
        self._stages[_Stage.PRELIM] = True

        # prelim_out = Prelim.run(
        #     self._metadata.,
        #     self._model.data.y,
        #     self._options,
        # )


    def sifted(self) -> None:
        if not self._stages[_Stage.PRELIM]:
            raise StageError

        self._stages[_Stage.SIFTED] = True

        raise NotImplementedError


    def pilot(self) -> None:
        if not self._stages[_Stage.SIFTED]:
            raise StageError

        self._stages[_Stage.PILOT] = True

        raise NotImplementedError


    def cloister(self) -> None:
        if not self._stages[_Stage.PILOT]:
            raise StageError

        self._stages[_Stage.CLOISTER] = True

        raise NotImplementedError


    def trace(self) -> None:
        if not self._stages[_Stage.PILOT]:
            raise StageError

        self._stages[_Stage.TRACE] = True

        raise NotImplementedError


    def pythia(self) -> None:
        if not self._stages[_Stage.PILOT]:
            raise StageError

        self._stages[_Stage.PYTHIA] = True

        raise NotImplementedError
