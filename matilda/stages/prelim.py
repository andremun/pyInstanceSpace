"""Performing preliminary data processing.

The main focus is on the `prelim` function, which prepares the input data for further
analysis and modeling.

The `prelim` function takes feature and performance data matrices along with a set of
processing options, and performs various preprocessing tasks such as normalization,
outlier detection and removal, and binary performance classification. These tasks are
guided by the options specified in the `InstanceSpaceOptions` object.
"""

import numpy as np
from numpy.typing import NDArray

from matilda.data.model import PrelimDataChanged, PrelimOut
from matilda.data.options import PrelimOptions


class Prelim:
    """See file docstring."""

    @staticmethod
    def run(
        x: NDArray[np.double],
        y: NDArray[np.double],
        opts: PrelimOptions,
    ) -> tuple[PrelimDataChanged, PrelimOut]:
        """Perform preliminary processing on the input data 'x' and 'y'.

        Args
            x: The feature matrix (instances x features) to process.
            y: The performance matrix (instances x algorithms) to
                process.
            opts: An object of type InstanceSpaceOptions containing options for
                processing.

        Returns
        -------
            A tuple containing the processed data (as 'Data' object) and
            preliminary output information (as 'PrelimOut' object).
        """
        # TODO: Rewrite PRELIM logic in python
        raise NotImplementedError
