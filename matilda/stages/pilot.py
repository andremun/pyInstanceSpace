"""PILOT: Obtaining a two-dimensional projection.

Projecting Instances with Linearly Observable Trends (PILOT)
is a dimensionality reduction algorithm which aims to facilitate
the identification of relationships between instances and
algorithms by unveiling linear trends in the data, increasing
from one edge of the space to the opposite.

"""

import numpy as np
from numpy.typing import NDArray

from matilda.data.model import PilotOut
from matilda.data.option import PilotOptions


class Pilot:
    """See file docstring."""

    @staticmethod
    def run(
        x: NDArray[np.double],
        y: NDArray[np.double],
        feat_labels: list[str],
        opts: PilotOptions,
    ) -> PilotOut:
        """Produce the final subset of features.

        opts.pilot.analytic determines whether the analytic (set as TRUE) or the
        numerical (set as FALSE) solution to be adopted.

        opts.pilot.ntries number of iterations that the numerical solution is attempted.
        """
        # TODO: Rewrite PILOT logic in python
        raise NotImplementedError