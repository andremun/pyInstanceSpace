"""CLOISTER is using correlation to estimate a boundary for the space.

The function uses the correlation between the features and the performance of the
algorithms to estimate a boundary for the space. It constructs edges based on the
correlation between the features. The function then uses these edges to construct
a convex hull, providing a boundary estimate for the dataset.
"""

import numpy as np
from numpy.typing import NDArray

from matilda.data.model import CloisterOut, Options


class Cloister:
    """See file docstring."""

    @staticmethod
    def run(
        x: NDArray[np.double],
        a: NDArray[np.double],
        opts: Options,
    ) -> CloisterOut:
        """Estimate a boundary for the space using correlation.

        Args
            x: The feature matrix (instances x features) to process.
            a: A matrix, probably the performance matrix (instances x
                algorithms) to process. (Not sure now)
            opts: An object of type Options containing options for
                processing.

        Returns
        -------
            A structure containing Zedge and Zecorr
        """
        # TODO: Rewrite PRELIM logic in python
        raise NotImplementedError
