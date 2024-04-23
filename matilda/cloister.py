"""
CLOISTER is using correlation to estimate a boundary for the space.

The function uses the correlation between the features and the performance of the
algorithms to estimate a boundary for the space. It constructs edges based on the
correlation between the features. The function then uses these edges to construct
a convex hull, providing a boundary estimate for the dataset.
"""

import numpy as np
from numpy.typing import NDArray

from matilda.data.model import CloisterOut, Options


def cloister(
    x: NDArray[np.double],
    a: NDArray[np.double],
    opts: Options,
) -> CloisterOut:
    """
    Estimate a boundary for the space using correlation.

    :param X: The feature matrix (instances x features) to process.
    :param A: A matrix, probably the performance matrix (instances x algorithms)
      to process. (Not sure now)
    :param opts: An object of type Options containing options for processing.
    :return: A structure containing Zedge and Zecorr

    """
    # TODO: Rewrite PRELIM logic in python
    raise NotImplementedError
