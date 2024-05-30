"""By Chen Note: NOT quite sure where this function was used.

In the buildIS, this function's x parameter's size could be changed,
so I treat x type as NDArray[np.double], rather than Model.data.X.
"""

import numpy as np
from numpy._typing import NDArray

from matilda.data.options import SelvarsOptions


class Filter:
    """See file docstring."""

    @staticmethod
    def run(
        x: NDArray[np.double],
        y: NDArray[np.double],
        y_bin: NDArray[np.bool_],
        opts: SelvarsOptions,
    ) -> tuple[NDArray[np.bool_], NDArray[np.bool_], NDArray[np.bool_], float]:
        """NOT quite sure where this function was used.

        Note that the return value, based on the original Matlab code,
        that is the List (or cell?) needs further justification about what
        data types will be adopted.
        """
        # TODO: Rewrite FILTER logic in python
        raise NotImplementedError
