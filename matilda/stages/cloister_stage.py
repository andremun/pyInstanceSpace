"""CLOISTER is using correlation to estimate a boundary for the space.

The function uses the correlation between the features and the performance of the
algorithms to estimate a boundary for the space. It constructs edges based on the
correlation between the features. The function then uses these edges to construct
a convex hull, providing a boundary estimate for the dataset.
"""

import numpy as np
from numpy.typing import NDArray
from stages.stage import Stage

from matilda.data.options import CloisterOptions


class CloisterStage(Stage):
    """See file docstring."""

    def __init__(self, x: NDArray[np.double], a: NDArray[np.double]) -> None:
        """See file docstring."""
        self.x = x
        self.a = a

    @staticmethod
    def _inputs() -> list[tuple[str, type]]:
        """See file docstring."""
        return [["x", NDArray[np.double]], ["a", NDArray[np.double]]]

    @staticmethod
    def _outputs() -> list[tuple[str, type]]:
        """See file docstring."""
        return [["z_edge", NDArray[np.double]], ["z_ecorr", NDArray[np.double]]]

    def _run(
        self,
        options: CloisterOptions,
    ) -> tuple[NDArray[np.double], NDArray[np.double]]:
        """See file docstring."""
        # Implement code that goes into buildIS here
        raise NotImplementedError

    @staticmethod
    def cloister(
        x: NDArray[np.double],
        a: NDArray[np.double],
    ) -> tuple[NDArray[np.double], NDArray[np.double]]:
        """See file docstring."""
        # Implement code from CLOISTER.m here
        raise NotImplementedError
