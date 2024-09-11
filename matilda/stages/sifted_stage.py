"""Stage for SIFTED algorithm."""

import numpy as np
from numpy.typing import NDArray
from stages.stage import Stage

from matilda.data.options import SiftedOptions


class SiftedStage(Stage):
    """See file docstring."""

    def __init__(
        self,
        x: NDArray[np.double],
        y: NDArray[np.double],
        y_bin: NDArray[np.bool_],
        opts: SiftedOptions,
    ) -> None:
        """See file docstring."""
        self.x = x
        self.y = y
        self.y_bin = y_bin
        self.opts = opts

    @staticmethod
    def _inputs() -> list[tuple[str, type]]:
        """See file docstring."""
        return [
            ["x", NDArray[np.double]],
            ["y", NDArray[np.double]],
            ["y_bin", NDArray[np.bool_]],
            ["opts", SiftedOptions],
        ]

    @staticmethod
    def _outputs() -> list[tuple[str, type]]:
        """See file docstring."""
        return [
            ["flag", int],
            ["rho", np.double],
            ["k", int],
            ["n_trees", int],
            ["max_lter", int],
            ["replicates", int],
            ["idx", NDArray[np.int_]],
        ]

    def _run(
        self,
        x: NDArray[np.double],
        y: NDArray[np.double],
        y_bin: NDArray[np.bool_],
        opts: SiftedOptions,
    ) -> tuple[int, np.double, int, int, int, int, NDArray[np.int_]]:
        """See file docstring."""
        raise NotImplementedError

    @staticmethod
    def sifted(
        x: NDArray[np.double],
        y: NDArray[np.double],
        y_bin: NDArray[np.bool_],
        opts: SiftedOptions,
    ) -> tuple[int, np.double, int, int, int, int, NDArray[np.int_]]:
        """See file docstring."""
        raise NotImplementedError
