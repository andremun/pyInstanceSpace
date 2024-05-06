import numpy as np
from numpy._typing import NDArray

from matilda.data.model import Data, PrelimOut
from matilda.data.option import Options


class PrePro:
    """See file docstring."""

    @staticmethod
    def run(
        x: NDArray[np.double],
        y: NDArray[np.double],
        opts: Options,
    ) -> tuple[Data, PrelimOut]:
        """Perform preliminary processing on the input data 'x' and 'y'.

        Args
            x: The feature matrix (instances x features) to process.
            y: The performance matrix (instances x algorithms) to
                process.
            opts: An object of type Options containing options for
                processing.

        Returns
        -------
            A tuple containing the processed data (as 'Data' object) and
            preliminary output information (as 'PrelimOut' object).
        """
        # TODO: Rewrite PRELIM logic in python
        raise NotImplementedError
