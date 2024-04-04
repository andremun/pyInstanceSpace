import numpy as np
from numpy.typing import NDArray

from matilda.data.model import Data, PrelimOut
from matilda.data.option import Opts


def prelim(
    x: NDArray[np.double],
    y: NDArray[np.double],
    opts: Opts,
) -> tuple[Data, PrelimOut]:
    """
    Perform preliminary processing on the input data 'x' and 'y'.
      
    The function preprocess the data by applying operations such as normalization,
    outlier handling, and binary performance classification based on the specified
    options in 'opts'.

    :param x: The feature matrix (instances x features) to process.
    :param y: The performance matrix (instances x algorithms) to process.
    :param opts: An object of type Opts containing options for processing.
    :return: A tuple containing the processed data (as 'Data' object) and preliminary
             output information (as 'PrelimOut' object).
    """
    # TODO: Rewrite PRELIM logic in python
    raise NotImplementedError
