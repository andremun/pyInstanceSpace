import numpy as np
from numpy._typing import NDArray

from matilda.data.model import Model, PilotOut
from matilda.data.option import Opts

"""
By Chen

"""


def pilot(
    x: NDArray[np.double],
    y: NDArray[np.double],
    feat_labels: Model.data.feat_labels,
    opts: Opts.pilot,
) -> PilotOut:
    """
    
    """
    # TODO: Rewrite PILOT logic in python
    raise NotImplementedError
