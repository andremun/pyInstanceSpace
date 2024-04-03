from typing import List

import numpy as np
from numpy._typing import NDArray

from matilda.data.model import Model
from matilda.data.option import Opts

"""
By Chen

    Note: NOT quite sure where this function were used
    In the buildIS, this function's x parameter's size could be changed,
    so I treat x type is NDArray[np.double], rather than Model.data.X

"""

def FILTER(x: NDArray[np.double], y: NDArray[np.double], opts: Opts.selvars) -> \
        List[NDArray[bool], NDArray[bool], NDArray[bool]]:

    # TODO: Rewrite FILTER logic in python
    raise NotImplementedError
