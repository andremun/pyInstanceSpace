from inspect import signature

import numpy as np
from numpy.typing import NDArray

from matilda.conductor import Conductor
from matilda.data.options import CloisterOptions
from matilda.stages.cloister import Cloister

a: NDArray[np.double] = np.array([])
b: NDArray[np.double] = np.array([])
c = CloisterOptions.default()

conductor = Conductor(
    [Cloister],
    [
        a,
        b,
        c,
    ],
)
