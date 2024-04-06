import numpy as np
from numpy.typing import NDArray

from matilda.data.model import SiftedOut
from matilda.data.option import SiftedOptions


def sifted(
    x: NDArray[np.double],
    y: NDArray[np.double],
    y_bin: NDArray[np.bool_],
    opts: SiftedOptions,
) -> tuple[NDArray[np.double], SiftedOut]:
    # TODO: rewrite SIFTED logic in python
    raise NotImplementedError


def cost_fcn(
    comb: NDArray[np.double], #not sure about the type
    x: NDArray[np.double],
    y_bin: NDArray[np.bool_],
    n_trees: int,
    n_wrokers: int,
) -> NDArray[np.double]:
    # TODO: rewrite SIFTED logic in python
    raise NotImplementedError


def fcn_forga(
    idx: NDArray[np.intc],
    x: NDArray[np.double],
    y_bin: NDArray[np.bool_],
    n_trees: int,
    clust: NDArray[np.bool_],
    n_workers: int,
) -> NDArray[np.double]:
    raise NotImplementedError
