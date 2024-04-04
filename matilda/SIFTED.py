import numpy as np
from numpy.typing import NDArray

from matilda.data.model import Opts, SiftedOut


def sifted(
    x: NDArray[np.double],
    y: NDArray[np.double],
    y_bin: NDArray[np.bool_],
    opts: Opts.sifted,
) -> tuple[NDArray[np.double], SiftedOut]:
    # TODO: rewrite SIFTED logic in python
    raise NotImplementedError


def cost_fcn(
    comb,
    x: NDArray[np.double],
    y_bin: NDArray[np.bool_],
    n_trees,
    n_wrokers,
) -> NDArray[np.double]:
    # TODO: rewrite SIFTED logic in python
    raise NotImplementedError


def fcn_forga(
    idx, x, y_bin: NDArray[np.bool_], n_trees, clust, n_workers,
) -> NDArray[np.double]:
    raise NotImplementedError
