import numpy as np
from numpy.typing import NDArray
from typing import List
from matilda.data.model import Opts, SiftedOut



def SIFTED(X: NDArray[np.double], Y: NDArray[np.double], Ybin: NDArray[np.bool_], opts: Opts.sifted) -> List[X,  SiftedOut] :
    # TODO: rewrite SIFTED logic in python
    raise NotImplementedError

def costfcn(comb, X:NDArray[np.double], Ybin:NDArray[np.bool_], ntrees, nwrokers) -> NDArray[np.double]:
    # TODO: rewrite SIFTED logic in python
    raise NotImplementedError

def fcnforga(idx, X, Ybin, ntrees, clust, nworkers) -> NDArray[np.double]:
    raise NotImplementedError