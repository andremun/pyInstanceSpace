import numpy as np
from numpy import double
from numpy._typing import NDArray
from typing import List, Set

from matilda.data.model import Model, Footprint, TraceOut, polyshape
from matilda.data.option import Opts, TraceOptions

"""
By Chen.
Note that for the parameter: algolabels, still deciding it's type.

"""
def TRACE(Z: NDArray[np.double] , Ybin: NDArray(bool), P:NDArray(double),
          beta:  NDArray[np.bool_], algolabels: List[str], opts: TraceOptions) -> TraceOut:
    # TODO: Rewrite TRACE logic in python
    raise NotImplementedError


"""
% =========================================================================
% SUBFUNCTIONS
% =========================================================================
"""

def TRACEbuild(Z: NDArray[np.double], Ybin: NDArray(bool), opts: TraceOptions)\
        -> Footprint:
    # TODO: Rewrite TRACEbuild logic in python
    raise NotImplementedError

def TRACEcontra(base:Footprint ,test:Footprint,Z:NDArray[np.double],
                Ybase: NDArray(bool),Ytest:NDArray(bool), opts:TraceOptions) -> [Footprint, Footprint]:
    # TODO: Rewrite TRACEcontra logic in python
    raise NotImplementedError

def TRACEtight(polygon:polyshape,Z:NDArray[np.double],Ybin:NDArray(bool),opts:TraceOptions) -> polyshape:
    # TODO: Rewrite TRACEtight logic in python
    raise NotImplementedError

#note that for polydata, it is  highly probably a 2 dimensional array
def TRACEfitpoly(polydata:NDArray[np.double] ,NDArray: [np.double],Ybin: NDArray(bool),opts:TraceOptions ) -> polyshape:
    # TODO: Rewrite TRACEfitpoly logic in python
    raise NotImplementedError


def TRACEsummary(footprint: Footprint, spaceArea: double, spaceDensity: double) ->[double,double,double,double,double]:
    # TODO: Rewrite TRACEsummary logic in python
    raise NotImplementedError

def TRACEthrow()-> Footprint:
    # TODO: Rewrite TRACEthrow logic in python
    #footprint.polygon = [], other contents will be 0
    raise NotImplementedError


def dbscan(x:NDArray[np.double] ,k: int, Eps: double) -> [NDArray[int],NDArray[int]]:
    # TODO: Rewrite dbscan logic in python
    # note that parameter:Eps could be dropped.
    raise NotImplementedError

def epsilon(x:NDArray[np.double] ,k: int) -> double:
    # TODO: Rewrite epsilon logic in python
    raise NotImplementedError

def dist(i:NDArray[np.double],x:NDArray[np.double]) -> NDArray[np.double]:
    # TODO: Rewrite dist logic in python
    raise NotImplementedError
