
import numpy as np
from numpy import double
from numpy._typing import NDArray

from matilda.data.model import Footprint, PolyShape, TraceOut
from matilda.data.option import TraceOptions

"""
By Chen.
Note that for the parameter: algolabels, still deciding it's type.

"""


def trace(
    z: NDArray[np.double],
    y_bin: NDArray[np.bool_],
    p: NDArray[np.double],
    beta: NDArray[np.bool_],
    algo_labels: list[str],
    opts: TraceOptions,
) -> TraceOut:

    # TODO: Rewrite TRACE logic in python
    raise NotImplementedError


"""
% =========================================================================
% SUBFUNCTIONS
% =========================================================================
"""


def trace_build(
    z: NDArray[np.double], 
    y_bin: NDArray[np.bool_], 
    opts: TraceOptions,
) -> Footprint:
    # TODO: Rewrite TRACEbuild logic in python
    raise NotImplementedError


def trace_contra(
    base: Footprint,
    test: Footprint,
    z: NDArray[np.double],
    y_base: NDArray[np.bool_],
    y_test: NDArray[np.bool_],
    opts: TraceOptions,
) -> tuple[Footprint, Footprint]:
    # not sure whether the returned value is tuple or list, needs further decision
    # TODO: Rewrite TRACEcontra logic in python
    raise NotImplementedError


def trace_tight(
    polygon: PolyShape,
    z: NDArray[np.double],
    y_bin: NDArray[np.bool_],
    opts: TraceOptions,
) -> PolyShape:
    # TODO: Rewrite TRACEtight logic in python
    raise NotImplementedError


# note that for polydata, it is  highly probably a 2 dimensional array
def trace_fitpoly(
    poly_data: NDArray[np.double],
    z: NDArray[np.double],
    y_bin: NDArray[np.bool_],
    opts: TraceOptions,
) -> PolyShape:
    # TODO: Rewrite TRACEfitpoly logic in python
    raise NotImplementedError


def trace_summary(
    footprint: Footprint, 
    space_area: double, 
    space_density: double,
) -> list[float]:
    # TODO: Rewrite TRACEsummary logic in python
    raise NotImplementedError


def trace_throw() -> Footprint:
    # TODO: Rewrite TRACEthrow logic in python
    # footprint.polygon = [], other contents will be 0
    raise NotImplementedError


def dbscan(
    x: NDArray[np.double], 
    k: int, eps: double,
) -> tuple[NDArray[np.intc], NDArray[np.intc]]:
    # TODO: Rewrite dbscan logic in python
    # not sure whether the returned value is tuple or list, needs further decision
    # note that parameter:Eps could be dropped.
    raise NotImplementedError


def epsilon(x: NDArray[np.double], k: int) -> double:
    # TODO: Rewrite epsilon logic in python
    raise NotImplementedError


def dist(i: NDArray[np.double], x: NDArray[np.double]) -> NDArray[np.double]:
    # TODO: Rewrite dist logic in python
    raise NotImplementedError
