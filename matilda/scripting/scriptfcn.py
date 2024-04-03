import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from typing import List, Set, Any
from numpy.typing import NDArray


def writeArray2CSV(
    data: NDArray[np.double], colnames: List[str], rownames: List[str], filename: str
) -> None:
    # TODO: implement the logic
    raise NotImplementedError


def writeCell2CSV(
    data: pd.DataFrame, colnames: List[str], rownames: List[str], filename: str
) -> None:
    # TODO: implement the logic
    raise NotImplementedError


def makeBndLabels(data: NDArray[np.double]) -> List[str]:
    # TODO: implement the logic
    raise NotImplementedError


def colorscale(data: NDArray[np.double]) -> NDArray[np.double]:
    # TODO: implement the logic
    raise NotImplementedError


def colorscaleg(data: NDArray[np.double]) -> NDArray[np.double]:
    # TODO: implement the logic
    raise NotImplementedError


def drawSources(Z: NDArray[np.double], S: Set[str]) -> None:
    # TODO: implement the logic
    raise NotImplementedError


def drawScatter(Z: NDArray[np.double], X: NDArray[np.double], titlelabel: str) -> None:
    # TODO: implement the logic
    raise NotImplementedError


def drawPortfolioSelections(
    Z: NDArray[np.double], P: NDArray[np.double], algolabels: List[str], titlelabel: str
) -> None:
    # TODO: implement the logic
    raise NotImplementedError


def drawPortfolioFootprint(
    Z: NDArray[np.double], best: List[Any], P: NDArray[np.double], algolabels: List[str]
) -> None:
    # TODO: update type declaration for 'best' with TraceOut.best type from model.py
    # TODO: implement the logic
    raise NotImplementedError


def drawGoodBadFootprint(
    Z: NDArray[np.double], good: List[Any], Ybin: NDArray[np.bool_], titlelabel: str
) -> None:
    # TODO: update type declaration for 'good' with TraceOut.good[i] type (individual instance)
    #       from model.py
    # TODO: implement the logic
    raise NotImplementedError


def drawFootprint(footprint: List[Any], color: List[float], alpha: float) -> None:
    # TODO: update type declaration for 'footprint' with TraceOut.good[i] or best[i] type
    #       (individual instance) from model.py
    # TODO: implement the logic
    raise NotImplementedError


def drawBinaryPerformance(
    Z: NDArray[np.double], Ybin: NDArray[np.bool_], titlelabel: str
) -> None:
    # TODO: implement the logic
    raise NotImplementedError
