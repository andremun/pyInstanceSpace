from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray


def write_array_to_csv(
    data: NDArray[np.double],
    col_names: list[str],
    row_names: list[str],
    file_name: str,
) -> None:
    """
    Write a NumPy array to a CSV file, including row and column headers.

    :param data: The 2D array of data to write.
    :param col_names: List of column names for the CSV.
    :param row_names: List of row names for the CSV.
    :param file_name: Name of the file to write the CSV.
    """
    # TODO: implement the logic
    raise NotImplementedError


def write_cell_to_csv(
    data: pd.DataFrame,
    col_names: list[str],
    row_names: list[str],
    file_name: str,
) -> None:
    """
    Write a pandas DataFrame to a CSV file, using specified row and column headers.

    :param data: The DataFrame to write.
    :param col_names: List of column names for the CSV.
    :param row_names: List of row names for the CSV.
    :param file_name: Name of the file to write the CSV.
    """
    # TODO: implement the logic
    raise NotImplementedError


def make_bnd_labels(data: NDArray[np.double]) -> list[str]:
    """
    Generate boundary labels for the given data array.

    :param data: The data array for which to generate labels.
    :return: A list of boundary labels.
    """
    # TODO: implement the logic
    raise NotImplementedError


def color_scale(data: NDArray[np.double]) -> NDArray[np.double]:
    """
    Apply a color scaling transformation to the given data.

    :param data: The data array to transform.
    :return: The color-scaled data array.
    """
    # TODO: implement the logic
    raise NotImplementedError


def color_scaleg(data: NDArray[np.double]) -> NDArray[np.double]:
    """
    Apply a grayscale color scaling transformation to the given data.

    :param data: The data array to transform.
    :return: The grayscale color-scaled data array.
    """
    # TODO: implement the logic
    raise NotImplementedError


def draw_sources(z: NDArray[np.double], s: set[str]) -> None:
    """
    Draw source points from the given set onto the specified data.

    :param z: The data array on which to draw.
    :param s: The set of sources to draw.
    """
    # TODO: implement the logic
    raise NotImplementedError


def draw_scatter(
    z: NDArray[np.double], x: NDArray[np.double], title_label: str,
) -> None:
    """
    Create a scatter plot of the given data.

    :param z: The data for the x-axis.
    :param x: The data for the y-axis.
    :param title_label: The title for the scatter plot.
    """
    # TODO: implement the logic
    raise NotImplementedError


def draw_portfolio_selections(
    z: NDArray[np.double],
    p: NDArray[np.double],
    algo_labels: list[str],
    title_label: str,
) -> None:
    """
    Draw a portfolio selection plot using the given data and algorithm labels.

    :param z: The data array for the portfolio.
    :param p: The performance data array.
    :param algolabels: The labels of the algorithms used.
    :param titlelabel: The title of the plot.
    """
    # TODO: implement the logic
    raise NotImplementedError


def draw_portfolio_footprint(
    z: NDArray[np.double],
    best: list[Any],
    p: NDArray[np.double],
    algo_labels: list[str],
) -> None:
    """
    Draw a footprint plot for the portfolio selections.

    :param z: The data array for the portfolio.
    :param best: A list representing the best selections.
    :param p: The performance data array.
    :param algolabels: The labels of the algorithms used.
    """
    # TODO: update type declaration for 'best' with TraceOut.best type from model.py
    # TODO: implement the logic
    raise NotImplementedError


def draw_good_bad_footprint(
    z: NDArray[np.double],
    good: list[Any],
    y_bin: NDArray[np.bool_],
    title_label: str,
) -> None:
    """
    Draw a footprint plot distinguishing good and bad selections.

    :param z: The data array for the footprint.
    :param good: A list of good selection data.
    :param Ybin: A binary array indicating good and bad selections.
    :param title_label: The title of the plot.
    """
    # TODO: update type declaration for 'good' with TraceOut.good[i] type
    #       (individual instance) from model.py
    # TODO: implement the logic
    raise NotImplementedError


def draw_footprint(footprint: list[Any], color: list[float], alpha: float) -> None:
    """
    Draw a footprint plot with specified color and transparency settings.

    :param footprint: A list of footprint data.
    :param color: A list defining the color of the footprint.
    :param alpha: The transparency level of the footprint.
    """
    # TODO: update type declaration for 'footprint' with TraceOut.good[i] or best[i]
    #       type (individual instance) from model.py
    # TODO: implement the logic
    raise NotImplementedError


def draw_binary_performance(
    z: NDArray[np.double],
    y_bin: NDArray[np.bool_],
    title_label: str,
) -> None:
    """
    Draw a binary performance plot based on the given data.

    :param z: The data array used for plotting.
    :param y_bin: A binary array indicating the performance outcome (good or bad).
    :param title_label: The title of the plot.
    """
    # TODO: implement the logic
    raise NotImplementedError
