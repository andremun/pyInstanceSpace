"""Filter data instances based on pairwise distances and compute uniformity.

This module implements a filtering mechanism to identify and classify data instances
based on pairwise distances between feature and response vectors. The filtering criteria
for identifying subsets include factors such as feature distances, response distances,
and binary classification labels.

The `_FilterType` enum class is used to differentiate between various filtering
strategies.
"""

from enum import Enum

import numpy as np
from numpy._typing import NDArray
from scipy.spatial.distance import cdist, pdist, squareform


class _FilterType(Enum):
    # similarity based on the features
    FTR = "Ftr"
    # both features and Algorithmic Performances (APs) with Euclidian distance
    FTR_AP = "Ftr&AP"
    # features with Euclidian distance and APs goodness
    FTR_GOOD = "Ftr&Good"
    # features with Euclidian distance and APs with both Euclidian distance and goodness
    FTR_AP_GOOD = "Ftr&AP&Good"


def filter_instance(
    x: NDArray[np.double],
    y: NDArray[np.double],
    y_bin: NDArray[np.bool_],
    selvars_type: str,
    min_distance: float,
) -> tuple[NDArray[np.bool_], NDArray[np.bool_], NDArray[np.bool_]]:
    """Filter instances based on distances and binary relations.

    Args
    ----
        x (np.ndarray): Feature instance matrix.
        y (np.ndarray): Algorithm performance matrix.
        y_bin (np.ndarray): Boolean performance matrix on algorithm from prelim.
        Options including 'min_distance' and 'selvars_type'.

    Returns
    -------
        subset_index (NDArray[np.bool_]): An array indicating whether each instance
            is excluded from the subset.
        is_dissimilar (NDArray[np.bool_]): An array indicating whether each instance
            is considered dissimilar.
        is_visa (NDArray[np.bool_]): An array indicating instances VISA flags.
    """
    n_insts, n_algos = y.shape
    n_feats = x.shape[1]

    subset_index = np.zeros(n_insts, dtype=bool)
    is_dissimilar = np.ones(n_insts, dtype=bool)
    is_visa = np.zeros(n_insts, dtype=bool)

    gamma = np.sqrt(n_algos / n_feats) * min_distance
    filter_type = _FilterType(selvars_type)

    for i in range(n_insts):
        if subset_index[i]:
            continue

        for j in range(i + 1, n_insts):
            if subset_index[j]:
                continue

            dx = cdist([x[i, :]], [x[j, :]]).item()
            dy = cdist([y[i, :]], [y[j, :]]).item()
            db = np.all(np.logical_and(y_bin[i, :], y_bin[j, :]))

            if dx <= min_distance:
                is_dissimilar[j] = False
                if filter_type == _FilterType.FTR:
                    subset_index[j] = True
                elif filter_type == _FilterType.FTR_AP:
                    subset_index[j], is_visa[j] = (
                        (True, False) if dy <= gamma else (False, True)
                    )
                elif filter_type == _FilterType.FTR_GOOD:
                    subset_index[j], is_visa[j] = (True, False) if db else (False, True)
                elif filter_type == _FilterType.FTR_AP_GOOD:
                    if db:
                        subset_index[j], is_visa[j] = (
                            (True, False) if dy <= gamma else (False, True)
                        )
                    else:
                        is_visa[j] = True
                else:
                    print("Invalid flag!")

    return subset_index, is_dissimilar, is_visa


def compute_uniformity(x: NDArray[np.double], subset_index: NDArray[np.bool_]) -> float:
    """Calculate the uniformity of the selected subset based on distances.

    The function computes pairwise distances between all selected instances that
    have not been excluded. It calculates the ratio between the standard deviation
    and mean of the nearest-neighbor distances and returns a uniformity score as
    1 minus this ratio.

    Args
    ----
        subset_index (NDArray[np.bool_]): An array indicating whether each instance
            is excluded from the subset.

    Returns
    -------
        uniformity (float): A score indicating the uniformity of the subset.
    """
    d = squareform(pdist(x[~subset_index, :]))
    np.fill_diagonal(d, np.nan)
    nearest = np.nanmin(d, axis=0)
    return float(1 - (np.std(nearest, ddof=1) / np.mean(nearest)))


def filter(
    x: NDArray[np.double],
    y: NDArray[np.double],
    y_bin: NDArray[np.bool_],
    selvars_type: str,
    min_distance: float,
) -> tuple[NDArray[np.bool_], NDArray[np.bool_], NDArray[np.bool_], float]:
    """Filter instances based on distances and binary relations.

    Args
    ----
        x (np.ndarray): Feature instance matrix.
        y (np.ndarray): Algorithm performance matrix.
        y_bin (np.ndarray): Boolean performance matrix on algorithm from prelim.
        Options including 'mindistance' and 'type'.

    Returns
    -------
        subset_index (NDArray[np.bool_]): An array indicating whether each instance
            is excluded from the subset.
        is_dissimilar (NDArray[np.bool_]): An array indicating whether each instance
            is considered dissimilar.
        is_visa (NDArray[np.bool_]): An array indicating instances VISA flags.
    """
    subset_index, is_dissimilar, is_visa = filter_instance(
        x,
        y,
        y_bin,
        selvars_type,
        min_distance,
    )
    uniformity = compute_uniformity(x, subset_index)

    print(f"Uniformity of the instance subset: {uniformity:.4f}")

    return subset_index, is_dissimilar, is_visa, uniformity
