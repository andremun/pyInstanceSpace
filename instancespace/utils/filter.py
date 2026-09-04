# SPDX-License-Identifier: LicenseRef-PolyForm-Noncommercial-1.0.0
# Copyright (c) 2024-2026 Mario Andrés Muñoz
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
from loguru import logger
from numpy._typing import NDArray
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist

MIN_KEPT_INSTANCES_FOR_UNIFORMITY = 2


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
    needs_ap = filter_type in (_FilterType.FTR_AP, _FilterType.FTR_AP_GOOD)

    # A KD-tree built once over x answers exactly the question this function
    # needs -- which pairs are within min_distance -- without the O(n_insts^2)
    # cost of computing (and discarding) a distance for every pair regardless
    # of how far apart it is. query_ball_point(x, min_distance) mirrors
    # MATLAB's own one-shot `rangesearch(X, X, opts.mindistance)`; both use
    # inclusive (<=) distance semantics (verified directly).
    tree = cKDTree(x)
    neighbours = tree.query_ball_point(x, min_distance)

    for i in range(n_insts):
        if subset_index[i]:
            continue

        jj_list = [j for j in neighbours[i] if j > i]
        if not jj_list:
            continue

        # Only compute algorithm-performance distances for the (typically
        # small) set of instances already known to be feature-close to i,
        # instead of a full O(n_insts^2) dy -- dy is never read for any pair
        # farther apart than min_distance.
        dy_i = cdist(y[i : i + 1, :], y[jj_list, :])[0] if needs_ap else None

        # The elimination itself stays sequential, not just the neighbour
        # lookup: which instances end up marked redundant depends on the
        # running state of subset_index (an instance already marked
        # redundant is skipped as both a future i and j), so this greedy
        # process isn't safe to vectorise away without changing which
        # instances get kept. Processing order among the jj's for a fixed i
        # doesn't matter -- each jj's assignments below are independent of
        # every other jj considered for the same i -- only the outer i order
        # (0..n_insts-1) does.
        for k, j in enumerate(jj_list):
            if subset_index[j]:
                continue

            is_dissimilar[j] = False
            db = np.all(np.logical_and(y_bin[i, :], y_bin[j, :]))

            if filter_type == _FilterType.FTR:
                subset_index[j] = True
            elif filter_type == _FilterType.FTR_AP:
                assert dy_i is not None
                subset_index[j], is_visa[j] = (
                    (True, False) if dy_i[k] <= gamma else (False, True)
                )
            elif filter_type == _FilterType.FTR_GOOD:
                subset_index[j], is_visa[j] = (True, False) if db else (False, True)
            elif filter_type == _FilterType.FTR_AP_GOOD:
                assert dy_i is not None
                if db:
                    subset_index[j], is_visa[j] = (
                        (True, False) if dy_i[k] <= gamma else (False, True)
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

    Uniformity is undefined (F12) when fewer than 2 instances are retained, or
    when every retained instance coincides in feature space (mean
    nearest-neighbour distance of 0, which would otherwise divide-by-zero into
    inf/NaN) - matches MATLAB's `ISA:FILTER:degenerateUniformity` guard
    (`core/FILTER.m`), returning NaN with a warning instead of a silent,
    numpy-raised `RuntimeWarning` and a meaningless value.

    Args
    ----
        subset_index (NDArray[np.bool_]): An array indicating whether each instance
            is excluded from the subset.

    Returns
    -------
        uniformity (float): A score indicating the uniformity of the subset, or
            NaN if undefined for the reasons above.
    """
    x_kept = x[~subset_index, :]
    if x_kept.shape[0] < MIN_KEPT_INSTANCES_FOR_UNIFORMITY:
        nearest = np.array([])
    else:
        # k=2: the nearest neighbour of any point in its own reference set is
        # always itself, at distance 0 (column 0) -- column 1 is the nearest
        # OTHER point, matching MATLAB's knnsearch(Xkept, Xkept, 'K', 2) and
        # avoiding the O(n_kept^2) dense distance matrix the previous
        # pdist/squareform version built just to discard everything but its
        # per-row minimum.
        dist, _ = cKDTree(x_kept).query(x_kept, k=2)
        nearest = dist[:, 1]

    if (
        nearest.size < MIN_KEPT_INSTANCES_FOR_UNIFORMITY
        or np.all(np.isnan(nearest))
        or np.nanmean(nearest) == 0
    ):
        logger.warning(
            "[FILTER] Uniformity is undefined for the retained instance subset "
            "(fewer than 2 instances, or all retained instances coincide in "
            "feature space); returning NaN.",
        )
        return float("nan")

    return float(1 - (np.nanstd(nearest, ddof=1) / np.nanmean(nearest)))


def do_filter(
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
