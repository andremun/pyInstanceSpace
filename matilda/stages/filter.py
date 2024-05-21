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

from matilda.data.option import SelvarsOptions


class _FilterType(Enum):
    FTR = "Ftr"
    FTR_AP = "Ftr&AP"
    FTR_GOOD = "Ftr&Good"
    FTR_AP_GOOD = "Ftr&AP&Good"


class Filter:
    """See file docstring."""

    x: NDArray[np.double]
    y: NDArray[np.double]
    y_bin: NDArray[np.bool_]
    opts: SelvarsOptions
    n_insts: int
    n_algos: int
    n_feats: int

    def __init__(
            self,
            x: NDArray[np.double],
            y: NDArray[np.double],
            y_bin: NDArray[np.bool_],
            opts: SelvarsOptions,
        ) -> None:
        """Initialize the Filter stage.

        Args
        ----
            x (NDArray): The feature matrix (instances x features).
            y (NDArray): The algorithm matrix (instances x algorithm performances).
            y_bin (NDArray): Binary labels for algorithm perfromance from prelim.
            opts (SelvarsOptions): Selvas options
        """
        self.x = x
        self.y = y
        self.y_bin = y_bin
        self.opts = opts
        self.n_insts, self.n_algos = y.shape
        self.n_feats = x.shape[1]

    @staticmethod
    def run(
        x: NDArray[np.double],
        y: NDArray[np.double],
        y_bin: NDArray[np.bool_],
        opts: SelvarsOptions,
    ) -> tuple[NDArray[np.bool_], NDArray[np.bool_], NDArray[np.bool_], float]:
        """Filter instances based on distances and binary relations.

        Args
        ----
            x (np.ndarray): Feature instance matrix.
            y (np.ndarray): Algorithm performance matrix.
            y_bin (np.ndarray): Boolean performance matrix on algorithm from prelim.
            opts (Options): Options including 'mindistance' and 'type'.

        Returns
        -------
            subset_index (NDArray[np.bool_]): An array indicating whether each instance
                is excluded from the subset.
            is_dissimilar (NDArray[np.bool_]): An array indicating whether each instance
                is considered dissimilar.
            is_visa (NDArray[np.bool_]): An array indicating instances VISA flags.
        """
        data_filter = Filter(x, y, y_bin, opts)
        subset_index, is_dissimilar, is_visa = data_filter.filter_instance()
        uniformity = data_filter.compute_uniformity(subset_index)

        print(f"Uniformity of the instance subset: {uniformity:.4f}")

        return subset_index, is_dissimilar, is_visa, uniformity

    """
    % =========================================================================
    % SUBFUNCTIONS
    % =========================================================================
    """

    def filter_instance(
            self,
        ) -> tuple[NDArray[np.bool_], NDArray[np.bool_], NDArray[np.bool_]]:
        """Filter instances based on distances between feature and response vectors.

        Returns
        -------
            subset_index (NDArray[np.bool_]): An array indicating whether each instance
                is excluded from the subset.
            is_dissimilar (NDArray[np.bool_]): An array indicating whether each instance
                is considered dissimilar.
            is_visa (NDArray[np.bool_]): An array indicating instances VISA flags.
        """
        subset_index = np.zeros(self.n_insts, dtype=bool)
        is_dissimilar = np.ones(self.n_insts, dtype=bool)
        is_visa = np.zeros(self.n_insts, dtype=bool)

        gamma = np.sqrt(self.n_algos / self.n_feats) * self.opts.min_distance
        filter_type = _FilterType(self.opts.selvars_type)

        for i in range(self.n_insts):
            if subset_index[i]:
                continue

            for j in range(i + 1, self.n_insts):
                if subset_index[j]:
                    continue

                dx = cdist([self.x[i, :]], [self.x[j, :]]).item()
                dy = cdist([self.y[i, :]], [self.y[j, :]]).item()
                db = np.all(np.logical_and(self.y_bin[i, :], self.y_bin[j, :]))

                if dx <= self.opts.min_distance:
                    is_dissimilar[j] = False
                    if filter_type == _FilterType.FTR:
                        subset_index[j] = True
                    elif filter_type == _FilterType.FTR_AP:
                        subset_index[j], is_visa[j] = (
                            (True, False) if dy <= gamma else (False, True)
                        )
                    elif filter_type == _FilterType.FTR_GOOD:
                        subset_index[j], is_visa[j] = (
                            (True, False) if db else (False, True)
                        )
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

    def compute_uniformity(self, subset_index: NDArray[np.bool_]) -> float:
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
        d = squareform(pdist(self.x[~subset_index, :]))
        np.fill_diagonal(d, np.nan)
        nearest = np.nanmin(d, axis=0)
        return float(1 - (np.std(nearest, ddof=1) / np.mean(nearest)))
