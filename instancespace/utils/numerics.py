# SPDX-License-Identifier: LicenseRef-PolyForm-Noncommercial-1.0.0
# Copyright (c) 2024-2026 Mario Andrés Muñoz
"""Numerical helpers whose semantics intentionally follow MATLAB."""

from decimal import ROUND_HALF_UP, Decimal
from typing import overload

import numpy as np
from numpy.typing import NDArray


@overload
def matlab_round(values: float, decimals: int = 0) -> float: ...


@overload
def matlab_round(
    values: NDArray[np.double],
    decimals: int = 0,
) -> NDArray[np.double]: ...


def matlab_round(
    values: float | NDArray[np.double],
    decimals: int = 0,
) -> float | NDArray[np.double]:
    """Round decimal ties away from zero, matching MATLAB ``round``."""
    source = np.asarray(values, dtype=np.double)
    rounded = source.copy()
    quantum = Decimal(1).scaleb(-decimals)
    for index in np.ndindex(source.shape):
        value = float(source[index])
        if np.isfinite(value):
            rounded[index] = float(
                Decimal(str(value)).quantize(quantum, rounding=ROUND_HALF_UP),
            )
    if rounded.ndim == 0:
        return float(rounded.item())
    return rounded
