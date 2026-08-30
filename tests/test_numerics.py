"""Tests for shared numerical compatibility helpers."""

import numpy as np

from instancespace.utils.numerics import matlab_round


def test_matlab_rounds_decimal_ties_away_from_zero() -> None:
    """Exact decimal ties follow MATLAB rather than NumPy's ties-to-even rule."""
    values = np.array([0.5, -0.5, 1.25, -1.25], dtype=np.double)

    np.testing.assert_array_equal(
        matlab_round(values, 1),
        [0.5, -0.5, 1.3, -1.3],
    )


def test_matlab_round_handles_scalars_and_nonfinite_values() -> None:
    """The shared helper preserves scalar and non-finite compatibility."""
    assert matlab_round(2.5) == 3.0
    assert matlab_round(-2.5) == -3.0
    values = np.array([np.nan, np.inf, -np.inf], dtype=np.double)

    np.testing.assert_array_equal(matlab_round(values), values)
