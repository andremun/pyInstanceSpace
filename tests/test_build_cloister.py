"""Test module for Cloister class to verify its functionality.

The file contains multiple unit tests to ensure that the `Cloister` class corretly
perform its tasks. The basic mechanism of the test is to compare its output against
output from MATLAB and check if the outputs are the same or reasonable similar. The
tests also include some boundary test where appropriate to test the boundary of the
statement within the methods to ensure they are implemented appropriately.

Tests include:
- Correlation calculations and boundary test.
- Generating binary matrix from decimal
- Running analysis from start to end
- Error handling from convex hull calculation
- Boundary generation and the boundary test.
"""

from pathlib import Path

import numpy as np
import pytest
from loguru import logger
from numpy.typing import NDArray

from instancespace.data.options import CloisterOptions
from instancespace.stages.cloister import CloisterInput, CloisterStage

script_dir = Path(__file__).parent


def _assert_same_hull_cycle(
    expected: NDArray[np.double],
    actual: NDArray[np.double],
    atol: float = 1e-6,
) -> None:
    """Assert two vertex lists describe the same closed convex-hull boundary.

    Convex hull vertex order is only defined up to a starting point and
    traversal direction - neither MATLAB's `convhull` nor SciPy's
    `ConvexHull` guarantees a canonical one, so a plain `np.allclose` on raw
    row order is fragile to a version-driven change in either. Checks every
    rotation of `actual`, and of its reversal, against `expected`; fails
    only if none match, i.e. the underlying point set or values actually
    differ, not just their starting vertex or direction.
    """
    if expected.shape != actual.shape:
        pytest.fail(f"shape mismatch: expected {expected.shape}, got {actual.shape}")

    for candidate in (actual, actual[::-1]):
        for offset in range(len(candidate)):
            if np.allclose(np.roll(candidate, -offset, axis=0), expected, atol=atol):
                return

    pytest.fail(
        "no rotation or reflection of actual matches expected - the "
        "boundary points themselves differ, not just their order",
    )


class CloisterMatlabInputs:
    """Class to store MATLAB input data for cloister tests."""

    def __init__(self) -> None:
        """Initialize the input data for the cloister tests."""
        csv_path_x = script_dir / "test_data/cloister/input/input_x.csv"
        csv_path_a = script_dir / "test_data/cloister/input/input_a.csv"

        self.input_x = np.genfromtxt(csv_path_x, delimiter=",")
        self.input_a = np.genfromtxt(csv_path_a, delimiter=",")


class CloisterMatlabOutput:
    """Class to store MATLAB output data for cloister tests."""

    def __init__(self) -> None:
        """Initialize the output data for the cloister tests."""
        csv_path_rho = script_dir / "test_data/cloister/output/rho.csv"
        csv_path_rho_zero = script_dir / "test_data/cloister/output/rho_zero_pval.csv"
        csv_path_z_edge = script_dir / "test_data/cloister/output/z_edge.csv"
        csv_path_z_ecorr = script_dir / "test_data/cloister/output/z_ecorr.csv"
        csv_path_x_edge = script_dir / "test_data/cloister/output/x_edge.csv"
        csv_path_remove = script_dir / "test_data/cloister/output/remove.csv"
        csv_path_index = script_dir / "test_data/cloister/output/index.csv"

        self.rho = np.genfromtxt(csv_path_rho, delimiter=",")
        self.rho_zero = np.genfromtxt(csv_path_rho_zero, delimiter=",")
        self.z_edge = np.genfromtxt(csv_path_z_edge, delimiter=",")
        self.z_ecorr = np.genfromtxt(csv_path_z_ecorr, delimiter=",")
        self.x_edge = np.genfromtxt(csv_path_x_edge, delimiter=",")
        self.remove = np.genfromtxt(csv_path_remove, delimiter=",")
        self.index = np.genfromtxt(csv_path_index, delimiter=",")


class TestCloister:
    """Test module for Cloister class to verify its functionality."""

    @pytest.fixture()
    def input_data(self) -> CloisterMatlabInputs:
        """Fixture to initialize MATLAB input data for cloister tests."""
        return CloisterMatlabInputs()

    @pytest.fixture()
    def output_data(self) -> CloisterMatlabOutput:
        """Fixture to initialize MATLAB output data for cloister tests."""
        return CloisterMatlabOutput()

    def test_correlation_calculation(
        self,
        input_data: CloisterMatlabInputs,
        output_data: CloisterMatlabOutput,
    ) -> None:
        """Test correlation calculation against MATLAB's correlation output.

        Compare the calculated rho value between MATLAB's and Python's computation
        using same pval value.
        """
        input_x = input_data.input_x
        options = CloisterOptions.default()

        rho_python = CloisterStage._compute_correlation(
            input_x,
            options,
        )
        rho_matlab = output_data.rho

        assert np.allclose(rho_matlab, rho_python)

    def test_correlation_calculation_boundary(
        self,
        input_data: CloisterMatlabInputs,
        output_data: CloisterMatlabOutput,
    ) -> None:
        """Test with pval value being zero.

        This will test both on point and off point boundary.
        """
        input_x = input_data.input_x
        option = CloisterOptions(p_val=0, c_thres=0.7)

        rho_python = CloisterStage._compute_correlation(input_x, option)
        rho_matlab = output_data.rho_zero

        assert np.allclose(rho_matlab, rho_python)

    def test_decimal_to_binary(
        self,
        input_data: CloisterMatlabInputs,
        output_data: CloisterMatlabOutput,
    ) -> None:
        """Test generating binary matrix from decimal against MATLAB's de2bi ouput.

        Compare the generated binary matrix between MATLAB's de2bi function and custom
        function implemented in Python. Python's matrix should be 1 less than MATLAB's
        output since MATLAB use 1 base indexing while python use 0 base indexing.
        """
        input_x = input_data.input_x
        nfeats = input_x.shape[1]

        index_python = CloisterStage._decimal_to_binary_matrix(nfeats)
        index_matlab = output_data.index

        assert np.all(index_matlab == index_python + 1)

    def test_decimal_to_binary_with_empty_x(self) -> None:
        """Test generating binary matrix with empty input."""
        empty_x = np.empty((0, 0))
        nfeats = empty_x.shape[1]

        index = CloisterStage._decimal_to_binary_matrix(nfeats)

        assert index.shape == (1, 1)
        assert index[0, 0] == 0

    def test_run(
        self,
        input_data: CloisterMatlabInputs,
        output_data: CloisterMatlabOutput,
    ) -> None:
        """Test run methods correctly run analysis from start to end.

        The test also tests convex hull calculation with valid input. MATLAB's
        `convhull` and SciPy's `ConvexHull` both produce a circular sequence of
        boundary vertices, but neither guarantees the same starting vertex or
        traversal direction as the other - that's an implementation detail of
        the underlying algorithm, not a documented contract either side
        promises. `_assert_same_hull_cycle` compares them as the same closed
        polygon rather than as literally identical arrays.
        """
        input_x = input_data.input_x
        input_a = input_data.input_a
        options = CloisterOptions.default()

        inputs = CloisterInput(
            input_x,
            input_a,
            options,
        )

        z_edge_python, z_ecorr_python = CloisterStage._run(inputs)
        z_edge_matlab = output_data.z_edge
        z_ecorr_matlab = output_data.z_ecorr

        _assert_same_hull_cycle(z_edge_matlab, z_edge_python)
        _assert_same_hull_cycle(z_ecorr_matlab, z_ecorr_python)

    def test_convex_hull_qhull_error(self) -> None:
        """Test convex hull function properly handles qhull error."""
        points_collinear = np.array([[0, 0], [1, 1], [2, 2]])
        output = CloisterStage._compute_convex_hull(points_collinear)
        assert output.size == 0

    def test_convex_hull_value_error(self) -> None:
        """Test convex hull function properly handles value error."""
        points_one_dimension = np.array([[1], [2], [3]])
        output = CloisterStage._compute_convex_hull(
            points_one_dimension,
        )
        assert output.size == 0

    def test_boundary_generation(
        self,
        input_data: CloisterMatlabInputs,
        output_data: CloisterMatlabOutput,
    ) -> None:
        """Test boundary generation against MATLAB's output.

        Compare the z_edge and z_ecorr vaules obtained from the function with MATLAB's.
        output to verify Python implementation produce out within acceptable range.
        """
        input_x = input_data.input_x
        options = CloisterOptions.default()

        rho = CloisterStage._compute_correlation(input_x, options)

        x_edge_python, remove_python = CloisterStage._generate_boundaries(
            input_x,
            rho,
            options,
        )
        x_edge_matlab = output_data.x_edge
        remove_matlab = output_data.remove

        assert np.allclose(x_edge_matlab, x_edge_python)
        assert np.all(remove_matlab == remove_python)

    def test_boundary_generation_cthres_boundary(
        self,
        input_data: CloisterMatlabInputs,
        output_data: CloisterMatlabOutput,
    ) -> None:
        """Test cthres boundary."""
        csv_path_rho = script_dir / "test_data/cloister/input/rho_boundary.csv"
        rho_boundary = np.genfromtxt(csv_path_rho, delimiter=",")

        input_x = input_data.input_x
        options = CloisterOptions.default()

        _, remove = CloisterStage._generate_boundaries(
            input_x,
            rho_boundary,
            options,
        )
        remove_matlab = output_data.remove

        assert np.all(remove_matlab == remove)

    def test_max_features_guard_uses_convex_hull_fallback(
        self,
        input_data: CloisterMatlabInputs,
    ) -> None:
        """Above `max_features`, CLOISTER must skip corner enumeration entirely.

        Regression test: `CloisterOptions` had no `max_features` field at
        all, and `_generate_boundaries` unconditionally enumerated
        `2**nfeats` corners - intractable for any realistic feature count
        above ~25, unlike MATLAB's `opts.maxFeatures` guard (default 20),
        which falls back to a plain convex hull of the projected instances.
        """
        input_x = input_data.input_x
        input_a = input_data.input_a
        options = CloisterOptions(p_val=0.05, c_thres=0.7, max_features=5)

        messages: list[str] = []
        sink_id = logger.add(messages.append, level="WARNING")
        try:
            z_edge, z_ecorr = CloisterStage.cloister(input_x, input_a, options)
        finally:
            logger.remove(sink_id)

        assert z_edge.shape[0] > 0
        # The fallback uses the same convex hull for both outputs.
        np.testing.assert_array_equal(z_edge, z_ecorr)
        assert any("skipped" in m and "convex hull" in m for m in messages)

    def test_correlation_ignores_sparse_nan(
        self,
        input_data: CloisterMatlabInputs,
    ) -> None:
        """A sparse NaN in one feature column must not corrupt its correlations.

        Regression test: `pearsonr` on a NaN-containing pair silently
        returns `(nan, nan)` for the *whole* pair instead of computing over
        the valid overlap - and that NaN then survived the significance
        filter (`nan > p_val` is `False`), leaking into the returned `rho`
        matrix, unlike MATLAB's NaN-tolerant design.
        """
        input_x = input_data.input_x.copy()
        options = CloisterOptions.default()

        rho_before = CloisterStage._compute_correlation(
            input_x,
            options,
        )

        input_x[3, 0] = np.nan  # sparse NaN in one feature column
        rho_after = CloisterStage._compute_correlation(input_x, options)

        assert not np.any(np.isnan(rho_after))
        # Only column 0's correlations can have shifted (fewer valid rows);
        # every other pair's correlation is unaffected by that NaN.
        untouched = [i for i in range(input_x.shape[1]) if i != 0]
        for i in untouched:
            for j in untouched:
                assert rho_after[i, j] == pytest.approx(rho_before[i, j])

    def test_generate_boundaries_bounds_are_nan_aware(
        self,
        input_data: CloisterMatlabInputs,
    ) -> None:
        """A sparse NaN must not turn a feature's bounds into NaN.

        Regression test: `_generate_boundaries` used plain `np.min`/`np.max`
        (NaN-propagating) instead of `np.nanmin`/`np.nanmax`, so a single
        NaN in a feature column would make that whole column's bounds NaN,
        propagating into `x_edge` and then failing `ConvexHull` outright.
        """
        input_x = input_data.input_x.copy()
        input_x[3, 0] = np.nan
        rho = CloisterStage._compute_correlation(
            input_x,
            CloisterOptions.default(),
        )

        x_edge, _ = CloisterStage._generate_boundaries(
            input_x,
            rho,
            CloisterOptions.default(),
        )

        assert not np.any(np.isnan(x_edge))

    def test_cloister_handles_sparse_nan_without_crashing(
        self,
        input_data: CloisterMatlabInputs,
    ) -> None:
        """End-to-end: a sparse NaN must not silently empty out the boundary.

        Regression test: the NaN propagated through bounds -> x_edge ->
        `ConvexHull`, which raises on NaN input; `_compute_convex_hull`
        caught that and returned an empty array for *both* z_edge and
        z_ecorr, with no indication anything had gone wrong beyond the
        (wrong, in this case) "correlation threshold too strict" message.
        """
        input_x = input_data.input_x.copy()
        input_x[3, 0] = np.nan
        input_a = input_data.input_a
        options = CloisterOptions.default()

        z_edge, z_ecorr = CloisterStage.cloister(input_x, input_a, options)

        assert z_edge.shape[0] > 0
        assert z_ecorr.shape[0] > 0

    def test_cloister_z_edge_failure_logs_distinct_error(
        self,
        input_data: CloisterMatlabInputs,
    ) -> None:
        """A genuinely-failed z_edge must log its own error, not be silent.

        Regression test: `cloister()` only special-cased an empty
        `z_ecorr` (interpreting it as "correlation threshold too strict");
        an empty `z_edge` - which MATLAB lets fail loudly rather than
        silently return - had no handling at all, and would have gotten
        the same (wrong) "threshold too strict" message if it reached that
        check by coincidence, or none at all as originally written.
        """
        input_x = input_data.input_x
        options = CloisterOptions.default()
        # Projects every instance to the same point - genuinely degenerate,
        # not a correlation-threshold issue.
        degenerate_a = np.zeros((2, input_x.shape[1]))

        messages: list[str] = []
        sink_id = logger.add(messages.append, level="ERROR")
        try:
            z_edge, z_ecorr = CloisterStage.cloister(input_x, degenerate_a, options)
        finally:
            logger.remove(sink_id)

        assert z_edge.size == 0
        assert z_ecorr.size == 0
        assert any("Could not construct a boundary polygon" in m for m in messages)

    def test_compute_convex_hull_hull_dims_all_matches_default(self) -> None:
        """`hull_dims=None` (i.e. `options.hull_dims="all"`) is a no-op restriction.

        #299 audit finding, issue 5, acceptance criterion: `hull_dims="all"`
        must produce identical output to today's unrestricted behaviour.
        """
        rng = np.random.default_rng(0)
        points = rng.random((20, 3))

        unrestricted = CloisterStage._compute_convex_hull(points)
        explicit_all = CloisterStage._compute_convex_hull(points, None)

        np.testing.assert_array_equal(unrestricted, explicit_all)

    def test_compute_convex_hull_hull_dims_restricts_geometry_not_output_columns(
        self,
    ) -> None:
        """`hull_dims=2` computes the hull on the first 2 columns only.

        #299 audit finding, issue 5, acceptance criterion: restricting
        `hull_dims` changes which points are selected as vertices (by
        computing hull geometry over fewer dimensions) but the returned
        vertices still carry every column of the input, matching MATLAB's
        `core/CLOISTER.m` (always a 2D hull, full-dimensional output points).
        """
        rng = np.random.default_rng(1)
        # Third column is pure noise uncorrelated with the first two, so
        # restricting to 2 dims can plausibly change which rows are on the
        # hull boundary.
        points = rng.random((30, 3))

        hull_2d = CloisterStage._compute_convex_hull(points, 2)
        hull_full = CloisterStage._compute_convex_hull(points)

        assert hull_2d.shape[1] == points.shape[1]
        assert hull_full.shape[1] == points.shape[1]

    def test_compute_convex_hull_hull_dims_exceeding_columns_does_not_crash(
        self,
    ) -> None:
        """`hull_dims` larger than the point set's column count must not crash.

        #299 audit finding, issue 5, acceptance criterion: NumPy's slicing
        (`points[:, :hull_dims]`) is a no-op past the array's own width, so
        this degrades gracefully to using every available column rather
        than raising.
        """
        rng = np.random.default_rng(2)
        points = rng.random((10, 2))

        output = CloisterStage._compute_convex_hull(points, 5)

        assert output.shape[1] == points.shape[1]
        assert output.shape[0] > 0

    def test_cloister_run_with_hull_dims_two_matches_default(
        self,
        input_data: CloisterMatlabInputs,
        output_data: CloisterMatlabOutput,
    ) -> None:
        """End-to-end: `hull_dims=2` matches MATLAB's reference output.

        #299 audit finding, issue 5. `input_a` projects to exactly 2 columns
        already, so `hull_dims=2` and the default `hull_dims="all"` are
        equivalent here - this is the currently-shipped, PILOT-is-2D-only
        case, not a claim that `hull_dims` has no effect in general.
        """
        input_x = input_data.input_x
        input_a = input_data.input_a
        options = CloisterOptions.default(hull_dims=2)

        inputs = CloisterInput(input_x, input_a, options)
        z_edge_python, z_ecorr_python = CloisterStage._run(inputs)

        _assert_same_hull_cycle(output_data.z_edge, z_edge_python)
        _assert_same_hull_cycle(output_data.z_ecorr, z_ecorr_python)

    def test_cloister_threshold_message_only_for_genuine_threshold_failure(
        self,
        input_data: CloisterMatlabInputs,
    ) -> None:
        """The "threshold too strict" message must not fire for a z_edge failure.

        Regression test companion to the above: when z_edge itself is
        empty, the misleading "correlation threshold was too strict"
        message (plus its "weakely" typo) must not appear at all - that
        message is reserved for when z_edge succeeds but z_ecorr's
        correlation-filtered corner set doesn't.
        """
        input_x = input_data.input_x
        options = CloisterOptions.default()
        degenerate_a = np.zeros((2, input_x.shape[1]))

        messages: list[str] = []
        sink_id = logger.add(messages.append, level="INFO")
        try:
            CloisterStage.cloister(input_x, degenerate_a, options)
        finally:
            logger.remove(sink_id)

        assert not any("threshold" in m.lower() for m in messages)
        assert not any("weakely" in m for m in messages)
