"""
Module for testing the TRACE analysis process using predefined datasets.

This module contains two test functions: `test_trace_pythia` and
`test_trace_simulation`.
Each function reads in algorithm labels and various datasets, runs the TRACE analysis,
and evaluates the performance footprints for different algorithms.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from numpy.typing import NDArray
from pandas.testing import assert_frame_equal
from shapely.geometry import Polygon

from instancespace.data.model import Footprint
from instancespace.data.options import GeneralOptions, ParallelOptions, TraceOptions
from instancespace.stages.trace import TraceInputs, TraceOutputs, TraceStage


def _trace_for_geometry(
    z: NDArray[np.double],
    *,
    purity: float = 0.55,
) -> TraceStage:
    """Create a minimal TRACE stage for focused geometry tests."""
    n_instances = z.shape[0]
    return TraceStage(
        z=z,
        y_bin=np.zeros((n_instances, 1), dtype=np.bool_),
        p=np.zeros(n_instances, dtype=np.int_),
        beta=np.zeros(n_instances, dtype=np.bool_),
        algo_labels=["algorithm"],
        trace_opts=TraceOptions.default(purity=purity),
        parallel_opts=ParallelOptions.default(flag=False, n_cores=1),
        general_opts=GeneralOptions.default(),
    )


def test_trace_pythia() -> None:
    """Test the TRACE analysis using the 'pythia' dataset.

    This function reads algorithm labels, instance space (z), binary performance
    indicators (y_bin), performance metrics (p), and beta thresholds from CSV files.
    It then runs the TRACE analysis using the `Trace` class and outputs the results.

    Data Source:
    ----------
    The data is read from CSV files located in the 'test_data/trace_csvs' directory.

    Returns:
    -------
    None
    """
    # Define the path to the file
    main_dir = Path(__file__).parent

    algo_labels_path = main_dir / "test_data/trace_csvs/algolabels.txt"

    # Use Path.open() to open the file
    with algo_labels_path.open() as f:
        algo_labels = f.read().split(",")

    # Reading instance space from Z.csv
    z = np.genfromtxt(
        main_dir / "test_data/trace_csvs/Z.csv",
        delimiter=",",
        dtype=np.double,
    )

    # Reading binary performance indicators from y_bin.csv
    y_bin = np.genfromtxt(
        main_dir / "test_data/trace_csvs/yhat.csv",
        delimiter=",",
        dtype=np.int_,
    ).astype(np.bool_)

    # Reading binary performance indicators from y_bin2.csv
    y_bin2 = np.genfromtxt(
        main_dir / "test_data/trace_csvs/yhat2.csv",
        delimiter=",",
        dtype=np.int_,
    ).astype(np.bool_)

    # Reading performance metrics from p.csv
    p1 = np.genfromtxt(
        main_dir / "test_data/trace_csvs/selection0.csv",
        delimiter=",",
        dtype=np.double,
    )
    p1 = p1 - 1  # Adjusting indices to be zero-based

    # Reading performance metrics from p2.csv
    p2 = np.genfromtxt(
        main_dir / "test_data/trace_csvs/dataP.csv",
        delimiter=",",
        dtype=np.double,
    )

    # Reading beta thresholds from beta.csv
    beta = np.genfromtxt(
        main_dir / "test_data/trace_csvs/beta.csv",
        delimiter=",",
        dtype=np.int_,
    ).astype(np.bool_)

    # Setting TRACE options with a purity value of 0.55 and enabling sim values
    trace_options = TraceOptions(True, 0.55)

    parallel_options = ParallelOptions(False, 3)

    # Initialising and running the TRACE analysis
    trace_inputs: TraceInputs = TraceInputs(
        z,
        p1.astype(np.double),
        p2.astype(np.double),
        beta,
        algo_labels,
        y_bin,
        y_bin2,
        trace_options,
        parallel_options,
        GeneralOptions.default(),
    )

    trace_output: TraceOutputs = TraceStage._run(trace_inputs)  # noqa: SLF001

    correct_result_path = main_dir / "test_data/trace_csvs/correct_results_pythia.csv"
    expected_output = pd.read_csv(correct_result_path).sort_values("Algorithm")
    received_output = trace_output.trace_summary.sort_values("Algorithm")

    # Use assert_frame_equal with tolerance
    assert_frame_equal(expected_output, received_output, rtol=1e-2, atol=1e-2)
    print("DataFrames are almost equal.")


def _load_trace_fixture() -> TraceInputs:
    """Build a `TraceInputs` from the same CSV fixture the two tests above use."""
    main_dir = Path(__file__).parent

    algo_labels_path = main_dir / "test_data/trace_csvs/algolabels.txt"
    with algo_labels_path.open() as f:
        algo_labels = f.read().split(",")

    z = np.genfromtxt(
        main_dir / "test_data/trace_csvs/Z.csv",
        delimiter=",",
        dtype=np.double,
    )
    y_bin = np.genfromtxt(
        main_dir / "test_data/trace_csvs/yhat.csv",
        delimiter=",",
        dtype=np.int_,
    ).astype(np.bool_)
    y_bin2 = np.genfromtxt(
        main_dir / "test_data/trace_csvs/yhat2.csv",
        delimiter=",",
        dtype=np.int_,
    ).astype(np.bool_)
    p1 = np.genfromtxt(
        main_dir / "test_data/trace_csvs/selection0.csv",
        delimiter=",",
        dtype=np.double,
    )
    p1 = p1 - 1
    p2 = np.genfromtxt(
        main_dir / "test_data/trace_csvs/dataP.csv",
        delimiter=",",
        dtype=np.double,
    )
    beta = np.genfromtxt(
        main_dir / "test_data/trace_csvs/beta.csv",
        delimiter=",",
        dtype=np.int_,
    ).astype(np.bool_)

    return TraceInputs(
        z,
        p1.astype(np.double),
        p2.astype(np.double),
        beta,
        algo_labels,
        y_bin,
        y_bin2,
        TraceOptions.default(),
        ParallelOptions(False, 3),
        GeneralOptions.default(),
    )


@pytest.mark.parametrize(
    ("use_sim", "expected_portfolio"),
    [
        (True, np.array([-1, 1], dtype=np.int_)),
        (False, np.array([0, 1], dtype=np.int_)),
    ],
)
def test_trace_run_normalises_only_the_experimental_portfolio(
    monkeypatch: pytest.MonkeyPatch,
    *,
    use_sim: bool,
    expected_portfolio: NDArray[np.int_],
) -> None:
    """TRACE shares zero-based internals without shifting PYTHIA selections."""
    observed_portfolios: list[NDArray[np.int_]] = []
    empty = Footprint(None, 0, 0, 0, 0, 0)
    expected_output = TraceOutputs(empty, [], [], empty, pd.DataFrame())

    def capture_trace(
        z: NDArray[np.double],
        y_bin: NDArray[np.bool_],
        p: NDArray[np.int_],
        beta: NDArray[np.bool_],
        algo_labels: list[str],
        trace_opts: TraceOptions,
        parallel_opts: ParallelOptions,
        general_opts: GeneralOptions,
        executor: object | None = None,
    ) -> TraceOutputs:
        del (
            z,
            y_bin,
            beta,
            algo_labels,
            trace_opts,
            parallel_opts,
            general_opts,
            executor,
        )
        observed_portfolios.append(p.copy())
        return expected_output

    monkeypatch.setattr(TraceStage, "trace", capture_trace)
    inputs = TraceInputs(
        z=np.zeros((2, 2), dtype=np.double),
        selection0=np.array([-1, 1], dtype=np.int_),
        p=np.array([1, 2], dtype=np.int_),
        beta=np.zeros(2, dtype=np.bool_),
        algo_labels=["first", "second"],
        y_hat=np.zeros((2, 2), dtype=np.bool_),
        y_bin=np.zeros((2, 2), dtype=np.bool_),
        trace_options=TraceOptions.default(use_sim=use_sim),
        parallel_options=ParallelOptions.default(flag=False),
        general_options=GeneralOptions.default(),
    )

    output = TraceStage._run(inputs)  # noqa: SLF001

    assert output is expected_output
    assert len(observed_portfolios) == 1
    np.testing.assert_array_equal(observed_portfolios[0], expected_portfolio)


@pytest.mark.parametrize(
    "portfolio",
    [
        np.array([0, 1], dtype=np.int_),
        np.array([1, 3], dtype=np.int_),
        np.array([1.0, 1.5], dtype=np.double),
        np.array([1.0, np.nan], dtype=np.double),
        np.array([1 + 0j, 2 + 0j], dtype=np.complex128),
        np.array([1, object()], dtype=np.object_),
    ],
)
def test_trace_rejects_invalid_one_based_experimental_portfolios(
    portfolio: NDArray[np.generic],
) -> None:
    """Malformed PRELIM portfolios fail at the explicit TRACE boundary."""
    with pytest.raises(ValueError, match="Experimental portfolio"):
        TraceStage._experimental_portfolio_indices(  # noqa: SLF001
            portfolio,  # type: ignore[arg-type]
            n_instances=2,
            n_algorithms=2,
        )


def test_trace3_method_dispatches_to_trace3(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The additive TRACE3 method no longer falls through to legacy."""
    inputs = _load_trace_fixture()._replace(
        trace_options=TraceOptions.default(method="trace3"),
    )
    empty = Footprint(None, 0, 0, 0, 0, 0)
    expected = TraceOutputs(empty, [], [], empty, pd.DataFrame())
    monkeypatch.setattr(TraceStage, "_trace3", lambda _self: expected)

    assert TraceStage._run(inputs) is expected  # noqa: SLF001


def test_trace_contra_false_skips_contradiction_removal() -> None:
    """F11: `contra=False` skips the contradiction-removal step entirely.

    Matches MATLAB legacy TRACE's `contra` option (default `True`, matching
    Python's previous unconditional behaviour). This fixture's footprints
    don't happen to overlap, so `contra=True` vs `False` produce identical
    numeric output either way here - the log trace is what actually proves
    the step ran (or didn't), so assert on that directly rather than on
    output that isn't guaranteed to differ for this particular dataset.
    """
    from loguru import logger

    def _collect_logs(inputs: TraceInputs) -> list[str]:
        messages: list[str] = []
        sink_id = logger.add(
            lambda msg: messages.append(msg.record["message"]),
            level="INFO",
        )
        try:
            TraceStage._run(inputs)  # noqa: SLF001
        finally:
            logger.remove(sink_id)
        return messages

    inputs_default = _load_trace_fixture()
    messages_true = _collect_logs(
        inputs_default._replace(trace_options=TraceOptions.default(contra=True)),
    )
    messages_false = _collect_logs(
        inputs_default._replace(trace_options=TraceOptions.default(contra=False)),
    )

    assert any("removing contradictory" in m for m in messages_true)
    assert not any("removing contradictory" in m for m in messages_false)
    assert any("skipping contradiction removal" in m for m in messages_false)


def test_trace_simulation() -> None:
    """Run the 'simulation' dataset against its Python regression baseline.

    This function reads algorithm labels, instance space (z), binary performance
    indicators (y_bin2), performance metrics (p2), and beta thresholds from CSV files.
    It then runs the TRACE analysis using the `Trace` class and outputs the results.

    The output CSV is a Python regression baseline, not a verified MATLAB export.
    Replacing it with a fresh MATLAB reference remains tracked by issue #278.

    Data Source:
    ----------
    The data is read from CSV files located in the 'test_data/trace_csvs' directory.

    Returns:
    -------
    None
    """
    # Define the path to the file
    script_dir = Path(__file__).parent

    algo_labels_path = script_dir / "test_data/trace_csvs/algolabels.txt"

    # Use Path.open() to open the file
    with algo_labels_path.open() as f:
        algo_labels = f.read().split(",")

    # Reading instance space from Z.csv
    z = np.genfromtxt(
        script_dir / "test_data/trace_csvs/Z.csv",
        delimiter=",",
        dtype=np.double,
    )

    # Reading binary performance indicators from y_bin.csv
    y_bin = np.genfromtxt(
        script_dir / "test_data/trace_csvs/yhat.csv",
        delimiter=",",
        dtype=np.int_,
    ).astype(np.bool_)

    # Reading binary performance indicators from y_bin2.csv
    y_bin2 = np.genfromtxt(
        script_dir / "test_data/trace_csvs/yhat2.csv",
        delimiter=",",
        dtype=np.int_,
    ).astype(np.bool_)

    # Reading performance metrics from p.csv
    p1 = np.genfromtxt(
        script_dir / "test_data/trace_csvs/selection0.csv",
        delimiter=",",
        dtype=np.double,
    )
    p1 = p1 - 1  # Adjusting indices to be zero-based

    # Reading performance metrics from p2.csv
    p2 = np.genfromtxt(
        script_dir / "test_data/trace_csvs/dataP.csv",
        delimiter=",",
        dtype=np.double,
    )
    assert np.all(p2 >= 1)  # PRELIM/Data keeps MATLAB-compatible one-based indices.

    # Reading beta thresholds from beta.csv
    beta = np.genfromtxt(
        script_dir / "test_data/trace_csvs/beta.csv",
        delimiter=",",
        dtype=np.int_,
    ).astype(np.bool_)

    # Setting TRACE options with a purity value of 0.55 and disabling sim values
    trace_options = TraceOptions(False, 0.1)

    parallel_options = ParallelOptions(False, 3)

    # Initialising and running the TRACE analysis
    trace_inputs: TraceInputs = TraceInputs(
        z,
        p1.astype(np.double),
        p2.astype(np.double),
        beta,
        algo_labels,
        y_bin,
        y_bin2,
        trace_options,
        parallel_options,
        GeneralOptions.default(),
    )

    trace_output: TraceOutputs = TraceStage._run(trace_inputs)  # noqa: SLF001
    regression_baseline_path = (
        script_dir / "test_data/trace_csvs/correct_results_simulation.csv"
    )
    expected_output = pd.read_csv(regression_baseline_path).sort_values("Algorithm")
    received_output = trace_output.trace_summary.sort_values("Algorithm")

    # Use assert_frame_equal with tolerance
    assert_frame_equal(expected_output, received_output, rtol=1e-2, atol=1e-2)
    print("DataFrames are almost equal.")


def test_tight_uses_pointwise_membership_and_refits_selected_cloud(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Refinement passes all selected points directly to the shared fitter."""
    z = np.array(
        [
            [0.0, 0.0],
            [0.2, 0.0],
            [0.2, 0.2],
            [0.0, 0.2],
            [2.0, 2.0],
        ],
        dtype=np.double,
    )
    y_bin = np.array([True, True, True, True, False], dtype=np.bool_)
    trace = _trace_for_geometry(z)
    container = Polygon(z[:4])
    fitted_clouds: list[NDArray[np.double]] = []

    def capture_fit(
        polydata: NDArray[np.double],
        _y_bin: NDArray[np.bool_],
    ) -> Polygon:
        fitted_clouds.append(polydata.copy())
        return Polygon(polydata)

    monkeypatch.setattr(trace, "fit_poly", capture_fit)

    refined = trace.tight(container, y_bin)

    assert not refined.is_empty
    assert len(fitted_clouds) == 1
    np.testing.assert_array_equal(fitted_clouds[0], z[:4])


def test_tight_returns_empty_polygon_when_no_region_can_be_refined() -> None:
    """An unsuccessful refinement remains a safe geometry, never ``None``."""
    z = np.array([[0.0, 0.0], [0.2, 0.0], [0.0, 0.2]], dtype=np.double)
    trace = _trace_for_geometry(z)
    polygon = Polygon([(-0.1, -0.1), (0.3, -0.1), (0.3, 0.3), (-0.1, 0.3)])

    refined = trace.tight(polygon, np.zeros(z.shape[0], dtype=np.bool_))

    assert isinstance(refined, Polygon)
    assert refined.is_empty


@pytest.mark.filterwarnings("error::RuntimeWarning")
def test_contra_keeps_both_footprints_when_overlap_has_no_evidence() -> None:
    """A contradiction with no enclosed instances is not assigned by NaN purity."""
    z = np.array([[0.5, 0.5], [2.5, 2.5]], dtype=np.double)
    y_base = np.array([True, False], dtype=np.bool_)
    y_test = np.array([False, True], dtype=np.bool_)
    trace = _trace_for_geometry(z)
    base_polygon = Polygon([(0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0)])
    test_polygon = Polygon([(1.0, 1.0), (3.0, 1.0), (3.0, 3.0), (1.0, 3.0)])
    base = Footprint.from_polygon(base_polygon, z, y_base)
    test = Footprint.from_polygon(test_polygon, z, y_test)

    refined_base, refined_test = trace.contra(base, test, y_base, y_test)

    assert refined_base.polygon is not None
    assert refined_test.polygon is not None
    assert refined_base.polygon.equals(base_polygon)
    assert refined_test.polygon.equals(test_polygon)


@pytest.mark.filterwarnings("error::RuntimeWarning")
def test_contra_counts_boundary_points_as_contradiction_evidence() -> None:
    """MATLAB ``isinterior`` treats overlap-boundary instances as evidence."""
    z = np.array([[1.0, 1.0]], dtype=np.double)
    y_base = np.array([True], dtype=np.bool_)
    y_test = np.array([False], dtype=np.bool_)
    trace = _trace_for_geometry(z)
    base_polygon = Polygon([(0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0)])
    test_polygon = Polygon([(1.0, 0.0), (3.0, 0.0), (3.0, 2.0), (1.0, 2.0)])
    base = Footprint.from_polygon(base_polygon, z, y_base)
    test = Footprint.from_polygon(test_polygon, z, y_test)

    refined_base, refined_test = trace.contra(base, test, y_base, y_test)

    assert refined_base.polygon is not None
    assert refined_base.polygon.equals(base_polygon)
    assert refined_test.polygon is None


@pytest.mark.filterwarnings("error::RuntimeWarning")
def test_contra_unequal_purity_refines_the_weaker_footprint() -> None:
    """Unequal purity takes the live ``tight`` path without losing the footprint."""
    base_overlap = np.array(
        [[1.2, 0.5], [1.2, 1.0], [1.2, 1.5]],
        dtype=np.double,
    )
    test_overlap = np.array([[1.8, 1.0]], dtype=np.double)
    base_only = np.array([[0.5, 1.0]], dtype=np.double)
    test_grid = np.array(
        [(x, y) for x in np.linspace(2.3, 3.7, 5) for y in np.linspace(0.3, 1.7, 5)],
        dtype=np.double,
    )
    z = np.vstack((base_overlap, test_overlap, base_only, test_grid))
    y_base = np.zeros(z.shape[0], dtype=np.bool_)
    y_base[:3] = True
    y_base[4] = True
    y_test = np.zeros(z.shape[0], dtype=np.bool_)
    y_test[3] = True
    y_test[5:] = True
    trace = _trace_for_geometry(z)
    base_polygon = Polygon([(0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0)])
    test_polygon = Polygon([(1.0, 0.0), (4.0, 0.0), (4.0, 2.0), (1.0, 2.0)])
    base = Footprint.from_polygon(base_polygon, z, y_base)
    test = Footprint.from_polygon(test_polygon, z, y_test)

    refined_base, refined_test = trace.contra(base, test, y_base, y_test)

    assert refined_base.polygon is not None
    assert refined_test.polygon is not None
    assert refined_base.polygon.equals(base_polygon)
    assert not refined_test.polygon.is_empty
    assert refined_test.area < test.area
    assert refined_base.polygon.intersection(refined_test.polygon).is_empty


def test_fit_poly_removes_triangles_without_supporting_instances() -> None:
    """A triangle containing no observed instance is removed like MATLAB TRACE."""
    z = np.array([[2.0, 2.0], [3.0, 3.0], [4.0, 4.0]], dtype=np.double)
    trace = _trace_for_geometry(z)
    polydata = np.array([[0.0, 0.0], [0.1, 0.0], [0.0, 0.1]], dtype=np.double)
    y_bin = np.array([True, False, False], dtype=np.bool_)

    polygon = trace.fit_poly(polydata, y_bin)

    assert polygon is not None
    assert polygon.is_empty


def test_fit_poly_counts_boundary_points_as_triangle_support() -> None:
    """Observed triangle vertices count as support under MATLAB ``isinterior``."""
    polydata = np.array([[0.0, 0.0], [0.1, 0.0], [0.0, 0.1]], dtype=np.double)
    z = np.vstack((polydata, np.array([[2.0, 2.0]], dtype=np.double)))
    trace = _trace_for_geometry(z)
    y_bin = np.array([True, True, True, False], dtype=np.bool_)

    polygon = trace.fit_poly(polydata, y_bin)

    assert polygon is not None
    assert not polygon.is_empty


def test_build_throws_when_dbscan_produces_no_polygon(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No fitted cluster uses the canonical empty footprint representation."""
    z = np.array([[0.0, 0.0], [0.1, 0.0], [0.0, 0.1]], dtype=np.double)
    trace = _trace_for_geometry(z)

    def no_clusters(
        _y_bin: NDArray[np.bool_],
        data: NDArray[np.double],
    ) -> NDArray[np.int_]:
        return np.full(data.shape[0], -1, dtype=np.int_)

    monkeypatch.setattr(trace, "run_dbscan", no_clusters)

    footprint = trace.build(np.ones(z.shape[0], dtype=np.bool_))

    assert footprint == trace.throw()
    assert footprint.polygon is None


def test_from_polygon_normalises_empty_geometry() -> None:
    """Empty Shapely geometries do not create a second empty-footprint state."""
    z = np.array([[0.0, 0.0]], dtype=np.double)

    footprint = Footprint.from_polygon(
        Polygon(),
        z,
        np.array([True], dtype=np.bool_),
        smoothen=True,
    )

    assert footprint == Footprint(None, 0, 0, 0, 0, 0)


def test_from_polygon_counts_boundary_points() -> None:
    """Footprint metrics include boundary instances like MATLAB ``isinterior``."""
    polygon = Polygon([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)])
    z = np.array([[0.0, 0.0], [0.5, 0.5], [2.0, 2.0]], dtype=np.double)
    y_bin = np.array([True, True, False], dtype=np.bool_)

    footprint = Footprint.from_polygon(polygon, z, y_bin)
    expected_members = 2

    assert footprint.elements == expected_members
    assert footprint.good_elements == expected_members
    assert footprint.purity == 1.0


def test_dist_returns_vector_for_one_dimensional_data() -> None:
    """One-dimensional DBSCAN distances retain one value per data row."""
    data = np.array([[0.0], [2.0], [5.0]], dtype=np.double)

    distances = TraceStage.dist(np.array([1.0], dtype=np.double), data)

    np.testing.assert_array_equal(distances, np.array([1.0, 1.0, 4.0]))
    assert distances.shape == (data.shape[0],)


def test_dbscan_returns_integer_cluster_labels() -> None:
    """DBSCAN cluster identifiers use an integer dtype, including noise labels."""
    data = np.array([[0.0], [0.1], [3.0]], dtype=np.double)

    labels = TraceStage.dbscan(data, k=1, eps=0.2)

    assert np.issubdtype(labels.dtype, np.integer)
    np.testing.assert_array_equal(labels, np.array([1, 1, -1], dtype=np.int_))
