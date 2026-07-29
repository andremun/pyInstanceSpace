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
from pandas.testing import assert_frame_equal

from instancespace.data.options import GeneralOptions, ParallelOptions, TraceOptions
from instancespace.stages.trace import TraceInputs, TraceOutputs, TraceStage


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
    p2 = p2 - 1  # Adjusting indices to be zero-based

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
    p2 = p2 - 1
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


def test_trace_method_other_than_legacy_raises() -> None:
    """F11: `TraceOptions.method` other than 'legacy' fails loudly, not silently.

    Only MATLAB's TRACE_legacy.m algorithm is ported; requesting 'trace3'
    (or any other value) must not silently run legacy anyway.
    """
    inputs = _load_trace_fixture()._replace(
        trace_options=TraceOptions.default(method="trace3"),
    )

    with pytest.raises(NotImplementedError, match="trace3"):
        TraceStage._run(inputs)  # noqa: SLF001


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
    """Test the TRACE analysis using the 'simulation' dataset.

    This function reads algorithm labels, instance space (z), binary performance
    indicators (y_bin2), performance metrics (p2), and beta thresholds from CSV files.
    It then runs the TRACE analysis using the `Trace` class and outputs the results.

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
    p2 = p2 - 1  # Adjusting indices to be zero-based

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
    correct_result_path = (
        script_dir / "test_data/trace_csvs/correct_results_simulation.csv"
    )
    expected_output = pd.read_csv(correct_result_path).sort_values("Algorithm")
    received_output = trace_output.trace_summary.sort_values("Algorithm")

    # Use assert_frame_equal with tolerance
    assert_frame_equal(expected_output, received_output, rtol=1e-2, atol=1e-2)
    print("DataFrames are almost equal.")
