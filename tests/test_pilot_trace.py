from pathlib import Path

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from matilda.data.options import TraceOptions
from matilda.stages.trace import TraceInputs, TraceOutputs, TraceStage

from matilda.data.options import PilotOptions
from matilda.stages.pilot import PilotStage, PilotOutput
from scipy.io import loadmat

script_dir = Path(__file__).parent


def test_pilot_analytical_trace_pythia() -> None:
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
    z = pilot_run_analytic().z

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
    )
    trace_output: TraceOutputs = TraceStage._run(trace_inputs)  # noqa: SLF001

    correct_result_path = main_dir / "test_data/trace_csvs/correct_results_pythia.csv"
    expected_output = pd.read_csv(correct_result_path)
    received_output = trace_output.summary


def test_pilot_analytical_trace_simulation() -> None:
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
    z = pilot_run_analytic().z

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
    trace_options = TraceOptions(False, 0.55)

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
    )
    trace_output: TraceOutputs = TraceStage._run(trace_inputs)  # noqa: SLF001
    correct_result_path = (
        script_dir / "test_data/trace_csvs/correct_results_simulation.csv"
    )
    expected_output = pd.read_csv(correct_result_path).sort_values("Algorithm")
    received_output = trace_output.summary.sort_values("Algorithm")


def test_pilot_numerical_trace_pythia() -> None:
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
    z = pilot_run_numerical().z

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
    )
    trace_output: TraceOutputs = TraceStage._run(trace_inputs)  # noqa: SLF001

    correct_result_path = main_dir / "test_data/trace_csvs/correct_results_pythia.csv"
    expected_output = pd.read_csv(correct_result_path)
    received_output = trace_output.summary


def test_pilot_numerical_trace_simulation() -> None:
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
    z = pilot_run_numerical().z

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
    trace_options = TraceOptions(False, 0.55)

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
    )
    trace_output: TraceOutputs = TraceStage._run(trace_inputs)  # noqa: SLF001
    correct_result_path = (
        script_dir / "test_data/trace_csvs/correct_results_simulation.csv"
    )
    expected_output = pd.read_csv(correct_result_path).sort_values("Algorithm")
    received_output = trace_output.summary.sort_values("Algorithm")


class SampleData:
    """Data class for testing the Pilot stage for analytic purposes.

    This class contains the data used for testing the Pilot stage for analytic purposes.
    """

    def __init__(self) -> None:
        """Initialize the sample data for the Pilot stage."""
        fp_sampledata = script_dir / "test_data/pilot/input/test_analytic.mat"
        data = loadmat(fp_sampledata)
        self.x_sample = data["X"]
        self.y_sample = data["Y"]
        feat_labels_sample = data["featlabels"][0]
        self.feat_labels_sample = [str(label[0]) for label in feat_labels_sample]


class SampleDataNum:
    """Data class for testing the Pilot stage for numerical purposes.

    This class contains the data used for testing the Pilot stage for
    numerical purposes.
    """

    def __init__(self) -> None:
        """Initialize the sample data for the Pilot stage."""
        fp_sampledata = script_dir / "test_data/pilot/input/test_numerical.mat"
        data = loadmat(fp_sampledata)
        self.x_sample = data["X_test"]
        self.y_sample = data["Y_test"]
        feat_labels = data["featlabels"][0]
        self.feat_labels_sample = [str(label[0]) for label in feat_labels]
        analytic = data["optsPilot"][0, 0]["analytic"][0, 0]
        n_tries = int(data["optsPilot"][0, 0]["ntries"][0, 0])
        self.opts_sample = PilotOptions(None, None, analytic, n_tries)


class MatlabResults:
    """Data class for verifying the output of the Pilot analytical method.

    This class contains the data used for verifying the output of the
    analytical Pilot stage.
    """

    def __init__(self) -> None:
        """Initialize the sample data for the Pilot stage."""
        fp_outdata = script_dir / "test_data/pilot/output/matlab_results_ana.mat"
        self.data = loadmat(fp_outdata)


class MatlabResultsNum:
    """Data class for verifying the output of the Pilot numerical method.

    This class contains the data used for verifying the output of the
    numerical Pilot stage.
    """

    def __init__(self) -> None:
        """Initialize the sample data for the Pilot stage."""
        fp_outdata = script_dir / "test_data/pilot/output/matlab_results_num.mat"
        self.data = loadmat(fp_outdata)


def pilot_run_analytic() -> PilotOutput:
    """Test the run function for the Pilot stage for analytic purposes."""
    sd = SampleData()
    mtr = MatlabResults()

    x_sample = sd.x_sample
    y_sample = sd.y_sample
    feat_labels_sample = sd.feat_labels_sample
    opts = PilotOptions(None, None, True, 5)
    pilot = PilotStage(x_sample, y_sample, feat_labels_sample)
    return pilot.pilot(x_sample, y_sample, feat_labels_sample, opts)


def pilot_run_numerical() -> PilotOutput:
    """Test the run function for the Pilot stage for numerical purposes."""
    sd = SampleDataNum()
    mtr = MatlabResultsNum()

    x_sample = sd.x_sample
    y_sample = sd.y_sample
    feat_labels_sample = sd.feat_labels_sample
    opts_sample = sd.opts_sample
    opts = PilotOptions(None, None, opts_sample.analytic, opts_sample.n_tries)
    pilot = PilotStage(x_sample, y_sample, feat_labels_sample)
    return pilot.pilot(x_sample, y_sample, feat_labels_sample, opts)

