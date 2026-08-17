"""Test module for Prelim class to verify its functionality.

The file contains multiple unit tests to ensure that the `Prelim` class corretly
performs its tasks. The basic mechanism of the test is to compare its output against
output from MATLAB and check if the outputs are the same or reasonable similar.

Tests include:
- Value of feature matrix after removing extreme outliers.
-- Verifying the values for IQR, median, upper and lower bounds.
- Normalisation of the feature matrix and performance matrix.
-- Verifying the values for lambda, min, mu, and sigma.
- Verifying the values of the data.model after running the Prelim class.
"""

import dataclasses
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from instancespace.data.options import (
    AutoOptions,
    GeneralOptions,
    PerformanceOptions,
    PrelimConfigOptions,
    PrelimOptions,
    SelvarsOptions,
)
from instancespace.stages.prelim import (
    PrelimInput,
    PrelimStage,
    compute_binary_performance,
)
from tests.utils.option_creator import create_option

script_dir = Path(__file__).parent

csv_path_x_input = script_dir / "test_data/prelim/input/model-data-x-input.csv"
csv_path_y_input = script_dir / "test_data/prelim/input/model-data-y.csv"

csv_path_beta = script_dir / "test_data/prelim/output/model-data-beta.csv"
csv_path_num_good_algos = (
    script_dir / "test_data/prelim/output/model-data-numGoodAlgos.csv"
)
csv_path_p = script_dir / "test_data/prelim/output/model-data-p.csv"
csv_path_ybest = script_dir / "test_data/prelim/output/model-data-ybest.csv"
csv_path_ybin = script_dir / "test_data/prelim/output/model-data-ybin.csv"
csv_path_x_output = script_dir / "test_data/prelim/output/model-data-x.csv"
csv_path_y_output = script_dir / "test_data/prelim/output/model-data-y.csv"
csv_path_x_output_after_bound = (
    script_dir / "test_data/prelim/output/model-data-x-after-bound.csv"
)
csv_path_prelim_output_hi_bound = (
    script_dir / "test_data/prelim/output/model-prelim-hibound.csv"
)
csv_path_prelim_output_iq_range = (
    script_dir / "test_data/prelim/output/model-prelim-iqrange.csv"
)
csv_path_prelim_output_med_val = (
    script_dir / "test_data/prelim/output/model-prelim-medval.csv"
)
csv_path_prelim_output_lo_bound = (
    script_dir / "test_data/prelim/output/model-prelim-lobound.csv"
)
csv_path_prelim_output_lambda_x = (
    script_dir / "test_data/prelim/output/model-prelim-lambdaX.csv"
)
csv_path_prelim_output_min_x = (
    script_dir / "test_data/prelim/output/model-prelim-minX.csv"
)
csv_path_prelim_output_lambda_y = (
    script_dir / "test_data/prelim/output/model-prelim-lambdaY.csv"
)
csv_path_prelim_output_min_y = (
    script_dir / "test_data/prelim/output/model-prelim-minY.csv"
)
csv_path_prelim_output_mu_x = (
    script_dir / "test_data/prelim/output/model-prelim-muX.csv"
)
csv_path_prelim_output_mu_y = (
    script_dir / "test_data/prelim/output/model-prelim-muY.csv"
)
csv_path_prelim_output_sigma_x = (
    script_dir / "test_data/prelim/output/model-prelim-sigmaX.csv"
)
csv_path_prelim_output_sigma_y = (
    script_dir / "test_data/prelim/output/model-prelim-sigmaY.csv"
)

csv_path_prelim_input_x_raw = (
    script_dir / "test_data/prelim/fractional/before/Xraw_split.txt"
)

csv_path_prelim_input_y_raw = (
    script_dir / "test_data/prelim/fractional/before/Yraw_split.txt"
)

csv_path_prelim_input_p = script_dir / "test_data/prelim/fractional/before/P_split.txt"

csv_path_prelim_inst_labels = (
    script_dir / "test_data/prelim/fractional/before/instlabels_split.txt"
)
# input data
x_input = pd.read_csv(csv_path_x_input, header=None).to_numpy()
y_input = pd.read_csv(csv_path_y_input, header=None).to_numpy()
x_raw = np.genfromtxt(csv_path_prelim_input_x_raw, delimiter=",")
y_raw = np.genfromtxt(csv_path_prelim_input_y_raw, delimiter=",")
s: pd.Series | None = None  # type: ignore[type-arg]
inst_labels = np.genfromtxt(csv_path_prelim_inst_labels, delimiter=",")

prelim_opts = PrelimOptions(
    abs_perf=True,
    beta_threshold=0.5500,
    epsilon=0.2000,
    max_perf=False,
    bound=True,
    norm=True,
)

selvars_opts = SelvarsOptions.default()


def test_bound() -> None:
    """Test the removal of outliers from the feature matrix."""
    prelim_hi_bound = np.genfromtxt(csv_path_prelim_output_hi_bound, delimiter=",")
    prelim_lo_bound = np.genfromtxt(csv_path_prelim_output_lo_bound, delimiter=",")
    prelim_med_val = np.genfromtxt(csv_path_prelim_output_med_val, delimiter=",")
    prelim_iq_range = np.genfromtxt(csv_path_prelim_output_iq_range, delimiter=",")
    prelim_x_after_bound = np.genfromtxt(csv_path_x_output_after_bound, delimiter=",")

    prelim = PrelimStage(
        x_input,
        y_input,
        x_raw,
        y_raw,
        s,
        pd.Series(inst_labels),
        prelim_opts,
        selvars_opts,
        GeneralOptions.default(),
    )
    prelim_bound = prelim._bound()  # noqa: SLF001
    x = prelim_bound.x
    hi_bound = prelim_bound.hi_bound
    lo_bound = prelim_bound.lo_bound
    med_val = prelim_bound.med_val
    iq_range = prelim_bound.iq_range

    assert np.allclose(x, prelim_x_after_bound)
    assert np.allclose(hi_bound, prelim_hi_bound)
    assert np.allclose(lo_bound, prelim_lo_bound)
    assert np.allclose(med_val, prelim_med_val)
    assert np.allclose(iq_range, prelim_iq_range)


def test_normalise() -> None:
    """Test the normalisation of the feature matrix and performance matrix."""
    prelim_lambda_x = np.genfromtxt(csv_path_prelim_output_lambda_x, delimiter=",")
    prelim_min_x = np.genfromtxt(csv_path_prelim_output_min_x, delimiter=",")
    prelim_mu_x = np.genfromtxt(csv_path_prelim_output_mu_x, delimiter=",")
    prelim_sigma_x = np.genfromtxt(csv_path_prelim_output_sigma_x, delimiter=",")
    prelim_lambda_y = np.genfromtxt(csv_path_prelim_output_lambda_y, delimiter=",")
    prelim_min_y = np.genfromtxt(csv_path_prelim_output_min_y, delimiter=",").item()
    prelim_mu_y = np.genfromtxt(csv_path_prelim_output_mu_y, delimiter=",")
    prelim_sigma_y = np.genfromtxt(csv_path_prelim_output_sigma_y, delimiter=",")

    (
        x,
        y,
        y_bin,
        y_best,
        p,
        num_good_algos,
        beta,
        med_val,
        iq_range,
        hi_bound,
        lo_bound,
        min_x,
        lambda_x,
        mu_x,
        sigma_x,
        min_y,
        lambda_y,
        sigma_y,
        mu_y,
    ) = PrelimStage.prelim(
        x_input,
        y_input,
        x_raw,
        y_raw,
        s,
        pd.Series(inst_labels),
        prelim_opts,
        selvars_opts,
        GeneralOptions.default(),
    )

    assert np.allclose(lambda_x, prelim_lambda_x)
    assert np.allclose(min_x, prelim_min_x)
    assert np.allclose(mu_x, prelim_mu_x)
    assert np.allclose(sigma_x, prelim_sigma_x)
    assert np.allclose(lambda_y, prelim_lambda_y)
    assert np.allclose(min_y, prelim_min_y)
    assert np.allclose(mu_y, prelim_mu_y)
    assert np.allclose(sigma_y, prelim_sigma_y)


def test_prelim() -> None:
    """Test the Prelim run method for the values of the data.model."""
    beta_output = pd.read_csv(csv_path_beta, sep=",", header=None).iloc[:, 0].values
    p_output = pd.read_csv(csv_path_p, sep=",", header=None).iloc[:, 0].values
    ybest_output = pd.read_csv(csv_path_ybest, sep=",", header=None).iloc[:, 0].values
    ybin_output = pd.read_csv(csv_path_ybin, sep=",", header=None)
    num_good_algos_output = pd.read_csv(csv_path_num_good_algos, header=None, sep=",")
    x_output = pd.read_csv(csv_path_x_output, header=None).to_numpy()
    y_output = pd.read_csv(csv_path_y_output, header=None).to_numpy()

    (
        x,
        y,
        y_bin,
        y_best,
        p,
        num_good_algos,
        beta,
        med_val,
        iq_range,
        hi_bound,
        lo_bound,
        min_x,
        lambda_x,
        mu_x,
        sigma_x,
        min_y,
        lambda_y,
        sigma_y,
        mu_y,
    ) = PrelimStage.prelim(
        x_input,
        y_input,
        x_raw,
        y_raw,
        s,
        pd.Series(inst_labels),
        prelim_opts,
        selvars_opts,
        GeneralOptions.default(),
    )

    assert np.allclose(x, x_output)
    assert np.allclose(y, y_output)
    assert np.allclose(y_bin, ybin_output)
    assert np.allclose(
        np.array(y_best).flatten(),
        np.array(ybest_output, dtype=np.float64),
    )
    assert np.allclose(p, np.array(p_output, dtype=np.float64))
    assert np.allclose(num_good_algos, num_good_algos_output.values.flatten())
    assert np.allclose(beta, np.array(beta_output, dtype=bool))


csv_input_prelim_x_run = script_dir / "test_data/prelim/run/input/input_X.csv"
csv_input_prelim_y_run = script_dir / "test_data/prelim/run/input/input_Y.csv"
csv_input_prelim_x_raw_run = script_dir / "test_data/prelim/run/input/input_Xraw.csv"
csv_input_prelim_y_raw_run = script_dir / "test_data/prelim/run/input/input_Yraw.csv"
csv_input_inst_labels_run = (
    script_dir / "test_data/prelim/run/input/input_instlabels.csv"
)

csv_output_prelim_beta_run = script_dir / "test_data/prelim/run/output/output_beta.csv"
csv_output_prelim_num_good_algos_run = (
    script_dir / "test_data/prelim/run/output/output_numGoodAlgos.csv"
)
csv_output_prelim_p_run = script_dir / "test_data/prelim/run/output/output_P.csv"
csv_output_prelim_ybest_run = (
    script_dir / "test_data/prelim/run/output/output_Ybest.csv"
)
csv_output_prelim_ybin_run = script_dir / "test_data/prelim/run/output/output_Ybin.csv"
csv_output_prelim_x_run = script_dir / "test_data/prelim/run/output/output_X.csv"
csv_output_prelim_y_run = script_dir / "test_data/prelim/run/output/output_Y.csv"


def test_prelim_run() -> None:
    """Test the Prelim run method for the values of the data.model."""
    x_input_run = pd.read_csv(csv_input_prelim_x_run, header=None).to_numpy()
    y_input_run = pd.read_csv(csv_input_prelim_y_run, header=None).to_numpy()
    x_raw_run = np.genfromtxt(csv_input_prelim_x_raw_run, delimiter=",")
    y_raw_run = np.genfromtxt(csv_input_prelim_y_raw_run, delimiter=",")
    inst_labels_input_run = np.genfromtxt(csv_input_inst_labels_run, delimiter=",")

    p_output_run = (
        pd.read_csv(csv_output_prelim_p_run, sep=",", header=None).iloc[:, 0].values
    )
    ybest_output_run = (
        pd.read_csv(csv_output_prelim_ybest_run, sep=",", header=None).iloc[:, 0].values
    )
    ybin_output_run = pd.read_csv(csv_output_prelim_ybin_run, sep=",", header=None)

    x_output_run = pd.read_csv(csv_output_prelim_x_run, header=None).to_numpy()
    y_output_run = pd.read_csv(csv_output_prelim_y_run, header=None).to_numpy()

    s: pd.Series | None = None  # type: ignore[type-arg]

    prelim_opts = PrelimOptions(
        abs_perf=True,
        beta_threshold=0.5500,
        epsilon=0.2000,
        max_perf=False,
        bound=True,
        norm=True,
    )

    selvars_opts = SelvarsOptions(
        small_scale_flag=False,
        small_scale=0.50,
        file_idx_flag=False,
        file_idx="",
        feats=None,
        algos=None,
        selvars_type="Ftr&Good",
        min_distance=0.1,
        density_flag=False,
    )

    inputs = PrelimInput(
        x=x_input_run,
        y=y_input_run,
        x_raw=x_raw_run,
        y_raw=y_raw_run,
        s=s,
        inst_labels=pd.Series(inst_labels_input_run),
        prelim_options=prelim_opts,
        selvars_options=selvars_opts,
        general_options=GeneralOptions.default(),
    )

    (
        med_val,
        iq_range,
        hi_bound,
        lo_bound,
        min_x,
        lambda_x,
        mu_x,
        sigma_x,
        min_y,
        lambda_y,
        sigma_y,
        mu_y,
        x,
        y,
        x_raw,
        y_raw,
        y_bin,
        y_best,
        p,
        num_good_algos,
        beta,
        inst_labels,
        data_dense,
        s,
    ) = PrelimStage._run(  # noqa: SLF001
        inputs,
    )

    assert np.allclose(x.shape, x_output_run.shape)
    assert np.allclose(y, y_output_run)
    assert np.allclose(y_bin, ybin_output_run)
    assert np.allclose(
        np.array(y_best).flatten(),
        np.array(ybest_output_run, dtype=np.float64),
    )
    assert np.allclose(p, np.array(p_output_run, dtype=np.float64))


def _collect_warnings(
    fn: Callable[..., None],
    *args: NDArray[np.double],
) -> list[str]:
    """Run fn(*args) and return the loguru WARNING-level messages it emitted."""
    from loguru import logger

    messages: list[str] = []
    sink_id = logger.add(
        lambda msg: messages.append(msg.record["message"]),
        level="WARNING",
    )
    try:
        fn(*args)
    finally:
        logger.remove(sink_id)
    return messages


def test_prelim_many_zero_best_warning_fires_above_threshold() -> None:
    """Regression test for F14: warn when >5% of instances have Ybest == 0."""
    prelim_stage = PrelimStage.__new__(PrelimStage)
    # 2 of 10 instances (20%) have a best-algorithm performance of exactly zero.
    y_best = np.concatenate([np.zeros(2), np.ones(8)])
    warn_many_zero_best = prelim_stage._warn_many_zero_best  # noqa: SLF001

    messages = _collect_warnings(warn_many_zero_best, y_best)

    assert any("best-algorithm performance of exactly zero" in m for m in messages)


def test_prelim_many_zero_best_warning_silent_below_threshold() -> None:
    """No warning when at most 5% of instances have Ybest == 0."""
    prelim_stage = PrelimStage.__new__(PrelimStage)
    y_best = np.ones(10)  # none are zero
    warn_many_zero_best = prelim_stage._warn_many_zero_best  # noqa: SLF001

    messages = _collect_warnings(warn_many_zero_best, y_best)

    assert messages == []


def _collect_info(fn: Callable[[], object]) -> list[str]:
    """Run fn() and return the loguru INFO-level messages it emitted."""
    from loguru import logger

    messages: list[str] = []
    sink_id = logger.add(
        lambda msg: messages.append(msg.record["message"]),
        level="INFO",
    )
    try:
        fn()
    finally:
        logger.remove(sink_id)
    return messages


def test_prelim_minimisation_branch_matches_matlab_formula() -> None:
    """Regression test: minimisation-branch relative performance is `y/best - 1`.

    Previously computed `1 - best/y` (closeness to the *worst*, not the
    *best*), which can flip an algorithm's "good"/"bad" classification
    relative to MATLAB's `Y/Ybest - 1` for the same data.
    """
    y = np.array([[2.0, 4.0], [3.0, 3.0]])
    x = np.ones((2, 1))
    opts = PrelimOptions(
        abs_perf=False,
        beta_threshold=0.55,
        epsilon=0.2,
        max_perf=False,  # minimisation: lower is better
        bound=False,
        norm=False,
    )

    result = PrelimStage.prelim(
        x,
        y.copy(),
        x.copy(),
        y.copy(),
        None,
        pd.Series(["i1", "i2"]),
        opts,
        selvars_opts,
        GeneralOptions.default(),
    )
    y_out = result[1]

    # y_best per instance (minimising): row0 -> 2.0, row1 -> 3.0.
    y_best = np.array([[2.0], [3.0]])
    expected = y / y_best - 1
    assert np.allclose(y_out, expected)
    # The old (wrong) formula would have produced 1 - y_best/y instead.
    wrong = 1 - y_best / y
    assert not np.allclose(y_out, wrong)


def test_prelim_nan_aware_statistics() -> None:
    """Regression test: median/IQR/min ignore NaNs instead of propagating them.

    A single NaN in a feature column must not turn that column's `med_val`/
    `iq_range`/`hi_bound`/`lo_bound`/`min_x` into NaN for every instance.
    """
    x = np.array([[1.0, 10.0], [2.0, np.nan], [3.0, 30.0], [4.0, 40.0]])
    y = np.array([[1.0], [2.0], [3.0], [4.0]])
    opts = PrelimOptions(
        abs_perf=True,
        beta_threshold=0.55,
        epsilon=0.2,
        max_perf=True,
        bound=False,
        norm=False,
    )

    result = PrelimStage.prelim(
        x,
        y,
        x.copy(),
        y.copy(),
        None,
        pd.Series(["i1", "i2", "i3", "i4"]),
        opts,
        selvars_opts,
        GeneralOptions.default(),
    )
    med_val, min_x = result[7], result[11]

    assert not np.isnan(med_val).any()
    assert not np.isnan(min_x).any()
    assert med_val[1] == np.nanmedian(x[:, 1])
    assert min_x[1] == np.nanmin(x[:, 1])


def test_prelim_iqr_multiplier_option_scales_bounds() -> None:
    """A non-default `iqr_multiplier` changes hi_bound/lo_bound proportionally."""
    x = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
    y = np.ones((5, 1))
    base_opts = PrelimOptions(
        abs_perf=True,
        beta_threshold=0.55,
        epsilon=0.2,
        max_perf=True,
        bound=False,
        norm=False,
    )
    wide_opts = dataclasses.replace(base_opts, iqr_multiplier=2.0)

    default_result = PrelimStage.prelim(
        x,
        y.copy(),
        x.copy(),
        y.copy(),
        None,
        pd.Series(["i1", "i2", "i3", "i4", "i5"]),
        base_opts,
        selvars_opts,
        GeneralOptions.default(),
    )
    wide_result = PrelimStage.prelim(
        x,
        y.copy(),
        x.copy(),
        y.copy(),
        None,
        pd.Series(["i1", "i2", "i3", "i4", "i5"]),
        wide_opts,
        selvars_opts,
        GeneralOptions.default(),
    )
    default_hi_bound, default_lo_bound = default_result[9], default_result[10]
    wide_hi_bound, wide_lo_bound = wide_result[9], wide_result[10]

    med_val = default_result[7]
    iq_range = default_result[8]
    assert np.allclose(default_hi_bound, med_val + 5.0 * iq_range)
    assert np.allclose(wide_hi_bound, med_val + 2.0 * iq_range)
    assert np.allclose(default_lo_bound, med_val - 5.0 * iq_range)
    assert np.allclose(wide_lo_bound, med_val - 2.0 * iq_range)


def test_prelim_zero_value_ties_are_detected() -> None:
    """Regression test: a tie at Ybest == 0 must register as a tie.

    Previously, tie detection compared raw performance against the
    *eps-substituted* best value, so a genuine zero-value tie between two
    algorithms was silently never counted - even though the final `p` value
    coincidentally comes out the same either way (both mechanisms pick the
    first tied index), the tie must still be *reported* correctly, and this
    is the plumbing a future (not-yet-implemented) smarter tie-break needs
    to actually run on zero-value ties instead of silently skipping them.

    Exercises `compute_binary_performance()` (F9's extraction, shared with
    `explore()`'s evaluation path) rather than the training-only method it
    replaced, since the tie-breaking logic now lives there.
    """
    # Instance 0: algorithms 0 and 1 both score the (minimum) best of 0.0 - a
    # zero-value tie. Instance 1: no tie.
    y_raw = np.array([[0.0, 0.0, 5.0], [1.0, 2.0, 3.0]])
    perf_opts = PerformanceOptions(
        max_perf=False,
        abs_perf=True,
        epsilon=0.0,
        beta_threshold=0.55,
    )

    messages: list[str] = []
    result = None

    def _run() -> None:
        nonlocal result
        result = compute_binary_performance(
            y_raw,
            perf_opts,
            GeneralOptions.default(),
        )

    messages = _collect_info(_run)

    assert result is not None
    assert result.p[0] == 1  # first tied algorithm (1-based), unchanged either way
    assert any("50" in m and "more than one best algorithm" in m for m in messages)


def test_prelim_options_copy_master_preprocessing_flag() -> None:
    """PRELIM receives MATLAB's auto.preproc master switch."""
    options = create_option(auto=AutoOptions(preproc=False))

    prelim = PrelimOptions.from_options(options)

    assert prelim.preproc is False


def test_prelim_options_preserve_positional_iqr_multiplier() -> None:
    """Existing positional IQR and master-switch arguments keep their meaning."""
    multiplier = 2.5
    nan_threshold = 0.20
    prelim = PrelimOptions(False, True, 0.2, 0.55, True, True, multiplier, False)

    assert prelim.iqr_multiplier == multiplier
    assert prelim.preproc is False
    assert prelim.nan_threshold == nan_threshold


def test_prelim_options_propagate_build_level_configuration() -> None:
    """The composed stage options carry both values from ``opts.prelim``."""
    iqr_multiplier = 3.5
    nan_threshold = 0.4
    options = dataclasses.replace(
        create_option(),
        prelim=PrelimConfigOptions(iqr_multiplier, nan_threshold),
    )

    prelim = PrelimOptions.from_options(options)

    assert prelim.iqr_multiplier == iqr_multiplier
    assert prelim.nan_threshold == nan_threshold


def test_prelim_master_switch_disables_bound_and_normalisation() -> None:
    """auto.preproc=False bypasses both optional preprocessing operations."""
    x = np.array([[1.0, 10.0], [2.0, 20.0], [100.0, 30.0], [4.0, 40.0]])
    y_raw = np.array([[2.0, 4.0], [4.0, 2.0], [3.0, 6.0], [6.0, 3.0]])
    opts = PrelimOptions(
        max_perf=False,
        abs_perf=False,
        epsilon=0.2,
        beta_threshold=0.55,
        bound=True,
        norm=True,
        preproc=False,
    )
    expected_y = compute_binary_performance(
        y_raw,
        PerformanceOptions(False, False, 0.2, 0.55),
        GeneralOptions.default(),
    ).y

    result = PrelimStage.prelim(
        x.copy(),
        y_raw.copy(),
        x.copy(),
        y_raw.copy(),
        None,
        pd.Series(["i1", "i2", "i3", "i4"]),
        opts,
        selvars_opts,
        GeneralOptions.default(),
    )

    np.testing.assert_array_equal(result[0], x)
    np.testing.assert_allclose(result[1], expected_y)
    np.testing.assert_array_equal(result[12], np.zeros(x.shape[1]))
    np.testing.assert_array_equal(result[16], np.zeros(y_raw.shape[1]))


def test_prelim_normalises_relative_performance_with_sparse_nans() -> None:
    """Normalisation consumes transformed Y and ignores isolated NaNs."""
    x = np.array(
        [
            [1.0, 7.0],
            [2.0, 6.0],
            [3.0, 5.0],
            [4.0, 4.0],
            [5.0, 3.0],
            [6.0, 2.0],
        ],
    )
    y_raw = np.array(
        [
            [2.0, 4.0],
            [4.0, 2.0],
            [3.0, np.nan],
            [6.0, 3.0],
            [5.0, 4.0],
            [8.0, 5.0],
        ],
    )
    opts = PrelimOptions(
        max_perf=False,
        abs_perf=False,
        epsilon=0.2,
        beta_threshold=0.55,
        bound=False,
        norm=True,
    )
    performance = compute_binary_performance(
        y_raw,
        PerformanceOptions(False, False, 0.2, 0.55),
        GeneralOptions.default(),
    )
    expected_stage = PrelimStage(
        x.copy(),
        performance.y.copy(),
        x.copy(),
        y_raw.copy(),
        None,
        pd.Series([f"i{i}" for i in range(len(x))]),
        opts,
        selvars_opts,
        GeneralOptions.default(),
    )
    expected = expected_stage._normalise()  # noqa: SLF001

    result = PrelimStage.prelim(
        x.copy(),
        y_raw.copy(),
        x.copy(),
        y_raw.copy(),
        None,
        pd.Series([f"i{i}" for i in range(len(x))]),
        opts,
        selvars_opts,
        GeneralOptions.default(),
    )

    assert result[15] == np.nanmin(performance.y)
    np.testing.assert_allclose(result[1], expected.y, equal_nan=True)
