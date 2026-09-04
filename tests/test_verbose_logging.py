"""Smoke tests for Q3: `general.verbose` gates per-trial/per-iteration log detail.

Each stage's `_log`/`_pilot_print` always logs its top-level narrative messages;
per-trial/per-iteration detail is only logged when `GeneralOptions.verbose` is True.
"""

from collections.abc import Callable
from typing import Any

import numpy as np
from loguru import logger
from numpy.typing import NDArray

from instancespace.data.options import GeneralOptions, PilotOptions
from instancespace.stages.pilot import PilotStage


def _collect_logs(
    fn: Callable[..., Any],
    *args: Any,  # noqa: ANN401
    level: str = "DEBUG",
) -> list[str]:
    """Run fn(*args) and return the loguru messages it emitted at/above level."""
    messages: list[str] = []
    sink_id = logger.add(
        lambda msg: messages.append(msg.record["message"]),
        level=level,
    )
    try:
        fn(*args)
    finally:
        logger.remove(sink_id)
    return messages


def _small_pilot_inputs() -> tuple[
    NDArray[np.double],
    NDArray[np.double],
    list[str],
    PilotOptions,
]:
    rng = np.random.default_rng(0)
    x = rng.random((20, 3))
    y = rng.random((20, 2))
    feat_labels = ["f0", "f1", "f2"]
    opts = PilotOptions(None, None, False, 2)
    return x, y, feat_labels, opts


def test_pilot_per_trial_detail_only_appears_when_verbose() -> None:
    """PILOT's per-trial message shows only when general.verbose is True."""
    x, y, feat_labels, opts = _small_pilot_inputs()

    verbose_messages = _collect_logs(
        PilotStage.pilot,
        x,
        y,
        feat_labels,
        opts,
        GeneralOptions(verbose=True, seed=0),
    )
    quiet_messages = _collect_logs(
        PilotStage.pilot,
        x,
        y,
        feat_labels,
        opts,
        GeneralOptions(verbose=False, seed=0),
    )

    assert any("completed trial" in m for m in verbose_messages)
    assert not any("completed trial" in m for m in quiet_messages)

    # The top-level narrative always shows, regardless of verbose.
    assert any("PILOT is solving numerically" in m for m in verbose_messages)
    assert any("PILOT is solving numerically" in m for m in quiet_messages)
