# SPDX-License-Identifier: LicenseRef-PolyForm-Noncommercial-1.0.0
# Copyright (c) 2024-2026 Mario Andrés Muñoz
"""Unit tests for TraceStage's pool-reuse hook (Q6).

Checks the actual mechanism `compute_algorithm_qualities` uses to decide
whether to submit work to a caller-supplied pool or create its own - not the
footprint computation itself, which the existing `test_trace.py` validation
tests already cover end to end.
"""

from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from typing import cast
from unittest.mock import patch

import numpy as np

from instancespace.data.model import Footprint
from instancespace.data.options import GeneralOptions, ParallelOptions
from instancespace.stages.trace import TraceStage


def _bare_trace_stage(n_algos, executor) -> TraceStage:  # type: ignore[no-untyped-def]
    # Params deliberately untyped (see test_instance_space_executor.py's
    # _bare_instance_space for why): a typed signature makes mypy check this
    # body, which then rejects the intentional attribute-monkeypatching below.
    stage = TraceStage.__new__(TraceStage)
    stage.algo_labels = [f"algo{i}" for i in range(n_algos)]
    stage.y_bin = np.zeros((3, n_algos), dtype=np.bool_)
    stage.p = np.zeros(3, dtype=np.int_)
    stage.parallel_opts = ParallelOptions(True, 2)
    stage.executor = executor
    stage.general_opts = cast(GeneralOptions, SimpleNamespace(verbose=False))
    stage.process_algorithm = lambda i: (  # type: ignore[method-assign]
        i,
        Footprint(None, 0, 0, 0, 0, 0),
        Footprint(None, 0, 0, 0, 0, 0),
    )
    return stage


def test_compute_algorithm_qualities_reuses_a_supplied_executor() -> None:
    shared_executor = ThreadPoolExecutor(max_workers=2)
    stage = _bare_trace_stage(n_algos=3, executor=shared_executor)

    with patch(
        "instancespace.stages.trace.ThreadPoolExecutor",
    ) as mock_pool_class:
        good, best = stage.compute_algorithm_qualities(3)

    mock_pool_class.assert_not_called()
    assert len(good) == 3
    assert len(best) == 3
    shared_executor.shutdown(wait=True)


def test_compute_algorithm_qualities_creates_its_own_pool_when_none_supplied() -> None:
    stage = _bare_trace_stage(n_algos=2, executor=None)

    good, best = stage.compute_algorithm_qualities(2)

    assert len(good) == 2
    assert len(best) == 2


def test_compute_algorithm_qualities_output_identical_with_and_without_reuse() -> None:
    # Same inputs, only the pool-sourcing differs - results must match exactly.
    own_pool_stage = _bare_trace_stage(n_algos=4, executor=None)
    own_good, own_best = own_pool_stage.compute_algorithm_qualities(4)

    shared_executor = ThreadPoolExecutor(max_workers=2)
    shared_stage = _bare_trace_stage(n_algos=4, executor=shared_executor)
    shared_good, shared_best = shared_stage.compute_algorithm_qualities(4)
    shared_executor.shutdown(wait=True)

    assert own_good == shared_good
    assert own_best == shared_best
