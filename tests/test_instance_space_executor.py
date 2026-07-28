# SPDX-License-Identifier: LicenseRef-PolyForm-Noncommercial-1.0.0
# Copyright (c) 2024-2026 Mario Andrés Muñoz
"""Unit tests for InstanceSpace's reused ThreadPoolExecutor (Q6).

Exercises `_get_executor()`/`close()` directly against a bare `InstanceSpace`
(`__new__` + only the attributes these methods touch), mirroring the stubbing
style already used in `tests/exploreIS/test_explore_stage_iter.py` -
constructing a full pipeline just to check pool-reuse bookkeeping would be
disproportionate.
"""

from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

from instancespace.data.options import ParallelOptions
from instancespace.instance_space import InstanceSpace


def _bare_instance_space(n_cores):
    # Deliberately untyped (matching test_explore_stage_iter.py's
    # `_stub_stages` pattern): a typed signature makes mypy check this body,
    # which then rejects the intentional attribute-monkeypatching below.
    space = InstanceSpace.__new__(InstanceSpace)
    space._options = SimpleNamespace(parallel=ParallelOptions(True, n_cores))
    space._executor = None
    space._executor_workers = None
    return space


def test_get_executor_reuses_the_pool_when_worker_count_is_unchanged():
    space = _bare_instance_space(n_cores=2)

    first = space._get_executor()
    second = space._get_executor()

    assert first is second
    space.close()


def test_get_executor_recreates_the_pool_when_worker_count_changes():
    space = _bare_instance_space(n_cores=2)

    first = space._get_executor()
    space._options = SimpleNamespace(parallel=ParallelOptions(True, 4))
    second = space._get_executor()

    assert first is not second
    assert first._shutdown  # the stale pool was shut down, not leaked
    space.close()


def test_close_shuts_down_and_unsets_the_pool():
    space = _bare_instance_space(n_cores=2)
    executor = space._get_executor()

    space.close()

    assert space._executor is None
    assert space._executor_workers is None
    assert executor._shutdown


def test_close_is_a_no_op_when_nothing_was_ever_built():
    space = _bare_instance_space(n_cores=2)

    space.close()  # must not raise

    assert space._executor is None


def test_get_executor_after_close_recreates_the_pool_lazily():
    space = _bare_instance_space(n_cores=2)
    first = space._get_executor()
    space.close()

    second = space._get_executor()

    assert isinstance(second, ThreadPoolExecutor)
    assert second is not first
    space.close()


def test_run_stage_passes_the_cached_executor_by_default():
    space = _bare_instance_space(n_cores=2)
    captured = {}

    def fake_run_stage(stage, **arguments):
        captured.update(arguments)
        return "ran"

    space._runner = SimpleNamespace(run_stage=fake_run_stage)

    result = space.run_stage("SomeStage")

    assert result == "ran"
    assert captured["executor"] is space._executor
    space.close()


def test_run_stage_does_not_override_a_caller_supplied_executor():
    space = _bare_instance_space(n_cores=2)
    captured = {}
    caller_executor = ThreadPoolExecutor(max_workers=1)

    def fake_run_stage(stage, **arguments):
        captured.update(arguments)
        return "ran"

    space._runner = SimpleNamespace(run_stage=fake_run_stage)

    space.run_stage("SomeStage", executor=caller_executor)

    assert captured["executor"] is caller_executor
    caller_executor.shutdown(wait=True)
    space.close()
