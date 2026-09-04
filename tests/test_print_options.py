"""Tests for the recursive options printer (Q4)."""

from instancespace.data.options import InstanceSpaceOptions
from instancespace.utils.print_options import format_options


def test_format_options_emits_one_line_per_leaf_field() -> None:
    options = InstanceSpaceOptions.default(*([None] * 12))

    lines = format_options(options)

    # One line per leaf field across every nested group, not one line per top-level
    # group (a nested dataclass's raw repr must not appear on a single line).
    assert any(line.strip().startswith("parallel.flag") for line in lines)
    assert any(line.strip().startswith("parallel.n_cores") for line in lines)
    assert any(line.strip().startswith("perf.max_perf") for line in lines)
    assert any(line.strip().startswith("pythia.cv_folds") for line in lines)
    assert not any("ParallelOptions(" in line for line in lines)
    assert not any("PerformanceOptions(" in line for line in lines)


def test_format_options_leaf_line_contains_value_repr() -> None:
    options = InstanceSpaceOptions.default(*([None] * 12))

    lines = format_options(options)

    flag_line = next(line for line in lines if line.strip().startswith("parallel.flag"))
    assert repr(options.parallel.flag) in flag_line
