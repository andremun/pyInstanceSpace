"""Recursive, compact formatting for nested InstanceSpaceOptions dataclasses.

Mirrors MATLAB's ``InstanceSpace.printOptions()``/``formatOptionValue()``: a nested
options dataclass (e.g. ``options.parallel``) is recursed into so every leaf setting
gets its own line (``parallel.flag``, ``parallel.n_cores``, ...), rather than printing
the nested dataclass's raw repr on a single line.
"""

from __future__ import annotations

import dataclasses
from typing import Any


def format_options(options: Any, prefix: str = "") -> list[str]:  # noqa: ANN401
    """Format a (possibly nested) dataclass into one line per leaf field.

    Args
    ----
        options : Any
            A dataclass instance (e.g. an ``InstanceSpaceOptions``, or one of its
            nested option groups during recursion).
        prefix : str
            Dotted-path prefix prepended to each field name; extended with the
            current field's name on each recursive call. Callers normally omit this.

    Returns
    -------
        list[str]
            One formatted line per leaf (non-dataclass) field.
    """
    lines: list[str] = []
    for field in dataclasses.fields(options):
        value = getattr(options, field.name)
        name = f"{prefix}{field.name}"
        if dataclasses.is_dataclass(value) and not isinstance(value, type):
            lines.extend(format_options(value, prefix=f"{name}."))
        else:
            lines.append(f"  {name:<28} {value!r}")
    return lines
