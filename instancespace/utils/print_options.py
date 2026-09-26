# SPDX-License-Identifier: LicenseRef-PolyForm-Noncommercial-1.0.0
# Copyright (c) 2024-2026 Mario Andrés Muñoz
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

    Parameters
    ----------
    options : Any
        A dataclass instance, for example an ``InstanceSpaceOptions``. During
        recursion, it is one of the nested option groups.
    prefix : str
        The dotted-path prefix for each field name. Each recursive call adds the
        name of the current field to the prefix. Callers usually omit this.

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
