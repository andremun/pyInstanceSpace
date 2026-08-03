# SPDX-License-Identifier: LicenseRef-PolyForm-Noncommercial-1.0.0
# Copyright (c) 2024-2026 Mario Andrés Muñoz
"""Tests for F13: eager option validation, matching MATLAB's `ISAvalidateOpts.m`.

`InstanceSpaceOptions.__post_init__` validates every recognised, currently-ported
option field at construction time - for every construction path (`default()`,
`from_dict()`, or direct construction), not just the two entry points that used
to be the only ones exercised - so a bad value fails loudly and immediately
instead of surfacing as a confusing crash deep inside a stage, or silently
producing a wrong-but-plausible result.
"""

import dataclasses
from collections.abc import Callable

import pytest

from instancespace.data.options import InstanceSpaceOptions, ParallelOptions


def test_fully_defaulted_options_construct_without_error(
    valid_options: InstanceSpaceOptions,
) -> None:
    """Sanity check: `valid_options` (T3's shared fixture) is itself valid."""


def test_none_seed_is_valid_not_rejected_as_a_bad_int(
    valid_options: InstanceSpaceOptions,
) -> None:
    """`general.seed` may be `None` (non-deterministic run) - not a MATLAB field."""
    general = dataclasses.replace(valid_options.general, seed=None)
    dataclasses.replace(valid_options, general=general)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda o: dataclasses.replace(
                o,
                parallel=dataclasses.replace(o.parallel, flag=1),
            ),
            "logical",
        ),
        (
            lambda o: dataclasses.replace(
                o,
                parallel=ParallelOptions(flag=False, n_cores=-1),
            ),
            "positive",
        ),
        (
            lambda o: dataclasses.replace(
                o,
                perf=dataclasses.replace(o.perf, epsilon=1.5),
            ),
            "unit range",
        ),
        (
            lambda o: dataclasses.replace(
                o,
                selvars=dataclasses.replace(o.selvars, selvars_type="Bogus"),
            ),
            "one of",
        ),
        (
            lambda o: dataclasses.replace(
                o,
                selvars=dataclasses.replace(o.selvars, feats=["a", 2, "c"]),
            ),
            "list of strings",
        ),
        (
            lambda o: dataclasses.replace(
                o,
                sifted=dataclasses.replace(o.sifted, k=0),
            ),
            "positive",
        ),
        (
            lambda o: dataclasses.replace(
                o,
                pythia=dataclasses.replace(o.pythia, classifier="not-a-classifier"),
            ),
            "one of",
        ),
        (
            lambda o: dataclasses.replace(
                o,
                trace=dataclasses.replace(o.trace, method="not-a-method"),
            ),
            "one of",
        ),
        (
            lambda o: dataclasses.replace(
                o,
                pilot=dataclasses.replace(o.pilot, method="not-a-method"),
            ),
            "one of",
        ),
        (
            lambda o: dataclasses.replace(
                o,
                outputs=dataclasses.replace(o.outputs, csv=0),
            ),
            "logical",
        ),
    ],
)
def test_invalid_field_raises_with_a_clear_message(
    valid_options: InstanceSpaceOptions,
    mutate: Callable[[InstanceSpaceOptions], InstanceSpaceOptions],
    message: str,
) -> None:
    """A bad value for a validated field raises `ValueError` immediately."""
    with pytest.raises(ValueError, match=message):
        mutate(valid_options)


def test_from_dict_runs_the_same_validation() -> None:
    """`from_dict()` (the JSON-loading entry point) hits the same `__post_init__`."""
    with pytest.raises(ValueError, match="unit range"):
        InstanceSpaceOptions.from_dict({"perf": {"epsilon": 3.0}})


def test_selvars_feats_and_algos_accept_none(
    valid_options: InstanceSpaceOptions,
) -> None:
    """`feats`/`algos` are optional - `None` must not be rejected as a bad list."""
    selvars = dataclasses.replace(valid_options.selvars, feats=None, algos=None)
    dataclasses.replace(valid_options, selvars=selvars)


def test_unrecognised_json_field_still_errors_before_validation_runs() -> None:
    """An unknown field name is still caught by the existing name-check, not F13's."""
    with pytest.raises(ValueError, match="not defined"):
        InstanceSpaceOptions.from_dict({"perf": {"not_a_real_field": 1}})
