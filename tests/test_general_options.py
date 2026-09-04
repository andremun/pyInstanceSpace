"""Tests for GeneralOptions (Q3/Q9): the new `verbose`/`seed` option group."""

from instancespace.data.options import GeneralOptions, InstanceSpaceOptions


def test_general_options_default_matches_previously_hardcoded_behaviour() -> None:
    general = GeneralOptions.default()
    assert general.verbose is True
    assert general.seed == 0


def test_instance_space_options_default_general_is_backward_compatible() -> None:
    # Existing callers that never mention `general` (positional *([None] * 12), or
    # direct InstanceSpaceOptions(...) construction predating this field) must keep
    # getting today's implicit behaviour: verbose logging on, seed 0.
    options = InstanceSpaceOptions.default(*([None] * 12))
    assert options.general.verbose is True
    assert options.general.seed == 0


def test_instance_space_options_accepts_explicit_general() -> None:
    custom = GeneralOptions.default(verbose=False, seed=42)
    options = InstanceSpaceOptions.default(
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        general=custom,
    )
    assert options.general is custom


def test_from_dict_defaults_general_when_absent() -> None:
    options = InstanceSpaceOptions.from_dict({})
    assert options.general.verbose is True
    assert options.general.seed == 0


def test_from_dict_parses_general_section() -> None:
    options = InstanceSpaceOptions.from_dict({"general": {"verbose": False, "seed": 7}})
    assert options.general.verbose is False
    assert options.general.seed == 7


def test_from_dict_allows_null_seed_for_nondeterministic_runs() -> None:
    options = InstanceSpaceOptions.from_dict({"general": {"seed": None}})
    assert options.general.seed is None
