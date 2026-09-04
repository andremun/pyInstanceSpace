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
from pathlib import Path
from typing import cast

import numpy as np
import pytest

from instancespace.data.default_options import DEFAULT_PILOT_N_TRIES
from instancespace.data.options import (
    AutoOptions,
    BoundOptions,
    CloisterOptions,
    GeneralOptions,
    InstanceSpaceOptions,
    NormOptions,
    OutputOptions,
    ParallelOptions,
    PerformanceOptions,
    PilotOptions,
    PrelimConfigOptions,
    PrelimOptions,
    PythiaOptions,
    SelvarsOptions,
    SiftedOptions,
    TraceOptions,
    from_json_file,
)


def _replace_nested_options(
    options: InstanceSpaceOptions,
    group: str,
    **changes: object,
) -> InstanceSpaceOptions:
    """Replace one nested option group and revalidate the aggregate."""
    nested = dataclasses.replace(getattr(options, group), **changes)
    return dataclasses.replace(options, **{group: nested})


def test_fully_defaulted_options_construct_without_error(
    valid_options: InstanceSpaceOptions,
) -> None:
    """Sanity check: `valid_options` (T3's shared fixture) is itself valid."""


def test_no_argument_default_factory_builds_the_complete_option_tree() -> None:
    """The documented public factory requires no placeholder arguments."""
    assert InstanceSpaceOptions.default() == InstanceSpaceOptions.from_dict({})


def test_prelim_configuration_defaults_match_matlab() -> None:
    """The aggregate exposes MATLAB's build-level PRELIM defaults."""
    iqr_multiplier = 5.0
    nan_threshold = 0.20
    options = InstanceSpaceOptions.default()

    assert options.prelim.iqr_multiplier == iqr_multiplier
    assert options.prelim.nan_threshold == nan_threshold


def test_matlab_prelim_json_aliases_are_configurable() -> None:
    """MATLAB's camel-case PRELIM keys load into the aggregate configuration."""
    iqr_multiplier = 7.5
    nan_threshold = 0.35
    options = InstanceSpaceOptions.from_dict(
        {
            "PrElIm": {
                "iqrMultiplier": iqr_multiplier,
                "nanThreshold": nan_threshold,
            },
        },
    )

    assert options.prelim.iqr_multiplier == iqr_multiplier
    assert options.prelim.nan_threshold == nan_threshold


@pytest.mark.parametrize(
    ("construct", "message"),
    [
        (lambda: PrelimConfigOptions.default(iqr_multiplier=0), "strictly positive"),
        (lambda: PrelimConfigOptions.default(nan_threshold=1.1), "unit range"),
        (
            lambda: PrelimOptions(
                False,
                True,
                0.2,
                0.55,
                True,
                True,
                iqr_multiplier=0,
            ),
            "strictly positive",
        ),
        (
            lambda: PrelimOptions(
                False,
                True,
                0.2,
                0.55,
                True,
                True,
                nan_threshold=-0.1,
            ),
            "unit range",
        ),
    ],
)
def test_prelim_configuration_validates_at_direct_boundaries(
    construct: Callable[[], object],
    message: str,
) -> None:
    """Both aggregate and composed PRELIM options reject invalid thresholds."""
    with pytest.raises(ValueError, match=message):
        construct()


def test_none_seed_is_valid_not_rejected_as_a_bad_int(
    valid_options: InstanceSpaceOptions,
) -> None:
    """General and inherited stage seeds may be None for non-deterministic runs."""
    general = dataclasses.replace(valid_options.general, seed=None)
    sifted = dataclasses.replace(valid_options.sifted, seed=None)
    pilot = dataclasses.replace(valid_options.pilot, seed=None)
    dataclasses.replace(valid_options, general=general, sifted=sifted, pilot=pilot)


def test_stage_seeds_load_from_matlab_option_names() -> None:
    """Resolved MATLAB stage seeds map directly to the public option fields."""
    options = InstanceSpaceOptions.from_dict(
        {"sifted": {"seed": 17}, "pilot": {"seed": 23}},
    )

    assert options.sifted.seed == 17
    assert options.pilot.seed == 23


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda o: dataclasses.replace(
                o,
                parallel=dataclasses.replace(o.parallel, flag=1),
            ),
            "True or False",
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
                perf=dataclasses.replace(o.perf, abs_perf=False, epsilon=1.5),
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
            "True or False",
        ),
        (
            lambda o: dataclasses.replace(
                o,
                pythia=dataclasses.replace(o.pythia, skip=1),
            ),
            "True or False",
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
        InstanceSpaceOptions.from_dict({"perf": {"absperf": False, "epsilon": 3.0}})


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


def test_pythia_skip_with_trace_use_sim_raises(
    valid_options: InstanceSpaceOptions,
) -> None:
    """#298 Issue 10: `pythia.skip=True` + `trace.use_sim=True` is rejected.

    This port's TRACE (legacy-only) clusters PYTHIA's `y_hat` predictions
    with DBSCAN to build compact footprints - `y_hat` fills the role
    DBSCAN's own density clustering plays on raw labels in true legacy
    TRACE. Skipping PYTHIA training removes that input entirely, so
    silently allowing this combination would degrade TRACE's footprints
    from compact regions to fragmented ones built from raw, noisy `y_bin`,
    unlike MATLAB's trace3 (which has an independent, Yhat-free compacting
    mechanism and so degrades gracefully instead).
    """
    with pytest.raises(ValueError, match="incompatible"):
        dataclasses.replace(
            valid_options,
            pythia=dataclasses.replace(valid_options.pythia, skip=True),
            trace=dataclasses.replace(valid_options.trace, use_sim=True),
        )


def test_pythia_skip_with_trace_use_sim_false_constructs(
    valid_options: InstanceSpaceOptions,
) -> None:
    """The safe pairing - `use_sim=False` - is not rejected."""
    dataclasses.replace(
        valid_options,
        pythia=dataclasses.replace(valid_options.pythia, skip=True),
        trace=dataclasses.replace(valid_options.trace, use_sim=False),
    )


def test_absolute_performance_epsilon_accepts_any_finite_real(
    valid_options: InstanceSpaceOptions,
) -> None:
    """Absolute thresholds are not restricted to a unit-scaled measure."""
    perf = dataclasses.replace(valid_options.perf, abs_perf=True, epsilon=12.5)
    dataclasses.replace(valid_options, perf=perf)


@pytest.mark.parametrize(
    ("group", "field", "value", "message"),
    [
        ("sifted", "flag", 1, "True or False"),
        ("sifted", "rho", np.nan, "finite real"),
        ("sifted", "k", 0, "positive"),
        ("sifted", "n_trees", 0, "positive"),
        ("sifted", "max_iter", 0, "positive"),
        ("sifted", "replicates", 0, "positive"),
        ("sifted", "num_generations", 0, "positive"),
        ("sifted", "num_parents_mating", 0, "positive"),
        ("sifted", "sol_per_pop", 0, "positive"),
        ("sifted", "parent_selection_type", "bad", "one of"),
        ("sifted", "k_tournament", 0, "positive"),
        ("sifted", "keep_elitism", True, "integer"),
        ("sifted", "crossover_type", "bad", "one of"),
        ("sifted", "cross_over_probability", np.inf, "finite real"),
        ("sifted", "mutation_type", "bad", "one of"),
        ("sifted", "mutation_probability", np.nan, "finite real"),
        ("sifted", "stop_criteria", "max_generations", "saturate_N"),
        ("sifted", "pval", -0.1, "unit range"),
        ("sifted", "dims", 2.0, "2 or 3"),
        ("pilot", "analytic", 1, "True or False"),
        ("pilot", "n_tries", 0, "positive"),
        ("pilot", "adjust_rotation", 1, "True or False"),
        ("pilot", "cost_weight", np.nan, "finite real"),
        ("pilot", "method", "bad", "one of"),
        ("pilot", "dims", 2.0, "2 or 3"),
        ("pilot", "view_groups", [[-1]], "zero-based"),
        ("cloister", "p_val", np.nan, "finite real"),
        ("cloister", "c_thres", 1.1, "unit range"),
        ("cloister", "max_features", 0, "positive"),
        ("cloister", "hull_dims", 1, "at least 2"),
        ("pythia", "cv_folds", 1, "at least 2"),
        ("pythia", "is_poly_krnl", 1, "True or False"),
        ("pythia", "use_weights", 1, "True or False"),
        ("pythia", "classifier", "bad", "one of"),
        ("pythia", "tuning", "bad", "one of"),
        ("pythia", "n_tuning_iter", 0, "positive"),
        ("pythia", "skip", 1, "True or False"),
        ("trace", "use_sim", 1, "True or False"),
        ("trace", "purity", np.nan, "finite real"),
        ("trace", "method", "bad", "one of"),
        ("trace", "contra", 1, "True or False"),
        ("trace", "min_instances", 0, "positive"),
        ("trace", "min_area_frac", np.inf, "finite real"),
    ],
)
def test_every_active_stage_option_rejects_invalid_values(
    valid_options: InstanceSpaceOptions,
    group: str,
    field: str,
    value: object,
    message: str,
) -> None:
    """Every option consumed by an active stage has a named eager check."""
    with pytest.raises(ValueError, match=message):
        _replace_nested_options(valid_options, group, **{field: value})


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("x0", np.ones(4), "two-dimensional"),
        ("precalc_alpha", np.array([[np.nan]]), "finite"),
        ("x0", np.array([[True]]), "non-Boolean"),
    ],
)
def test_pilot_option_matrices_are_validated_when_created(
    valid_options: InstanceSpaceOptions,
    field: str,
    value: object,
    message: str,
) -> None:
    """PILOT matrices reject invalid rank, value, and scalar types early."""
    with pytest.raises(ValueError, match=message):
        dataclasses.replace(
            valid_options.pilot,
            **{field: value},  # type: ignore[arg-type]
        )


def test_pilot_allows_precalculated_solution_alongside_start_matrix(
    valid_options: InstanceSpaceOptions,
) -> None:
    """MATLAB accepts both and gives a valid precalcAlpha precedence."""
    options = dataclasses.replace(
        valid_options.pilot,
        x0=np.ones((2, 1)),
        precalc_alpha=np.ones((2, 1)),
    )

    assert options.x0 is not None
    assert options.precalc_alpha is not None


def test_pilot_restart_default_matches_matlab() -> None:
    """MATLAB uses ten restarts while callers can request the old Python count."""
    matlab_default_n_tries = 10
    assert (
        PilotOptions.default().n_tries
        == DEFAULT_PILOT_N_TRIES
        == matlab_default_n_tries
    )
    assert PilotOptions.default(n_tries=5).n_tries == 5


def test_pilot_3d_options_load_and_round_trip_in_canonical_form() -> None:
    """MATLAB names load while the public object stores immutable Python indices."""
    dims = 3
    options = InstanceSpaceOptions.from_dict(
        {
            "pilot": {
                "dims": dims,
                "viewGroups": [[0, 2, 2], [1]],
            },
        },
    )

    assert options.pilot.dims == dims
    assert options.pilot.view_groups == ((0, 2, 2), (1,))

    round_tripped = InstanceSpaceOptions.from_dict(dataclasses.asdict(options))
    assert round_tripped.pilot.view_groups == options.pilot.view_groups


def test_pilot_dimension_additions_preserve_positional_2d_defaults() -> None:
    """Existing four-argument construction remains 2D with one global view."""
    dims = 2
    options = PilotOptions(None, None, False, 5)

    assert options.dims == dims
    assert options.view_groups == ()


@pytest.mark.parametrize(
    ("value", "expected_dims"),
    [(False, 2), (True, 3)],
)
def test_legacy_isa3d_json_maps_unambiguously_to_dims(
    value: bool,
    expected_dims: int,
) -> None:
    """The MATLAB legacy boolean remains a JSON-only dimensionality alias."""
    options = InstanceSpaceOptions.from_dict({"pilot": {"ISA3D": value}})

    assert options.pilot.dims == expected_dims


def test_consistent_legacy_isa3d_and_dims_are_accepted() -> None:
    """Equivalent legacy and current fields have one unambiguous meaning."""
    dims = 3
    options = InstanceSpaceOptions.from_dict(
        {"pilot": {"ISA3D": True, "dims": dims}},
    )

    assert options.pilot.dims == dims


def test_conflicting_legacy_isa3d_and_dims_are_rejected() -> None:
    """Conflicting dimensionality fields cannot be resolved by precedence."""
    with pytest.raises(ValueError, match="different projection dimensions"):
        InstanceSpaceOptions.from_dict(
            {"pilot": {"ISA3D": False, "dims": 3}},
        )


@pytest.mark.parametrize("value", [0, 1, "true", None])
def test_legacy_isa3d_requires_a_json_boolean(value: object) -> None:
    """Numeric and text truthiness must not silently select a dimension."""
    with pytest.raises(ValueError, match="True or False"):
        InstanceSpaceOptions.from_dict({"pilot": {"ISA3D": value}})


@pytest.mark.parametrize(
    "view_groups",
    [None, [0, 1], [[]], [[-1]], [[True]], [[1.0]], "0,1"],
)
def test_pilot_view_groups_reject_malformed_or_non_zero_based_indices(
    view_groups: object,
) -> None:
    """Every supplied viewpoint group is a non-empty integer index vector."""
    with pytest.raises(ValueError, match="viewGroups"):
        PilotOptions.default(view_groups=view_groups)


def test_pilot_view_groups_accept_lists_and_tuples_without_deduplicating() -> None:
    """MATLAB-permitted overlaps and duplicate indices retain their ordering."""
    options = PilotOptions.default(view_groups=[[0, 0], (0, 2)])

    assert options.view_groups == ((0, 0), (0, 2))


def test_adjust_rotation_is_rejected_for_a_3d_projection() -> None:
    """The existing centroid-angle rotation is defined only for 2D PILOT."""
    with pytest.raises(ValueError, match="only supported.*dims=2"):
        PilotOptions.default(dims=3, adjust_rotation=True)


@pytest.mark.parametrize(
    ("params", "message"),
    [
        (np.ones(2), "two-dimensional"),
        (np.array([[np.nan, 1.0]]), "finite"),
        (np.array([[1.0]]), "2 columns"),
        (np.array([[0.0, 1.0]]), "strictly positive"),
    ],
)
def test_pythia_parameter_matrix_contract(
    valid_options: InstanceSpaceOptions,
    params: np.ndarray,  # type: ignore[type-arg]
    message: str,
) -> None:
    """PYTHIA parameters are finite, 2-D, and classifier-aware."""
    with pytest.raises(ValueError, match=message):
        _replace_nested_options(valid_options, "pythia", params=params)


@pytest.mark.parametrize(
    ("classifier", "params"),
    [
        ("knn", np.array([[-2.5, 5.0]])),
        ("tree", np.array([[-2.5]])),
        ("ensemble", np.array([[0.0, -2.5]])),
    ],
)
def test_pythia_discrete_precalculated_params_allow_matlab_normalization(
    classifier: str,
    params: np.ndarray,  # type: ignore[type-arg]
) -> None:
    """Raw discrete values may reach MATLAB-compatible stage normalization."""
    options = PythiaOptions.default(
        classifier=classifier,
        tuning="none",
        params=params,
    )

    assert options.params is not None
    np.testing.assert_array_equal(options.params, params)


@pytest.mark.parametrize(
    ("classifier", "params", "parameter_name"),
    [
        ("svm", np.array([[0.0, 1.0]]), "BoxConstraint"),
        ("svm", np.array([[1.0, -1.0]]), "KernelScale"),
        ("nb", np.array([[0.0]]), "Bandwidth"),
        ("linear", np.array([[-1.0]]), "Lambda"),
    ],
)
def test_pythia_continuous_precalculated_params_remain_strictly_positive(
    classifier: str,
    params: np.ndarray,  # type: ignore[type-arg]
    parameter_name: str,
) -> None:
    """Continuous estimator parameters fail eagerly with their MATLAB name."""
    with pytest.raises(ValueError, match=rf"{parameter_name}.*strictly positive"):
        PythiaOptions.default(
            classifier=classifier,
            tuning="none",
            params=params,
        )


def test_pythia_none_tuning_requires_precalculated_parameters(
    valid_options: InstanceSpaceOptions,
) -> None:
    """Disabling tuning without supplying parameters is incomplete."""
    with pytest.raises(ValueError, match="requires"):
        _replace_nested_options(valid_options, "pythia", tuning="none")


def test_adaptive_mutation_accepts_two_unit_probabilities(
    valid_options: InstanceSpaceOptions,
) -> None:
    """PyGAD adaptive mutation uses one probability for each fitness group."""
    sifted = dataclasses.replace(
        valid_options.sifted,
        mutation_type="adaptive",
        mutation_probability=[0.2, 0.1],
    )
    dataclasses.replace(valid_options, sifted=sifted)


@pytest.mark.parametrize(
    "field",
    ["num_parents_mating", "k_tournament", "keep_elitism"],
)
def test_sifted_population_dependent_values_cannot_exceed_population(
    valid_options: InstanceSpaceOptions,
    field: str,
) -> None:
    """GA population relationships fail before PyGAD receives the options."""
    with pytest.raises(ValueError, match="cannot exceed"):
        _replace_nested_options(
            valid_options,
            "sifted",
            sol_per_pop=2,
            **{field: 3},
        )


@pytest.mark.parametrize(
    ("method", "expected"),
    [("legacy", 0.55), ("trace3", 0.60)],
)
def test_trace_omitted_purity_is_method_aware(method: str, expected: float) -> None:
    """Omitted purity follows each MATLAB TRACE method's own default."""
    direct_trace = TraceOptions.default(method=method)
    loaded = InstanceSpaceOptions.from_dict({"trace": {"method": method}})

    assert direct_trace.purity == expected
    assert loaded.trace.purity == expected


def test_trace3_json_aliases_load_new_thresholds() -> None:
    """MATLAB-style TRACE3 threshold keys map to Python option fields."""
    min_instances = 7
    min_area_frac = 0.2
    options = InstanceSpaceOptions.from_dict(
        {
            "trace": {
                "method": "trace3",
                "minInstances": min_instances,
                "minAreaFrac": min_area_frac,
            },
        },
    )

    assert options.trace.min_instances == min_instances
    assert options.trace.min_area_frac == min_area_frac


def test_trace_explicit_purity_wins_over_method_default() -> None:
    """A caller-provided threshold is not overwritten by method selection."""
    explicit_purity = 0.72
    options = InstanceSpaceOptions.from_dict(
        {"trace": {"method": "trace3", "purity": explicit_purity}},
    )

    assert options.trace.purity == explicit_purity


def test_trace3_allows_pythia_skip_with_similarity_enabled(
    valid_options: InstanceSpaceOptions,
) -> None:
    """TRACE3 can fall back to truth when PYTHIA predictions are unavailable."""
    pythia = dataclasses.replace(valid_options.pythia, skip=True)
    trace = dataclasses.replace(valid_options.trace, method="trace3", use_sim=True)

    dataclasses.replace(valid_options, pythia=pythia, trace=trace)


def test_json_option_matrices_are_normalized_to_double_arrays() -> None:
    """JSON nested lists enter the same matrix validation path as direct arrays."""
    options = InstanceSpaceOptions.from_dict(
        {
            "pilot": {"x0": [[1.0], [2.0]]},
            "pythia": {"params": [[1.0, 2.0]]},
        },
    )

    assert isinstance(options.pilot.x0, np.ndarray)
    assert options.pilot.x0.dtype == np.double
    assert isinstance(options.pythia.params, np.ndarray)
    assert options.pythia.params.dtype == np.double


def test_current_matlab_option_aliases_load() -> None:
    """Current MATLAB option names map to their canonical Python fields."""
    cost_weight = 2.5
    corr_threshold = 0.6
    max_features = 8
    cv_folds = 3
    options = InstanceSpaceOptions.from_dict(
        {
            "pilot": {"alpha": cost_weight},
            "cloister": {
                "corrThreshold": corr_threshold,
                "maxFeatures": max_features,
                "hullDims": "ALL",
            },
            "pythia": {"kFold": cv_folds},
        },
    )

    assert options.pilot.cost_weight == cost_weight
    assert options.cloister.c_thres == corr_threshold
    assert options.cloister.max_features == max_features
    assert options.cloister.hull_dims == "all"
    assert options.pythia.cv_folds == cv_folds


def test_aliases_cannot_define_one_option_twice() -> None:
    """Equivalent MATLAB/Python names cannot silently overwrite each other."""
    with pytest.raises(ValueError, match="Conflicting fields"):
        InstanceSpaceOptions.from_dict(
            {"pilot": {"alpha": 2.0, "costWeight": 3.0}},
        )


def test_case_insensitive_member_values_are_normalized() -> None:
    """MATLAB-style enum spellings are accepted and stored canonically."""
    trace3_purity = 0.60
    sifted = SiftedOptions.default(
        parent_selection_type="TOURNAMENT",
        crossover_type="SCATTERED",
        mutation_type="RANDOM",
        stop_criteria="SATURATE_5",
    )
    pilot = PilotOptions.default(method="PLS")
    cloister = CloisterOptions.default(hull_dims="ALL")  # type: ignore[arg-type]
    pythia = PythiaOptions.default(classifier="SVM", tuning="SOBOL")
    trace = TraceOptions.default(method="TRACE3")
    selvars = SelvarsOptions.default(selvars_type="ftr&ap")

    assert sifted.parent_selection_type == "tournament"
    assert sifted.crossover_type == "scattered"
    assert sifted.mutation_type == "random"
    assert sifted.stop_criteria == "saturate_5"
    assert pilot.method == "pls"
    assert cloister.hull_dims == "all"
    assert pythia.classifier == "svm"
    assert pythia.tuning == "sobol"
    assert trace.method == "trace3"
    assert trace.purity == trace3_purity
    assert selvars.selvars_type == "Ftr&AP"


def test_pythia_skip_bypasses_inactive_training_options() -> None:
    """Skip mode validates its classifier but ignores tuning-only fields."""
    options = PythiaOptions.default(
        classifier="SVM",
        tuning="none",
        n_tuning_iter=0,
        params=np.ones(1),
        skip=True,
    )

    assert options.skip is True
    assert options.classifier == "svm"
    assert options.tuning == "none"
    assert options.params is not None


def test_pythia_skip_rejects_an_unknown_classifier_before_stage_execution() -> None:
    """Skip still consumes classifier selection before its training bypass."""
    with pytest.raises(ValueError, match="pythia.classifier.*one of"):
        PythiaOptions.default(classifier="not-used", skip=True)


def test_pythia_skip_still_validates_non_tuning_configuration() -> None:
    """Skip bypasses training inputs, not the shared option schema."""
    with pytest.raises(ValueError, match="at least 2"):
        PythiaOptions.default(cv_folds=1, skip=True)


def test_pythia_valid_precalculated_params_make_tuning_count_irrelevant() -> None:
    """No tuning iterations are required when valid parameters are supplied."""
    options = PythiaOptions.default(
        params=np.ones((1, 2)),
        n_tuning_iter=0,
    )

    assert options.params is not None
    assert options.params.shape == (1, 2)


@pytest.mark.parametrize(
    ("construct", "message"),
    [
        (lambda: GeneralOptions.default(seed=-1), "non-negative"),
        (lambda: ParallelOptions.default(n_cores=0), "positive"),
        (
            lambda: PerformanceOptions.default(abs_perf=False, epsilon=2.0),
            "unit range",
        ),
        (lambda: AutoOptions.default(preproc=cast(bool, 1)), "True or False"),
        (lambda: BoundOptions.default(flag=cast(bool, 1)), "True or False"),
        (lambda: NormOptions.default(flag=cast(bool, 1)), "True or False"),
        (lambda: SelvarsOptions.default(selvars_type="bad"), "one of"),
        (lambda: SiftedOptions.default(k=0), "positive"),
        (lambda: SiftedOptions.default(seed=-1), "non-negative"),
        (lambda: PilotOptions.default(n_tries=0), "positive"),
        (lambda: PilotOptions.default(seed=-1), "non-negative"),
        (lambda: CloisterOptions.default(max_features=0), "positive"),
        (lambda: PythiaOptions.default(cv_folds=1), "at least 2"),
        (lambda: TraceOptions.default(min_instances=0), "positive"),
        (lambda: OutputOptions.default(csv=cast(bool, 1)), "True or False"),
    ],
)
def test_direct_nested_option_construction_validates_active_fields(
    construct: Callable[[], object],
    message: str,
) -> None:
    """Nested option objects fail at their own construction boundary."""
    with pytest.raises(ValueError, match=message):
        construct()


@pytest.mark.parametrize("contents", [None, [], "not-an-object", 4])
def test_malformed_json_root_has_a_contextual_error(contents: object) -> None:
    """A malformed JSON root reports the root contract, not an attribute crash."""
    with pytest.raises(ValueError, match="JSON root must be an object"):
        InstanceSpaceOptions.from_dict(contents)


def test_non_text_json_root_key_has_a_contextual_error() -> None:
    """Top-level group names must be JSON strings."""
    with pytest.raises(ValueError, match="JSON root keys must be strings"):
        InstanceSpaceOptions.from_dict({1: {}})


def test_top_level_json_groups_are_case_insensitive() -> None:
    """Top-level groups follow the same case-insensitive contract as leaves."""
    epsilon = 0.25
    options = InstanceSpaceOptions.from_dict(
        {
            "PeRf": {"AbsPerf": False, "Epsilon": epsilon},
            "TRACE": {"Method": "TRACE3"},
        },
    )

    assert options.perf.abs_perf is False
    assert options.perf.epsilon == epsilon
    assert options.trace.method == "trace3"


def test_nested_json_fields_use_unicode_casefolding() -> None:
    """Nested keys use the loader's Unicode case-insensitive contract."""
    seed = 19
    long_s_seed = "\N{LATIN SMALL LETTER LONG S}eed"

    options = InstanceSpaceOptions.from_dict({"general": {long_s_seed: seed}})

    assert options.general.seed == seed


def test_casefold_equivalent_nested_json_fields_are_rejected() -> None:
    """Unicode-equivalent keys cannot silently overwrite one option."""
    long_s_seed = "\N{LATIN SMALL LETTER LONG S}eed"

    with pytest.raises(ValueError, match="Conflicting fields") as exc_info:
        InstanceSpaceOptions.from_dict(
            {"general": {"seed": 1, long_s_seed: 2}},
        )

    assert long_s_seed in str(exc_info.value)


def test_case_only_top_level_duplicates_are_rejected() -> None:
    """Equivalent group names cannot silently replace one another."""
    with pytest.raises(ValueError, match="Conflicting top-level fields"):
        InstanceSpaceOptions.from_dict({"trace": {}, "TRACE": {}})


def test_case_insensitive_top_level_groups_load_from_file(tmp_path: Path) -> None:
    """The file-loading entry point uses canonical top-level group names."""
    epsilon = 0.4
    path = tmp_path / "options.json"
    path.write_text(
        f'{{"PERF":{{"AbsPerf":false,"epsilon":{epsilon}}},'
        '"Trace":{"method":"TRACE3"}}',
        encoding="utf-8",
    )

    options = from_json_file(path)

    assert options is not None
    assert options.perf.epsilon == epsilon
    assert options.trace.method == "trace3"


def test_top_level_conflicts_from_file_fail_cleanly(tmp_path: Path) -> None:
    """File loading converts case-only group conflicts into a clean failure."""
    path = tmp_path / "options.json"
    path.write_text('{"trace":{},"TRACE":{}}', encoding="utf-8")

    assert from_json_file(path) is None


def test_malformed_json_root_from_file_fails_cleanly(tmp_path: Path) -> None:
    """A non-object JSON document is rejected by the file-loading boundary."""
    path = tmp_path / "options.json"
    path.write_text("[]", encoding="utf-8")

    assert from_json_file(path) is None


@pytest.mark.parametrize(
    ("contents", "group"),
    [
        ({"pilot": []}, "PilotOptions"),
        ({"cloister": "not-an-object"}, "CloisterOptions"),
        ({"pythia": None}, "PythiaOptions"),
    ],
)
def test_malformed_json_group_has_a_contextual_error(
    contents: object,
    group: str,
) -> None:
    """Malformed nested groups identify the affected option class."""
    with pytest.raises(ValueError, match=rf"{group} JSON group must be an object"):
        InstanceSpaceOptions.from_dict(contents)


def test_non_text_json_group_key_has_a_contextual_error() -> None:
    """Nested option names must be JSON strings."""
    with pytest.raises(
        ValueError,
        match="PilotOptions JSON group keys must be strings",
    ):
        InstanceSpaceOptions.from_dict(
            {"pilot": {1: 2}},
        )
