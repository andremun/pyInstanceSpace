# SPDX-License-Identifier: LicenseRef-PolyForm-Noncommercial-1.0.0
# Copyright (c) 2024-2026 Mario Andrés Muñoz
"""Defines a collection of data classes that represent configuration options.

These classes provide a structured way to specify and manage settings for different
aspects of the model's execution and behaviour.
"""

from __future__ import annotations

import json
import numbers
import re
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Literal, Self, TypeVar, cast

import numpy as np
from loguru import logger
from numpy.typing import NDArray

from instancespace.data.default_options import (
    DEFAULT_AUTO_PREPROC,
    DEFAULT_BOUND_FLAG,
    DEFAULT_CLOISTER_C_THRES,
    DEFAULT_CLOISTER_HULL_DIMS,
    DEFAULT_CLOISTER_MAX_FEATURES,
    DEFAULT_CLOISTER_P_VAL,
    DEFAULT_GENERAL_SEED,
    DEFAULT_GENERAL_VERBOSE,
    DEFAULT_NORM_FLAG,
    DEFAULT_OUTPUTS_CSV,
    DEFAULT_OUTPUTS_PNG,
    DEFAULT_OUTPUTS_WEB,
    DEFAULT_PARALLEL_FLAG,
    DEFAULT_PARALLEL_N_CORES,
    DEFAULT_PERFORMANCE_ABS_PERF,
    DEFAULT_PERFORMANCE_BETA_THRESHOLD,
    DEFAULT_PERFORMANCE_EPSILON,
    DEFAULT_PERFORMANCE_MAX_PERF,
    DEFAULT_PILOT_ADJUST_ROTATION,
    DEFAULT_PILOT_ANALYTICS,
    DEFAULT_PILOT_COST_WEIGHT,
    DEFAULT_PILOT_METHOD,
    DEFAULT_PILOT_N_TRIES,
    DEFAULT_PRELIM_IQR_MULTIPLIER,
    DEFAULT_PRELIM_NAN_THRESHOLD,
    DEFAULT_PYTHIA_CLASSIFIER,
    DEFAULT_PYTHIA_CV_FOLDS,
    DEFAULT_PYTHIA_IS_POLY_KRNL,
    DEFAULT_PYTHIA_N_TUNING_ITER,
    DEFAULT_PYTHIA_SKIP,
    DEFAULT_PYTHIA_TUNING,
    DEFAULT_PYTHIA_USE_WEIGHTS,
    DEFAULT_SELVARS_DENSITY_FLAG,
    DEFAULT_SELVARS_FILE_IDX,
    DEFAULT_SELVARS_FILE_IDX_FLAG,
    DEFAULT_SELVARS_MIN_DISTANCE,
    DEFAULT_SELVARS_SMALL_SCALE,
    DEFAULT_SELVARS_SMALL_SCALE_FLAG,
    DEFAULT_SELVARS_TYPE,
    DEFAULT_SIFTED_CROSSOVER_PROBABILITY,
    DEFAULT_SIFTED_CROSSOVER_TYPE,
    DEFAULT_SIFTED_DIMS,
    DEFAULT_SIFTED_FLAG,
    DEFAULT_SIFTED_K,
    DEFAULT_SIFTED_K_TOURNAMENT,
    DEFAULT_SIFTED_KEEP_ELITISM,
    DEFAULT_SIFTED_MAX_ITER,
    DEFAULT_SIFTED_MUTATION_PROBABILITY,
    DEFAULT_SIFTED_MUTATION_TYPE,
    DEFAULT_SIFTED_NTREES,
    DEFAULT_SIFTED_NUM_GENERATION,
    DEFAULT_SIFTED_NUM_PARENTS_MATING,
    DEFAULT_SIFTED_PARENT_SELECTION_TYPE,
    DEFAULT_SIFTED_PVAL,
    DEFAULT_SIFTED_REPLICATES,
    DEFAULT_SIFTED_RHO,
    DEFAULT_SIFTED_SOL_PER_POP,
    DEFAULT_SIFTED_STOP_CRITERIA,
    DEFAULT_TRACE3_PURITY,
    DEFAULT_TRACE_CONTRA,
    DEFAULT_TRACE_METHOD,
    DEFAULT_TRACE_MIN_AREA_FRAC,
    DEFAULT_TRACE_MIN_INSTANCES,
    DEFAULT_TRACE_PURITY,
    DEFAULT_TRACE_USE_SIM,
)

MATRIX_DIMENSIONS = 2
MIN_CV_FOLDS = 2
MIN_HULL_DIMENSIONS = 2
ADAPTIVE_PROBABILITY_COUNT = 2
PYTHIA_TWO_PARAMETER_COUNT = 2
MIN_OPTION_ROWS = 1


@dataclass(frozen=True)
class GeneralOptions:
    """General options not specific to any one stage.

    Mirrors MATLAB's ``opts.general.*`` namespace.

    Attributes
    ----------
    verbose : bool
        Whether stages log per-trial/per-iteration detail (e.g. PYTHIA's
        per-classifier tuning progress), or only their top-level `[STAGE] message`
        lines.
    seed : int | None
        Seed threaded through every stage's random-number generation (both
        `np.random.default_rng(seed=...)` and scikit-learn's `random_state=...`),
        replacing the hardcoded `0` previously scattered across `pilot.py`,
        `sifted.py`, `prelim.py`, and `pythia.py`. `None` requests a
        non-deterministic run; the default `0` exactly matches that previously
        hardcoded value, so leaving this unset changes nothing for existing callers.
    """

    verbose: bool
    seed: int | None

    def __post_init__(self) -> None:
        """Validate general options at their public construction boundary."""
        _validate_general_options(self)

    @staticmethod
    def default(
        verbose: bool = DEFAULT_GENERAL_VERBOSE,
        seed: int | None = DEFAULT_GENERAL_SEED,
    ) -> GeneralOptions:
        """Instantiate with default values."""
        return GeneralOptions(
            verbose=verbose,
            seed=seed,
        )


@dataclass(frozen=True)
class ParallelOptions:
    """Configuration options for parallel computing."""

    flag: bool
    n_cores: int

    def __post_init__(self) -> None:
        """Validate parallel options at their public construction boundary."""
        _validate_parallel_options(self)

    @staticmethod
    def default(
        flag: bool = DEFAULT_PARALLEL_FLAG,
        n_cores: int = DEFAULT_PARALLEL_N_CORES,
    ) -> ParallelOptions:
        """Instantiate with default values."""
        return ParallelOptions(
            flag=flag,
            n_cores=n_cores,
        )


@dataclass(frozen=True)
class PerformanceOptions:
    """Options related to performance thresholds and criteria for model evaluation."""

    max_perf: bool
    abs_perf: bool
    epsilon: float
    beta_threshold: float

    def __post_init__(self) -> None:
        """Validate performance options at their public construction boundary."""
        _validate_performance_options(self)

    @staticmethod
    def default(
        max_perf: bool = DEFAULT_PERFORMANCE_MAX_PERF,
        abs_perf: bool = DEFAULT_PERFORMANCE_ABS_PERF,
        epsilon: float = DEFAULT_PERFORMANCE_EPSILON,
        beta_threshold: float = DEFAULT_PERFORMANCE_BETA_THRESHOLD,
    ) -> PerformanceOptions:
        """Instantiate with default values."""
        return PerformanceOptions(
            max_perf=max_perf,
            abs_perf=abs_perf,
            epsilon=epsilon,
            beta_threshold=beta_threshold,
        )


@dataclass(frozen=True)
class AutoOptions:
    """Options for automatic processing steps in the model pipeline."""

    preproc: bool

    def __post_init__(self) -> None:
        """Validate automatic preprocessing options when constructed."""
        _validate_auto_options(self)

    @staticmethod
    def default(
        preproc: bool = DEFAULT_AUTO_PREPROC,
    ) -> AutoOptions:
        """Instantiate with default values."""
        return AutoOptions(
            preproc=preproc,
        )


@dataclass(frozen=True)
class BoundOptions:
    """Options for applying bounds in the model calculations or evaluations."""

    flag: bool

    def __post_init__(self) -> None:
        """Validate bounding options when constructed."""
        _validate_bound_options(self)

    @staticmethod
    def default(
        flag: bool = DEFAULT_BOUND_FLAG,
    ) -> BoundOptions:
        """Instantiate with default values."""
        return BoundOptions(
            flag=flag,
        )


@dataclass(frozen=True)
class NormOptions:
    """Options to control normalization processes within the model."""

    flag: bool

    def __post_init__(self) -> None:
        """Validate normalization options when constructed."""
        _validate_norm_options(self)

    @staticmethod
    def default(
        flag: bool = DEFAULT_NORM_FLAG,
    ) -> NormOptions:
        """Instantiate with default values."""
        return NormOptions(
            flag=flag,
        )


@dataclass(frozen=True)
class SelvarsOptions:
    """Options for selecting variables, including criteria and file indices."""

    small_scale_flag: bool
    small_scale: float
    file_idx_flag: bool
    file_idx: str
    feats: list[str] | None
    algos: list[str] | None
    selvars_type: str
    min_distance: float
    density_flag: bool

    def __post_init__(self) -> None:
        """Normalize and validate selection options when constructed."""
        _validate_selvars_options(self)

    @staticmethod
    def default(
        small_scale_flag: bool = DEFAULT_SELVARS_SMALL_SCALE_FLAG,
        small_scale: float = DEFAULT_SELVARS_SMALL_SCALE,
        file_idx_flag: bool = DEFAULT_SELVARS_FILE_IDX_FLAG,
        file_idx: str = DEFAULT_SELVARS_FILE_IDX,
        feats: list[str] | None = None,
        algos: list[str] | None = None,
        selvars_type: str = DEFAULT_SELVARS_TYPE,
        min_distance: float = DEFAULT_SELVARS_MIN_DISTANCE,
        density_flag: bool = DEFAULT_SELVARS_DENSITY_FLAG,
    ) -> SelvarsOptions:
        """Instantiate with default values."""
        return SelvarsOptions(
            small_scale_flag=small_scale_flag,
            small_scale=small_scale,
            file_idx_flag=file_idx_flag,
            file_idx=file_idx,
            feats=feats,
            algos=algos,
            selvars_type=selvars_type,
            min_distance=min_distance,
            density_flag=density_flag,
        )


@dataclass(frozen=True)
class SiftedOptions:
    """Options specific to the sifting process in data analysis."""

    flag: bool
    rho: float
    k: int
    n_trees: int
    max_iter: int
    replicates: int
    num_generations: int
    num_parents_mating: int
    sol_per_pop: int
    parent_selection_type: str
    k_tournament: int
    keep_elitism: int
    crossover_type: str
    cross_over_probability: float
    mutation_type: str
    mutation_probability: float | list[float] | tuple[float, float]
    stop_criteria: str
    # Significance threshold for the correlation filter, matching MATLAB's
    # opts.pval (core/SIFTED.m) - was a hardcoded class constant.
    pval: float = DEFAULT_SIFTED_PVAL
    # Projection dimensionality for the GA fitness function's internal KNN
    # neighbour count (dims + 1), matching MATLAB's opts.dims. PILOT itself
    # is 2D-only in this port, so 3 is accepted but currently has no effect
    # on PILOT's actual output - see default_options.py's DEFAULT_SIFTED_DIMS.
    dims: int = DEFAULT_SIFTED_DIMS

    def __post_init__(self) -> None:
        """Normalize and validate SIFTED options."""
        _validate_sifted_options(self)

    @staticmethod
    def default(
        flag: bool = DEFAULT_SIFTED_FLAG,
        rho: float = DEFAULT_SIFTED_RHO,
        k: int = DEFAULT_SIFTED_K,
        n_trees: int = DEFAULT_SIFTED_NTREES,
        max_iter: int = DEFAULT_SIFTED_MAX_ITER,
        replicates: int = DEFAULT_SIFTED_REPLICATES,
        num_generations: int = DEFAULT_SIFTED_NUM_GENERATION,
        num_parents_mating: int = DEFAULT_SIFTED_NUM_PARENTS_MATING,
        sol_per_pop: int = DEFAULT_SIFTED_SOL_PER_POP,
        parent_selection_type: str = DEFAULT_SIFTED_PARENT_SELECTION_TYPE,
        k_tournament: int = DEFAULT_SIFTED_K_TOURNAMENT,
        keep_elitism: int = DEFAULT_SIFTED_KEEP_ELITISM,
        crossover_type: str = DEFAULT_SIFTED_CROSSOVER_TYPE,
        cross_over_probability: float = DEFAULT_SIFTED_CROSSOVER_PROBABILITY,
        mutation_type: str = DEFAULT_SIFTED_MUTATION_TYPE,
        mutation_probability: float | list[float] | tuple[float, float] = (
            DEFAULT_SIFTED_MUTATION_PROBABILITY
        ),
        stop_criteria: str = DEFAULT_SIFTED_STOP_CRITERIA,
        pval: float = DEFAULT_SIFTED_PVAL,
        dims: int = DEFAULT_SIFTED_DIMS,
    ) -> SiftedOptions:
        """Instantiate with default values."""
        return SiftedOptions(
            flag=flag,
            rho=rho,
            k=k,
            n_trees=n_trees,
            max_iter=max_iter,
            replicates=replicates,
            num_generations=num_generations,
            num_parents_mating=num_parents_mating,
            sol_per_pop=sol_per_pop,
            parent_selection_type=parent_selection_type,
            k_tournament=k_tournament,
            keep_elitism=keep_elitism,
            crossover_type=crossover_type,
            cross_over_probability=cross_over_probability,
            mutation_type=mutation_type,
            mutation_probability=mutation_probability,
            stop_criteria=stop_criteria,
            pval=pval,
            dims=dims,
        )


@dataclass(frozen=True)
class PilotOptions:
    """Options for pilot studies or preliminary analysis phases."""

    x0: NDArray[np.double] | None
    # Optional precomputed optimisation solution vector, shape (2*m + 2*n,)
    # (MATLAB's opts.precalcAlpha). Distinct from cost_weight below - this
    # used to be conflated under one `alpha` field (#301 issue 1).
    precalc_alpha: NDArray[np.double] | None
    analytic: bool
    n_tries: int
    adjust_rotation: bool = DEFAULT_PILOT_ADJUST_ROTATION
    # Scalar performance-reconstruction cost weight (MATLAB's opts.alpha,
    # also called costWeight). Weights the performance block relative to the
    # feature block in both the analytic and numerical solvers. 1.0 (default)
    # weights both blocks equally.
    cost_weight: float = DEFAULT_PILOT_COST_WEIGHT
    # 'standard' (analytic/numeric, `analytic` selects which) or 'pls'
    # (Partial Least Squares - F2, #262). `analytic` is only consulted when
    # method='standard'; 'pls' ignores it entirely, matching MATLAB's own
    # opts.method dispatch (core/PILOT.m).
    method: str = DEFAULT_PILOT_METHOD

    def __post_init__(self) -> None:
        """Normalize optional matrices and reject an ambiguous solver setup."""
        x0 = _coerce_optional_matrix("pilot.x0", self.x0)
        precalc_alpha = _coerce_optional_matrix(
            "pilot.precalcAlpha",
            self.precalc_alpha,
        )
        if x0 is not None and precalc_alpha is not None:
            msg = "opts.pilot.x0 and opts.pilot.precalcAlpha cannot both be set."
            raise ValueError(msg)
        object.__setattr__(self, "x0", x0)
        object.__setattr__(self, "precalc_alpha", precalc_alpha)
        _validate_pilot_options(self)

    @staticmethod
    def default(
        analytic: bool = DEFAULT_PILOT_ANALYTICS,
        n_tries: int = DEFAULT_PILOT_N_TRIES,
        x0: NDArray[np.double] | None = None,
        precalc_alpha: NDArray[np.double] | None = None,
        adjust_rotation: bool = DEFAULT_PILOT_ADJUST_ROTATION,
        cost_weight: float = DEFAULT_PILOT_COST_WEIGHT,
        method: str = DEFAULT_PILOT_METHOD,
    ) -> PilotOptions:
        """Instantiate with default values."""
        return PilotOptions(
            analytic=analytic,
            n_tries=n_tries,
            x0=x0,
            precalc_alpha=precalc_alpha,
            adjust_rotation=adjust_rotation,
            cost_weight=cost_weight,
            method=method,
        )


@dataclass(frozen=True)
class CloisterOptions:
    """Options for cloistering in the model."""

    p_val: float
    c_thres: float
    # Feature-count guard before corner enumeration (2**nfeats corners)
    # becomes intractable, matching MATLAB's opts.maxFeatures. Above this,
    # CLOISTER skips enumeration and uses a plain convex hull of the
    # projected instances as the boundary instead.
    max_features: int = DEFAULT_CLOISTER_MAX_FEATURES
    # "all" (default) uses every projected dimension for the convex hull;
    # 2 mimics MATLAB's always-2D-on-the-first-two-columns hull (core/
    # CLOISTER.m) while still returning full-dimensional vertices. #299
    # audit finding, issue 5 - see default_options.py for why "all" and 2
    # are currently equivalent in practice (PILOT is 2D-only in this port).
    hull_dims: int | Literal["all"] = DEFAULT_CLOISTER_HULL_DIMS

    def __post_init__(self) -> None:
        """Normalize and validate CLOISTER options."""
        _validate_cloister_options(self)

    @staticmethod
    def default(
        p_val: float = DEFAULT_CLOISTER_P_VAL,
        c_thres: float = DEFAULT_CLOISTER_C_THRES,
        max_features: int = DEFAULT_CLOISTER_MAX_FEATURES,
        hull_dims: int | Literal["all"] = DEFAULT_CLOISTER_HULL_DIMS,
    ) -> CloisterOptions:
        """Instantiate with default values."""
        return CloisterOptions(
            p_val=p_val,
            c_thres=c_thres,
            max_features=max_features,
            hull_dims=hull_dims,
        )


@dataclass(frozen=True)
class PythiaOptions:
    """Configuration for the Pythia component of the model."""

    cv_folds: int
    is_poly_krnl: bool
    use_weights: bool
    params: NDArray[np.double] | None
    classifier: str = DEFAULT_PYTHIA_CLASSIFIER
    tuning: str = DEFAULT_PYTHIA_TUNING
    n_tuning_iter: int = DEFAULT_PYTHIA_N_TUNING_ITER
    # Bypass classifier training entirely, matching core/PYTHIA.m's
    # opts.skip. Legacy TRACE requires true-label footprints when this is set.
    # TRACE3 can fall back to true labels without changing trace.use_sim.
    skip: bool = DEFAULT_PYTHIA_SKIP

    def __post_init__(self) -> None:
        """Normalize and validate active PYTHIA options."""
        _check_logical("pythia.skip", self.skip)
        if not self.skip:
            params = _coerce_optional_matrix("pythia.params", self.params)
            object.__setattr__(self, "params", params)
        _validate_pythia_options(self)

    @staticmethod
    def default(
        cv_folds: int = DEFAULT_PYTHIA_CV_FOLDS,
        is_poly_krnl: bool = DEFAULT_PYTHIA_IS_POLY_KRNL,
        use_weights: bool = DEFAULT_PYTHIA_USE_WEIGHTS,
        classifier: str = DEFAULT_PYTHIA_CLASSIFIER,
        tuning: str = DEFAULT_PYTHIA_TUNING,
        n_tuning_iter: int = DEFAULT_PYTHIA_N_TUNING_ITER,
        skip: bool = DEFAULT_PYTHIA_SKIP,
        params: NDArray[np.double] | None = None,
    ) -> PythiaOptions:
        """Instantiate with default values."""
        return PythiaOptions(
            cv_folds=cv_folds,
            is_poly_krnl=is_poly_krnl,
            use_weights=use_weights,
            params=params,
            classifier=classifier,
            tuning=tuning,
            n_tuning_iter=n_tuning_iter,
            skip=skip,
        )


@dataclass(frozen=True, init=False)
class TraceOptions:
    """Options for trace analysis in the model."""

    use_sim: bool
    purity: float
    method: str = DEFAULT_TRACE_METHOD
    contra: bool = DEFAULT_TRACE_CONTRA
    min_instances: int = DEFAULT_TRACE_MIN_INSTANCES
    min_area_frac: float = DEFAULT_TRACE_MIN_AREA_FRAC

    def __init__(
        self,
        use_sim: bool = DEFAULT_TRACE_USE_SIM,
        purity: float | None = None,
        method: str = DEFAULT_TRACE_METHOD,
        contra: bool = DEFAULT_TRACE_CONTRA,
        min_instances: int = DEFAULT_TRACE_MIN_INSTANCES,
        min_area_frac: float = DEFAULT_TRACE_MIN_AREA_FRAC,
    ) -> None:
        """Resolve the method-aware purity default."""
        resolved_method = _normalize_member(
            "trace.method",
            method,
            ("trace3", "legacy"),
        )
        resolved_purity = (
            DEFAULT_TRACE3_PURITY
            if purity is None and resolved_method == "trace3"
            else DEFAULT_TRACE_PURITY if purity is None else purity
        )
        object.__setattr__(self, "use_sim", use_sim)
        object.__setattr__(self, "purity", resolved_purity)
        object.__setattr__(self, "method", resolved_method)
        object.__setattr__(self, "contra", contra)
        object.__setattr__(self, "min_instances", min_instances)
        object.__setattr__(self, "min_area_frac", min_area_frac)
        _validate_trace_options(self)

    @staticmethod
    def default(
        use_sim: bool = DEFAULT_TRACE_USE_SIM,
        purity: float | None = None,
        method: str = DEFAULT_TRACE_METHOD,
        contra: bool = DEFAULT_TRACE_CONTRA,
        min_instances: int = DEFAULT_TRACE_MIN_INSTANCES,
        min_area_frac: float = DEFAULT_TRACE_MIN_AREA_FRAC,
    ) -> TraceOptions:
        """Instantiate with default values."""
        return TraceOptions(
            use_sim=use_sim,
            purity=purity,
            method=method,
            contra=contra,
            min_instances=min_instances,
            min_area_frac=min_area_frac,
        )


@dataclass(frozen=True)
class OutputOptions:
    """Options for controlling the output format."""

    csv: bool
    web: bool
    png: bool

    def __post_init__(self) -> None:
        """Validate output options at their public construction boundary."""
        _validate_output_options(self)

    @staticmethod
    def default(
        csv: bool = DEFAULT_OUTPUTS_CSV,
        web: bool = DEFAULT_OUTPUTS_WEB,
        png: bool = DEFAULT_OUTPUTS_PNG,
    ) -> OutputOptions:
        """Instantiate with default values."""
        return OutputOptions(
            csv=csv,
            web=web,
            png=png,
        )


def _check_logical(name: str, value: object) -> None:
    if not isinstance(value, bool):
        msg = f"opts.{name} must be True or False. Got {value!r}."
        raise ValueError(msg)


def _finite_real(name: str, value: object) -> float:
    """Return a finite, real, non-Boolean option value."""
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        msg = f"opts.{name} must be a finite real number. Got {value!r}."
        raise ValueError(msg)
    numeric_value = float(value)
    if not np.isfinite(numeric_value):
        msg = f"opts.{name} must be a finite real number. Got {value!r}."
        raise ValueError(msg)
    return numeric_value


def _check_unit_range(name: str, value: object) -> None:
    numeric_value = _finite_real(name, value)
    if not 0 <= numeric_value <= 1:
        msg = f"opts.{name} must be in the unit range [0, 1]. Got {value!r}."
        raise ValueError(msg)


def _check_positive(name: str, value: object, *, zero_allowed: bool = False) -> None:
    numeric_value = _finite_real(name, value)
    if zero_allowed:
        if numeric_value < 0:
            msg = f"opts.{name} must be non-negative. Got {value!r}."
            raise ValueError(msg)
    elif numeric_value <= 0:
        msg = f"opts.{name} must be strictly positive. Got {value!r}."
        raise ValueError(msg)


def _check_pos_int(name: str, value: object, *, zero_allowed: bool = False) -> None:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        msg = f"opts.{name} must be an integer. Got {value!r}."
        raise ValueError(msg)
    _check_positive(name, value, zero_allowed=zero_allowed)


def _normalize_member(name: str, value: object, valid: tuple[str, ...]) -> str:
    """Return the canonical spelling of a case-insensitive option member."""
    if isinstance(value, str):
        normalized = value.casefold()
        for member in valid:
            if member.casefold() == normalized:
                return member
    msg = f"opts.{name} must be one of {valid}. Got {value!r}."
    raise ValueError(msg)


def _canonical_json_option_key(
    json_field: str,
    field_mapping: dict[str, str],
) -> str:
    """Return the canonical dataclass field for a JSON option key."""
    normalized_field = json_field.casefold()
    return field_mapping.get(normalized_field, normalized_field)


def _check_sifted_dims(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        msg = f"opts.{name} must be 2 or 3. Got {value!r}."
        raise ValueError(msg)
    if value not in (2, 3):
        msg = f"opts.{name} must be 2 or 3. Got {value!r}."
        raise ValueError(msg)


def _check_cloister_hull_dims(name: str, value: object) -> None:
    if value == "all":
        return
    if (
        isinstance(value, bool)
        or not isinstance(value, numbers.Integral)
        or value < MIN_HULL_DIMENSIONS
    ):
        msg = f"opts.{name} must be 'all' or an integer of at least 2. Got {value!r}."
        raise ValueError(msg)


def _check_text(name: str, value: object) -> None:
    if not isinstance(value, str):
        msg = f"opts.{name} must be text. Got {value!r}."
        raise ValueError(msg)


def _check_text_list(name: str, value: object) -> None:
    if value is not None and (
        not isinstance(value, list) or not all(isinstance(item, str) for item in value)
    ):
        msg = f"opts.{name} must be a list of strings or None. Got {value!r}."
        raise ValueError(msg)


def _coerce_optional_matrix(
    name: str,
    value: object,
) -> NDArray[np.double] | None:
    """Normalize a finite real option matrix."""
    if value is None:
        return None
    array = np.asarray(value)
    is_real_numeric = np.issubdtype(array.dtype, np.number) and np.isrealobj(array)
    if array.dtype == np.bool_ or not is_real_numeric:
        msg = f"opts.{name} must be a numeric, non-Boolean matrix."
        raise ValueError(msg)
    if array.ndim != MATRIX_DIMENSIONS:
        msg = f"opts.{name} must be a two-dimensional matrix."
        raise ValueError(msg)
    matrix = np.asarray(array, dtype=np.double)
    if not np.isfinite(matrix).all():
        msg = f"opts.{name} must contain only finite values."
        raise ValueError(msg)
    return matrix


def _check_mutation_probability(
    mutation_type: str,
    probability: object,
) -> None:
    """Validate PyGAD's scalar or adaptive mutation probability."""
    if mutation_type == "adaptive":
        if (
            not isinstance(probability, list | tuple)
            or len(probability) != ADAPTIVE_PROBABILITY_COUNT
        ):
            msg = (
                "opts.sifted.mutation_probability must contain two values for "
                "adaptive mutation."
            )
            raise ValueError(msg)
        for value in probability:
            _check_unit_range("sifted.mutation_probability", value)
        return
    _check_unit_range("sifted.mutation_probability", probability)


def _check_stop_criteria(value: object) -> None:
    """Validate the supported PyGAD stopping criteria."""
    if not isinstance(value, str):
        msg = "opts.sifted.stop_criteria must be a stopping criterion string."
        raise ValueError(msg)
    saturate_match = re.fullmatch(r"saturate_([1-9][0-9]*)", value)
    if saturate_match is not None:
        return
    reach_match = re.fullmatch(r"reach_(.+)", value)
    if reach_match is not None:
        try:
            target = float(reach_match.group(1))
        except ValueError:
            target = float("nan")
        if np.isfinite(target):
            return
    msg = (
        "opts.sifted.stop_criteria must use 'saturate_N' or 'reach_VALUE'. "
        f"Got {value!r}."
    )
    raise ValueError(msg)


def _validate_general_options(options: GeneralOptions) -> None:
    """Validate general execution configuration."""
    _check_logical("general.verbose", options.verbose)
    if options.seed is not None:
        _check_pos_int("general.seed", options.seed, zero_allowed=True)


def _validate_parallel_options(options: ParallelOptions) -> None:
    """Validate parallel execution configuration."""
    _check_logical("general.parallel", options.flag)
    _check_pos_int("general.ncores", options.n_cores)


def _validate_performance_options(options: PerformanceOptions) -> None:
    """Validate performance-threshold configuration."""
    _check_logical("perf.MaxPerf", options.max_perf)
    _check_logical("perf.AbsPerf", options.abs_perf)
    if options.abs_perf:
        _finite_real("perf.epsilon", options.epsilon)
    else:
        _check_unit_range("perf.epsilon", options.epsilon)
    _check_unit_range("perf.betaThreshold", options.beta_threshold)


def _validate_prelim_config_options(options: PrelimConfigOptions) -> None:
    """Validate build-level preliminary preprocessing configuration."""
    _check_positive("prelim.iqrMultiplier", options.iqr_multiplier)
    _check_unit_range("prelim.nanThreshold", options.nan_threshold)


def _validate_auto_options(options: AutoOptions) -> None:
    """Validate automatic preprocessing configuration."""
    _check_logical("auto.preproc", options.preproc)


def _validate_bound_options(options: BoundOptions) -> None:
    """Validate bounding configuration."""
    _check_logical("bound.flag", options.flag)


def _validate_norm_options(options: NormOptions) -> None:
    """Validate normalization configuration."""
    _check_logical("norm.flag", options.flag)


def _validate_selvars_options(options: SelvarsOptions) -> None:
    """Normalize and validate selection configuration."""
    _check_logical("selvars.smallscaleflag", options.small_scale_flag)
    _check_unit_range("selvars.smallscale", options.small_scale)
    _check_logical("selvars.fileidxflag", options.file_idx_flag)
    _check_text("selvars.fileidx", options.file_idx)
    _check_logical("selvars.densityflag", options.density_flag)
    _check_positive("selvars.mindistance", options.min_distance)
    selvars_type = _normalize_member(
        "selvars.type",
        options.selvars_type,
        ("Ftr", "Ftr&AP", "Ftr&Good", "Ftr&AP&Good"),
    )
    object.__setattr__(options, "selvars_type", selvars_type)
    _check_text_list("selvars.feats", options.feats)
    _check_text_list("selvars.algos", options.algos)


def _validate_output_options(options: OutputOptions) -> None:
    """Validate output configuration."""
    _check_logical("outputs.csv", options.csv)
    _check_logical("outputs.png", options.png)
    _check_logical("outputs.web", options.web)


def _validate_sifted_options(options: SiftedOptions) -> None:
    """Validate SIFTED and PyGAD configuration."""
    parent_selection_type = _normalize_member(
        "sifted.parent_selection_type",
        options.parent_selection_type,
        (
            "sss",
            "rws",
            "sus",
            "random",
            "tournament",
            "tournament_nsga2",
            "nsga2",
            "rank",
        ),
    )
    crossover_type = _normalize_member(
        "sifted.crossover_type",
        options.crossover_type,
        ("single_point", "two_points", "uniform", "scattered"),
    )
    mutation_type = _normalize_member(
        "sifted.mutation_type",
        options.mutation_type,
        ("random", "swap", "scramble", "inversion", "adaptive"),
    )
    if not isinstance(options.stop_criteria, str):
        msg = "opts.sifted.stop_criteria must be a stopping criterion string."
        raise ValueError(msg)
    object.__setattr__(options, "parent_selection_type", parent_selection_type)
    object.__setattr__(options, "crossover_type", crossover_type)
    object.__setattr__(options, "mutation_type", mutation_type)
    object.__setattr__(options, "stop_criteria", options.stop_criteria.casefold())

    _check_logical("sifted.flag", options.flag)
    _check_unit_range("sifted.rho", options.rho)
    _check_pos_int("sifted.K", options.k)
    _check_pos_int("sifted.NTREES", options.n_trees)
    _check_pos_int("sifted.MaxIter", options.max_iter)
    _check_pos_int("sifted.Replicates", options.replicates)
    _check_pos_int("sifted.num_generations", options.num_generations)
    _check_pos_int("sifted.num_parents_mating", options.num_parents_mating)
    _check_pos_int("sifted.sol_per_pop", options.sol_per_pop)
    if options.num_parents_mating > options.sol_per_pop:
        msg = "opts.sifted.num_parents_mating cannot exceed sol_per_pop."
        raise ValueError(msg)
    _check_pos_int("sifted.k_tournament", options.k_tournament)
    if options.k_tournament > options.sol_per_pop:
        msg = "opts.sifted.k_tournament cannot exceed sol_per_pop."
        raise ValueError(msg)
    _check_pos_int("sifted.keep_elitism", options.keep_elitism, zero_allowed=True)
    if options.keep_elitism > options.sol_per_pop:
        msg = "opts.sifted.keep_elitism cannot exceed sol_per_pop."
        raise ValueError(msg)
    _check_unit_range(
        "sifted.cross_over_probability",
        options.cross_over_probability,
    )
    _check_mutation_probability(options.mutation_type, options.mutation_probability)
    _check_stop_criteria(options.stop_criteria)
    _check_unit_range("sifted.pval", options.pval)
    _check_sifted_dims("sifted.dims", options.dims)


def _validate_pilot_options(options: PilotOptions) -> None:
    """Validate PILOT configuration."""
    method = _normalize_member(
        "pilot.method",
        options.method,
        ("standard", "pls"),
    )
    object.__setattr__(options, "method", method)
    _check_logical("pilot.analytic", options.analytic)
    _check_pos_int("pilot.ntries", options.n_tries)
    _check_logical("pilot.adjustRotation", options.adjust_rotation)
    _check_positive("pilot.costWeight", options.cost_weight)
    if options.x0 is not None and options.x0.shape[1] < MIN_OPTION_ROWS:
        msg = "opts.pilot.x0 must contain at least one starting point."
        raise ValueError(msg)


def _validate_cloister_options(options: CloisterOptions) -> None:
    """Validate CLOISTER configuration."""
    if isinstance(options.hull_dims, str) and options.hull_dims.casefold() == "all":
        object.__setattr__(options, "hull_dims", "all")
    _check_unit_range("cloister.pval", options.p_val)
    _check_unit_range("cloister.corrThreshold", options.c_thres)
    _check_pos_int("cloister.maxFeatures", options.max_features)
    _check_cloister_hull_dims("cloister.hullDims", options.hull_dims)


def _validate_pythia_options(options: PythiaOptions) -> None:
    """Validate PYTHIA configuration and precomputed parameter columns."""
    _check_logical("pythia.skip", options.skip)
    classifier = _normalize_member(
        "pythia.classifier",
        options.classifier,
        ("knn", "svm", "tree", "nb", "linear", "ensemble"),
    )
    object.__setattr__(options, "classifier", classifier)
    _check_logical("pythia.isPolyKrnl", options.is_poly_krnl)
    _check_logical("pythia.useWeights", options.use_weights)
    _check_pos_int("pythia.kFold", options.cv_folds)
    if options.cv_folds < MIN_CV_FOLDS:
        msg = f"opts.pythia.kFold must be at least {MIN_CV_FOLDS}."
        raise ValueError(msg)
    if options.skip:
        return

    tuning = _normalize_member(
        "pythia.tuning",
        options.tuning,
        ("sobol", "bayes", "none"),
    )
    object.__setattr__(options, "tuning", tuning)
    if options.params is not None:
        expected_columns = (
            PYTHIA_TWO_PARAMETER_COUNT
            if options.classifier in ("knn", "svm", "ensemble")
            else 1
        )
        if options.params.shape[0] < MIN_OPTION_ROWS:
            msg = "opts.pythia.params must contain at least one row."
            raise ValueError(msg)
        if options.params.shape[1] != expected_columns:
            msg = (
                f"opts.pythia.params must have {expected_columns} columns for "
                f"classifier {options.classifier!r}."
            )
            raise ValueError(msg)
        positive_parameters = {
            "svm": ("BoxConstraint", "KernelScale"),
            "nb": ("Bandwidth",),
            "linear": ("Lambda",),
        }.get(options.classifier, ())
        for column_index, parameter_name in enumerate(positive_parameters):
            invalid_rows = np.flatnonzero(options.params[:, column_index] <= 0)
            if invalid_rows.size > 0:
                row_index = int(invalid_rows[0])
                value = float(options.params[row_index, column_index])
                msg = (
                    f"opts.pythia.params[{row_index}, {column_index}] "
                    f"({parameter_name}) must be strictly positive. Got {value!r}."
                )
                raise ValueError(msg)
        return

    if options.tuning == "none":
        msg = "opts.pythia.tuning='none' requires opts.pythia.params."
        raise ValueError(msg)
    _check_pos_int("pythia.nTuningIter", options.n_tuning_iter)


def _validate_trace_options(options: TraceOptions) -> None:
    """Validate TRACE configuration."""
    method = _normalize_member(
        "trace.method",
        options.method,
        ("trace3", "legacy"),
    )
    object.__setattr__(options, "method", method)
    _check_unit_range("trace.PI", options.purity)
    _check_logical("trace.useSim", options.use_sim)
    _check_logical("trace.contra", options.contra)
    _check_pos_int("trace.minInstances", options.min_instances)
    _check_unit_range("trace.minAreaFrac", options.min_area_frac)


@dataclass(frozen=True)
class PrelimConfigOptions:
    """Build-level PRELIM configuration matching MATLAB's ``opts.prelim``."""

    iqr_multiplier: float = DEFAULT_PRELIM_IQR_MULTIPLIER
    nan_threshold: float = DEFAULT_PRELIM_NAN_THRESHOLD

    def __post_init__(self) -> None:
        """Validate PRELIM configuration at its public construction boundary."""
        _validate_prelim_config_options(self)

    @staticmethod
    def default(
        iqr_multiplier: float = DEFAULT_PRELIM_IQR_MULTIPLIER,
        nan_threshold: float = DEFAULT_PRELIM_NAN_THRESHOLD,
    ) -> PrelimConfigOptions:
        """Instantiate with MATLAB-compatible defaults."""
        return PrelimConfigOptions(
            iqr_multiplier=iqr_multiplier,
            nan_threshold=nan_threshold,
        )


@dataclass(frozen=True)
class InstanceSpaceOptions:
    """Aggregates all options into a single configuration object for the model."""

    parallel: ParallelOptions
    perf: PerformanceOptions
    auto: AutoOptions
    bound: BoundOptions
    norm: NormOptions
    selvars: SelvarsOptions
    sifted: SiftedOptions
    pilot: PilotOptions
    cloister: CloisterOptions
    pythia: PythiaOptions
    trace: TraceOptions
    outputs: OutputOptions
    # Defaulted fields stay at the end so existing direct positional construction
    # that predates them keeps working unchanged.
    general: GeneralOptions = field(default_factory=GeneralOptions.default)
    prelim: PrelimConfigOptions = field(default_factory=PrelimConfigOptions.default)

    def __post_init__(self: Self) -> None:
        """Validate every active option before stage execution."""
        _validate_general_options(self.general)
        _validate_parallel_options(self.parallel)
        _validate_performance_options(self.perf)
        _validate_prelim_config_options(self.prelim)
        _validate_auto_options(self.auto)
        _validate_bound_options(self.bound)
        _validate_norm_options(self.norm)
        _validate_selvars_options(self.selvars)

        _validate_sifted_options(self.sifted)
        _validate_pilot_options(self.pilot)
        _validate_cloister_options(self.cloister)
        _validate_pythia_options(self.pythia)
        _validate_trace_options(self.trace)

        if self.trace.method == "legacy" and self.pythia.skip and self.trace.use_sim:
            msg = "pythia.skip=True is incompatible with legacy trace.use_sim=True."
            raise ValueError(msg)

        _validate_output_options(self.outputs)

    @staticmethod
    def from_dict(file_contents: object) -> InstanceSpaceOptions:
        """Load configuration options from a JSON file into an object.

        This function reads a JSON file from `filepath`, checks for expected
        top-level fields as defined in InstanceSpaceOptions, initializes each part of
        the InstanceSpaceOptions with data from the file, and sets missing optional
        fields using their default values.

        Args:
        ----
        file_contents
            Content of the dict with configuration options.

        Returns:
        -------
        InstanceSpaceOptions
            InstanceSpaceOptions object populated with data from the file.

        Raises:
        ------
        ValueError
            If the JSON file contains undefined sub options.

        """
        if not isinstance(file_contents, dict):
            msg = "Options JSON root must be an object."
            raise ValueError(msg)
        if not all(isinstance(key, str) for key in file_contents):
            msg = "Options JSON root keys must be strings."
            raise ValueError(msg)
        raw_contents = cast(dict[str, Any], file_contents)

        # Normalize top-level group names just like nested option names. Preserve
        # conflict information before building a canonical dictionary so case-only
        # duplicates cannot silently overwrite one another.
        options_fields = {f.name for f in fields(InstanceSpaceOptions)}
        contents: dict[str, Any] = {}
        original_names: dict[str, str] = {}
        for group_name, group_values in raw_contents.items():
            canonical_name = group_name.casefold()
            if canonical_name in contents:
                msg = (
                    "Conflicting top-level fields in JSON: "
                    f"{original_names[canonical_name]!r} and {group_name!r} "
                    f"both map to {canonical_name!r}."
                )
                raise ValueError(msg)
            contents[canonical_name] = group_values
            original_names[canonical_name] = group_name

        # Validate if the canonical top-level fields match the aggregate dataclass.
        extra_canonical_fields = set(contents) - options_fields
        extra_fields = {
            original_names[canonical_name] for canonical_name in extra_canonical_fields
        }

        if extra_fields:
            raise ValueError(
                f"Extra fields in JSON are not defined in InstanceSpaceOptions: "
                f" {extra_fields}",
            )

        # Initialize each part of InstanceSpaceOptions, using default values for missing
        # fields
        return InstanceSpaceOptions(
            parallel=InstanceSpaceOptions._load_dataclass(
                ParallelOptions,
                contents.get("parallel", {}),
                {
                    "ncores": "n_cores",
                },
            ),
            perf=InstanceSpaceOptions._load_dataclass(
                PerformanceOptions,
                contents.get("perf", {}),
                {
                    "maxperf": "max_perf",
                    "absperf": "abs_perf",
                    "betathreshold": "beta_threshold",
                },
            ),
            prelim=InstanceSpaceOptions._load_dataclass(
                PrelimConfigOptions,
                contents.get("prelim", {}),
                {
                    "iqrmultiplier": "iqr_multiplier",
                    "nanthreshold": "nan_threshold",
                },
            ),
            auto=InstanceSpaceOptions._load_dataclass(
                AutoOptions,
                contents.get("auto", {}),
            ),
            bound=InstanceSpaceOptions._load_dataclass(
                BoundOptions,
                contents.get("bound", {}),
            ),
            norm=InstanceSpaceOptions._load_dataclass(
                NormOptions,
                contents.get("norm", {}),
            ),
            selvars=InstanceSpaceOptions._load_dataclass(
                SelvarsOptions,
                contents.get("selvars", {}),
                {
                    "smallscaleflag": "small_scale_flag",
                    "smallscale": "small_scale",
                    "fileidxflag": "file_idx_flag",
                    "fileidx": "file_idx",
                    "densityflag": "density_flag",
                    "mindistance": "min_distance",
                    "type": "selvars_type",
                },
            ),
            sifted=InstanceSpaceOptions._load_dataclass(
                SiftedOptions,
                contents.get("sifted", {}),
                {
                    "ntrees": "n_trees",
                    "maxiter": "max_iter",
                    "replicates": "replicates",
                    "numgenerations": "num_generations",
                    "numparentsmating": "num_parents_mating",
                    "solperpop": "sol_per_pop",
                    "parentselectiontype": "parent_selection_type",
                    "ktournament": "k_tournament",
                    "keepelitism": "keep_elitism",
                    "crossovertype": "crossover_type",
                    "crossoverprobability": "cross_over_probability",
                    "mutationtype": "mutation_type",
                    "mutationprobability": "mutation_probability",
                    "stopcriteria": "stop_criteria",
                },
            ),
            pilot=InstanceSpaceOptions._load_dataclass(
                PilotOptions,
                contents.get("pilot", {}),
                {
                    "ntries": "n_tries",
                    "adjustrotation": "adjust_rotation",
                    "costweight": "cost_weight",
                    "alpha": "cost_weight",
                    "precalcalpha": "precalc_alpha",
                },
            ),
            cloister=InstanceSpaceOptions._load_dataclass(
                CloisterOptions,
                contents.get("cloister", {}),
                {
                    "pval": "p_val",
                    "cthres": "c_thres",
                    "corrthreshold": "c_thres",
                    "maxfeatures": "max_features",
                    "hulldims": "hull_dims",
                },
            ),
            pythia=InstanceSpaceOptions._load_dataclass(
                PythiaOptions,
                contents.get("pythia", {}),
                field_mapping={
                    "cvfolds": "cv_folds",
                    "kfold": "cv_folds",
                    "ispolykrnl": "is_poly_krnl",
                    "useweights": "use_weights",
                    "uselibsvm": "_",  # deprecated MATLAB flag - genuinely ignored
                    "ntuningiter": "n_tuning_iter",
                },
            ),
            trace=InstanceSpaceOptions._load_dataclass(
                TraceOptions,
                contents.get("trace", {}),
                field_mapping={
                    "pi": "purity",
                    "usesim": "use_sim",
                    "mininstances": "min_instances",
                    "minareafrac": "min_area_frac",
                },  # mapping the 'pi' in JSON to the 'purity' in TraceOptions
            ),
            outputs=InstanceSpaceOptions._load_dataclass(
                OutputOptions,
                contents.get("outputs", {}),
            ),
            general=InstanceSpaceOptions._load_dataclass(
                GeneralOptions,
                contents.get("general", {}),
            ),
        )

    @staticmethod
    def default(
        parallel: ParallelOptions | None = None,
        perf: PerformanceOptions | None = None,
        auto: AutoOptions | None = None,
        bound: BoundOptions | None = None,
        norm: NormOptions | None = None,
        selvars: SelvarsOptions | None = None,
        sifted: SiftedOptions | None = None,
        pilot: PilotOptions | None = None,
        cloister: CloisterOptions | None = None,
        pythia: PythiaOptions | None = None,
        trace: TraceOptions | None = None,
        outputs: OutputOptions | None = None,
        general: GeneralOptions | None = None,
        prelim: PrelimConfigOptions | None = None,
    ) -> InstanceSpaceOptions:
        """Instantiate with default values."""
        return InstanceSpaceOptions(
            parallel=parallel or ParallelOptions.default(),
            perf=perf or PerformanceOptions.default(),
            auto=auto or AutoOptions.default(),
            bound=bound or BoundOptions.default(),
            norm=norm or NormOptions.default(),
            selvars=selvars or SelvarsOptions.default(),
            sifted=sifted or SiftedOptions.default(),
            pilot=pilot or PilotOptions.default(),
            cloister=cloister or CloisterOptions.default(),
            pythia=pythia or PythiaOptions.default(),
            trace=trace or TraceOptions.default(),
            outputs=outputs or OutputOptions.default(),
            general=general or GeneralOptions.default(),
            prelim=prelim or PrelimConfigOptions.default(),
        )

    T = TypeVar(
        "T",
        ParallelOptions,
        PerformanceOptions,
        PrelimConfigOptions,
        AutoOptions,
        BoundOptions,
        NormOptions,
        SelvarsOptions,
        SiftedOptions,
        PilotOptions,
        CloisterOptions,
        PythiaOptions,
        TraceOptions,
        OutputOptions,
        GeneralOptions,
    )

    @staticmethod
    def _validate_fields(
        data_class: type[T],
        data: dict[str, Any],
        field_mapping: dict[str, str] | None = None,
    ) -> None:
        """Validate all keys in the provided dictionary are valid fields in dataclass.

        Args:
        ----
        data_class : type[T]
            The dataclass type to validate against.
        data : dict
            The dictionary whose keys are to be validated.
        field_mapping : Optional[dict[str, str]], optional
            An optional dictionary that maps field names from the input JSON
            to the corresponding field names in the dataclass.
            For example, if the dataclass has a field `purity`, but the input
            dictionary uses the key `pi`, this mapping
            would be `{"pi": "purity"}`.

        Raises:
        ------
        ValueError
            If an undefined field is found in the dictionary or

        """
        if field_mapping is None:
            field_mapping = {}

        # Get all valid field names from the dataclass
        known_fields = {f.name for f in fields(data_class)}

        # Collect JSON fields and apply mapping (map pi to purity, etc.)
        mapped_json_fields = {}

        value_errors = []

        for json_field, value in data.items():
            # Use field mapping if available, otherwise keep the original field name
            mapped_field = _canonical_json_option_key(json_field, field_mapping)

            # Check for conflicts, i.e., if the JSON contains both 'pi' and 'purity'
            if mapped_field in mapped_json_fields:
                raise ValueError(
                    f"Conflicting fields in JSON: " f"'{json_field}' was defined twice",
                )

            # Check if the mapped field is valid (exists in the dataclass)
            if mapped_field not in known_fields and mapped_field != "_":
                value_errors.append(mapped_field)

            mapped_json_fields[mapped_field] = value

        if len(value_errors) > 0:
            raise ValueError(
                "The following fields from JSON are not defined in the data class "
                + data_class.__name__
                + "\n"
                + "\n".join(map(lambda x: f"   {x}", value_errors)),
            )

    @staticmethod
    def _load_dataclass(
        data_class: type[T],
        data: object,
        field_mapping: dict[str, str] | None = None,
    ) -> T:
        """Load data into a dataclass from a dictionary.

        Ensures all dictionary keys match dataclass fields and fills in fields
        with available data. If a field is missing in the dictionary, the default
        value from the dataclass is used.

        Args:
        ----
        data_class : type[T]
            The dataclass type to populate.
        data : dict
            Dictionary containing data to load into the dataclass.
        field_mapping : Optional[dict[str, str]], optional
            An optional dictionary that maps field names from the input JSON
            to the corresponding field names in the dataclass.
            For example, if the dataclass has a field `purity`, but the input
            dictionary uses the key `pi`, this mapping
            would be `{"pi": "purity"}`.

        Returns:
        -------
        T
            An instance of the dataclass populated with data.

        Raises:
        ------
        ValueError
            If the dictionary contains keys that are not valid fields in the dataclass.

        """
        if field_mapping is None:
            field_mapping = {}
        if not isinstance(data, dict):
            msg = f"{data_class.__name__} JSON group must be an object."
            raise ValueError(msg)
        if not all(isinstance(key, str) for key in data):
            msg = f"{data_class.__name__} JSON group keys must be strings."
            raise ValueError(msg)
        typed_data = cast(dict[str, Any], data)

        # Get the default values for the dataclass fields
        mapped_data: dict[str, Any] = {
            f.name: getattr(data_class.default(), f.name) for f in fields(data_class)
        }

        InstanceSpaceOptions._validate_fields(
            data_class,
            typed_data,
            field_mapping,
        )
        mapped_fields: set[str] = set()
        for json_field, value in typed_data.items():
            mapped_field = _canonical_json_option_key(json_field, field_mapping)
            if mapped_field == "_":
                continue
            mapped_data[mapped_field] = value
            mapped_fields.add(mapped_field)

        if data_class is TraceOptions and "purity" not in mapped_fields:
            mapped_data["purity"] = None

        return data_class(**mapped_data)


# InstanceSpaceOptions not part of the main InstanceSpaceOptions class


def _validate_prelim_options(options: PrelimOptions) -> None:
    """Validate the stage-composed PRELIM options."""
    _check_logical("perf.MaxPerf", options.max_perf)
    _check_logical("perf.AbsPerf", options.abs_perf)
    if options.abs_perf:
        _finite_real("perf.epsilon", options.epsilon)
    else:
        _check_unit_range("perf.epsilon", options.epsilon)
    _check_unit_range("perf.betaThreshold", options.beta_threshold)
    _check_logical("bound.flag", options.bound)
    _check_logical("norm.flag", options.norm)
    _check_positive("prelim.iqrMultiplier", options.iqr_multiplier)
    _check_logical("auto.preproc", options.preproc)
    _check_unit_range("prelim.nanThreshold", options.nan_threshold)


@dataclass(frozen=True)
class PrelimOptions:
    """Options for running PRELIM."""

    max_perf: bool
    abs_perf: bool
    epsilon: float
    beta_threshold: float
    bound: bool
    norm: bool
    # Multiplier applied to the IQR when computing outlier bounds
    # (hi_bound/lo_bound = med_val +/- iqr_multiplier * iq_range), matching
    # MATLAB's opts.iqrMultiplier. Default matches the value this port
    # previously hard-coded.
    iqr_multiplier: float = DEFAULT_PRELIM_IQR_MULTIPLIER
    # Master switch matching MATLAB's opts.auto.preproc. Keep this enabled by
    # default and after the existing multiplier to preserve positional callers.
    preproc: bool = DEFAULT_AUTO_PREPROC
    # Fraction of missing feature values at which preprocessing drops a column.
    # Keep this after every existing field to preserve positional callers.
    nan_threshold: float = DEFAULT_PRELIM_NAN_THRESHOLD

    def __post_init__(self) -> None:
        """Validate the composed PRELIM options at direct construction."""
        _validate_prelim_options(self)

    @staticmethod
    def default(
        max_perf: bool = DEFAULT_PERFORMANCE_MAX_PERF,
        abs_perf: bool = DEFAULT_PERFORMANCE_ABS_PERF,
        epsilon: float = DEFAULT_PERFORMANCE_EPSILON,
        beta_threshold: float = DEFAULT_PERFORMANCE_BETA_THRESHOLD,
        bound: bool = DEFAULT_BOUND_FLAG,
        norm: bool = DEFAULT_NORM_FLAG,
        iqr_multiplier: float = DEFAULT_PRELIM_IQR_MULTIPLIER,
        preproc: bool = DEFAULT_AUTO_PREPROC,
        nan_threshold: float = DEFAULT_PRELIM_NAN_THRESHOLD,
    ) -> PrelimOptions:
        """Instantiate the composed PRELIM options with aggregate defaults."""
        return PrelimOptions(
            max_perf=max_perf,
            abs_perf=abs_perf,
            epsilon=epsilon,
            beta_threshold=beta_threshold,
            bound=bound,
            norm=norm,
            iqr_multiplier=iqr_multiplier,
            preproc=preproc,
            nan_threshold=nan_threshold,
        )

    @staticmethod
    def from_options(options: InstanceSpaceOptions) -> PrelimOptions:
        """Get a prelim options object from an existing InstanceSpaceOptions object."""
        return PrelimOptions(
            max_perf=options.perf.max_perf,
            abs_perf=options.perf.abs_perf,
            epsilon=options.perf.epsilon,
            beta_threshold=options.perf.beta_threshold,
            bound=options.bound.flag,
            norm=options.norm.flag,
            iqr_multiplier=options.prelim.iqr_multiplier,
            preproc=options.auto.preproc,
            nan_threshold=options.prelim.nan_threshold,
        )


def from_json_file(file_path: Path | str) -> InstanceSpaceOptions | None:
    """Parse options from a JSON file and construct an InstanceSpaceOptions object.

    Args:
    ----
    file_path : Path | str
        The path to the JSON file containing the options.

    Returns:
    -------
    InstanceSpaceOptions or None
        An InstanceSpaceOptions object constructed from the parsed JSON data, or None
        if an error occurred during file reading or parsing.

    Raises:
    ------
    FileNotFoundError
        If the specified file does not exist.
    json.JSONDecodeError
        If the specified file contains invalid JSON.
    OSError
        If an I/O error occurred while reading the file.
    ValueError
        If the parsed JSON data contains invalid options.

    """
    if isinstance(file_path, str):
        file_path = Path(file_path)

    try:
        with file_path.open() as o:
            options_contents = o.read()
        opts_dict = json.loads(options_contents)

        return InstanceSpaceOptions.from_dict(opts_dict)
    except (FileNotFoundError, json.JSONDecodeError, OSError) as e:
        logger.error(f"{file_path}: {e!s}")
        return None
    except ValueError as e:
        logger.error(f"Error: Invalid options data in the file '{file_path}'.")
        logger.error(f"Error details: {e!s}")
        return None
