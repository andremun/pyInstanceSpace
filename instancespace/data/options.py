# SPDX-License-Identifier: LicenseRef-PolyForm-Noncommercial-1.0.0
# Copyright (c) 2024-2026 Mario Andrés Muñoz
"""Defines a collection of data classes that represent configuration options.

These classes provide a structured way to specify and manage settings for different
aspects of the model's execution and behaviour.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Literal, Self, TypeVar

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
    DEFAULT_PYTHIA_CLASSIFIER,
    DEFAULT_PYTHIA_CV_FOLDS,
    DEFAULT_PYTHIA_IS_POLY_KRNL,
    DEFAULT_PYTHIA_N_TUNING_ITER,
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
    DEFAULT_TRACE_CONTRA,
    DEFAULT_TRACE_METHOD,
    DEFAULT_TRACE_PURITY,
    DEFAULT_TRACE_USE_SIM,
)


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
    mutation_probability: float
    stop_criteria: str
    # Significance threshold for the correlation filter, matching MATLAB's
    # opts.pval (core/SIFTED.m) - was a hardcoded class constant.
    pval: float = DEFAULT_SIFTED_PVAL
    # Projection dimensionality for the GA fitness function's internal KNN
    # neighbour count (dims + 1), matching MATLAB's opts.dims. PILOT itself
    # is 2D-only in this port, so 3 is accepted but currently has no effect
    # on PILOT's actual output - see default_options.py's DEFAULT_SIFTED_DIMS.
    dims: int = DEFAULT_SIFTED_DIMS

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
        mutation_probability: float = DEFAULT_SIFTED_MUTATION_PROBABILITY,
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

    @staticmethod
    def default(
        cv_folds: int = DEFAULT_PYTHIA_CV_FOLDS,
        is_poly_krnl: bool = DEFAULT_PYTHIA_IS_POLY_KRNL,
        use_weights: bool = DEFAULT_PYTHIA_USE_WEIGHTS,
        classifier: str = DEFAULT_PYTHIA_CLASSIFIER,
        tuning: str = DEFAULT_PYTHIA_TUNING,
        n_tuning_iter: int = DEFAULT_PYTHIA_N_TUNING_ITER,
    ) -> PythiaOptions:
        """Instantiate with default values."""
        return PythiaOptions(
            cv_folds=cv_folds,
            is_poly_krnl=is_poly_krnl,
            use_weights=use_weights,
            params=None,
            classifier=classifier,
            tuning=tuning,
            n_tuning_iter=n_tuning_iter,
        )


@dataclass(frozen=True)
class TraceOptions:
    """Options for trace analysis in the model."""

    use_sim: bool
    purity: float
    method: str = DEFAULT_TRACE_METHOD
    contra: bool = DEFAULT_TRACE_CONTRA

    @staticmethod
    def default(
        use_sim: bool = DEFAULT_TRACE_USE_SIM,
        purity: float = DEFAULT_TRACE_PURITY,
        method: str = DEFAULT_TRACE_METHOD,
        contra: bool = DEFAULT_TRACE_CONTRA,
    ) -> TraceOptions:
        """Instantiate with default values."""
        return TraceOptions(
            use_sim=use_sim,
            purity=purity,
            method=method,
            contra=contra,
        )


@dataclass(frozen=True)
class OutputOptions:
    """Options for controlling the output format."""

    csv: bool
    web: bool
    png: bool

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
        msg = f"opts.{name} must be a logical scalar (True/False); got {value!r}."
        raise ValueError(msg)


def _check_unit_range(name: str, value: float) -> None:
    if not (0 <= value <= 1):
        msg = f"opts.{name} must be in the unit range [0, 1]; got {value!r}."
        raise ValueError(msg)


def _check_positive(name: str, value: float, *, zero_allowed: bool = False) -> None:
    if zero_allowed:
        if value < 0:
            msg = f"opts.{name} must be non-negative; got {value!r}."
            raise ValueError(msg)
    elif value <= 0:
        msg = f"opts.{name} must be strictly positive; got {value!r}."
        raise ValueError(msg)


def _check_pos_int(name: str, value: int, *, zero_allowed: bool = False) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        msg = f"opts.{name} must be an integer; got {value!r}."
        raise ValueError(msg)
    _check_positive(name, value, zero_allowed=zero_allowed)


def _check_member(name: str, value: str, valid: tuple[str, ...]) -> None:
    if value not in valid:
        msg = f"opts.{name} must be one of {valid}; got {value!r}."
        raise ValueError(msg)


def _check_sifted_dims(name: str, value: int) -> None:
    if value not in (2, 3):
        msg = f"opts.{name} must be 2 or 3; got {value!r}."
        raise ValueError(msg)


def _check_cloister_hull_dims(name: str, value: int | str) -> None:
    if value == "all":
        return
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        msg = f"opts.{name} must be 'all' or a positive integer; got {value!r}."
        raise ValueError(msg)


def _check_text_list(name: str, value: list[str] | None) -> None:
    if value is not None and not all(isinstance(v, str) for v in value):
        msg = f"opts.{name} must be a list of strings, or None; got {value!r}."
        raise ValueError(msg)


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
    # Defaulted (rather than required like the groups above) so existing direct
    # `InstanceSpaceOptions(...)` construction that predates this field keeps working
    # unchanged - this field must stay last for that reason.
    general: GeneralOptions = field(default_factory=GeneralOptions.default)

    def __post_init__(self: Self) -> None:
        """Validate every recognised option field, matching `ISAvalidateOpts.m` (F13).

        Fails loudly and immediately - at construction time, for every
        construction path, not just `from_dict()`/`default()` - rather than
        letting a bad value surface many stages later as a confusing crash
        deep inside PRELIM/PILOT/PYTHIA/etc., or silently produce a
        numerically-valid-looking but wrong result. Only checks fields that
        exist in this port today; MATLAB fields with no Python equivalent
        yet (`pilot.method`/`dims`/`topoWeight`/`viewGroups`, `pythia.skip`/
        `ensembleMethod`, `trace.minInstances`/`minAreaFrac`,
        `outputs.fig`) are out of scope until those options themselves are
        ported (see F2/F5/F8/F9). `sifted.pval`/`sifted.dims` (#300 audit
        findings, issues 2 and 4) and `cloister.hullDims` (#299 audit
        finding, issue 5) are now real fields, validated below.
        """
        _check_logical("general.verbose", self.general.verbose)
        if self.general.seed is not None:
            _check_pos_int("general.seed", self.general.seed, zero_allowed=True)

        _check_logical("general.parallel", self.parallel.flag)
        _check_pos_int("general.ncores", self.parallel.n_cores)

        _check_logical("perf.MaxPerf", self.perf.max_perf)
        _check_logical("perf.AbsPerf", self.perf.abs_perf)
        _check_unit_range("perf.epsilon", self.perf.epsilon)
        _check_unit_range("perf.betaThreshold", self.perf.beta_threshold)

        _check_logical("auto.preproc", self.auto.preproc)
        _check_logical("bound.flag", self.bound.flag)
        _check_logical("norm.flag", self.norm.flag)

        _check_logical("selvars.smallscaleflag", self.selvars.small_scale_flag)
        _check_unit_range("selvars.smallscale", self.selvars.small_scale)
        _check_logical("selvars.fileidxflag", self.selvars.file_idx_flag)
        _check_logical("selvars.densityflag", self.selvars.density_flag)
        _check_positive("selvars.mindistance", self.selvars.min_distance)
        _check_member(
            "selvars.type",
            self.selvars.selvars_type,
            ("Ftr", "Ftr&AP", "Ftr&Good", "Ftr&AP&Good"),
        )
        _check_text_list("selvars.feats", self.selvars.feats)
        _check_text_list("selvars.algos", self.selvars.algos)

        _check_logical("sifted.flag", self.sifted.flag)
        _check_unit_range("sifted.rho", self.sifted.rho)
        _check_pos_int("sifted.K", self.sifted.k)
        _check_pos_int("sifted.MaxIter", self.sifted.max_iter)
        _check_pos_int("sifted.Replicates", self.sifted.replicates)
        _check_unit_range("sifted.pval", self.sifted.pval)
        _check_sifted_dims("sifted.dims", self.sifted.dims)

        _check_logical("pilot.analytic", self.pilot.analytic)
        _check_pos_int("pilot.ntries", self.pilot.n_tries)
        _check_positive("pilot.costWeight", self.pilot.cost_weight)
        _check_member("pilot.method", self.pilot.method, ("standard", "pls"))

        _check_unit_range("cloister.pval", self.cloister.p_val)
        _check_unit_range("cloister.corrThreshold", self.cloister.c_thres)
        _check_cloister_hull_dims("cloister.hullDims", self.cloister.hull_dims)

        _check_member(
            "pythia.classifier",
            self.pythia.classifier,
            ("knn", "svm", "tree", "nb", "linear", "ensemble"),
        )
        _check_pos_int("pythia.kFold", self.pythia.cv_folds)

        _check_member("trace.method", self.trace.method, ("trace3", "legacy"))
        _check_unit_range("trace.PI", self.trace.purity)
        _check_logical("trace.contra", self.trace.contra)

        _check_logical("outputs.csv", self.outputs.csv)
        _check_logical("outputs.png", self.outputs.png)
        _check_logical("outputs.web", self.outputs.web)

    @staticmethod
    def from_dict(file_contents: dict[str, Any]) -> InstanceSpaceOptions:
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
        # Validate if the top-level fields match those in the InstanceSpaceOptions class
        options_fields = {f.name for f in fields(InstanceSpaceOptions)}
        extra_fields = set(file_contents.keys()) - options_fields

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
                file_contents.get("parallel", {}),
                {
                    "ncores": "n_cores",
                },
            ),
            perf=InstanceSpaceOptions._load_dataclass(
                PerformanceOptions,
                file_contents.get("perf", {}),
                {
                    "maxperf": "max_perf",
                    "absperf": "abs_perf",
                    "betathreshold": "beta_threshold",
                },
            ),
            auto=InstanceSpaceOptions._load_dataclass(
                AutoOptions,
                file_contents.get("auto", {}),
            ),
            bound=InstanceSpaceOptions._load_dataclass(
                BoundOptions,
                file_contents.get("bound", {}),
            ),
            norm=InstanceSpaceOptions._load_dataclass(
                NormOptions,
                file_contents.get("norm", {}),
            ),
            selvars=InstanceSpaceOptions._load_dataclass(
                SelvarsOptions,
                file_contents.get("selvars", {}),
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
                file_contents.get("sifted", {}),
                {
                    "ntrees": "n_trees",
                    "maxiter": "max_iter",
                    "replicates": "replicates",
                    # "k": "k",
                },
            ),
            pilot=InstanceSpaceOptions._load_dataclass(
                PilotOptions,
                file_contents.get("pilot", {}),
                {
                    "ntries": "n_tries",
                    "adjustrotation": "adjust_rotation",
                    "costweight": "cost_weight",
                    # "x0": "x0"
                },
            ),
            cloister=InstanceSpaceOptions._load_dataclass(
                CloisterOptions,
                file_contents.get("cloister", {}),
                {
                    "pval": "p_val",
                    "cthres": "c_thres",
                },
            ),
            pythia=InstanceSpaceOptions._load_dataclass(
                PythiaOptions,
                file_contents.get("pythia", {}),
                field_mapping={
                    "cvfolds": "cv_folds",
                    "ispolykrnl": "is_poly_krnl",
                    "useweights": "use_weights",
                    "uselibsvm": "_",  # deprecated MATLAB flag - genuinely ignored
                    "ntuningiter": "n_tuning_iter",
                },
            ),
            trace=InstanceSpaceOptions._load_dataclass(
                TraceOptions,
                file_contents.get("trace", {}),
                field_mapping={
                    "pi": "purity",
                    "usesim": "use_sim",
                },  # mapping the 'pi' in JSON to the 'purity' in TraceOptions
            ),
            outputs=InstanceSpaceOptions._load_dataclass(
                OutputOptions,
                file_contents.get("outputs", {}),
            ),
            general=InstanceSpaceOptions._load_dataclass(
                GeneralOptions,
                file_contents.get("general", {}),
            ),
        )

    @staticmethod
    def default(
        parallel: ParallelOptions | None,
        perf: PerformanceOptions | None,
        auto: AutoOptions | None,
        bound: BoundOptions | None,
        norm: NormOptions | None,
        selvars: SelvarsOptions | None,
        sifted: SiftedOptions | None,
        pilot: PilotOptions | None,
        cloister: CloisterOptions | None,
        pythia: PythiaOptions | None,
        trace: TraceOptions | None,
        outputs: OutputOptions | None,
        general: GeneralOptions | None = None,
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
        )

    T = TypeVar(
        "T",
        ParallelOptions,
        PerformanceOptions,
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
            mapped_field = field_mapping.get(json_field.lower(), json_field.lower())

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
        data: dict[str, Any],
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

        # Get the default values for the dataclass fields
        default_values = {
            f.name: getattr(data_class.default(), f.name) for f in fields(data_class)
        }

        mapped_data = {}

        data_lowercase = {k.lower(): v for k, v in data.items()}
        # Loop through each field in the dataclass, applying field mappings if needed
        for field_name, default_value in default_values.items():
            # If the field name is found in the dictionary, directly use its value
            if field_name.lower() in data_lowercase:
                mapped_data[field_name] = data_lowercase[field_name.lower()]
            else:
                # The field is explicitly mapped, use the mapped field name
                json_field_name = next(
                    (k for k, v in field_mapping.items() if v == field_name),
                    field_name,
                )

                # Fetch the value from the input dictionary, or fall back to the default
                mapped_data[field_name] = data_lowercase.get(
                    json_field_name,
                    default_value,
                )

        # Validate the fields before returning the dataclass instance
        InstanceSpaceOptions._validate_fields(data_class, data, field_mapping)

        return data_class(**mapped_data)


# InstanceSpaceOptions not part of the main InstanceSpaceOptions class


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
    iqr_multiplier: float = 5.0

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
