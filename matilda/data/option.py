"""Defines a collection of data classes that represent configuration options.

These classes provide a structured way to specify and manage settings for different
aspects of the model's execution and behaviour.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


class MissingOptionsError(Exception):
    """A required option wasn't set.

    An error raised when a stage is ran that requires an option to be set, and the
    option isn't present.
    """

    pass

@dataclass(frozen=True)
class ParallelOptions:
    """Configuration options for parallel computing."""

    flag: bool
    n_cores: int | None

    @staticmethod
    def default(
        flag: bool = False,
        n_cores: int | None = None,
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
    epsilon: float | None
    beta_threshold: float | None

    @staticmethod
    def default(
        max_perf: bool = False,
        abs_perf: bool = False,
        epsilon: float | None = None,
        beta_threshold: float | None = None,
    ) -> PerformanceOptions:
        """Instantiate with default values."""
        return PerformanceOptions(
            max_perf=max_perf,
            abs_perf=abs_perf,
            epsilon=epsilon,
            beta_threshold=beta_threshold,
        )

@dataclass(frozen=True)
class PrelimOptions:
    """Options for running PRELIM."""

    max_perf: bool
    abs_perf: bool
    epsilon: float | None
    beta_threshold: float | None
    bound: bool
    norm: bool

    @staticmethod
    def from_options(options: Options) -> PrelimOptions:
        """Get a prelim options object from an existing Options object."""
        return PrelimOptions(
            max_perf=options.perf.max_perf,
            abs_perf=options.perf.abs_perf,
            epsilon=options.perf.epsilon,
            beta_threshold=options.perf.beta_threshold,
            bound=options.bound.flag,
            norm=options.norm.flag,
        )


@dataclass(frozen=True)
class AutoOptions:
    """Options for automatic processing steps in the model pipeline."""

    preproc: bool

    @staticmethod
    def default(
        preproc: bool = False,
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
        flag: bool = False,
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
        flag: bool = False,
    ) -> NormOptions:
        """Instantiate with default values."""
        return NormOptions(
            flag=flag,
        )


@dataclass(frozen=True)
class SelvarsOptions:
    """Options for selecting variables, including criteria and file indices."""

    small_scale_flag: bool
    small_scale: float | None
    file_idx_flag: bool
    file_idx: str | None
    feats: pd.DataFrame | None
    algos: pd.DataFrame | None
    selvars_type: str | None
    min_distance: float | None
    density_flag: bool

    @staticmethod
    def default(
        small_scale_flag: bool = False,
        small_scale: float = False,
        file_idx_flag: bool = False,
        file_idx: str | None = None,
        feats: pd.DataFrame | None = None,
        algos: pd.DataFrame | None = None,
        selvars_type: str | None = None,
        min_distance: float | None = None,
        density_flag: bool = False,
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
    rho: float | None
    k: int | None
    n_trees: int | None
    max_iter: int | None
    replicates: int | None

    @staticmethod
    def default(
        flag: bool = False,
        rho: float | None = None,
        k: int | None = None,
        n_trees: int | None = None,
        max_iter: int | None = None,
        replicates: int | None = None,
    ) -> SiftedOptions:
        """Instantiate with default values."""
        return SiftedOptions(
            flag=flag,
            rho=rho,
            k=k,
            n_trees=n_trees,
            max_iter=max_iter,
            replicates=replicates,
        )


@dataclass(frozen=True)
class PilotOptions:
    """Options for pilot studies or preliminary analysis phases."""

    analytic: bool
    n_tries: int | None

    @staticmethod
    def default(
        analytic: bool = False,
        n_tries: int | None = None,
    ) -> PilotOptions:
        """Instantiate with default values."""
        return PilotOptions(
            analytic=analytic,
            n_tries=n_tries,
        )


@dataclass(frozen=True)
class CloisterOptions:
    """Options for cloistering in the model."""

    p_val: float | None
    c_thres: float | None

    @staticmethod
    def default(
        p_val: float | None = None,
        c_thres: float | None = None,
    ) -> CloisterOptions:
        """Instantiate with default values."""
        return CloisterOptions(
            p_val=p_val,
            c_thres=c_thres,
        )


@dataclass(frozen=True)
class PythiaOptions:
    """Configuration for the Pythia component of the model."""

    cv_folds: int | None
    is_poly_krnl: bool
    use_weights: bool
    use_lib_svm: bool

    @staticmethod
    def default(
        cv_folds: int | None = None,
        is_poly_krnl: bool = False,
        use_weights: bool = False,
        use_lib_svm: bool = False,
    ) -> PythiaOptions:
        """Instantiate with default values."""
        return PythiaOptions(
            cv_folds=cv_folds,
            is_poly_krnl=is_poly_krnl,
            use_weights=use_weights,
            use_lib_svm=use_lib_svm,
        )


@dataclass(frozen=True)
class TraceOptions:
    """Options for trace analysis in the model."""

    use_sim: bool
    pi: float | None

    @staticmethod
    def default(
        use_sim: bool = False,
        pi: float | None = None,
    ) -> TraceOptions:
        """Instantiate with default values."""
        return TraceOptions(
            use_sim=use_sim,
            pi=pi,
        )


@dataclass(frozen=True)
class OutputOptions:
    """Options for controlling the output format."""

    csv: bool
    web: bool
    png: bool

    @staticmethod
    def default(
        csv: bool = False,
        web: bool = False,
        png: bool = False,
    ) -> OutputOptions:
        """Instantiate with default values."""
        return OutputOptions(
            csv=csv,
            web=web,
            png=png,
        )


@dataclass(frozen=True)
class GeneralOptions:
    """General options that affect the whole system."""

    beta_threshold: float | None

    @staticmethod
    def default(
        beta_threshold: float | None = None,
    ) -> GeneralOptions:
        """Instantiate with default values."""
        return GeneralOptions(
            beta_threshold=beta_threshold,
        )


@dataclass(frozen=True)
class Options:
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
    general: GeneralOptions

    @staticmethod
    def from_file(file_contents: str) -> Options:
        """Parse options from a file, and construct an Options object.

        Args
        ----
        file_contents (str): The contents of a json file containing the options.

        Returns
        -------
        An Options object.
        """
        raise NotImplementedError

    def to_file(self) -> str:
        """Store options in a file from an Options object.

        Returns
        -------
        The options object serialised into a string.
        """
        raise NotImplementedError

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
        general: GeneralOptions | None,
    ) -> Options:
        """Instantiate with default values."""
        return Options(
            parallel= parallel or ParallelOptions.default(),
            perf= perf or PerformanceOptions.default(),
            auto= auto or AutoOptions.default(),
            bound= bound or BoundOptions.default(),
            norm= norm or NormOptions.default(),
            selvars= selvars or SelvarsOptions.default(),
            sifted= sifted or SiftedOptions.default(),
            pilot= pilot or PilotOptions.default(),
            cloister= cloister or CloisterOptions.default(),
            pythia= pythia or PythiaOptions.default(),
            trace= trace or TraceOptions.default(),
            outputs= outputs or OutputOptions.default(),
            general= general or GeneralOptions.default(),
        )
