"""Helper module to create Option object for unit testing.

Examples
--------
>>> cloister_opt = CloisterOptions(p_val=0.1, c_thres=0.8)
>>> options = create_option(cloister=cloister_opt)
"""

import pandas as pd

from matilda.data.options import (
    AutoOptions,
    BoundOptions,
    CloisterOptions,
    InstanceSpaceOptions,
    NormOptions,
    OutputOptions,
    ParallelOptions,
    PerformanceOptions,
    PilotOptions,
    PythiaOptions,
    SelvarsOptions,
    SiftedOptions,
    TraceOptions,
)

default_parallel = ParallelOptions(flag=False, n_cores=2)
default_perf = PerformanceOptions(
    max_perf=False,
    abs_perf=False,
    epsilon=0.2,
    beta_threshold=0.2,
)
default_auto = AutoOptions(preproc=True)
default_bound = BoundOptions(flag=True)
default_norm = NormOptions(flag=True)
default_selvars = SelvarsOptions(
    small_scale_flag=False,
    small_scale=0.5,
    file_idx_flag=False,
    file_idx="",
    feats=pd.get_dummies(pd.Series(list("abc")), dtype=float),
    algos=pd.get_dummies(pd.Series(list("abc")), dtype=float),
    selvars_type="Ftr&Good",
    density_flag=False,
    min_distance=0.1,
)
default_sifted = SiftedOptions(
    flag=True,
    rho=0.1,
    k=10,
    n_trees=50,
    max_iter=1000,
    replicates=100,
    num_generations=10,
    num_parents_mating=5,
    sol_per_pop=50,
    parent_selection_type="tournament",
    k_tournament=3,
    keep_elitism=True,
    crossover_type="single_point",
    cross_over_probability=0.8,
    mutation_type="random",
    mutation_probability=0.2,
    stop_criteria="max_generations",
)
default_pilot = PilotOptions(
    analytic=False,
    n_tries=5,
)
default_cloister = CloisterOptions(
    p_val=0.05,
    c_thres=0.7,
)
default_pythia = PythiaOptions(
    cv_folds=5,
    is_poly_krnl=False,
    use_weights=False,
    use_lib_svm=False,
)
default_trace = TraceOptions(
    use_sim=True,
    purity=0.55,
)
default_outputs = OutputOptions(
    csv=True,
    web=False,
    png=True,
)


def create_option(
    parallel: ParallelOptions = default_parallel,
    perf: PerformanceOptions = default_perf,
    auto: AutoOptions = default_auto,
    bound: BoundOptions = default_bound,
    norm: NormOptions = default_norm,
    selvars: SelvarsOptions = default_selvars,
    sifted: SiftedOptions = default_sifted,
    pilot: PilotOptions = default_pilot,
    cloister: CloisterOptions = default_cloister,
    pythia: PythiaOptions = default_pythia,
    trace: TraceOptions = default_trace,
    outputs: OutputOptions = default_outputs,
) -> InstanceSpaceOptions:
    """Create option object based on the argument given.

    InstanceSpaceOptions that is not specified will use corresponding default option
    object.
    """
    return InstanceSpaceOptions(
        parallel=parallel,
        perf=perf,
        auto=auto,
        bound=bound,
        norm=norm,
        selvars=selvars,
        sifted=sifted,
        pilot=pilot,
        cloister=cloister,
        pythia=pythia,
        trace=trace,
        outputs=outputs,
    )
