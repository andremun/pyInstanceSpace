import pandas as pd

from dataclasses import dataclass, field

@dataclass
class ParallelOptions:
    flag: bool
    ncores: int

@dataclass
class PerformanceOptions:
    MaxPerf: bool
    AbsPerf: bool
    epsilon: float
    betaThreshold: float

@dataclass
class AutoOptions:
    preproc: bool

@dataclass
class BoundOptions:
    flag: bool

@dataclass
class NormOptions:
    flag: bool

@dataclass
class SelvarsOptions:
    """By Chen """
    smallscaleflag: bool
    smallscale: float
    fileidxflag: bool
    fileidx: str
    """fileidx length is not sure, char or str"""
    feats: pd.DataFrame
    algos: pd.DataFrame

    #based on the usage of FILTER in buildIS, following type could have:
    type: str # Value is one of: Ftr,Ftr&AP,Ftr&Good,Ftr&AP&Good
    mindistance: float
    densityflag: bool

@dataclass
class SiftedOptions:
    flag: bool
    rho: float
    K: int
    NTREES: int
    MaxIter: int
    Replicates: int

@dataclass
class PilotOptions:
    analytic: bool
    ntries: int

@dataclass
class CloisterOptions:
    pval: float
    cthres: float

@dataclass
class PythiaOptions:
    cvfolds: int
    ispolykrnl: bool
    useweights: bool
    uselibsvm: bool

@dataclass
class TraceOptions:
    usesim: bool
    PI: float

@dataclass
class OutputOptions:
    csv: bool
    web: bool
    png: bool

@dataclass
class Opts:
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
