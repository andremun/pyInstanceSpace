
import numpy as np
import pandas as pd
import stage
from numpy.typing import NDArray

from matilda.data.metadata import Metadata
from matilda.data.model import (
    Data,
)
from matilda.data.options import (
    AutoOptions,
    BoundOptions,
    CloisterOptions,
    NormOptions,
    OutputOptions,
    PerformanceOptions,
    PilotOptions,
    PythiaOptions,
    SelvarsOptions,
    SiftedOptions,
    TraceOptions,
)


class preprocessingStage(stage):
    def __init__(self, feature_names: list[str], algorithm_names: list[str], 
                 instance_labels: pd.Series, instance_sources: pd.Series | None,
                 features: NDArray[np.double], algorithms: NDArray[np.double],
                 perf: PerformanceOptions, auto: AutoOptions, bound: BoundOptions,
                 norm: NormOptions, selvars: SelvarsOptions, sifted: SiftedOptions,
                 pilot: PilotOptions, cloister: CloisterOptions, pythia: PythiaOptions,
                trace: TraceOptions, outputs: OutputOptions):
        self.feature_names = feature_names
        self.algorithm_names = algorithm_names
        self.instance_labels = instance_labels
        self.instance_sources = instance_sources
        self.features = features
        self.algorithms = algorithms
        self.perf = perf
        self.auto = auto
        self.bound = bound
        self.norm = norm
        self.selvars = selvars
        self.sifted = sifted
        self.pilot = pilot
        self.cloister = cloister
        self.pythia = pythia
        self.trace = trace
        self.outputs = outputs


    @staticmethod
    def _inputs() -> list[tuple[str, type]]:
        return [
            ["feature_names", list[str]],
            ["algorithm_names", list[str]],
            ["instance_labels", pd.Series],
            ["instance_sources", pd.Series | None],
            ["features", NDArray[np.double]],
            ["algorithms", NDArray[np.double]],
            ["perf", PerformanceOptions],
            ["auto", AutoOptions],
            ["bound", BoundOptions],
            ["norm", NormOptions],
            ["selvars", SelvarsOptions],
            ["sifted", SiftedOptions],
            ["pilot", PilotOptions],
            ["cloister", CloisterOptions],
            ["pythia", PythiaOptions],
            ["trace", TraceOptions],
            ["outputs", OutputOptions],
        ]

    @staticmethod
    def _outputs() -> list[tuple[str, type]]:
        return [
            ["inst_labels", pd.Series],  # type: ignore[type-arg]
            ["feat_labels", list[str]],
            ["algo_labels", list[str]],
            ["x", NDArray[np.double]],
            ["y", NDArray[np.double]],
            ["s", pd.Series | None],
        ]

    def _run(
        self,
        perf: PerformanceOptions,
        auto: AutoOptions,
        bound: BoundOptions,
        norm: NormOptions,
        selvars: SelvarsOptions,
        sifted: SiftedOptions,
        pilot: PilotOptions,
        cloister: CloisterOptions,
        pythia: PythiaOptions,
        trace: TraceOptions,
        outputs: OutputOptions,
    ) -> tuple[pd.Series, list[str], list[str], NDArray[np.double], NDArray[np.double],
                pd.Series | None]:
        # All the code including the code in the buildIS should be here
        raise NotImplementedError

    @staticmethod
    def preprocessing(
        self,
        feature_names: list[str],
        algorithm_names: list[str],
        instance_labels: pd.Series,
        instance_sources: pd.Series | None,
        features: NDArray[np.double],
        algorithms: NDArray[np.double],
        perf: PerformanceOptions,
        auto: AutoOptions,
        bound: BoundOptions,
        norm: NormOptions,
        selvars: SelvarsOptions,
        sifted: SiftedOptions,
        pilot: PilotOptions,
        cloister: CloisterOptions,
        pythia: PythiaOptions,
        trace: TraceOptions,
        outputs: OutputOptions,
    ) -> tuple[pd.Series, list[str], list[str], NDArray[np.double], NDArray[np.double],
                pd.Series | None]:
        # This has code specific to preprocessing.m
        raise NotImplementedError

