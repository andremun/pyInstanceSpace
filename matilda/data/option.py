"""
Defines a collection of data classes that represent configuration options.

These classes provide a structured way to specify and manage settings for different
aspects of the model's execution and behaviour.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Self

import pandas as pd


@dataclass
class ParallelOptions:
    """Configuration options for parallel computing."""

    flag: bool
    n_cores: int


@dataclass
class PerformanceOptions:
    """Options related to performance thresholds and criteria for model evaluation."""

    max_perf: bool
    abs_perf: bool
    epsilon: float
    beta_threshold: float


@dataclass
class AutoOptions:
    """Options for automatic processing steps in the model pipeline."""

    preproc: bool


@dataclass
class BoundOptions:
    """Options for applying bounds in the model calculations or evaluations."""

    flag: bool


@dataclass
class NormOptions:
    """Options to control normalization processes within the model."""

    flag: bool


@dataclass
class SelvarsOptions:
    """Options for selecting variables, including criteria and file indices."""

    small_scale_flag: bool
    small_scale: float
    file_idx_flag: bool
    file_idx: str
    feats: pd.DataFrame
    algos: pd.DataFrame
    type: str
    min_distance: float
    density_flag: bool


@dataclass
class SiftedOptions:
    """Options specific to the sifting process in data analysis."""

    flag: bool
    rho: float
    k: int
    n_trees: int
    max_iter: int
    replicates: int


@dataclass
class PilotOptions:
    """Options for pilot studies or preliminary analysis phases."""

    analytic: bool
    n_tries: int


@dataclass
class CloisterOptions:
    """Options for cloistering in the model."""

    p_val: float
    c_thres: float


@dataclass
class PythiaOptions:
    """Configuration for the Pythia component of the model."""

    cv_folds: int
    is_poly_krnl: bool
    use_weights: bool
    use_lib_svm: bool


@dataclass
class TraceOptions:
    """Options for trace analysis in the model."""

    use_sim: bool
    PI: float


@dataclass
class OutputOptions:
    """Options for controlling the output format."""

    csv: bool
    web: bool
    png: bool


@dataclass
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
    "we probably need to have this 'general' field"  # general: GeneralOptions

    @staticmethod
    def from_file(filepath: Path) -> Options:
        if not filepath.is_file():
            raise FileNotFoundError \
                (f"Please place the options.json in the directory '{filepath.parent}'")

        with open(filepath) as file:
            opts_dict = json.load(file)

        # 验证顶级字段是否与Options中定义的字段相匹配
        options_fields = {f.name for f in fields(Options)}
        extra_fields = set(opts_dict.keys()) - options_fields
        if extra_fields:
            raise ValueError \
                (f"Extra fields in JSON not defined in Options: {extra_fields}")

        # 初始化Options中的每一个部分
        options = Options(
            parallel=load_dataclass(ParallelOptions, opts_dict["parallel"]) if "parallel" in opts_dict else None,
            perf=load_dataclass(PerformanceOptions, opts_dict["perf"]) if "perf" in opts_dict else None,
            auto=load_dataclass(AutoOptions, opts_dict["auto"]) if "auto" in opts_dict else None,
            bound=load_dataclass(BoundOptions, opts_dict["bound"]) if "bound" in opts_dict else None,
            norm=load_dataclass(NormOptions, opts_dict["norm"]) if "norm" in opts_dict else None,
            selvars=load_dataclass(SelvarsOptions, opts_dict["selvars"]) if "selvars" in opts_dict else None,
            sifted=load_dataclass(SiftedOptions, opts_dict["sifted"]) if "sifted" in opts_dict else None,
            pilot=load_dataclass(PilotOptions, opts_dict["pilot"]) if "pilot" in opts_dict else None,
            cloister=load_dataclass(CloisterOptions, opts_dict["cloister"]) if "cloister" in opts_dict else None,
            pythia=load_dataclass(PythiaOptions, opts_dict["pythia"]) if "pythia" in opts_dict else None,
            trace=load_dataclass(TraceOptions, opts_dict["trace"]) if "trace" in opts_dict else None,
            outputs=load_dataclass(OutputOptions, opts_dict["outputs"]) if "outputs" in opts_dict else None
        )

        print("-------------------------------------------------------------------------")
        print("-> Listing options to be used:")
        for field_name in fields(Options):
            field_value = getattr(options, field_name.name)
            print(f"{field_name.name}: {field_value}")

        return options

    def to_file(self: Self, filepath: Path) -> None:
        """
        Store options in a file from an Options object.

        :param filepath: The path of the resulting json file containing the options.
        """
        raise NotImplementedError


def validate_fields(data_class, data) -> None:
    # 获取数据类中定义的所有字段
    known_fields = {f.name for f in fields(data_class)}
    # 检测JSON中的字段是否都在数据类中有定义
    for key in data.keys():
        if key not in known_fields:
            raise ValueError \
                (f"Field '{key}' in JSON is not defined in the dataclass '{data_class.__name__}'")


def load_dataclass(data_class, data):
    validate_fields(data_class, data)  # 验证字段
    # 对每个数据类字段，从 JSON 数据中提取值，如果不存在，则使用 None
    init_args = {f.name: data.get(f.name, None) for f in fields(data_class)}
    return data_class(**init_args)


if __name__ == "__main__":
    metadata_file = Path("/Users/junhengchen/Documents/GitHub/MT-Updating-Matilda/tests/Trial_files/options.json")
    opt = Options.from_file(metadata_file)
    print("fine")
