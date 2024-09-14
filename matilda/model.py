"""Data about the output of running InstanceSpace."""

from dataclasses import dataclass
from pathlib import Path

from matilda._serializers import save_instance_space_for_web, save_instance_space_to_csv
from matilda.data.model import (
    CloisterOut,
    Data,
    FeatSel,
    PilotOut,
    PrelimOut,
    PythiaOut,
    SiftedOut,
    TraceOut,
)
from matilda.data.options import InstanceSpaceOptions


@dataclass(frozen=True)
class Model:
    """The output of running InstanceSpace."""

    data: Data
    data_dense: Data
    feat_sel: FeatSel
    prelim: PrelimOut
    sifted: SiftedOut
    pilot: PilotOut
    cloister: CloisterOut
    pythia: PythiaOut
    trace: TraceOut
    opts: InstanceSpaceOptions

    def save_to_csv(self, output_directory: Path) -> None:
        """Save csv outputs to a directory."""
        print(
            "=========================================================================",
        )
        print("-> Writing the data on CSV files for posterior analysis.")

        save_instance_space_to_csv(
            output_directory,
            self.data,
            self.sifted,
            self.trace,
            self.pilot,
            self.cloister,
            self.pythia,
        )

    def save_for_web(self, output_directory: Path) -> None:
        """Save csv outputs used for the web frontend to a directory."""
        print(
            "=========================================================================",
        )
        print("-> Writing the data for the web interface.")

        save_instance_space_for_web(
            output_directory,
            self.data,
            self.prelim,
            self.sifted,
            self.feat_sel,
        )
