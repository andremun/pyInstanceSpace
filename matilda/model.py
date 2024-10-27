"""Data about the output of running InstanceSpace."""

import os
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

from matilda._serialisers import (
    save_instance_space_for_web,
    save_instance_space_graphs,
    save_instance_space_output_mat,
    save_instance_space_to_csv,
)
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

DEFAULT_DIRECTARY_NAME = "output"


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

    T = TypeVar("T", bound="Model")

    @classmethod
    def from_stage_runner_output(
        cls: type[T],
        stage_runner_output: dict[str, Any],
        options: InstanceSpaceOptions,
    ) -> T:
        """Initialise a Model object from the output of an InstanceSpace StageRunner.

        Args
        ----
            cls (type[T]): the class
            stage_runner_output (dict[str, Any]): output of StageRunner for an
                InstanceSpace

        Returns
        -------
            Model: a Model object
        """
        data = Data.from_stage_runner_output(stage_runner_output)

        return cls(
            data=data,
            data_dense=data,  # TODO: Work out what data_dense is
            feat_sel=FeatSel.from_stage_runner_output(stage_runner_output),
            prelim=PrelimOut.from_stage_runner_output(stage_runner_output),
            sifted=SiftedOut.from_stage_runner_output(stage_runner_output),
            pilot=PilotOut.from_stage_runner_output(stage_runner_output),
            cloister=CloisterOut.from_stage_runner_output(stage_runner_output),
            pythia=PythiaOut.from_stage_runner_output(stage_runner_output),
            trace=TraceOut.from_stage_runner_output(stage_runner_output),
            opts=options,
        )

    def save_to_csv(self, output_directory: Path | str) -> None:
        """Save csv outputs to a directory."""
        print(
            "=========================================================================",
        )
        print("-> Writing the data on CSV files for posterior analysis.")

        if isinstance(output_directory, str):
            output_directory = Path(output_directory)

        save_instance_space_to_csv(
            output_directory,
            self.data,
            self.sifted,
            self.trace,
            self.pilot,
            self.cloister,
            self.pythia,
        )

    def save_for_web(self, output_directory: Path | str) -> None:
        """Save csv outputs used for the web frontend to a directory."""
        print(
            "=========================================================================",
        )
        print("-> Writing the data for the web interface.")

        if isinstance(output_directory, str):
            output_directory = Path(output_directory)

        save_instance_space_for_web(
            output_directory,
            self.data,
            self.feat_sel,
        )

    def save_graphs(self, output_directory: Path | str) -> None:
        """Save csv outputs used for the web frontend to a directory."""
        print(
            "=========================================================================",
        )
        print("-> Producing the plots.")

        if isinstance(output_directory, str):
            output_directory = Path(output_directory)

        save_instance_space_graphs(
            output_directory,
            self.data,
            self.opts,
            self.pythia,
            self.pilot,
            self.trace,
        )

    def save_to_mat(self, output_directory: Path | str) -> None:
        """Save csv outputs used for the web frontend to a directory."""
        print(
            "=========================================================================",
        )
        print("-> Writing the data for the web interface.")

        if isinstance(output_directory, str):
            output_directory = Path(output_directory)

        save_instance_space_output_mat(
            output_directory,
            self.data,
        )

    def save_zip(self, zip_filename: str, output_directory: Path | str) -> None:
        """Save serializer outputs into a zip used for the web frontend."""
        print(
            "=========================================================================",
        )

        if isinstance(output_directory, str):
            output_directory = Path(output_directory)

        dir_name = DEFAULT_DIRECTARY_NAME
        ignored_files = [".gitignore", zip_filename]
        with zipfile.ZipFile(
            output_directory / zip_filename,
            "w",
            zipfile.ZIP_DEFLATED,
        ) as zf:
            for root, _, files in os.walk(output_directory):
                for filename in files:
                    if filename in ignored_files:
                        continue
                    zf.write(Path(root) / filename, arcname=Path(dir_name) / filename)
        print(f"-> Successfully saved files into {zip_filename}.")
