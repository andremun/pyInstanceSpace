# SPDX-License-Identifier: LicenseRef-PolyForm-Noncommercial-1.0.0
# Copyright (c) 2024-2026 Mario Andrés Muñoz
"""Data about the output of running InstanceSpace."""

import hashlib
import hmac
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, TypeVar

import joblib

from instancespace._serialisers import (
    SerializationError,
    save_instance_space_for_web,
    save_instance_space_graphs,
    save_instance_space_output_mat,
    save_instance_space_to_csv,
)
from instancespace.data.model import (
    CloisterOut,
    Data,
    DataDense,
    FeatSel,
    PilotOut,
    PrelimOut,
    PythiaOut,
    SiftedOut,
    TraceOut,
)
from instancespace.data.options import InstanceSpaceOptions

DEFAULT_DIRECTORY_NAME = "output"
_INVALID_ARCHIVE_NAME = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_WINDOWS_RESERVED_ARCHIVE_STEMS = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}


class ModelSignatureError(Exception):
    """A `Model` file's signature could not be verified, or its presence is invalid.

    Covers a mismatched signature, a missing signature when `secret_key` was
    given, and a present signature when `secret_key` was not given (the
    downgrade-attack case - see `Model.load()`).
    """


def _validate_archive_name(zip_filename: str) -> None:
    """Reject archive names that are unsafe on common filesystems."""
    if (
        not zip_filename
        or zip_filename in {".", ".."}
        or _INVALID_ARCHIVE_NAME.search(zip_filename) is not None
        or zip_filename.rstrip(" .") != zip_filename
        or zip_filename.split(".", maxsplit=1)[0].upper()
        in _WINDOWS_RESERVED_ARCHIVE_STEMS
    ):
        raise ValueError("zip_filename must be one safe filename without a path.")


@dataclass(frozen=True)
class Model:
    """The output of running InstanceSpace."""

    data: Data
    data_dense: DataDense | None
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
            data_dense=stage_runner_output["data_dense"],
            feat_sel=FeatSel.from_stage_runner_output(stage_runner_output),
            prelim=PrelimOut.from_stage_runner_output(stage_runner_output),
            sifted=SiftedOut.from_stage_runner_output(stage_runner_output),
            pilot=PilotOut.from_stage_runner_output(stage_runner_output),
            cloister=CloisterOut.from_stage_runner_output(stage_runner_output),
            pythia=PythiaOut.from_stage_runner_output(stage_runner_output),
            trace=TraceOut.from_stage_runner_output(stage_runner_output),
            opts=options,
        )

    def save(self, path: Path | str, secret_key: bytes | None = None) -> None:
        """Serialise this Model to `path` via `joblib`, mirroring MATLAB's persistence.

        Trained scikit-learn/shapely objects (PYTHIA's classifiers, TRACE's
        footprint polygons) round-trip natively - no flattening step.

        If `secret_key` is given, an HMAC-SHA256 signature of the serialised
        bytes is written alongside the model at `<path>.sig`, and `load()`
        will refuse to deserialise unless the same key verifies it. If
        `secret_key` is `None` (the default - the local/desktop-development
        case with no server-managed secret), no signature is written and
        `load()` performs no verification, the same trust caveat any other
        unsigned `pickle`/`joblib` file already carries.

        On the production web platform, every server code path that calls
        `load()` must always supply `secret_key` and must never accept a
        user-supplied `path` - that invariant lives in the deployment, not
        in this method.
        """
        if isinstance(path, str):
            path = Path(path)

        joblib.dump(self, path)

        sig_path = path.with_name(path.name + ".sig")
        if secret_key is not None:
            signature = hmac.new(secret_key, path.read_bytes(), hashlib.sha256)
            sig_path.write_bytes(signature.digest())
        elif sig_path.exists():
            # Leaving a stale signature from a previous signed save() to this
            # same path would make an unrelated future load() see a
            # signature that doesn't belong to the file it's next to.
            sig_path.unlink()

    @classmethod
    def load(cls: type[T], path: Path | str, secret_key: bytes | None = None) -> T:
        """Deserialise a `Model` previously written by `save()`.

        Four cases, matching `save()`'s two modes:
        - `secret_key` given, signature present: verified before
          deserialising - raises `ModelSignatureError` on mismatch.
        - `secret_key` given, signature absent: raises - a caller expecting
          a verified load must never silently fall through to an unverified
          one.
        - `secret_key` is `None`, signature absent: deserialises directly
          (the desktop/dev path).
        - `secret_key` is `None`, signature present: raises. This is the
          downgrade-attack guard - a signed file must not become
          loadable-unverified just because the caller omitted the key.
        """
        if isinstance(path, str):
            path = Path(path)

        sig_path = path.with_name(path.name + ".sig")
        sig_exists = sig_path.exists()

        if secret_key is not None and not sig_exists:
            raise ModelSignatureError(
                f"secret_key was given but no signature file exists at "
                f"{sig_path}; refusing to load an unverifiable file.",
            )
        if secret_key is None and sig_exists:
            raise ModelSignatureError(
                f"A signature file exists at {sig_path} but no secret_key "
                "was given; refusing to load a signed file without "
                "verification.",
            )

        if secret_key is not None:
            expected_signature = sig_path.read_bytes()
            actual_signature = hmac.new(
                secret_key,
                path.read_bytes(),
                hashlib.sha256,
            ).digest()
            if not hmac.compare_digest(actual_signature, expected_signature):
                raise ModelSignatureError(
                    f"Signature verification failed for {path}; refusing to "
                    "deserialise.",
                )

        model = joblib.load(path)
        if not isinstance(model, cls):
            raise TypeError(
                f"{path} does not contain a {cls.__name__} (got "
                f"{type(model).__name__!r}).",
            )
        return model

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
        """Save serializer outputs in a structured ZIP archive."""
        print(
            "=========================================================================",
        )

        if isinstance(output_directory, str):
            output_directory = Path(output_directory)

        if not output_directory.is_dir():
            raise ValueError("output_directory must be an existing directory.")
        _validate_archive_name(zip_filename)

        output_root = output_directory.resolve()
        archive_path = output_root / zip_filename
        if archive_path.is_symlink():
            raise ValueError(f"ZIP target must not be a symlink: '{archive_path}'.")
        members: list[tuple[Path, str]] = []
        member_names: set[str] = set()

        for source in sorted(output_root.rglob("*"), key=lambda path: path.as_posix()):
            if source == archive_path:
                continue
            if source.is_symlink():
                raise ValueError(f"ZIP input must not be a symlink: '{source}'.")
            if not source.is_file() or source.name == ".gitignore":
                continue

            relative = source.relative_to(output_root)
            member_name = str(PurePosixPath(DEFAULT_DIRECTORY_NAME, *relative.parts))
            if member_name in member_names:
                raise ValueError(f"Duplicate ZIP member: '{member_name}'.")
            member_names.add(member_name)
            members.append((source, member_name))

        try:
            with zipfile.ZipFile(
                archive_path,
                "w",
                zipfile.ZIP_DEFLATED,
            ) as archive:
                for source, member_name in members:
                    archive.write(source, arcname=member_name)
        except Exception as exc:
            raise SerializationError(
                f"Could not write ZIP file '{archive_path}'.",
            ) from exc
        print(f"-> Successfully saved files into {zip_filename}.")
