"""Validate MATLAB fixture bundles and classify historical test data."""

from __future__ import annotations

import argparse
import csv
import fnmatch
import hashlib
import json
import math
import re
import shutil
import tempfile
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any, Final, cast

BUNDLE_SCHEMA: Final = "pyinstancespace.matlab-fixtures/v1"
REFERENCE_PROFILE_V1: Final = "pyinstancespace.reference-export/v1"
REFERENCE_PROFILE: Final = "pyinstancespace.reference-export/v2"
RESOLVED_OPTIONS_INDEX_SCHEMA: Final = "pyinstancespace.resolved-options-index/v1"
RESOLVED_OPTIONS_SCHEMA: Final = "pyinstancespace.resolved-options/v1"
INVENTORY_SCHEMA: Final = "pyinstancespace.fixture-inventory/v1"
VERIFIED_TRUST: Final = "matlab-verified"
DIAGNOSTIC_TRUST: Final = "matlab-diagnostic"
_ALLOWED_TRUST: Final = {
    VERIFIED_TRUST,
    DIAGNOSTIC_TRUST,
    "legacy-unknown",
    "python-regression",
    "python-synthetic",
    "test-scratch",
}
_ALLOWED_PHASES: Final = {"shared", "build", "explore"}
_ALLOWED_STAGES: Final = {
    "preprocessing",
    "prelim",
    "sifted",
    "pilot",
    "cloister",
    "pythia",
    "trace",
}
_SHA256_RE: Final = re.compile(r"[0-9a-f]{64}")
_COMMIT_RE: Final = re.compile(r"[0-9a-f]{40}")
_RELEASE_RE: Final = re.compile(r"R(?P<year>\d{4})(?P<half>[ab])")
_HOLE_RE: Final = re.compile(r"hole_(?P<number>[1-9]\d*)")
_GEOMETRY_HEADER: Final = (
    "part",
    "ring",
    "vertex",
    "is_hole",
    "z_1",
    "z_2",
)
_MIN_RING_VERTICES: Final = 3
_SHARED_INPUT_PATH_DEPTH: Final = 3
_RESOLVED_OPTION_PATH_DEPTH: Final = 2
_STAGE_PATH_DEPTH: Final = 5
_DOWNSTREAM_VARIANTS: Final = (
    "trace3_default",
    "trace3_pythia_skip",
    "legacy_svm",
)
_PILOT_EVIDENCE_VARIANTS: Final = (
    "pilot_standard_analytic_3d",
    "pilot_standard_numerical_3d_x0",
    "pilot_standard_numerical_3d_precalc",
    "pilot_pls_2d",
    "pilot_pls_3d_grouped",
)
_REFERENCE_VARIANTS: Final = _DOWNSTREAM_VARIANTS + _PILOT_EVIDENCE_VARIANTS
_PILOT_VARIANT_DIMS: Final = {
    "pilot_standard_analytic_3d": 3,
    "pilot_standard_numerical_3d_x0": 3,
    "pilot_standard_numerical_3d_precalc": 3,
    "pilot_pls_2d": 2,
    "pilot_pls_3d_grouped": 3,
}
_PILOT_OPTIONAL_FIELDS: Final = {
    "pilot_standard_numerical_3d_x0": {"X0"},
    "pilot_standard_numerical_3d_precalc": {"precalcAlpha"},
}
_REFERENCE_REQUIRED_TOOLBOXES: Final = {
    "MATLAB",
    "Statistics and Machine Learning Toolbox",
    "Optimization Toolbox",
    "Global Optimization Toolbox",
    "Financial Toolbox",
}
_GOLD_MATLAB_COMMIT: Final = "34c01293fef99b4eabd53323c393cb184cc95a8e"
_CANONICAL_DATASET_SHA256: Final = {
    "shared_inputs/reference/metadata.csv": (
        "961c65397b619a6e8e40df0ea6f90fbda448b8deb8a56e5a319e1be8f442bf0c"
    ),
    "shared_inputs/reference/metadata_test.csv": (
        "b1100ac00b60400faf354c95246ec57172bb53ed9963e5fb1b4cf34c613669ae"
    ),
}
_EXPORTER_SCRIPT: Final = "tests/matlab_export/pyis_export_reference_data.m"
_REFERENCE_V2_EXPORTER_SHA256: Final = (
    "7a5d0e26f14fd858770b21f24e0ac028aebb38ed29c84a52691a06b878246182"
)
_BASE_STAGE_VARIANTS: Final = {
    ("build", "prelim", "default"),
    ("build", "sifted", "default"),
    ("build", "pilot", "default"),
    ("build", "cloister", "default"),
}
_OPTION_FIELDS: Final = {
    "general": {"seed", "verbose", "parallel", "ncores"},
    "perf": {"MaxPerf", "AbsPerf", "epsilon", "betaThreshold"},
    "prelim": {"iqrMultiplier", "nanThreshold"},
    "auto": {"preproc"},
    "bound": {"flag"},
    "norm": {"flag"},
    "selvars": {
        "smallscaleflag",
        "smallscale",
        "fileidxflag",
        "fileidx",
        "densityflag",
        "mindistance",
        "type",
    },
    "sifted": {"flag", "rho", "pval", "K", "MaxIter", "Replicates"},
    "pilot": {
        "analytic",
        "ntries",
        "dims",
        "method",
        "alpha",
        "viewGroups",
        "topoWeight",
        "verbose",
    },
    "cloister": {"pval", "corrThreshold", "maxFeatures"},
    "pythia": {
        "flag",
        "kFold",
        "tuning",
        "nTuningIter",
        "params",
        "skip",
        "ispolykrnl",
        "useweights",
        "ensembleMethod",
        "verbose",
        "seed",
        "classifier",
    },
    "trace": {"method", "PI", "minInstances", "minAreaFrac", "contra"},
    "outputs": {"csv", "png", "fig", "web"},
}
_BOOL_OPTION_FIELDS: Final = {
    ("general", "verbose"),
    ("general", "parallel"),
    ("perf", "MaxPerf"),
    ("perf", "AbsPerf"),
    ("auto", "preproc"),
    ("bound", "flag"),
    ("norm", "flag"),
    ("selvars", "smallscaleflag"),
    ("selvars", "fileidxflag"),
    ("selvars", "densityflag"),
    ("sifted", "flag"),
    ("pilot", "analytic"),
    ("pilot", "verbose"),
    ("pythia", "flag"),
    ("pythia", "skip"),
    ("pythia", "ispolykrnl"),
    ("pythia", "useweights"),
    ("pythia", "verbose"),
    ("trace", "contra"),
    ("outputs", "csv"),
    ("outputs", "png"),
    ("outputs", "fig"),
    ("outputs", "web"),
}
_INT_OPTION_FIELDS: Final = {
    ("general", "seed"),
    ("general", "ncores"),
    ("sifted", "K"),
    ("sifted", "MaxIter"),
    ("sifted", "Replicates"),
    ("pilot", "ntries"),
    ("pilot", "dims"),
    ("cloister", "maxFeatures"),
    ("pythia", "kFold"),
    ("pythia", "nTuningIter"),
    ("pythia", "seed"),
    ("trace", "minInstances"),
}
_LIST_OPTION_FIELDS: Final = {
    ("pilot", "viewGroups"),
    ("pilot", "X0"),
    ("pilot", "precalcAlpha"),
    ("pythia", "params"),
}
_TEXT_OPTION_FIELDS: Final = {
    ("selvars", "fileidx"),
    ("selvars", "type"),
    ("pilot", "method"),
    ("pythia", "tuning"),
    ("pythia", "ensembleMethod"),
    ("pythia", "classifier"),
    ("trace", "method"),
}


class ProvenanceError(ValueError):
    """Report an invalid, incomplete, or altered fixture contract."""


@dataclass(frozen=True)
class BundleReport:
    """Summarize one validated fixture bundle."""

    root: Path
    trust: str
    matlab_release: str
    file_count: int
    total_bytes: int


@dataclass(frozen=True)
class InventoryReport:
    """Summarize the trust classes assigned to historical fixtures."""

    file_count: int
    counts: dict[str, int]


def install_verified_bundle(source: Path, destination: Path) -> BundleReport:
    """Atomically install one verified bundle without flattening its layout."""
    source_root = source.resolve()
    report = validate_bundle(source_root)
    if report.trust != VERIFIED_TRUST:
        raise ProvenanceError("Only verified MATLAB bundles can be installed")

    destination_root = destination.resolve()
    if destination_root.exists():
        raise ProvenanceError(
            f"Fixture destination already exists: {destination_root}",
        )
    if destination_root == source_root or destination_root.is_relative_to(source_root):
        raise ProvenanceError("Fixture destination cannot be inside the source bundle")

    destination_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{destination_root.name}.staging-",
            dir=destination_root.parent,
        ),
    )
    try:
        shutil.copytree(source_root, staging, dirs_exist_ok=True)
        installed_report = validate_bundle(staging)
        staging.rename(destination_root)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return BundleReport(
        root=destination_root,
        trust=installed_report.trust,
        matlab_release=installed_report.matlab_release,
        file_count=installed_report.file_count,
        total_bytes=installed_report.total_bytes,
    )


def sha256_file(path: Path) -> str:
    """Return the lowercase SHA-256 digest for a regular file."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_bundle(  # noqa: PLR0912
    root: Path,
    *,
    allow_diagnostic: bool = False,
) -> BundleReport:
    """Validate a generated MATLAB fixture bundle against its manifest."""
    bundle_root = root.resolve()
    if not bundle_root.is_dir():
        raise ProvenanceError(f"Fixture bundle is not a directory: {bundle_root}")

    manifest_path = bundle_root / "manifest.json"
    manifest = _load_object(manifest_path, "fixture manifest")
    _expect_equal(manifest, "schema_version", BUNDLE_SCHEMA)
    profile = _expect_text(manifest, "profile")
    if profile not in {REFERENCE_PROFILE_V1, REFERENCE_PROFILE}:
        raise ProvenanceError(f"Unsupported reference-export profile: {profile!r}")
    trust = _expect_text(manifest, "trust")
    if trust not in {VERIFIED_TRUST, DIAGNOSTIC_TRUST}:
        raise ProvenanceError(f"Unsupported generated fixture trust class: {trust!r}")
    if trust == DIAGNOSTIC_TRUST and not allow_diagnostic:
        raise ProvenanceError(
            "Diagnostic MATLAB fixtures are not accepted as parity oracles. "
            "Pass allow_diagnostic=True only for exporter diagnostics.",
        )

    _validate_timestamp(_expect_text(manifest, "generated_at"))
    _expect_text(manifest, "bundle_id")
    dataset = _expect_object(manifest, "dataset")
    options = _expect_object(manifest, "resolved_options")
    _expect_equal(options, "schema_version", RESOLVED_OPTIONS_INDEX_SCHEMA)

    matlab = _expect_object(manifest, "matlab")
    matlab_release = _expect_text(matlab, "release")
    _parse_release(matlab_release)
    _validate_commit(_expect_text(matlab, "repo_commit"), "matlab.repo_commit")
    repo_dirty = _expect_bool(matlab, "repo_dirty")
    _expect_text(matlab, "toolkit_version")
    _expect_text(matlab, "version")
    _expect_text(matlab, "platform")
    installed = _text_set(
        _expect_list(matlab, "installed_toolboxes"),
        "installed_toolboxes",
    )
    required = _text_set(
        _expect_list(matlab, "required_toolboxes"),
        "required_toolboxes",
    )
    if required != _REFERENCE_REQUIRED_TOOLBOXES:
        raise ProvenanceError(
            "required_toolboxes must name exactly the reference-export dependencies",
        )
    missing_toolboxes = sorted(required - installed)
    if missing_toolboxes:
        raise ProvenanceError(
            "MATLAB fixture environment is missing required toolboxes: "
            + ", ".join(missing_toolboxes),
        )

    generator = _expect_object(manifest, "generator")
    _validate_commit(_expect_text(generator, "repo_commit"), "generator.repo_commit")
    generator_dirty = _expect_bool(generator, "repo_dirty")
    _validate_sha256(
        _expect_text(generator, "script_sha256"),
        "generator.script_sha256",
    )
    _expect_text(generator, "script")

    if trust == VERIFIED_TRUST:
        if repo_dirty or generator_dirty:
            raise ProvenanceError(
                "Verified fixtures require clean MATLAB and generator repositories",
            )
        if profile == REFERENCE_PROFILE and matlab_release != "R2026a":
            raise ProvenanceError(
                "Reference-export/v2 requires MATLAB R2026a, " f"got {matlab_release}",
            )
        if profile == REFERENCE_PROFILE_V1 and _release_key(
            matlab_release,
        ) < _release_key("R2025a"):
            raise ProvenanceError(
                "Verified fixtures require MATLAB R2025a or newer, "
                f"got {matlab_release}",
            )

    entries = _expect_list(manifest, "files")
    if not entries:
        raise ProvenanceError("Fixture manifest must describe at least one file")

    expected_paths: set[str] = set()
    folded_paths: set[str] = set()
    entries_by_path: dict[str, dict[str, Any]] = {}
    total_bytes = 0
    for index, raw_entry in enumerate(entries):
        if not isinstance(raw_entry, dict):
            raise ProvenanceError(f"files[{index}] must be an object")
        entry = cast(dict[str, Any], raw_entry)
        relative = _safe_relative_path(_expect_text(entry, "path"))
        relative_text = relative.as_posix()
        folded = relative_text.casefold()
        if relative_text in expected_paths or folded in folded_paths:
            raise ProvenanceError(
                f"Duplicate or case-colliding fixture path: {relative_text}",
            )
        expected_paths.add(relative_text)
        folded_paths.add(folded)
        entries_by_path[relative_text] = entry

        _validate_sha256(_expect_text(entry, "sha256"), f"files[{index}].sha256")
        size_bytes = _expect_nonnegative_int(entry, "size_bytes")
        total_bytes += size_bytes
        media_type = _expect_text(entry, "media_type")
        _expect_text(entry, "role")
        phase = _expect_text(entry, "phase")
        if phase not in _ALLOWED_PHASES:
            raise ProvenanceError(f"files[{index}].phase is invalid: {phase!r}")
        stage = entry.get("stage")
        if phase == "shared" and stage == "":
            stage = None
        if stage is not None and stage not in _ALLOWED_STAGES:
            raise ProvenanceError(f"files[{index}].stage is invalid: {stage!r}")
        if phase != "shared" and stage is None:
            raise ProvenanceError(
                f"files[{index}] must name a stage for phase {phase!r}",
            )
        _expect_text(entry, "variant")
        empty = _expect_bool(entry, "empty")

        target = bundle_root.joinpath(*relative.parts)
        if target.is_symlink() or not target.is_file():
            raise ProvenanceError(
                "Manifest target is missing, not regular, or a symlink: "
                f"{relative_text}",
            )
        if target.stat().st_size != size_bytes:
            raise ProvenanceError(f"Size mismatch for {relative_text}")
        if sha256_file(target) != entry["sha256"]:
            raise ProvenanceError(f"SHA-256 mismatch for {relative_text}")

        if media_type == "text/csv":
            rows, columns = _csv_shape(target)
            if rows != _expect_nonnegative_int(entry, "rows"):
                raise ProvenanceError(f"CSV row-count mismatch for {relative_text}")
            if columns != _expect_nonnegative_int(entry, "columns"):
                raise ProvenanceError(f"CSV column-count mismatch for {relative_text}")
            if empty != (rows == 0):
                raise ProvenanceError(
                    f"CSV empty flag must match its data-row count for {relative_text}",
                )
            if stage == "trace" and (
                relative.name.startswith(("good_", "best_"))
                or relative.name == "hard.csv"
            ):
                _validate_trace_geometry_csv(target)

    actual_paths = {
        path.relative_to(bundle_root).as_posix()
        for path in bundle_root.rglob("*")
        if path.is_file() and path != manifest_path
    }
    symlinks = [
        path.relative_to(bundle_root).as_posix()
        for path in bundle_root.rglob("*")
        if path.is_symlink()
    ]
    if symlinks:
        raise ProvenanceError(
            "Fixture bundles cannot contain symlinks: " + ", ".join(symlinks),
        )
    missing = sorted(expected_paths - actual_paths)
    extra = sorted(actual_paths - expected_paths)
    if missing or extra:
        raise ProvenanceError(
            f"Manifest file-set mismatch. Missing={missing}; extra={extra}",
        )

    canonical_algorithm_labels: list[str] | None = None
    if trust == VERIFIED_TRUST and profile == REFERENCE_PROFILE:
        canonical_algorithm_labels = _validate_verified_v2_identity(
            bundle_root,
            matlab,
            generator,
        )

    _validate_reference_profile(
        bundle_root,
        dataset,
        options,
        entries_by_path,
        profile=profile,
        canonical_algorithm_labels=canonical_algorithm_labels,
    )

    return BundleReport(
        root=bundle_root,
        trust=trust,
        matlab_release=matlab_release,
        file_count=len(entries),
        total_bytes=total_bytes,
    )


def _validate_reference_profile(  # noqa: PLR0912
    bundle_root: Path,
    dataset: dict[str, Any],
    resolved_index: dict[str, Any],
    entries_by_path: dict[str, dict[str, Any]],
    *,
    profile: str,
    canonical_algorithm_labels: list[str] | None,
) -> None:
    """Enforce the complete, canonical reference-export profile."""
    _expect_equal(resolved_index, "schema_version", RESOLVED_OPTIONS_INDEX_SCHEMA)
    evidence_enabled = profile == REFERENCE_PROFILE
    expected_variants = (
        _REFERENCE_VARIANTS if evidence_enabled else _DOWNSTREAM_VARIANTS
    )
    records = _resolved_option_records(resolved_index, expected_variants)

    _expect_equal(
        dataset,
        "training_input",
        "shared_inputs/reference/metadata.csv",
    )
    _expect_equal(
        dataset,
        "test_input",
        "shared_inputs/reference/metadata_test.csv",
    )
    _expect_text(dataset, "name")
    dataset_seed = _expect_nonnegative_int(dataset, "seed")

    for relative, entry in entries_by_path.items():
        _validate_profile_entry(relative, entry)

    required = (
        _fixed_reference_paths() if evidence_enabled else _fixed_reference_paths_v1()
    )
    missing = sorted(required - entries_by_path.keys())
    if missing:
        raise ProvenanceError(
            f"Reference export profile is incomplete. Missing={missing}",
        )

    options_by_variant: dict[str, dict[str, Any]] = {}
    for variant, record in records.items():
        option_path = _expect_text(record, "path")
        expected_path = f"resolved_options/{variant}.json"
        if option_path != expected_path:
            raise ProvenanceError(
                f"Resolved-options path for {variant!r} must be {expected_path!r}",
            )
        artifact = _load_object(bundle_root / option_path, "resolved options")
        if set(artifact) != {"schema_version", "name", "description", "options"}:
            raise ProvenanceError(
                f"Resolved-options artifact for {variant!r} has an invalid structure",
            )
        _expect_equal(artifact, "schema_version", RESOLVED_OPTIONS_SCHEMA)
        if _expect_text(artifact, "name") != variant:
            raise ProvenanceError(
                f"Resolved-options artifact name does not match {variant!r}",
            )
        if _expect_text(artifact, "description") != _expect_text(
            record,
            "description",
        ):
            raise ProvenanceError(
                f"Resolved-options description does not match {variant!r}",
            )
        effective = _expect_object(artifact, "options")
        _validate_effective_options(effective, variant)
        options_by_variant[variant] = effective

    _validate_variant_option_relationships(options_by_variant, dataset_seed)

    reference_labels: list[str] | None = None
    for variant in _DOWNSTREAM_VARIANTS:
        trace_labels = _read_algorithm_labels(
            bundle_root / f"build_data/trace/{variant}/inputs/algorithm_labels.csv",
        )
        pythia_labels = _read_algorithm_labels(
            bundle_root / f"build_data/pythia/{variant}/inputs/algorithm_labels.csv",
        )
        if trace_labels != pythia_labels:
            raise ProvenanceError(
                f"PYTHIA and TRACE algorithm labels differ for {variant!r}",
            )
        for stage in ("pythia", "trace"):
            explore_labels = _read_algorithm_labels(
                bundle_root
                / f"explore_data/{stage}/{variant}/inputs/algorithm_labels.csv",
            )
            if explore_labels != trace_labels:
                raise ProvenanceError(
                    f"Build and explore algorithm labels differ for {variant!r}",
                )
        if reference_labels is None:
            reference_labels = trace_labels
        elif trace_labels != reference_labels:
            raise ProvenanceError("Reference variants use different algorithm labels")

    assert reference_labels is not None
    prelim_labels = _read_algorithm_labels(
        bundle_root / "build_data/prelim/default/inputs/algorithm_labels.csv",
    )
    if prelim_labels != reference_labels:
        raise ProvenanceError("PRELIM and downstream algorithm labels differ")
    if (
        canonical_algorithm_labels is not None
        and reference_labels != canonical_algorithm_labels
    ):
        raise ProvenanceError(
            "Downstream algorithm labels do not match the canonical metadata headers",
        )

    if evidence_enabled:
        _validate_pilot_evidence_profile(
            bundle_root,
            options_by_variant,
            reference_labels,
        )

    required.update(_geometry_paths(reference_labels))
    missing = sorted(required - entries_by_path.keys())
    extra = sorted(entries_by_path.keys() - required)
    if missing or extra:
        raise ProvenanceError(
            "Reference export profile file-set mismatch. "
            f"Missing={missing}; extra={extra}",
        )


def _validate_verified_v2_identity(
    bundle_root: Path,
    matlab: dict[str, Any],
    generator: dict[str, Any],
) -> list[str]:
    """Pin a verified v2 oracle to the audited source, data, and exporter."""
    matlab_commit = _expect_text(matlab, "repo_commit")
    if matlab_commit != _GOLD_MATLAB_COMMIT:
        raise ProvenanceError(
            "Verified v2 fixtures must use the gold MATLAB commit "
            f"{_GOLD_MATLAB_COMMIT}",
        )

    for relative, expected_hash in _CANONICAL_DATASET_SHA256.items():
        target = bundle_root / relative
        if not target.is_file():
            raise ProvenanceError(
                "Reference export profile is missing the canonical dataset: "
                f"{relative}",
            )
        actual_hash = sha256_file(target)
        if actual_hash != expected_hash:
            raise ProvenanceError(
                f"Verified v2 fixtures must use the canonical dataset: {relative}",
            )

    _expect_equal(generator, "script", _EXPORTER_SCRIPT)
    if _expect_text(generator, "script_sha256") != _REFERENCE_V2_EXPORTER_SHA256:
        raise ProvenanceError(
            "Verified v2 fixtures do not match the pinned exporter script hash",
        )

    training_labels = _read_metadata_algorithm_labels(
        bundle_root / "shared_inputs/reference/metadata.csv",
    )
    test_labels = _read_metadata_algorithm_labels(
        bundle_root / "shared_inputs/reference/metadata_test.csv",
    )
    if training_labels != test_labels:
        raise ProvenanceError(
            "Canonical training and test metadata use different algorithm headers",
        )
    return training_labels


def _read_metadata_algorithm_labels(path: Path) -> list[str]:
    header, _ = _read_csv_rows(path)
    labels = [
        column[len("algo_") :]
        for column in header
        if column.casefold().startswith("algo_")
    ]
    if not labels or any(not label for label in labels):
        raise ProvenanceError(f"Metadata has no valid algorithm headers: {path}")
    if len({label.casefold() for label in labels}) != len(labels):
        raise ProvenanceError(f"Metadata has duplicate algorithm headers: {path}")
    return labels


def _resolved_option_records(
    resolved_index: dict[str, Any],
    expected_variants: tuple[str, ...],
) -> dict[str, dict[str, Any]]:
    if set(resolved_index) != {"schema_version", "variants"}:
        raise ProvenanceError("resolved_options index has an invalid structure")
    records: dict[str, dict[str, Any]] = {}
    for index, value in enumerate(_expect_list(resolved_index, "variants")):
        if not isinstance(value, dict):
            raise ProvenanceError(
                f"resolved_options.variants[{index}] must be an object",
            )
        record = cast(dict[str, Any], value)
        if set(record) != {"name", "description", "path"}:
            raise ProvenanceError(
                f"resolved_options.variants[{index}] has an invalid structure",
            )
        name = _expect_text(record, "name")
        if name in records:
            raise ProvenanceError(f"Duplicate resolved-options variant: {name!r}")
        records[name] = record
    if set(records) != set(expected_variants):
        raise ProvenanceError(
            "resolved_options.variants must name exactly "
            f"{list(expected_variants)!r}",
        )
    return records


def _validate_profile_entry(  # noqa: PLR0912
    relative: str,
    entry: dict[str, Any],
) -> None:
    path = PurePosixPath(relative)
    parts = path.parts
    expected_stage: str | None
    if parts[0] == "shared_inputs":
        if len(parts) != _SHARED_INPUT_PATH_DEPTH or parts[1] != "reference":
            raise ProvenanceError(f"Noncanonical reference-export path: {relative}")
        expected_phase, expected_stage, expected_variant = "shared", None, "reference"
    elif parts[0] == "resolved_options":
        if len(parts) != _RESOLVED_OPTION_PATH_DEPTH or path.suffix != ".json":
            raise ProvenanceError(f"Noncanonical reference-export path: {relative}")
        expected_phase, expected_stage, expected_variant = "shared", None, path.stem
    elif parts[0] in {"build_data", "explore_data"}:
        if len(parts) != _STAGE_PATH_DEPTH or parts[3] not in {"inputs", "outputs"}:
            raise ProvenanceError(f"Noncanonical reference-export path: {relative}")
        expected_phase = parts[0].removesuffix("_data")
        expected_stage, expected_variant = parts[1], parts[2]
        stage_variant = (expected_phase, expected_stage, expected_variant)
        downstream = (
            expected_phase in {"build", "explore"}
            and expected_stage in {"pythia", "trace"}
            and expected_variant in _DOWNSTREAM_VARIANTS
        )
        pilot_evidence = (
            expected_phase in {"build", "explore"}
            and expected_stage == "pilot"
            and expected_variant in _PILOT_EVIDENCE_VARIANTS
        )
        if (
            stage_variant not in _BASE_STAGE_VARIANTS
            and not downstream
            and not pilot_evidence
        ):
            raise ProvenanceError(
                f"Unsupported reference-export stage/variant: {stage_variant!r}",
            )
        if expected_phase == "explore" and expected_stage not in {
            "pilot",
            "pythia",
            "trace",
        }:
            raise ProvenanceError(f"Unsupported explore stage: {expected_stage!r}")
    else:
        raise ProvenanceError(f"Noncanonical reference-export path: {relative}")

    actual_stage = entry.get("stage")
    if actual_stage == "":
        actual_stage = None
    actual = (
        _expect_text(entry, "phase"),
        actual_stage,
        _expect_text(entry, "variant"),
    )
    expected = (expected_phase, expected_stage, expected_variant)
    if actual != expected:
        raise ProvenanceError(
            f"Manifest metadata for {relative} must be {expected!r}, got {actual!r}",
        )
    expected_media = "application/json" if path.suffix == ".json" else "text/csv"
    if _expect_text(entry, "media_type") != expected_media:
        raise ProvenanceError(f"Manifest media type does not match {relative}")
    if _expect_text(entry, "role") != relative:
        raise ProvenanceError(f"Manifest role does not match {relative}")


def _validate_effective_options(options: dict[str, Any], variant: str) -> None:
    if set(options) != set(_OPTION_FIELDS):
        raise ProvenanceError(
            f"Resolved options for {variant!r} do not contain the exact option groups",
        )
    for group, expected_fields in _OPTION_FIELDS.items():
        values = _expect_object(options, group)
        variant_fields = set(expected_fields)
        if group == "pilot":
            variant_fields.update(_PILOT_OPTIONAL_FIELDS.get(variant, set()))
        if set(values) != variant_fields:
            raise ProvenanceError(
                f"Resolved options {variant!r}.{group} do not match the MATLAB schema",
            )
        for field, value in values.items():
            key = (group, field)
            if key in _BOOL_OPTION_FIELDS:
                valid = isinstance(value, bool)
            elif key in _INT_OPTION_FIELDS:
                valid = isinstance(value, int) and not isinstance(value, bool)
            elif key in _LIST_OPTION_FIELDS:
                valid = isinstance(value, list)
            elif key in _TEXT_OPTION_FIELDS:
                valid = isinstance(value, str) and (field == "fileidx" or bool(value))
            else:
                valid = (
                    isinstance(value, int | float)
                    and not isinstance(value, bool)
                    and math.isfinite(value)
                )
            if not valid:
                raise ProvenanceError(
                    f"Resolved option {variant!r}.{group}.{field} has an invalid type",
                )


def _validate_variant_option_relationships(  # noqa: PLR0912
    options_by_variant: dict[str, dict[str, Any]],
    dataset_seed: int,
) -> None:
    baseline = options_by_variant["trace3_default"]
    if baseline["general"]["seed"] != dataset_seed:
        raise ProvenanceError("Resolved general.seed does not match dataset.seed")
    if baseline["general"]["parallel"] or baseline["general"]["verbose"]:
        raise ProvenanceError(
            "Reference export must disable parallel and verbose modes",
        )
    if baseline["pilot"]["dims"] != 2:  # noqa: PLR2004
        raise ProvenanceError(
            "Reference export profile requires a two-dimensional PILOT",
        )
    if any(baseline["outputs"].values()):
        raise ProvenanceError("Reference export must disable toolkit output writers")

    for variant in _DOWNSTREAM_VARIANTS[1:]:
        effective = options_by_variant[variant]
        for group in set(_OPTION_FIELDS) - {"pythia", "trace"}:
            if effective[group] != baseline[group]:
                raise ProvenanceError(
                    f"Resolved option group {group!r} differs in {variant!r}",
                )

    expected_changes: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {
        "trace3_default": (
            {"classifier": "knn", "tuning": "sobol", "skip": False},
            {"method": "trace3", "PI": 0.6, "contra": False},
        ),
        "trace3_pythia_skip": (
            {"classifier": "knn", "tuning": "sobol", "skip": True},
            {"method": "trace3", "PI": 0.6, "contra": False},
        ),
        "legacy_svm": (
            {"classifier": "svm", "tuning": "sobol", "skip": False},
            {"method": "legacy", "PI": 0.55, "contra": True},
        ),
    }
    for variant, (pythia_changes, trace_changes) in expected_changes.items():
        effective = options_by_variant[variant]
        expected_pythia = dict(baseline["pythia"])
        expected_pythia.update(pythia_changes)
        expected_trace = dict(baseline["trace"])
        expected_trace.update(trace_changes)
        if effective["pythia"] != expected_pythia:
            raise ProvenanceError(f"Resolved PYTHIA options mismatch for {variant!r}")
        if effective["trace"] != expected_trace:
            raise ProvenanceError(f"Resolved TRACE options mismatch for {variant!r}")

    if set(options_by_variant) == set(_DOWNSTREAM_VARIANTS):
        return

    expected_pilot_settings = {
        "pilot_standard_analytic_3d": ("standard", 3, True),
        "pilot_standard_numerical_3d_x0": ("standard", 3, False),
        "pilot_standard_numerical_3d_precalc": ("standard", 3, False),
        "pilot_pls_2d": ("pls", 2, False),
        "pilot_pls_3d_grouped": ("pls", 3, True),
    }
    for variant, (method, dims, analytic) in expected_pilot_settings.items():
        effective = options_by_variant[variant]
        for group in set(_OPTION_FIELDS) - {"pilot", "pythia", "trace"}:
            if effective[group] != baseline[group]:
                raise ProvenanceError(
                    f"Resolved option group {group!r} differs in {variant!r}",
                )

        expected_pythia = dict(baseline["pythia"])
        expected_pythia["skip"] = True
        if effective["pythia"] != expected_pythia:
            raise ProvenanceError(f"Resolved PYTHIA options mismatch for {variant!r}")
        if effective["trace"] != baseline["trace"]:
            raise ProvenanceError(f"Resolved TRACE options mismatch for {variant!r}")

        pilot = effective["pilot"]
        for key in _OPTION_FIELDS["pilot"] - {
            "method",
            "dims",
            "analytic",
            "ntries",
            "viewGroups",
            "alpha",
        }:
            if pilot[key] != baseline["pilot"][key]:
                raise ProvenanceError(
                    f"Resolved PILOT option {key!r} differs in {variant!r}",
                )
        if (
            pilot["method"] != method
            or pilot["dims"] != dims
            or pilot["analytic"] is not analytic
            or pilot["ntries"] != 1
            or pilot["alpha"]
            != (
                3.0 if variant == "pilot_pls_3d_grouped" else baseline["pilot"]["alpha"]
            )
        ):
            raise ProvenanceError(f"Resolved PILOT options mismatch for {variant!r}")
        grouped = variant == "pilot_pls_3d_grouped"
        if grouped != bool(pilot["viewGroups"]):
            raise ProvenanceError(
                f"Resolved PILOT viewpoint groups mismatch for {variant!r}",
            )


def _read_csv_rows(path: Path) -> tuple[list[str], list[list[str]]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream)
        try:
            header = next(reader)
        except StopIteration as error:
            raise ProvenanceError(f"CSV has no header: {path}") from error
        rows = list(reader)
    if any(len(row) != len(header) for row in rows):
        raise ProvenanceError(f"CSV has ragged rows: {path}")
    return header, rows


def _read_numeric_csv(
    path: Path,
    *,
    expected_header: list[str] | None = None,
    row_labels: bool = False,
) -> list[list[float]]:
    header, rows = _read_csv_rows(path)
    if expected_header is not None and header != expected_header:
        raise ProvenanceError(f"CSV has an invalid dimensional header: {path}")
    offset = 1 if row_labels else 0
    values: list[list[float]] = []
    for row_number, row in enumerate(rows, start=2):
        try:
            numeric = [float(value) for value in row[offset:]]
        except ValueError as error:
            raise ProvenanceError(
                f"CSV row {row_number} is not numeric: {path}",
            ) from error
        if not all(math.isfinite(value) for value in numeric):
            raise ProvenanceError(
                f"CSV row {row_number} contains non-finite data: {path}",
            )
        values.append(numeric)
    return values


def _json_numeric_matrix(value: object, name: str) -> list[list[float]]:
    if not isinstance(value, list) or not value:
        raise ProvenanceError(f"{name} must be a nonempty numeric matrix")
    if all(
        isinstance(item, int | float) and not isinstance(item, bool) for item in value
    ):
        rows = [[float(item)] for item in value]
    elif all(isinstance(item, list) and item for item in value):
        nested = cast(list[list[Any]], value)
        width = len(nested[0])
        if any(len(row) != width for row in nested):
            raise ProvenanceError(f"{name} must be a rectangular numeric matrix")
        rows = []
        for row in nested:
            if not all(
                isinstance(item, int | float) and not isinstance(item, bool)
                for item in row
            ):
                raise ProvenanceError(f"{name} must be a numeric matrix")
            rows.append([float(item) for item in row])
    else:
        raise ProvenanceError(f"{name} must be a numeric matrix")
    if not all(math.isfinite(item) for row in rows for item in row):
        raise ProvenanceError(f"{name} must contain finite values")
    return rows


def _matrices_equal(
    left: list[list[float]],
    right: list[list[float]],
) -> bool:
    return len(left) == len(right) and all(
        len(left_row) == len(right_row)
        and all(a == b for a, b in zip(left_row, right_row, strict=True))
        for left_row, right_row in zip(left, right, strict=True)
    )


def _matrices_close(
    left: list[list[float]],
    right: list[list[float]],
    *,
    tolerance: float = 1e-11,
) -> bool:
    return len(left) == len(right) and all(
        len(left_row) == len(right_row)
        and all(
            math.isclose(a, b, rel_tol=tolerance, abs_tol=tolerance)
            for a, b in zip(left_row, right_row, strict=True)
        )
        for left_row, right_row in zip(left, right, strict=True)
    )


def _transpose(matrix: list[list[float]]) -> list[list[float]]:
    return [list(column) for column in zip(*matrix, strict=True)]


def _matmul(
    left: list[list[float]],
    right: list[list[float]],
) -> list[list[float]]:
    right_columns = _transpose(right)
    return [
        [
            sum(a * b for a, b in zip(row, column, strict=True))
            for column in right_columns
        ]
        for row in left
    ]


def _column_means(matrix: list[list[float]]) -> list[float]:
    return [sum(column) / len(matrix) for column in zip(*matrix, strict=True)]


def _center(matrix: list[list[float]]) -> list[list[float]]:
    means = _column_means(matrix)
    return [[value - means[index] for index, value in enumerate(row)] for row in matrix]


def _correlation_squared(left: list[float], right: list[float]) -> float:
    return _correlation(left, right) ** 2


def _correlation(left: list[float], right: list[float]) -> float:
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    left_delta = [value - left_mean for value in left]
    right_delta = [value - right_mean for value in right]
    numerator = sum(a * b for a, b in zip(left_delta, right_delta, strict=True))
    denominator = math.sqrt(
        sum(value * value for value in left_delta)
        * sum(value * value for value in right_delta),
    )
    if denominator == 0:
        raise ProvenanceError("PILOT correlation evidence has a constant vector")
    return numerator / denominator


def _pairwise_distances(matrix: list[list[float]]) -> list[float]:
    return [
        math.sqrt(
            sum(
                (left_value - right_value) ** 2
                for left_value, right_value in zip(left, right, strict=True)
            ),
        )
        for left_index, left in enumerate(matrix)
        for right in matrix[left_index + 1 :]
    ]


def _decode_pilot_solution(
    theta: list[float],
    dims: int,
    n_features: int,
    n_algorithms: int,
) -> tuple[list[list[float]], list[list[float]], list[list[float]]]:
    split = dims * n_features
    a_flat = theta[:split]
    b_flat = theta[split:]
    total_columns = n_features + n_algorithms
    a = [
        [a_flat[column * dims + row] for column in range(n_features)]
        for row in range(dims)
    ]
    combined = [
        [b_flat[dimension * total_columns + row] for dimension in range(dims)]
        for row in range(total_columns)
    ]
    b = combined[:n_features]
    c = _transpose(combined[n_features:])
    return a, b, c


def _pilot_numerical_trial_metrics(
    alpha: list[list[float]],
    x: list[list[float]],
    y: list[list[float]],
    dims: int,
    cost_weight: float,
) -> tuple[list[float], list[float]]:
    """Recompute MATLAB PILOT's per-trial loss and topology score."""
    n_features = len(x[0])
    n_algorithms = len(y[0])
    trial_count = len(alpha[0])
    x_bar = [[*x_row, *y_row] for x_row, y_row in zip(x, y, strict=True)]
    source_distances = _pairwise_distances(x)
    objectives: list[float] = []
    topology_scores: list[float] = []

    for trial in range(trial_count):
        theta = [row[trial] for row in alpha]
        a, b, c = _decode_pilot_solution(
            theta,
            dims,
            n_features,
            n_algorithms,
        )
        z = _matmul(x, _transpose(a))
        reconstruction_factors = [*b, *_transpose(c)]
        reconstructed = _matmul(z, _transpose(reconstruction_factors))
        column_errors = [
            sum(
                (actual[column] - estimate[column]) ** 2
                for actual, estimate in zip(x_bar, reconstructed, strict=True)
            )
            / len(x_bar)
            for column in range(n_features + n_algorithms)
        ]
        weighted = [
            value * (1.0 if column < n_features else cost_weight)
            for column, value in enumerate(column_errors)
        ]
        objectives.append(sum(weighted) / len(weighted))
        topology_scores.append(
            _correlation(source_distances, _pairwise_distances(z)),
        )

    return objectives, topology_scores


def _validate_pilot_evidence_profile(  # noqa: PLR0912
    bundle_root: Path,
    options_by_variant: dict[str, dict[str, Any]],
    algorithm_labels: list[str],
) -> None:
    """Validate the exact dimensional and solver-input PILOT evidence contract."""
    x_by_variant: dict[str, list[list[float]]] = {}
    y_by_variant: dict[str, list[list[float]]] = {}
    outputs_by_variant: dict[str, dict[str, list[list[float]]]] = {}
    recomputed_x0_perf: list[float] | None = None
    for variant in _PILOT_EVIDENCE_VARIANTS:
        dims = _PILOT_VARIANT_DIMS[variant]
        coordinate_header = [f"z_{index}" for index in range(1, dims + 1)]
        build_root = bundle_root / "build_data" / "pilot" / variant
        explore_root = bundle_root / "explore_data" / "pilot" / variant

        feature_header, feature_rows = _read_csv_rows(
            build_root / "inputs" / "feature_labels.csv",
        )
        if (
            feature_header != ["feature_name"]
            or not feature_rows
            or any(len(row) != 1 or not row[0] for row in feature_rows)
        ):
            raise ProvenanceError(
                f"PILOT feature labels have an invalid schema for {variant!r}",
            )
        feature_labels = [row[0] for row in feature_rows]
        n_features = len(feature_labels)
        context = _load_object(
            build_root / "inputs" / "stage_context.json",
            "PILOT stage context",
        )
        if set(context) != {
            "schema_version",
            "scope",
            "upstream_snapshot",
            "sifted_effective_pilot_dims",
            "input_transform",
            "feature_shift",
            "algorithm_shift",
            "explore_projection",
        }:
            raise ProvenanceError(f"PILOT stage context mismatch for {variant!r}")
        _expect_equal(
            context,
            "schema_version",
            "pyinstancespace.pilot-evidence-context/v1",
        )
        _expect_equal(context, "scope", "pilot-stage")
        _expect_equal(
            context,
            "upstream_snapshot",
            "build_data/pilot/default/inputs",
        )
        if context.get("sifted_effective_pilot_dims") != 2:  # noqa: PLR2004
            raise ProvenanceError(f"PILOT upstream dimensions mismatch for {variant!r}")
        _expect_equal(
            context,
            "explore_projection",
            "InstanceSpace.explore: Z=X*A' (uncentred)",
        )
        is_pls = variant.startswith("pilot_pls_")
        expected_feature_shift = (
            [0.25 * index for index in range(1, n_features + 1)] if is_pls else []
        )
        expected_algorithm_shift = (
            [0.4 * index for index in range(1, len(algorithm_labels) + 1)]
            if is_pls
            else []
        )
        expected_transform = "deterministic-column-shift" if is_pls else "none"
        if (
            context.get("input_transform") != expected_transform
            or context.get("feature_shift") != expected_feature_shift
            or context.get("algorithm_shift") != expected_algorithm_shift
        ):
            raise ProvenanceError(f"PILOT input transform mismatch for {variant!r}")
        x = _read_numeric_csv(
            build_root / "inputs" / "x.csv",
            expected_header=["Row", *feature_labels],
            row_labels=True,
        )
        y = _read_numeric_csv(
            build_root / "inputs" / "y.csv",
            expected_header=["Row", *algorithm_labels],
            row_labels=True,
        )
        if not x or len(x) != len(y):
            raise ProvenanceError(f"PILOT build inputs mismatch for {variant!r}")
        x_by_variant[variant] = x
        y_by_variant[variant] = y

        common_shapes = {
            "pilot_a_raw.csv": (dims, n_features),
            "pilot_b.csv": (n_features, dims),
            "pilot_c.csv": (dims, len(algorithm_labels)),
            "pilot_z.csv": (len(x), dims),
            "pilot_r2.csv": (n_features + len(algorithm_labels), 1),
            "pilot_error.csv": (1, 1),
        }
        output_matrices: dict[str, list[list[float]]] = {}
        for filename, expected_shape in common_shapes.items():
            expected_headers = {
                "pilot_a_raw.csv": [
                    f"col_{index}" for index in range(1, n_features + 1)
                ],
                "pilot_b.csv": [f"col_{index}" for index in range(1, dims + 1)],
                "pilot_c.csv": [
                    f"col_{index}" for index in range(1, len(algorithm_labels) + 1)
                ],
                "pilot_z.csv": coordinate_header,
                "pilot_r2.csv": ["r2"],
                "pilot_error.csv": ["error"],
            }
            matrix = _read_numeric_csv(
                build_root / "outputs" / filename,
                expected_header=expected_headers[filename],
            )
            actual_shape = (len(matrix), len(matrix[0]) if matrix else 0)
            if actual_shape != expected_shape:
                raise ProvenanceError(
                    f"PILOT artifact {filename!r} has the wrong shape for {variant!r}",
                )
            output_matrices[filename] = matrix
        outputs_by_variant[variant] = output_matrices

        pilot_matrix_header, pilot_matrix_rows = _read_csv_rows(
            build_root / "outputs" / "pilot_matrix.csv",
        )
        if (
            pilot_matrix_header != ["Row", *feature_labels]
            or len(
                pilot_matrix_rows,
            )
            != dims
        ):
            raise ProvenanceError(
                f"PILOT summary matrix has the wrong dimensions for {variant!r}",
            )
        summary_values: list[list[float]] = []
        for index, row in enumerate(pilot_matrix_rows, start=1):
            if row[0] != f"Z_{{{index}}}":
                raise ProvenanceError(
                    f"PILOT summary labels mismatch for {variant!r}",
                )
            try:
                summary_values.append([float(value) for value in row[1:]])
            except ValueError as error:
                raise ProvenanceError(
                    f"PILOT summary is not numeric for {variant!r}",
                ) from error
        rounded_a = [
            [round(value, 4) for value in row]
            for row in output_matrices["pilot_a_raw.csv"]
        ]
        if not _matrices_close(summary_values, rounded_a, tolerance=1e-12):
            raise ProvenanceError(f"PILOT summary values mismatch for {variant!r}")

        projection_input = _center(x) if variant.startswith("pilot_pls_") else x
        expected_z = _matmul(
            projection_input,
            _transpose(output_matrices["pilot_a_raw.csv"]),
        )
        if not _matrices_close(expected_z, output_matrices["pilot_z.csv"]):
            raise ProvenanceError(f"PILOT build projection mismatch for {variant!r}")

        x_bar = [[*x_row, *y_row] for x_row, y_row in zip(x, y, strict=True)]
        reconstruction_factors = [
            *output_matrices["pilot_b.csv"],
            *_transpose(output_matrices["pilot_c.csv"]),
        ]
        reconstructed = _matmul(
            output_matrices["pilot_z.csv"],
            _transpose(reconstruction_factors),
        )
        if variant.startswith("pilot_pls_"):
            means = _column_means(x_bar)
            reconstructed = [
                [value + means[index] for index, value in enumerate(row)]
                for row in reconstructed
            ]
            if max(abs(value) for value in means) < 0.1:  # noqa: PLR2004
                raise ProvenanceError("PLS evidence does not exercise centring")
        expected_error = sum(
            (actual - estimate) ** 2
            for actual_row, estimate_row in zip(x_bar, reconstructed, strict=True)
            for actual, estimate in zip(actual_row, estimate_row, strict=True)
        )
        actual_error = output_matrices["pilot_error.csv"][0][0]
        if not math.isclose(expected_error, actual_error, rel_tol=1e-10, abs_tol=1e-10):
            raise ProvenanceError(
                f"PILOT reconstruction error mismatch for {variant!r}",
            )
        actual_r2 = [row[0] for row in output_matrices["pilot_r2.csv"]]
        expected_r2 = [
            _correlation_squared(list(actual), list(estimate))
            for actual, estimate in zip(
                zip(*x_bar, strict=True),
                zip(*reconstructed, strict=True),
                strict=True,
            )
        ]
        if not _matrices_close(
            [[value] for value in expected_r2],
            [[value] for value in actual_r2],
        ):
            raise ProvenanceError(f"PILOT R2 mismatch for {variant!r}")

        pilot_options = options_by_variant[variant]["pilot"]
        solver_rows = dims * (2 * n_features + len(algorithm_labels))
        if variant == "pilot_standard_numerical_3d_x0":
            option_x0 = _json_numeric_matrix(pilot_options["X0"], f"{variant}.X0")
            trial_header = [f"col_{index}" for index in range(1, len(option_x0[0]) + 1)]
            input_x0 = _read_numeric_csv(
                build_root / "inputs" / "x0.csv",
                expected_header=trial_header,
            )
            output_x0 = _read_numeric_csv(
                build_root / "outputs" / "pilot_x0.csv",
                expected_header=trial_header,
            )
            if (
                len(option_x0) != solver_rows
                or not option_x0
                or len(option_x0[0]) != 3  # noqa: PLR2004
                or not _matrices_close(option_x0, input_x0, tolerance=1e-14)
                or not _matrices_close(option_x0, output_x0, tolerance=1e-14)
            ):
                raise ProvenanceError("PILOT X0 evidence is inconsistent")
            alpha = _read_numeric_csv(
                build_root / "outputs" / "pilot_alpha.csv",
                expected_header=trial_header,
            )
            eoptim = _read_numeric_csv(
                build_root / "outputs" / "pilot_eoptim.csv",
                expected_header=["eoptim"],
            )
            perf = _read_numeric_csv(
                build_root / "outputs" / "pilot_perf.csv",
                expected_header=["perf"],
            )
            if (
                len(alpha) != solver_rows
                or len(alpha[0]) != len(option_x0[0])
                or len(eoptim) != len(option_x0[0])
                or len(perf) != len(option_x0[0])
            ):
                raise ProvenanceError("PILOT numerical diagnostics are incomplete")
            recomputed_eoptim, recomputed_perf = _pilot_numerical_trial_metrics(
                alpha,
                x,
                y,
                dims,
                float(pilot_options["alpha"]),
            )
            if not _matrices_close(
                [[value] for value in recomputed_eoptim],
                eoptim,
                tolerance=1e-12,
            ):
                raise ProvenanceError("PILOT numerical trial objective mismatch")
            if not _matrices_close(
                [[value] for value in recomputed_perf],
                perf,
                tolerance=1e-12,
            ):
                raise ProvenanceError("PILOT numerical trial topology mismatch")
            recomputed_x0_perf = recomputed_perf
        elif variant == "pilot_standard_numerical_3d_precalc":
            option_alpha = _json_numeric_matrix(
                pilot_options["precalcAlpha"],
                f"{variant}.precalcAlpha",
            )
            input_alpha = _read_numeric_csv(
                build_root / "inputs" / "precalc_alpha.csv",
                expected_header=["precalc_alpha"],
            )
            output_alpha = _read_numeric_csv(
                build_root / "outputs" / "pilot_alpha.csv",
                expected_header=["col_1"],
            )
            if (
                len(option_alpha) != solver_rows
                or len(option_alpha[0]) != 1
                or not _matrices_close(option_alpha, input_alpha, tolerance=1e-14)
                or not _matrices_close(option_alpha, output_alpha, tolerance=1e-14)
            ):
                raise ProvenanceError("PILOT precalculated evidence is inconsistent")

        view_groups = pilot_options["viewGroups"]
        if dims == 3:  # noqa: PLR2004
            _validate_pilot_viewpoint_artifacts(
                build_root / "outputs",
                view_groups,
                algorithm_labels,
            )

        explore_x = _read_numeric_csv(
            explore_root / "inputs" / "x.csv",
            expected_header=["Row", *feature_labels],
            row_labels=True,
        )
        projection = _read_numeric_csv(
            explore_root / "inputs" / "projection_a.csv",
            expected_header=["Row", *feature_labels],
            row_labels=True,
        )
        explore_z = _read_numeric_csv(
            explore_root / "outputs" / "pilot_z.csv",
            expected_header=["Row", *coordinate_header],
            row_labels=True,
        )
        explore_x_rows = _read_csv_rows(explore_root / "inputs" / "x.csv")[1]
        explore_z_rows = _read_csv_rows(
            explore_root / "outputs" / "pilot_z.csv",
        )[1]
        projection_rows = _read_csv_rows(
            explore_root / "inputs" / "projection_a.csv",
        )[1]
        if (
            len(projection) != dims
            or len(explore_x) != len(explore_z)
            or not explore_x
            or any(len(row) != dims for row in explore_z)
            or not _matrices_close(
                projection,
                output_matrices["pilot_a_raw.csv"],
            )
        ):
            raise ProvenanceError(f"PILOT explore evidence mismatch for {variant!r}")
        if [row[0] for row in explore_x_rows] != [row[0] for row in explore_z_rows] or [
            row[0] for row in projection_rows
        ] != coordinate_header:
            raise ProvenanceError(
                f"PILOT explore row labels mismatch for {variant!r}",
            )
        projected = [
            [
                sum(
                    value * projection[dimension][index]
                    for index, value in enumerate(row)
                )
                for dimension in range(dims)
            ]
            for row in explore_x
        ]
        if not _matrices_close(projected, explore_z):
            raise ProvenanceError(
                f"PILOT explore projection is not reproducible for {variant!r}",
            )

    standard_variants = (
        "pilot_standard_analytic_3d",
        "pilot_standard_numerical_3d_x0",
        "pilot_standard_numerical_3d_precalc",
    )
    standard_x = x_by_variant[standard_variants[0]]
    standard_y = y_by_variant[standard_variants[0]]
    if any(
        not _matrices_equal(x_by_variant[variant], standard_x)
        or not _matrices_equal(y_by_variant[variant], standard_y)
        for variant in standard_variants[1:]
    ):
        raise ProvenanceError("Standard PILOT evidence variants use different inputs")
    default_inputs = bundle_root / "build_data/pilot/default/inputs"
    evidence_inputs = bundle_root / "build_data/pilot/pilot_standard_analytic_3d/inputs"
    for filename in ("x.csv", "y.csv", "feature_labels.csv"):
        if (default_inputs / filename).read_bytes() != (
            evidence_inputs / filename
        ).read_bytes():
            raise ProvenanceError("PILOT evidence is not linked to the SIFTED snapshot")
    default_feature_labels = (default_inputs / "feature_labels.csv").read_bytes()
    default_x_rows = _read_csv_rows(default_inputs / "x.csv")[1]
    default_y_rows = _read_csv_rows(default_inputs / "y.csv")[1]
    for variant in _PILOT_EVIDENCE_VARIANTS:
        inputs = bundle_root / "build_data" / "pilot" / variant / "inputs"
        if (inputs / "feature_labels.csv").read_bytes() != default_feature_labels:
            raise ProvenanceError("PILOT evidence variants use different features")
        variant_x_rows = _read_csv_rows(inputs / "x.csv")[1]
        variant_y_rows = _read_csv_rows(inputs / "y.csv")[1]
        if [row[0] for row in variant_x_rows] != [row[0] for row in default_x_rows] or [
            row[0] for row in variant_y_rows
        ] != [row[0] for row in default_y_rows]:
            raise ProvenanceError("PILOT evidence variants use different instances")

    pls_2d = "pilot_pls_2d"
    pls_3d = "pilot_pls_3d_grouped"
    if not _matrices_equal(x_by_variant[pls_2d], x_by_variant[pls_3d]) or not (
        _matrices_equal(y_by_variant[pls_2d], y_by_variant[pls_3d])
    ):
        raise ProvenanceError("PLS evidence variants use different shifted inputs")
    expected_pls_x = [
        [value + 0.25 * (index + 1) for index, value in enumerate(row)]
        for row in standard_x
    ]
    expected_pls_y = [
        [value + 0.4 * (index + 1) for index, value in enumerate(row)]
        for row in standard_y
    ]
    if not _matrices_close(x_by_variant[pls_2d], expected_pls_x) or not (
        _matrices_close(y_by_variant[pls_2d], expected_pls_y)
    ):
        raise ProvenanceError("PLS evidence shift is not reproducible")

    pls_2d_outputs = outputs_by_variant[pls_2d]
    pls_3d_outputs = outputs_by_variant[pls_3d]
    component_relations = {
        "pilot_a_raw.csv": pls_3d_outputs["pilot_a_raw.csv"][:2],
        "pilot_b.csv": [row[:2] for row in pls_3d_outputs["pilot_b.csv"]],
        "pilot_c.csv": pls_3d_outputs["pilot_c.csv"][:2],
        "pilot_z.csv": [row[:2] for row in pls_3d_outputs["pilot_z.csv"]],
    }
    for filename, expected in component_relations.items():
        if not _matrices_close(pls_2d_outputs[filename], expected):
            raise ProvenanceError(
                f"PILOT PLS 2D/3D component relation mismatch in {filename!r}",
            )

    x0_outputs = bundle_root / "build_data/pilot/pilot_standard_numerical_3d_x0/outputs"
    precalc_root = bundle_root / "build_data/pilot/pilot_standard_numerical_3d_precalc"
    x0_alpha = _read_numeric_csv(x0_outputs / "pilot_alpha.csv")
    if recomputed_x0_perf is None:
        raise ProvenanceError("PILOT numerical trial metrics were not validated")
    best_index = max(
        range(len(recomputed_x0_perf)),
        key=recomputed_x0_perf.__getitem__,
    )
    selected = [[row[best_index]] for row in x0_alpha]
    replayed = _read_numeric_csv(precalc_root / "inputs/precalc_alpha.csv")
    if not _matrices_close(selected, replayed, tolerance=1e-14):
        raise ProvenanceError(
            "PILOT precalculated evidence is not the best exported X0 solution",
        )
    decoded_a, decoded_b, decoded_c = _decode_pilot_solution(
        [row[0] for row in selected],
        3,
        len(standard_x[0]),
        len(algorithm_labels),
    )
    for filename, decoded in {
        "pilot_a_raw.csv": decoded_a,
        "pilot_b.csv": decoded_b,
        "pilot_c.csv": decoded_c,
    }.items():
        if not _matrices_close(decoded, _read_numeric_csv(x0_outputs / filename)):
            raise ProvenanceError(
                f"PILOT column-major solution decode mismatch in {filename!r}",
            )
    for filename in (
        "pilot_a_raw.csv",
        "pilot_b.csv",
        "pilot_c.csv",
        "pilot_z.csv",
        "pilot_r2.csv",
        "pilot_error.csv",
    ):
        if not _matrices_close(
            _read_numeric_csv(x0_outputs / filename),
            _read_numeric_csv(precalc_root / "outputs" / filename),
        ):
            raise ProvenanceError(
                f"PILOT precalculated replay differs in {filename!r}",
            )
    x0_viewpoint = x0_outputs
    precalc_viewpoint = precalc_root / "outputs"
    if (x0_viewpoint / "viewpoint_groups.csv").read_bytes() != (
        precalc_viewpoint / "viewpoint_groups.csv"
    ).read_bytes():
        raise ProvenanceError("PILOT replay changed the global viewpoint group")
    for filename in ("viewpoint_a.csv", "viewpoint_angles.csv"):
        if not _matrices_close(
            _read_numeric_csv(x0_viewpoint / filename),
            _read_numeric_csv(precalc_viewpoint / filename),
        ):
            raise ProvenanceError(
                f"PILOT X0-shape fallback differs in {filename!r}",
            )


def _validate_pilot_viewpoint_artifacts(  # noqa: PLR0912
    outputs: Path,
    configured_groups: object,
    algorithm_labels: list[str],
) -> None:
    if configured_groups == []:
        expected_groups = [list(range(1, len(algorithm_labels) + 1))]
    else:
        if not isinstance(configured_groups, list) or not all(
            isinstance(group, list) and group for group in configured_groups
        ):
            raise ProvenanceError("PILOT viewGroups must be a list of groups")
        expected_groups = cast(list[list[int]], configured_groups)
        if not all(
            isinstance(index, int)
            and not isinstance(index, bool)
            and 1 <= index <= len(algorithm_labels)
            for group in expected_groups
            for index in group
        ):
            raise ProvenanceError("PILOT viewGroups contain invalid algorithm indices")
        split = max(1, len(algorithm_labels) // 2 - 1)
        if expected_groups != [
            list(range(1, split + 1)),
            list(range(split + 1, len(algorithm_labels) + 1)),
        ]:
            raise ProvenanceError("Grouped PILOT viewpoint partition is not canonical")

    header, rows = _read_csv_rows(outputs / "viewpoint_groups.csv")
    if header != ["group", "member", "algorithm_index", "algorithm"]:
        raise ProvenanceError("PILOT viewpoint group artifact has an invalid header")
    expected_rows = [
        [
            str(group_index),
            str(member_index),
            str(algorithm_index),
            algorithm_labels[algorithm_index - 1],
        ]
        for group_index, group in enumerate(expected_groups, start=1)
        for member_index, algorithm_index in enumerate(group, start=1)
    ]
    if rows != expected_rows:
        raise ProvenanceError("PILOT viewpoint group artifact is inconsistent")

    group_count = len(expected_groups)
    viewpoint_a = _read_numeric_csv(
        outputs / "viewpoint_a.csv",
        expected_header=["group", "view_dimension", "z_1", "z_2", "z_3"],
    )
    angles = _read_numeric_csv(
        outputs / "viewpoint_angles.csv",
        expected_header=["group", "azimuth", "elevation"],
    )
    if len(viewpoint_a) != 2 * group_count or len(angles) != group_count:
        raise ProvenanceError("PILOT viewpoint artifacts have the wrong group count")
    if [int(row[0]) for row in angles] != list(range(1, group_count + 1)):
        raise ProvenanceError("PILOT viewpoint angle groups are not contiguous")
    expected_matrix_groups = [
        group for group in range(1, group_count + 1) for _ in range(2)
    ]
    if [int(row[0]) for row in viewpoint_a] != expected_matrix_groups or [
        int(row[1]) for row in viewpoint_a
    ] != [dimension for _ in range(group_count) for dimension in (1, 2)]:
        raise ProvenanceError("PILOT viewpoint matrix groups are not canonical")
    for group_index in range(group_count):
        first = viewpoint_a[2 * group_index][2:]
        second = viewpoint_a[2 * group_index + 1][2:]
        if not math.isclose(
            math.sqrt(sum(value * value for value in first)),
            1.0,
            rel_tol=1e-10,
            abs_tol=1e-10,
        ) or not math.isclose(
            math.sqrt(sum(value * value for value in second)),
            1.0,
            rel_tol=1e-10,
            abs_tol=1e-10,
        ):
            raise ProvenanceError("PILOT viewpoint rows are not unit vectors")
        cross = [
            first[1] * second[2] - first[2] * second[1],
            first[2] * second[0] - first[0] * second[2],
            first[0] * second[1] - first[1] * second[0],
        ]
        horizontal = math.hypot(cross[0], cross[1])
        if horizontal == 0 and cross[2] == 0:
            raise ProvenanceError("PILOT viewpoint rows are collinear")
        expected_azimuth = math.atan2(cross[1], cross[0])
        expected_elevation = math.atan2(cross[2], horizontal)
        actual = angles[group_index]
        if not (
            math.isclose(
                actual[1],
                expected_azimuth,
                rel_tol=1e-10,
                abs_tol=1e-10,
            )
            and math.isclose(
                actual[2],
                expected_elevation,
                rel_tol=1e-10,
                abs_tol=1e-10,
            )
        ):
            raise ProvenanceError("PILOT viewpoint angles are inconsistent")


def _fixed_reference_paths() -> set[str]:
    paths = {
        "shared_inputs/reference/metadata.csv",
        "shared_inputs/reference/metadata_test.csv",
        *{f"resolved_options/{variant}.json" for variant in _REFERENCE_VARIANTS},
    }
    stage_files = {
        "prelim/default/inputs": {
            "x_raw.csv",
            "y_raw.csv",
            "x_processed.csv",
            "y_processed.csv",
            "y_bin.csv",
            "y_best.csv",
            "p.csv",
            "beta.csv",
            "feature_labels.csv",
            "algorithm_labels.csv",
        },
        "prelim/default/outputs": {
            "prelim_feature_params.csv",
            "prelim_algo_params.csv",
            "prelim_scalars.csv",
            "prelim_instance_outputs.csv",
            "prelim_ybin.csv",
        },
        "sifted/default/inputs": {"x.csv", "y.csv", "y_bin.csv", "feature_labels.csv"},
        "sifted/default/outputs": {
            "correlation_rho.csv",
            "correlation_pval.csv",
            "sifted_indices.csv",
            "selected_indices.csv",
        },
        "pilot/default/inputs": {"x.csv", "y.csv", "feature_labels.csv"},
        "pilot/default/outputs": {
            "pilot_matrix.csv",
            "pilot_a_raw.csv",
            "pilot_b.csv",
            "pilot_c.csv",
            "pilot_z.csv",
            "pilot_r2.csv",
            "pilot_error.csv",
            "pilot_eoptim.csv",
            "pilot_perf.csv",
            "pilot_alpha.csv",
            "pilot_x0.csv",
        },
        "cloister/default/inputs": {"x.csv", "projection_a.csv"},
        "cloister/default/outputs": {"z_edge.csv", "z_ecorr.csv"},
    }
    for stage_path, filenames in stage_files.items():
        paths.update(f"build_data/{stage_path}/{name}" for name in filenames)

    pythia_inputs = {
        "z.csv",
        "y_raw.csv",
        "y_bin.csv",
        "y_best.csv",
        "algorithm_labels.csv",
    }
    pythia_outputs = {
        "summary.csv",
        "ysub.csv",
        "yhat.csv",
        "pr0sub.csv",
        "pr0hat.csv",
        "selection0.csv",
        "selection1.csv",
        "normalization_mu.csv",
        "normalization_sigma.csv",
        "raw_metrics.csv",
        "hyperparameters.csv",
    }
    trace_inputs = {
        "z.csv",
        "y_bin.csv",
        "y_hat.csv",
        "p.csv",
        "beta.csv",
        "algorithm_labels.csv",
    }
    for variant in _DOWNSTREAM_VARIANTS:
        paths.update(
            f"build_data/pythia/{variant}/inputs/{name}" for name in pythia_inputs
        )
        paths.update(
            f"build_data/pythia/{variant}/outputs/{name}" for name in pythia_outputs
        )
        paths.update(
            f"build_data/trace/{variant}/inputs/{name}" for name in trace_inputs
        )
        paths.update(
            f"build_data/trace/{variant}/outputs/{name}"
            for name in ("summary.csv", "raw_metrics.csv", "hard.csv")
        )
        paths.update(
            f"explore_data/pythia/{variant}/inputs/{name}" for name in pythia_inputs
        )
        paths.update(
            f"explore_data/pythia/{variant}/outputs/{name}"
            for name in ("eval_summary.csv", "predictions.csv", "probabilities.csv")
        )
        paths.update(
            f"explore_data/trace/{variant}/inputs/{name}" for name in trace_inputs
        )
        paths.update(
            f"explore_data/trace/{variant}/outputs/{name}"
            for name in ("eval_summary.csv", "membership.csv")
        )

    pilot_common_inputs = {
        "x.csv",
        "y.csv",
        "feature_labels.csv",
        "stage_context.json",
    }
    pilot_common_outputs = {
        "pilot_matrix.csv",
        "pilot_a_raw.csv",
        "pilot_b.csv",
        "pilot_c.csv",
        "pilot_z.csv",
        "pilot_r2.csv",
        "pilot_error.csv",
    }
    for variant in _PILOT_EVIDENCE_VARIANTS:
        inputs = set(pilot_common_inputs)
        outputs = set(pilot_common_outputs)
        if variant == "pilot_standard_numerical_3d_x0":
            inputs.add("x0.csv")
            outputs.update(
                {
                    "pilot_eoptim.csv",
                    "pilot_perf.csv",
                    "pilot_alpha.csv",
                    "pilot_x0.csv",
                },
            )
        elif variant == "pilot_standard_numerical_3d_precalc":
            inputs.add("precalc_alpha.csv")
            outputs.add("pilot_alpha.csv")
        if _PILOT_VARIANT_DIMS[variant] == 3:  # noqa: PLR2004
            outputs.update(
                {
                    "viewpoint_groups.csv",
                    "viewpoint_a.csv",
                    "viewpoint_angles.csv",
                },
            )
        paths.update(f"build_data/pilot/{variant}/inputs/{name}" for name in inputs)
        paths.update(f"build_data/pilot/{variant}/outputs/{name}" for name in outputs)
        paths.update(
            {
                f"explore_data/pilot/{variant}/inputs/x.csv",
                f"explore_data/pilot/{variant}/inputs/projection_a.csv",
                f"explore_data/pilot/{variant}/outputs/pilot_z.csv",
            },
        )
    return paths


def _fixed_reference_paths_v1() -> set[str]:
    """Return the frozen 229-file profile installed before PILOT #262 evidence."""
    paths = _fixed_reference_paths()
    return {
        relative
        for relative in paths
        if not (
            relative.startswith("resolved_options/pilot_")
            or (
                len(PurePosixPath(relative).parts) >= 3  # noqa: PLR2004
                and PurePosixPath(relative).parts[2] in _PILOT_EVIDENCE_VARIANTS
            )
        )
    }


def _geometry_paths(labels: list[str]) -> set[str]:
    return {
        f"build_data/trace/{variant}/outputs/{kind}_{label}.csv"
        for variant in _DOWNSTREAM_VARIANTS
        for kind in ("good", "best")
        for label in labels
    }


def _read_algorithm_labels(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream)
        try:
            header = next(reader)
        except StopIteration as error:
            raise ProvenanceError(
                f"Algorithm-label CSV has no header: {path}",
            ) from error
        rows = list(reader)
    if header != ["algorithm_name"] or not rows:
        raise ProvenanceError(f"Algorithm-label CSV has an invalid schema: {path}")
    labels = [row[0] for row in rows if len(row) == 1 and row[0]]
    folded_labels = {label.casefold() for label in labels}
    if len(labels) != len(rows) or len(folded_labels) != len(labels):
        raise ProvenanceError(f"Algorithm-label CSV has invalid labels: {path}")
    for label in labels:
        candidate = PurePosixPath(f"good_{label}.csv")
        if len(candidate.parts) != 1 or label in {".", ".."} or "\\" in label:
            raise ProvenanceError(
                f"Algorithm label is unsafe for geometry filenames: {label!r}",
            )
    return labels


def validate_inventory(  # noqa: PLR0912
    repo_root: Path,
    inventory_path: Path,
) -> InventoryReport:
    """Prove that every historical fixture has exactly one declared trust class."""
    root = repo_root.resolve()
    inventory = _load_object(inventory_path, "fixture inventory")
    _expect_equal(inventory, "schema_version", INVENTORY_SCHEMA)
    roots = [_safe_relative_path(item) for item in _text_list(inventory, "roots")]
    ignored = _text_list(inventory, "ignore")
    rules_raw = _expect_list(inventory, "rules")
    if not rules_raw:
        raise ProvenanceError(
            "Fixture inventory needs at least one classification rule",
        )

    rules: list[tuple[str, tuple[str, ...], str]] = []
    for index, raw_rule in enumerate(rules_raw):
        if not isinstance(raw_rule, dict):
            raise ProvenanceError(f"rules[{index}] must be an object")
        rule = cast(dict[str, Any], raw_rule)
        pattern = _expect_text(rule, "pattern")
        excludes_raw = rule.get("exclude", [])
        if not isinstance(excludes_raw, list) or not all(
            isinstance(item, str) and item for item in excludes_raw
        ):
            raise ProvenanceError(
                f"rules[{index}].exclude must be a list of text patterns",
            )
        trust = _expect_text(rule, "trust")
        if trust not in _ALLOWED_TRUST:
            raise ProvenanceError(f"rules[{index}].trust is invalid: {trust!r}")
        rules.append((pattern, tuple(cast(list[str], excludes_raw)), trust))

    counts: Counter[str] = Counter()
    unmatched: list[str] = []
    ambiguous: list[str] = []
    files: set[str] = set()
    for relative_root in roots:
        fixture_root = root.joinpath(*relative_root.parts)
        if not fixture_root.is_dir():
            raise ProvenanceError(
                f"Inventory root does not exist: {relative_root.as_posix()}",
            )
        for path in fixture_root.rglob("*"):
            if not path.is_file() or path.is_symlink():
                continue
            relative = path.relative_to(root).as_posix()
            if any(fnmatch.fnmatchcase(relative, pattern) for pattern in ignored):
                continue
            files.add(relative)
            matches = [
                trust
                for pattern, excludes, trust in rules
                if fnmatch.fnmatchcase(relative, pattern)
                and not any(
                    fnmatch.fnmatchcase(relative, exclusion) for exclusion in excludes
                )
            ]
            if not matches:
                unmatched.append(relative)
            elif len(matches) > 1:
                ambiguous.append(relative)
            else:
                counts[matches[0]] += 1

    if unmatched or ambiguous:
        raise ProvenanceError(
            f"Fixture inventory is incomplete. Unmatched={sorted(unmatched)}; "
            f"ambiguous={sorted(ambiguous)}",
        )
    return InventoryReport(file_count=len(files), counts=dict(sorted(counts.items())))


def _load_object(path: Path, description: str) -> dict[str, Any]:
    if not path.is_file():
        raise ProvenanceError(f"Missing {description}: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ProvenanceError(f"Cannot read {description} {path}: {error}") from error
    if not isinstance(value, dict):
        raise ProvenanceError(
            f"{description.capitalize()} must contain one JSON object",
        )
    return cast(dict[str, Any], value)


def _expect_object(parent: dict[str, Any], key: str) -> dict[str, Any]:
    value = parent.get(key)
    if not isinstance(value, dict):
        raise ProvenanceError(f"{key} must be an object")
    return cast(dict[str, Any], value)


def _expect_list(parent: dict[str, Any], key: str) -> list[Any]:
    value = parent.get(key)
    if not isinstance(value, list):
        raise ProvenanceError(f"{key} must be a list")
    return value


def _expect_text(parent: dict[str, Any], key: str) -> str:
    value = parent.get(key)
    if not isinstance(value, str) or not value:
        raise ProvenanceError(f"{key} must be nonempty text")
    return value


def _expect_bool(parent: dict[str, Any], key: str) -> bool:
    value = parent.get(key)
    if not isinstance(value, bool):
        raise ProvenanceError(f"{key} must be Boolean")
    return value


def _expect_nonnegative_int(parent: dict[str, Any], key: str) -> int:
    value = parent.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ProvenanceError(f"{key} must be a nonnegative integer")
    return value


def _expect_equal(parent: dict[str, Any], key: str, expected: str) -> None:
    actual = _expect_text(parent, key)
    if actual != expected:
        raise ProvenanceError(f"{key} must be {expected!r}, got {actual!r}")


def _text_list(parent: dict[str, Any], key: str) -> list[str]:
    values = _expect_list(parent, key)
    if not all(isinstance(value, str) and value for value in values):
        raise ProvenanceError(f"{key} must contain nonempty text values")
    return cast(list[str], values)


def _text_set(values: list[Any], name: str) -> set[str]:
    if not all(isinstance(value, str) and value for value in values):
        raise ProvenanceError(f"{name} must contain nonempty text values")
    result = set(cast(list[str], values))
    if len(result) != len(values):
        raise ProvenanceError(f"{name} cannot contain duplicates")
    return result


def _safe_relative_path(value: str) -> PurePosixPath:
    if not isinstance(value, str) or not value:
        raise ProvenanceError("Fixture paths must be nonempty text")
    path = PurePosixPath(value)
    if (
        "\\" in value
        or path.is_absolute()
        or value != path.as_posix()
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ProvenanceError(f"Unsafe or noncanonical fixture path: {value!r}")
    return path


def _validate_sha256(value: str, name: str) -> None:
    if _SHA256_RE.fullmatch(value) is None:
        raise ProvenanceError(f"{name} must be a lowercase SHA-256 digest")


def _validate_commit(value: str, name: str) -> None:
    if _COMMIT_RE.fullmatch(value) is None:
        raise ProvenanceError(f"{name} must be a full lowercase Git commit")


def _validate_timestamp(value: str) -> None:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ProvenanceError("generated_at must be an ISO-8601 timestamp") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ProvenanceError("generated_at must include a timezone")


def _parse_release(value: str) -> tuple[int, int]:
    match = _RELEASE_RE.fullmatch(value)
    if match is None:
        raise ProvenanceError(f"Invalid MATLAB release: {value!r}")
    return int(match.group("year")), 0 if match.group("half") == "a" else 1


def _release_key(value: str) -> tuple[int, int]:
    return _parse_release(value)


def _csv_shape(path: Path) -> tuple[int, int]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream)
        try:
            header = next(reader)
        except StopIteration:
            return 0, 0
        rows = list(reader)
    if any(len(row) != len(header) for row in rows):
        raise ProvenanceError(f"CSV has ragged rows: {path}")
    return len(rows), len(header)


def _validate_trace_geometry_csv(path: Path) -> None:  # noqa: PLR0912
    """Validate the region/ring schema used for lossless TRACE geometry."""
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream)
        try:
            header = tuple(next(reader))
        except StopIteration as error:
            raise ProvenanceError(f"TRACE geometry has no header: {path}") from error
        if header != _GEOMETRY_HEADER:
            raise ProvenanceError(
                f"TRACE geometry has an invalid header: {path}",
            )

        groups: dict[tuple[int, str], list[tuple[int, float, float]]] = {}
        part_rings: dict[int, set[str]] = {}
        for row_number, row in enumerate(reader, start=2):
            if len(row) != len(_GEOMETRY_HEADER):
                raise ProvenanceError(
                    f"TRACE geometry row {row_number} has the wrong width: {path}",
                )
            try:
                part = int(row[0])
                vertex = int(row[2])
                is_hole = int(row[3])
                z_1 = float(row[4])
                z_2 = float(row[5])
            except ValueError as error:
                raise ProvenanceError(
                    f"TRACE geometry row {row_number} has invalid values: {path}",
                ) from error
            ring = row[1]
            hole_match = _HOLE_RE.fullmatch(ring)
            if (
                part < 1
                or vertex < 1
                or is_hole not in {0, 1}
                or (ring != "exterior" and hole_match is None)
                or is_hole != int(ring != "exterior")
                or not math.isfinite(z_1)
                or not math.isfinite(z_2)
            ):
                raise ProvenanceError(
                    "TRACE geometry row "
                    f"{row_number} violates the ring contract: {path}",
                )
            groups.setdefault((part, ring), []).append((vertex, z_1, z_2))
            part_rings.setdefault(part, set()).add(ring)

    if not groups:
        return
    parts = sorted(part_rings)
    if parts != list(range(1, parts[-1] + 1)):
        raise ProvenanceError(f"TRACE geometry parts are not contiguous: {path}")
    for part, rings in part_rings.items():
        if "exterior" not in rings:
            raise ProvenanceError(
                f"TRACE geometry part {part} has no exterior ring: {path}",
            )
        hole_numbers = sorted(
            int(match.group("number"))
            for ring in rings
            if (match := _HOLE_RE.fullmatch(ring)) is not None
        )
        if hole_numbers != list(range(1, len(hole_numbers) + 1)):
            raise ProvenanceError(
                f"TRACE geometry part {part} has noncontiguous holes: {path}",
            )
    for (part, ring), vertices in groups.items():
        vertex_numbers = [vertex[0] for vertex in vertices]
        if len(vertices) < _MIN_RING_VERTICES or vertex_numbers != list(
            range(1, len(vertices) + 1),
        ):
            raise ProvenanceError(
                "TRACE geometry part "
                f"{part} ring {ring!r} has invalid vertices: {path}",
            )
        if vertices[0][1:] == vertices[-1][1:]:
            raise ProvenanceError(
                f"TRACE geometry part {part} ring {ring!r} repeats its closure: {path}",
            )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    verify = subparsers.add_parser("verify", help="validate one generated bundle")
    verify.add_argument("root", type=Path)
    verify.add_argument("--allow-diagnostic", action="store_true")

    inventory = subparsers.add_parser(
        "inventory",
        help="validate historical classification",
    )
    inventory.add_argument("repo_root", type=Path)
    inventory.add_argument("inventory", type=Path)

    install = subparsers.add_parser(
        "install",
        help="atomically install a verified bundle in the unified layout",
    )
    install.add_argument("source", type=Path)
    install.add_argument("destination", type=Path)
    return parser


def main() -> int:
    """Run the fixture validation command-line interface."""
    arguments = _build_parser().parse_args()
    if arguments.command == "verify":
        report = validate_bundle(
            arguments.root,
            allow_diagnostic=arguments.allow_diagnostic,
        )
        result = {
            "file_count": report.file_count,
            "matlab_release": report.matlab_release,
            "root": str(report.root),
            "total_bytes": report.total_bytes,
            "trust": report.trust,
        }
    elif arguments.command == "inventory":
        report_inventory = validate_inventory(arguments.repo_root, arguments.inventory)
        result = {
            "counts": report_inventory.counts,
            "file_count": report_inventory.file_count,
        }
    else:
        report_installed = install_verified_bundle(
            arguments.source,
            arguments.destination,
        )
        result = {
            "file_count": report_installed.file_count,
            "matlab_release": report_installed.matlab_release,
            "root": str(report_installed.root),
            "total_bytes": report_installed.total_bytes,
            "trust": report_installed.trust,
        }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
