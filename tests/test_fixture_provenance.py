"""Tests for MATLAB fixture provenance and historical data classification."""

from __future__ import annotations

import csv
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from tools.fixture_provenance import (
    BUNDLE_SCHEMA,
    DIAGNOSTIC_TRUST,
    VERIFIED_TRUST,
    ProvenanceError,
    install_verified_bundle,
    sha256_file,
    validate_bundle,
    validate_inventory,
)

_COMMIT = "1" * 40
_SCRIPT_HASH = "2" * 64
_BUNDLE_FILE_COUNT = 2
_MIN_HISTORICAL_FILES = 300


def _write_bundle(
    root: Path,
    *,
    trust: str = VERIFIED_TRUST,
    release: str = "R2025a",
    matlab_dirty: bool = False,
) -> dict[str, Any]:
    data_path = root / "shared_inputs" / "study" / "metadata.csv"
    data_path.parent.mkdir(parents=True)
    with data_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["instance", "feature_a"])
        writer.writerow(["one", "1.0"])

    empty_path = root / "build" / "trace" / "trace3" / "good_empty.csv"
    empty_path.parent.mkdir(parents=True)
    empty_path.write_text(
        "part,ring,vertex,is_hole,z_1,z_2\n",
        encoding="utf-8",
    )

    files = [
        {
            "path": data_path.relative_to(root).as_posix(),
            "sha256": sha256_file(data_path),
            "size_bytes": data_path.stat().st_size,
            "media_type": "text/csv",
            "role": "training metadata",
            "phase": "shared",
            "stage": None,
            "variant": "study",
            "empty": False,
            "rows": 1,
            "columns": 2,
        },
        {
            "path": empty_path.relative_to(root).as_posix(),
            "sha256": sha256_file(empty_path),
            "size_bytes": empty_path.stat().st_size,
            "media_type": "text/csv",
            "role": "explicit empty good footprint",
            "phase": "build",
            "stage": "trace",
            "variant": "trace3",
            "empty": True,
            "rows": 0,
            "columns": 6,
        },
    ]
    manifest: dict[str, Any] = {
        "schema_version": BUNDLE_SCHEMA,
        "bundle_id": "study-defaults",
        "trust": trust,
        "generated_at": "2026-08-17T00:00:00Z",
        "dataset": {"name": "study", "seed": 42},
        "resolved_options": {
            "variants": [{"name": "trace3", "trace": {"method": "trace3"}}],
        },
        "matlab": {
            "repo_commit": _COMMIT,
            "repo_dirty": matlab_dirty,
            "toolkit_version": "0.9.0",
            "release": release,
            "version": "25.1",
            "platform": "maca64",
            "installed_toolboxes": [
                "MATLAB",
                "Statistics and Machine Learning Toolbox",
            ],
            "required_toolboxes": ["MATLAB", "Statistics and Machine Learning Toolbox"],
        },
        "generator": {
            "repo_commit": _COMMIT,
            "repo_dirty": False,
            "script": "tests/matlab_export/pyis_export_reference_data.m",
            "script_sha256": _SCRIPT_HASH,
        },
        "files": files,
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return manifest


def _rewrite_manifest(root: Path, manifest: dict[str, Any]) -> None:
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_verified_bundle_passes_full_integrity_check(tmp_path: Path) -> None:
    """Accept an intact current-release bundle with complete provenance."""
    _write_bundle(tmp_path)

    report = validate_bundle(tmp_path)

    assert report.trust == VERIFIED_TRUST
    assert report.matlab_release == "R2025a"
    assert report.file_count == _BUNDLE_FILE_COUNT
    assert report.total_bytes > 0


@pytest.mark.parametrize("mutation", ["content", "extra", "missing"])
def test_file_set_or_hash_changes_are_rejected(tmp_path: Path, mutation: str) -> None:
    """Reject altered, missing, and untracked fixture content."""
    manifest = _write_bundle(tmp_path)
    target = tmp_path / manifest["files"][0]["path"]
    if mutation == "content":
        target.write_text("changed", encoding="utf-8")
    elif mutation == "extra":
        (tmp_path / "untracked.csv").write_text("x\n1\n", encoding="utf-8")
    else:
        target.unlink()

    with pytest.raises(ProvenanceError):
        validate_bundle(tmp_path)


def test_duplicate_casefolded_paths_are_rejected(tmp_path: Path) -> None:
    """Reject archive paths that collide on case-insensitive filesystems."""
    manifest = _write_bundle(tmp_path)
    duplicate = deepcopy(manifest["files"][0])
    duplicate["path"] = duplicate["path"].upper()
    manifest["files"].append(duplicate)
    _rewrite_manifest(tmp_path, manifest)

    with pytest.raises(ProvenanceError, match="case-colliding"):
        validate_bundle(tmp_path)


@pytest.mark.parametrize(
    "unsafe",
    ["../escape.csv", "/absolute.csv", "a/../b.csv", "a\\b.csv"],
)
def test_noncanonical_manifest_paths_are_rejected(tmp_path: Path, unsafe: str) -> None:
    """Reject paths that are unsafe or platform-dependent."""
    manifest = _write_bundle(tmp_path)
    manifest["files"][0]["path"] = unsafe
    _rewrite_manifest(tmp_path, manifest)

    with pytest.raises(ProvenanceError, match="Unsafe|noncanonical"):
        validate_bundle(tmp_path)


def test_verified_bundle_requires_clean_current_matlab(tmp_path: Path) -> None:
    """Require clean sources and the declared gold MATLAB release."""
    _write_bundle(tmp_path, release="R2024a", matlab_dirty=True)

    with pytest.raises(ProvenanceError, match="clean MATLAB"):
        validate_bundle(tmp_path)


def test_diagnostic_bundle_is_explicitly_opt_in(tmp_path: Path) -> None:
    """Keep older diagnostic exports out of parity assertions by default."""
    _write_bundle(tmp_path, trust=DIAGNOSTIC_TRUST, release="R2024a", matlab_dirty=True)

    with pytest.raises(ProvenanceError, match="not accepted as parity oracles"):
        validate_bundle(tmp_path)
    report = validate_bundle(tmp_path, allow_diagnostic=True)
    assert report.trust == DIAGNOSTIC_TRUST


def test_verified_bundle_installs_atomically_with_layout_intact(tmp_path: Path) -> None:
    """Install only a validated bundle and preserve every relative path."""
    source = tmp_path / "source"
    source.mkdir()
    _write_bundle(source)
    destination = tmp_path / "fixtures" / "matlab" / "current"

    report = install_verified_bundle(source, destination)

    assert report.root == destination
    assert report.file_count == _BUNDLE_FILE_COUNT
    assert (destination / "manifest.json").is_file()
    assert (destination / "shared_inputs" / "study" / "metadata.csv").read_bytes() == (
        source / "shared_inputs" / "study" / "metadata.csv"
    ).read_bytes()
    assert (destination / "build" / "trace" / "trace3" / "good_empty.csv").is_file()
    validate_bundle(destination)


def test_install_rejects_diagnostic_or_existing_destination(tmp_path: Path) -> None:
    """Never promote diagnostics or overwrite an existing fixture tree."""
    diagnostic = tmp_path / "diagnostic"
    diagnostic.mkdir()
    _write_bundle(
        diagnostic,
        trust=DIAGNOSTIC_TRUST,
        release="R2024a",
        matlab_dirty=True,
    )
    destination = tmp_path / "current"

    with pytest.raises(ProvenanceError, match="not accepted as parity oracles"):
        install_verified_bundle(diagnostic, destination)

    verified = tmp_path / "verified"
    verified.mkdir()
    _write_bundle(verified)
    destination.mkdir()
    with pytest.raises(ProvenanceError, match="already exists"):
        install_verified_bundle(verified, destination)


def test_missing_required_toolbox_is_rejected(tmp_path: Path) -> None:
    """Reject an export that lacks a required MATLAB toolbox."""
    manifest = _write_bundle(tmp_path)
    manifest["matlab"]["required_toolboxes"].append("Optimization Toolbox")
    _rewrite_manifest(tmp_path, manifest)

    with pytest.raises(ProvenanceError, match="Optimization Toolbox"):
        validate_bundle(tmp_path)


def test_csv_shape_and_empty_status_are_verified(tmp_path: Path) -> None:
    """Verify semantic shape metadata in addition to file hashes."""
    manifest = _write_bundle(tmp_path)
    manifest["files"][0]["rows"] = 2
    _rewrite_manifest(tmp_path, manifest)

    with pytest.raises(ProvenanceError, match="row-count"):
        validate_bundle(tmp_path)


def test_trace_geometry_schema_accepts_parts_and_holes(tmp_path: Path) -> None:
    """Accept reconstructable region/ring geometry without closure vertices."""
    manifest = _write_bundle(tmp_path)
    entry = manifest["files"][1]
    target = tmp_path / entry["path"]
    target.write_text(
        "part,ring,vertex,is_hole,z_1,z_2\n"
        "1,exterior,1,0,0,0\n"
        "1,exterior,2,0,3,0\n"
        "1,exterior,3,0,0,3\n"
        "1,hole_1,1,1,0.5,0.5\n"
        "1,hole_1,2,1,0.5,1\n"
        "1,hole_1,3,1,1,0.5\n",
        encoding="utf-8",
    )
    entry.update(
        sha256=sha256_file(target),
        size_bytes=target.stat().st_size,
        rows=6,
        empty=False,
    )
    _rewrite_manifest(tmp_path, manifest)

    validate_bundle(tmp_path)


@pytest.mark.parametrize(
    "geometry",
    [
        "part,ring,vertex,z_1,z_2\n",
        (
            "part,ring,vertex,is_hole,z_1,z_2\n"
            "1,exterior,1,1,0,0\n"
            "1,exterior,2,1,1,0\n"
            "1,exterior,3,1,0,1\n"
        ),
        (
            "part,ring,vertex,is_hole,z_1,z_2\n"
            "1,exterior,1,0,0,0\n"
            "1,exterior,3,0,1,0\n"
            "1,exterior,4,0,0,1\n"
        ),
    ],
)
def test_trace_geometry_schema_rejects_ambiguous_data(
    tmp_path: Path,
    geometry: str,
) -> None:
    """Reject geometry that cannot be reconstructed without guessing."""
    manifest = _write_bundle(tmp_path)
    entry = manifest["files"][1]
    target = tmp_path / entry["path"]
    target.write_text(geometry, encoding="utf-8")
    with target.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.reader(stream))
    entry.update(
        sha256=sha256_file(target),
        size_bytes=target.stat().st_size,
        rows=max(0, len(rows) - 1),
        columns=len(rows[0]),
        empty=len(rows) == 1,
    )
    _rewrite_manifest(tmp_path, manifest)

    with pytest.raises(ProvenanceError, match="TRACE geometry"):
        validate_bundle(tmp_path)


def test_repository_inventory_classifies_every_historical_fixture() -> None:
    """Assign every historical data artifact one explicit trust class."""
    repo_root = Path(__file__).resolve().parents[1]

    report = validate_inventory(
        repo_root,
        repo_root / "tests" / "fixture_inventory.json",
    )

    assert report.file_count >= _MIN_HISTORICAL_FILES
    assert report.counts["legacy-unknown"] > 0
    assert report.counts["python-regression"] > 0
    assert report.counts["python-synthetic"] > 0


def test_inventory_rejects_an_unclassified_file(tmp_path: Path) -> None:
    """Reject an inventory that leaves any fixture unclassified."""
    (tmp_path / "fixtures").mkdir()
    (tmp_path / "fixtures" / "data.csv").write_text("x\n1\n", encoding="utf-8")
    inventory = {
        "schema_version": "pyinstancespace.fixture-inventory/v1",
        "roots": ["fixtures"],
        "ignore": [],
        "rules": [],
    }
    inventory_path = tmp_path / "inventory.json"
    inventory_path.write_text(json.dumps(inventory), encoding="utf-8")

    with pytest.raises(ProvenanceError, match="at least one"):
        validate_inventory(tmp_path, inventory_path)
