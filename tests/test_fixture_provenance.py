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
    REFERENCE_PROFILE,
    RESOLVED_OPTIONS_INDEX_SCHEMA,
    RESOLVED_OPTIONS_SCHEMA,
    VERIFIED_TRUST,
    ProvenanceError,
    _fixed_reference_paths,
    _geometry_paths,
    install_verified_bundle,
    sha256_file,
    validate_bundle,
    validate_inventory,
)

_COMMIT = "1" * 40
_SCRIPT_HASH = "2" * 64
_MIN_HISTORICAL_FILES = 300
_VARIANTS = ("trace3_default", "trace3_pythia_skip", "legacy_svm")
_ALGORITHM_LABELS = ["algo_a"]
_REFERENCE_ALGORITHM_LABELS = [
    "NB",
    "LDA",
    "QDA",
    "CART",
    "J48",
    "KNN",
    "L_SVM",
    "poly_SVM",
    "RBF_SVM",
    "RandF",
]


def _effective_options(variant: str) -> dict[str, Any]:
    options: dict[str, Any] = {
        "general": {"seed": 42, "verbose": False, "parallel": False, "ncores": 18},
        "perf": {
            "MaxPerf": False,
            "AbsPerf": False,
            "epsilon": 0.05,
            "betaThreshold": 0.55,
        },
        "prelim": {"iqrMultiplier": 5, "nanThreshold": 0.2},
        "auto": {"preproc": True},
        "bound": {"flag": True},
        "norm": {"flag": True},
        "selvars": {
            "smallscaleflag": False,
            "smallscale": 0.3,
            "fileidxflag": False,
            "fileidx": "",
            "densityflag": False,
            "mindistance": 0.1,
            "type": "Ftr&Good",
        },
        "sifted": {
            "flag": True,
            "rho": 0.1,
            "pval": 0.05,
            "K": 10,
            "MaxIter": 1000,
            "Replicates": 100,
        },
        "pilot": {
            "analytic": False,
            "ntries": 10,
            "dims": 2,
            "method": "standard",
            "alpha": 1.0,
            "viewGroups": [],
            "topoWeight": 0,
            "verbose": False,
        },
        "cloister": {"pval": 0.05, "corrThreshold": 0.7, "maxFeatures": 20},
        "pythia": {
            "flag": True,
            "kFold": 5,
            "tuning": "sobol",
            "nTuningIter": 20,
            "params": [],
            "skip": False,
            "ispolykrnl": False,
            "useweights": False,
            "ensembleMethod": "Bag",
            "verbose": False,
            "seed": 42,
            "classifier": "knn",
        },
        "trace": {
            "method": "trace3",
            "PI": 0.6,
            "minInstances": 4,
            "minAreaFrac": 0.01,
            "contra": False,
        },
        "outputs": {"csv": False, "png": False, "fig": False, "web": False},
    }
    if variant == "trace3_pythia_skip":
        options["pythia"]["skip"] = True
    elif variant == "legacy_svm":
        options["pythia"]["classifier"] = "svm"
        options["trace"].update(method="legacy", PI=0.55, contra=True)
    return options


def _entry_metadata(relative: str) -> tuple[str, str | None, str]:
    parts = Path(relative).parts
    if parts[0] == "shared_inputs":
        return "shared", None, "reference"
    if parts[0] == "resolved_options":
        return "shared", None, Path(relative).stem
    return parts[0].removesuffix("_data"), parts[1], parts[2]


def _write_profile_file(root: Path, relative: str) -> None:
    target = root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    if relative.startswith("resolved_options/"):
        variant = target.stem
        artifact = {
            "schema_version": RESOLVED_OPTIONS_SCHEMA,
            "name": variant,
            "description": f"{variant} reference variant",
            "options": _effective_options(variant),
        }
        target.write_text(json.dumps(artifact), encoding="utf-8")
    elif relative.endswith("algorithm_labels.csv"):
        target.write_text("algorithm_name\nalgo_a\n", encoding="utf-8")
    elif "/trace/" in relative and (
        target.name.startswith(("good_", "best_")) or target.name == "hard.csv"
    ):
        target.write_text("part,ring,vertex,is_hole,z_1,z_2\n", encoding="utf-8")
    elif relative == "shared_inputs/reference/metadata.csv":
        target.write_text("instance,feature_a\none,1.0\n", encoding="utf-8")
    elif relative == "shared_inputs/reference/metadata_test.csv":
        target.write_text("instance,feature_a\ntest,2.0\n", encoding="utf-8")
    else:
        target.write_text("value\n1\n", encoding="utf-8")


def _manifest_entry(root: Path, relative: str) -> dict[str, Any]:
    target = root / relative
    phase, stage, variant = _entry_metadata(relative)
    media_type = "application/json" if target.suffix == ".json" else "text/csv"
    rows = 0
    columns = 0
    empty = False
    if media_type == "text/csv":
        with target.open("r", encoding="utf-8", newline="") as stream:
            csv_rows = list(csv.reader(stream))
        rows = max(0, len(csv_rows) - 1)
        columns = len(csv_rows[0]) if csv_rows else 0
        empty = rows == 0
    return {
        "path": relative,
        "sha256": sha256_file(target),
        "size_bytes": target.stat().st_size,
        "media_type": media_type,
        "role": relative,
        "phase": phase,
        "stage": stage,
        "variant": variant,
        "empty": empty,
        "rows": rows,
        "columns": columns,
    }


def _write_bundle(
    root: Path,
    *,
    trust: str = VERIFIED_TRUST,
    release: str = "R2025a",
    matlab_dirty: bool = False,
) -> dict[str, Any]:
    profile_paths = _fixed_reference_paths() | _geometry_paths(_ALGORITHM_LABELS)
    for relative in sorted(profile_paths):
        _write_profile_file(root, relative)
    files = [_manifest_entry(root, relative) for relative in sorted(profile_paths)]
    manifest: dict[str, Any] = {
        "schema_version": BUNDLE_SCHEMA,
        "profile": REFERENCE_PROFILE,
        "bundle_id": "study-defaults",
        "trust": trust,
        "generated_at": "2026-08-17T00:00:00Z",
        "dataset": {
            "name": "study",
            "seed": 42,
            "training_input": "shared_inputs/reference/metadata.csv",
            "test_input": "shared_inputs/reference/metadata_test.csv",
        },
        "resolved_options": {
            "schema_version": RESOLVED_OPTIONS_INDEX_SCHEMA,
            "variants": [
                {
                    "name": variant,
                    "description": f"{variant} reference variant",
                    "path": f"resolved_options/{variant}.json",
                }
                for variant in _VARIANTS
            ],
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
                "Optimization Toolbox",
                "Global Optimization Toolbox",
                "Financial Toolbox",
            ],
            "required_toolboxes": [
                "MATLAB",
                "Statistics and Machine Learning Toolbox",
                "Optimization Toolbox",
                "Global Optimization Toolbox",
                "Financial Toolbox",
            ],
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


def _entry_for(manifest: dict[str, Any], relative: str) -> dict[str, Any]:
    return next(entry for entry in manifest["files"] if entry["path"] == relative)


def _refresh_entry(root: Path, entry: dict[str, Any]) -> None:
    refreshed = _manifest_entry(root, entry["path"])
    entry.update(refreshed)


def test_verified_bundle_passes_full_integrity_check(tmp_path: Path) -> None:
    """Accept an intact current-release bundle with complete provenance."""
    _write_bundle(tmp_path)

    report = validate_bundle(tmp_path)

    assert report.trust == VERIFIED_TRUST
    assert report.matlab_release == "R2025a"
    assert report.file_count == len(
        _fixed_reference_paths() | _geometry_paths(_ALGORITHM_LABELS),
    )
    assert report.total_bytes > 0


def test_reference_profile_declares_the_documented_file_count() -> None:
    """Keep the fixed reference-study profile and documentation synchronized."""
    profile = _fixed_reference_paths() | _geometry_paths(_REFERENCE_ALGORITHM_LABELS)

    assert len(profile) == 229  # noqa: PLR2004


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
    assert report.file_count == len(
        _fixed_reference_paths() | _geometry_paths(_ALGORITHM_LABELS),
    )
    assert (destination / "manifest.json").is_file()
    assert (
        destination / "shared_inputs" / "reference" / "metadata.csv"
    ).read_bytes() == (
        source / "shared_inputs" / "reference" / "metadata.csv"
    ).read_bytes()
    assert (
        destination
        / "build_data"
        / "trace"
        / "trace3_default"
        / "outputs"
        / "good_algo_a.csv"
    ).is_file()
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
    manifest["matlab"]["installed_toolboxes"].remove("Financial Toolbox")
    _rewrite_manifest(tmp_path, manifest)

    with pytest.raises(ProvenanceError, match="Financial Toolbox"):
        validate_bundle(tmp_path)


def test_profile_cannot_omit_a_required_toolbox_from_both_lists(
    tmp_path: Path,
) -> None:
    """Require the known exporter dependencies, not only internal list agreement."""
    manifest = _write_bundle(tmp_path)
    manifest["matlab"]["installed_toolboxes"].remove("Financial Toolbox")
    manifest["matlab"]["required_toolboxes"].remove("Financial Toolbox")
    _rewrite_manifest(tmp_path, manifest)

    with pytest.raises(ProvenanceError, match="reference-export dependencies"):
        validate_bundle(tmp_path)


def test_csv_shape_and_empty_status_are_verified(tmp_path: Path) -> None:
    """Verify semantic shape metadata in addition to file hashes."""
    manifest = _write_bundle(tmp_path)
    manifest["files"][0]["rows"] = 2
    _rewrite_manifest(tmp_path, manifest)

    with pytest.raises(ProvenanceError, match="row-count"):
        validate_bundle(tmp_path)


@pytest.mark.parametrize(
    "relative",
    [
        "shared_inputs/reference/metadata_test.csv",
        "resolved_options/legacy_svm.json",
        "build_data/pilot/default/outputs/pilot_z.csv",
        "build_data/pythia/trace3_default/outputs/raw_metrics.csv",
        "build_data/trace/trace3_default/outputs/raw_metrics.csv",
        "build_data/trace/trace3_default/outputs/good_algo_a.csv",
        "explore_data/trace/trace3_default/inputs/y_bin.csv",
        "explore_data/pythia/legacy_svm/outputs/predictions.csv",
        "explore_data/trace/trace3_pythia_skip/outputs/membership.csv",
    ],
)
def test_profile_rejects_deleting_file_and_manifest_entry(
    tmp_path: Path,
    relative: str,
) -> None:
    """Reject self-consistent manifests that omit a required profile artifact."""
    manifest = _write_bundle(tmp_path)
    (tmp_path / relative).unlink()
    manifest["files"] = [
        entry for entry in manifest["files"] if entry["path"] != relative
    ]
    _rewrite_manifest(tmp_path, manifest)

    with pytest.raises(ProvenanceError, match="Reference export profile"):
        validate_bundle(tmp_path)


def test_profile_rejects_an_internally_consistent_two_file_bundle(
    tmp_path: Path,
) -> None:
    """Require the exporter profile even when hashes and inventory agree."""
    manifest = _write_bundle(tmp_path)
    retained = {
        "shared_inputs/reference/metadata.csv",
        "shared_inputs/reference/metadata_test.csv",
    }
    for entry in manifest["files"]:
        if entry["path"] not in retained:
            (tmp_path / entry["path"]).unlink()
    manifest["files"] = [
        entry for entry in manifest["files"] if entry["path"] in retained
    ]
    _rewrite_manifest(tmp_path, manifest)

    with pytest.raises(ProvenanceError, match="Reference export profile"):
        validate_bundle(tmp_path)


def test_profile_rejects_self_consistent_stale_artifact(tmp_path: Path) -> None:
    """Reject an extra exporter artifact even when it is correctly manifested."""
    manifest = _write_bundle(tmp_path)
    relative = "build_data/pilot/default/outputs/stale.csv"
    _write_profile_file(tmp_path, relative)
    manifest["files"].append(_manifest_entry(tmp_path, relative))
    _rewrite_manifest(tmp_path, manifest)

    with pytest.raises(ProvenanceError, match="file-set mismatch.*stale.csv"):
        validate_bundle(tmp_path)


@pytest.mark.parametrize("missing", ["perf", "outputs"])
def test_resolved_options_require_every_top_level_group(
    tmp_path: Path,
    missing: str,
) -> None:
    """Reject partial option artifacts even when their hash is refreshed."""
    manifest = _write_bundle(tmp_path)
    relative = "resolved_options/trace3_default.json"
    target = tmp_path / relative
    artifact = json.loads(target.read_text(encoding="utf-8"))
    artifact["options"].pop(missing)
    target.write_text(json.dumps(artifact), encoding="utf-8")
    _refresh_entry(root=tmp_path, entry=_entry_for(manifest, relative))
    _rewrite_manifest(tmp_path, manifest)

    with pytest.raises(ProvenanceError, match="exact option groups"):
        validate_bundle(tmp_path)


def test_resolved_options_require_every_nested_field(tmp_path: Path) -> None:
    """Reject a complete-looking tree with one effective MATLAB field omitted."""
    manifest = _write_bundle(tmp_path)
    relative = "resolved_options/trace3_pythia_skip.json"
    target = tmp_path / relative
    artifact = json.loads(target.read_text(encoding="utf-8"))
    artifact["options"]["pythia"].pop("skip")
    target.write_text(json.dumps(artifact), encoding="utf-8")
    _refresh_entry(root=tmp_path, entry=_entry_for(manifest, relative))
    _rewrite_manifest(tmp_path, manifest)

    with pytest.raises(ProvenanceError, match="pythia.*MATLAB schema"):
        validate_bundle(tmp_path)


def test_resolved_options_index_must_link_the_matching_artifact(
    tmp_path: Path,
) -> None:
    """Reject a variant index that points at a different options artifact."""
    manifest = _write_bundle(tmp_path)
    manifest["resolved_options"]["variants"][0][
        "path"
    ] = "resolved_options/legacy_svm.json"
    _rewrite_manifest(tmp_path, manifest)

    with pytest.raises(ProvenanceError, match="Resolved-options path"):
        validate_bundle(tmp_path)


def test_resolved_options_index_requires_complete_records(tmp_path: Path) -> None:
    """Reject a variant index that omits its artifact description."""
    manifest = _write_bundle(tmp_path)
    manifest["resolved_options"]["variants"][0].pop("description")
    _rewrite_manifest(tmp_path, manifest)

    with pytest.raises(ProvenanceError, match="invalid structure"):
        validate_bundle(tmp_path)


def test_resolved_options_artifact_name_must_match_variant(tmp_path: Path) -> None:
    """Reject an artifact whose internal identity disagrees with its index."""
    manifest = _write_bundle(tmp_path)
    relative = "resolved_options/legacy_svm.json"
    target = tmp_path / relative
    artifact = json.loads(target.read_text(encoding="utf-8"))
    artifact["name"] = "trace3_default"
    target.write_text(json.dumps(artifact), encoding="utf-8")
    _refresh_entry(root=tmp_path, entry=_entry_for(manifest, relative))
    _rewrite_manifest(tmp_path, manifest)

    with pytest.raises(ProvenanceError, match="artifact name"):
        validate_bundle(tmp_path)


def test_resolved_variant_values_must_match_export_profile(tmp_path: Path) -> None:
    """Reject a structurally complete tree describing the wrong variant."""
    manifest = _write_bundle(tmp_path)
    relative = "resolved_options/trace3_pythia_skip.json"
    target = tmp_path / relative
    artifact = json.loads(target.read_text(encoding="utf-8"))
    artifact["options"]["pythia"]["skip"] = False
    target.write_text(json.dumps(artifact), encoding="utf-8")
    _refresh_entry(root=tmp_path, entry=_entry_for(manifest, relative))
    _rewrite_manifest(tmp_path, manifest)

    with pytest.raises(ProvenanceError, match="PYTHIA options mismatch"):
        validate_bundle(tmp_path)


def test_profile_rejects_path_metadata_mismatch(tmp_path: Path) -> None:
    """Require manifest phase, stage, and variant to agree with the path."""
    manifest = _write_bundle(tmp_path)
    entry = _entry_for(
        manifest,
        "explore_data/trace/trace3_default/outputs/membership.csv",
    )
    entry["phase"] = "build"
    _rewrite_manifest(tmp_path, manifest)

    with pytest.raises(ProvenanceError, match="Manifest metadata"):
        validate_bundle(tmp_path)


def test_trace_geometry_schema_accepts_parts_and_holes(tmp_path: Path) -> None:
    """Accept reconstructable region/ring geometry without closure vertices."""
    manifest = _write_bundle(tmp_path)
    entry = _entry_for(
        manifest,
        "build_data/trace/trace3_default/outputs/good_algo_a.csv",
    )
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
    entry = _entry_for(
        manifest,
        "build_data/trace/trace3_default/outputs/good_algo_a.csv",
    )
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
