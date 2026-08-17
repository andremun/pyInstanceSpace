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
    _expect_object(manifest, "dataset")
    options = _expect_object(manifest, "resolved_options")
    variants = _expect_list(options, "variants")
    if not variants:
        raise ProvenanceError("resolved_options.variants must not be empty")

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
        if _release_key(matlab_release) < _release_key("R2025a"):
            raise ProvenanceError(
                "Verified fixtures require MATLAB R2025a or newer, "
                f"got {matlab_release}",
            )

    entries = _expect_list(manifest, "files")
    if not entries:
        raise ProvenanceError("Fixture manifest must describe at least one file")

    expected_paths: set[str] = set()
    folded_paths: set[str] = set()
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

    return BundleReport(
        root=bundle_root,
        trust=trust,
        matlab_release=matlab_release,
        file_count=len(entries),
        total_bytes=total_bytes,
    )


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
