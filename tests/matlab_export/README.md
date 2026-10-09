# MATLAB fixture export

This directory provides the reproducible MATLAB reference-data workflow for issues
[#278](https://github.com/andremun/pyInstanceSpace/issues/278) and
[#310](https://github.com/andremun/pyInstanceSpace/issues/310), extended with PILOT
evidence for [#262](https://github.com/andremun/pyInstanceSpace/issues/262) and native
3D TRACE evidence for [#265](https://github.com/andremun/pyInstanceSpace/issues/265).

## Trust contract

`pyis_export_reference_data.m` has two modes:

- `verified` requires clean MATLAB and Python repositories, MATLAB R2026a,
  the required toolboxes, full Git commits, and a new output path. Only this mode may
  produce parity fixtures.
- `diagnostic` permits an older or dirty environment. It exercises the exporter but
  its output is never a MATLAB oracle.

Both modes preflight MATLAB plus Statistics and Machine Learning, Optimization, Global
Optimization, and Financial Toolbox. PRELIM calls `boxcox`, so Financial Toolbox is an
execution dependency rather than optional provenance metadata.

The exporter writes to scratch space and publishes atomically. `manifest.json`
records the repositories, exporter hash, MATLAB environment, dataset, file hashes,
CSV shapes, semantic roles, and explicit empty artifacts. Each variant links a separate
JSON artifact containing the complete effective option tree after MATLAB validation and
default resolution; partial `pythia`/`trace` overrides are not provenance.

The verifier keeps the 229-file `reference-export/v1` contract frozen and readable.
The canonical installed oracle uses the additive 423-file `reference-export/v2`
contract. V2 requires eight complete option artifacts, every declared build/explore
stage, PILOT solver inputs and lineage, raw metrics, memberships, and all 2D/3D
algorithm geometry. Deleting a file and its manifest entry therefore remains an error.

Verified v2 also pins the audited MATLAB commit, both canonical input hashes, their
algorithm headers, and a versioned exporter-script hash. A gold-source, dataset, or
exporter change requires an explicit verifier-profile update and fixture regeneration.
Diagnostic exports remain flexible and v1 remains frozen.

## Candidate validation and promotion

MATLAB `master` is the current implementation target. The committed oracle is an
approved snapshot of a particular commit; release validation checks that immutable
revision. A separate, non-blocking CI job reports when master has advanced.

New exports must pass candidate validation before numerical review. Use the full
MATLAB and Python generator commits requested for the export, and the exporter
script from that generator checkout. Do not simply copy claimed identities from
the candidate manifest:

```bash
python tools/fixture_provenance.py candidate /path/to/new-export \
  --matlab-commit "$MATLAB_COMMIT" \
  --generator-commit "$GENERATOR_COMMIT" \
  --exporter-script /path/to/generator/tests/matlab_export/pyis_export_reference_data.m
```

The candidate must be a clean verified-mode R2026a v2 export. Validation retains
the canonical dataset hashes, complete file set, per-file hashes, effective-option
checks, geometry checks and PILOT lineage checks. It compares source identities
with the requested run, rather than the old approval, and computes a new content
root without requiring byte equality with the old bundle. Its report explicitly
says `matlab-candidate` and `approved: false`; this does not change the export's
manifest. Hash consistency is not a claim of scientific equivalence or independent
authentication of the export job.

Prepare an immutable review package at a **new** destination with:

```bash
python tools/fixture_provenance.py prepare-promotion /path/to/new-export \
  /path/to/promotion-review \
  --matlab-commit "$MATLAB_COMMIT" \
  --generator-commit "$GENERATOR_COMMIT" \
  --exporter-script /path/to/generator/tests/matlab_export/pyis_export_reference_data.m
```

This copies and revalidates the candidate, then publishes a directory containing
`bundle/`, `promotion.json` (previous/proposed identities and manifest hash), and
review instructions. It never edits approved fixtures or verifier pins. An existing
destination is rejected. Failed preparation cleans up its staging directory.

Promotion is a reviewed Git commit, not an automatic consequence of validation:

1. Review numerical and semantic differences, including the stage-local tests.
2. Replace `tests/fixtures/matlab/current` with the reviewed bundle and update
   `_GOLD_MATLAB_COMMIT`, `_REFERENCE_V2_EXPORTER_SHA256`, and
   `_VERIFIED_V2_CONTENT_ROOT_SHA256` in `tools/fixture_provenance.py` from the
   review record. Include the matching exporter if it changed.
3. Update `tests/fixture_inventory.json` if paths changed, and review affected
   reference assertions. Schema or option-contract changes require an explicit
   verifier-profile update, not a bypass flag.
4. Run `verify`, `inventory`, and the parity suite; commit all changes together.
   Do not publish a pin-only or manifest-only update.

The existing `verify` and `install` commands retain their approved-identity checks.
An unapproved new candidate remains ineligible for installation. The diagnostic
mode is not a substitute for candidate validation.

For numerical PILOT evidence, verification decodes every MATLAB-order solution column,
recomputes its weighted reconstruction objective and topology score, and selects the
precalculated replay from those recomputed scores rather than trusting the exported
diagnostic vectors.

Historical files remain classified as `legacy-unknown`, `python-regression`,
`python-synthetic`, or `test-scratch` in `tests/fixture_inventory.json`. They are not
silently promoted to MATLAB references.

## Output layout

```text
<bundle>/
├── manifest.json
├── shared_inputs/reference/{metadata.csv,metadata_test.csv}
├── resolved_options/<variant>.json
├── build_data/<stage>/<variant>/{inputs,outputs}/
└── explore_data/<stage>/<variant>/{inputs,outputs}/
```

Both build and explore stages carry their explicit numeric inputs. A reviewed bundle is
installed unchanged at `tests/fixtures/matlab/current`; no alternate flattened tree is
supported.

The downstream variants are:

- `trace3_default`: current KNN/Sobol/TRACE3 path;
- `trace3_pythia_skip`: TRACE3 true-label fallback;
- `legacy_svm`: retained legacy TRACE regression.

V2 adds five stage-level PILOT variants:

- standard analytic 3D with one global viewpoint;
- standard numerical 3D from explicit three-column `X0` while `ntries=1`;
- exact replay of that run's best `precalcAlpha` solution;
- shifted-input MATLAB SIMPLS in 2D; and
- the same shifted-input SIMPLS in 3D with uneven grouped viewpoints.

The PLS shift makes MATLAB's internal centring observable. The current exporter
rebuilds SIFTED for each variant's PILOT dimensions using the public stage API.
Stage-context v2 records that rebuild, the dimensions and the fitted feature mean;
exploration reconstructs `Z=(X-Xmean)*A'`. The canonical dataset must still yield
the same selected inputs across variants, checked by the validator, so the PLS
2D/3D component comparison remains meaningful. Coordinate columns are emitted as
`z_1` through `z_d`.

The approved bundle retains its historical stage-context v1 (retained 2D SIFTED,
uncentred exploration). Its source/hash pins are unchanged. The validator accepts
both explicit context versions; new exports remain candidates until numerical
review and promotion.

The already-built `pilot_standard_analytic_3d` model also supplies the TRACE3 build
and explore evidence. It does not add a duplicate resolved-options variant. Every
good, best, and hard footprint has four explicit artifacts: alpha-shape vertices,
tetrahedra, outward boundary faces, and the descending alpha spectrum. Indices are
one-based. Empty footprints keep all four headers and no rows. Raw metrics retain the
final alpha, stored `RegionThreshold`, region/tetrahedron/face counts, volume, surface
area, and empty state; verification recomputes topology, orientation, geometry,
membership, and rescored summaries without trusting row order.

TRACE geometry is region-aware: each row identifies the part, ring, vertex, and
whether the ring is a hole. Empty geometry is a header-only CSV. Raw footprint
metrics and membership are exported separately; rounded summaries are not the
numeric oracle.

## Generate and verify

Run from MATLAB with paths adjusted for the two checkouts:

```matlab
addpath('/path/to/pyInstanceSpace/tests/matlab_export');
pyis_export_reference_data( ...
    '/path/to/InstanceSpace', ...
    '/new/path/reference-bundle', ...
    'generatorRoot', '/path/to/pyInstanceSpace', ...
    'mode', 'verified');
```

Then validate independently in Python:

```bash
python -m tools.fixture_provenance verify /new/path/reference-bundle
```

For exporter debugging only, use MATLAB mode `diagnostic` and add
`--allow-diagnostic` to the verifier command.
Diagnostic parity-reader calibration additionally requires the explicit
`PYIS_ALLOW_DIAGNOSTIC_FIXTURES=1` environment opt-in; readers reject diagnostic
bundles by default.

After scientific review, install a verified bundle without flattening or overwriting
paths:

```bash
python -m tools.fixture_provenance install \
  /new/path/reference-bundle tests/fixtures/matlab/current
```

## Current execution status

The canonical oracle at `tests/fixtures/matlab/current/` is a reviewed, installed
`reference-export/v2` bundle with 423 files and `matlab-verified` trust. It was generated
under MATLAB R2026a Update 4 from clean MATLAB
`98a01ac0513c0dd0f8a9bd91ed2926c871334d7b` (InstanceSpace v0.9.1) and clean
Python generator `4816b8cf23ad9392e7a7f5aa85bfbc32080dfe84`. The exporter identity is pinned to
`d11293556b12beb63e3320094a2340ba3f7f8b7a58677ff404f20c0ba3b7350c`.

Collection contains 86 provenance tests and 41 current-gold scientific readers. The
local CI-equivalent gate passed all 1,046 collected tests with 92.08% branch coverage
and no uncaught warnings under `-W error`. Frozen v1 bundles remain verifiable, but they
are not the installed current oracle. Diagnostic and `legacy-unknown` snapshots remain
non-oracles.

### Explicit TRACE boundary tolerance

Current MATLAB exports use `trace.boundaryTolerance` (default zero) for both
TRACE metrics and exported membership. Python's corresponding option is
`trace.boundary_tolerance`; both specify an absolute Euclidean distance in
projection units. The fitted value controls inference. The fixture validator
uses the declared tolerance when independently reconstructing 3D memberships.
Historical bundles without the option retain exact semantics. New 2D TRACE
coordinates and ring vertices are written at full double precision. This does
not change the approved bundle or its source/hash pins.
