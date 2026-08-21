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

The PLS shift makes MATLAB's internal centring observable. Each PILOT variant records
that it reuses the default 2D SIFTED snapshot; it proves the PILOT/viewpoint and public
explore projection paths on fixed inputs, not a separate end-to-end SIFTED-3D run.
Coordinate columns are emitted as `z_1` through `z_d`. Explore keeps MATLAB's public
uncentred `Z=X*A'` inference behavior, including for PLS.

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
`34c01293fef99b4eabd53323c393cb184cc95a8e` and clean Python generator
`cf3cde0da5a3067300bd94a48d4d09ff5cf20b0c`. The exporter identity is pinned to
`d11293556b12beb63e3320094a2340ba3f7f8b7a58677ff404f20c0ba3b7350c`.

Collection contains 86 provenance tests and 40 current-gold scientific readers. The
CI-equivalent Linux gate passed all 1,039 collected tests with 92.00% branch coverage
and no uncaught warnings under `-W error`. Frozen v1 bundles remain verifiable, but they
are not the installed current oracle. Diagnostic and `legacy-unknown` snapshots remain
non-oracles.
