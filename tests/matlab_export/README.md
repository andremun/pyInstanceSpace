# MATLAB fixture export

This directory provides the reproducible MATLAB reference-data workflow for issues
[#278](https://github.com/andremun/pyInstanceSpace/issues/278) and
[#310](https://github.com/andremun/pyInstanceSpace/issues/310), extended with PILOT
evidence for [#262](https://github.com/andremun/pyInstanceSpace/issues/262).

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

The verifier keeps the installed 229-file `reference-export/v1` contract frozen and
enforces the additive `reference-export/v2` contract for new exports. V2 requires eight
complete option artifacts, every declared build/explore stage, PILOT solver inputs and
lineage, raw metrics, memberships, and all algorithm geometry. Deleting a file and its
manifest entry therefore remains an error.

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

The installed current oracle is the reviewed 229-file v1 bundle generated on 2026-08-18
under MATLAB R2026a Update 4 from clean MATLAB `34c0129` and Python generator `b87179f`.
On 2026-08-20 the prospective 323-file v2 profile completed all eight variants under
R2026a and passed the strict verifier in diagnostic mode. It remains non-oracle until a
clean verified run is reviewed and installed atomically. Historical snapshots retain
their non-oracle trust classes.
