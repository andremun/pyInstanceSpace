# MATLAB fixture export

This directory provides the reproducible MATLAB reference-data workflow for issues
[#278](https://github.com/andremun/pyInstanceSpace/issues/278) and
[#310](https://github.com/andremun/pyInstanceSpace/issues/310).

## Trust contract

`pyis_export_reference_data.m` has two modes:

- `verified` requires clean MATLAB and Python repositories, MATLAB R2025a or newer,
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

The verifier enforces the `reference-export/v1` profile as well as manifest hashes. It
requires the shared inputs, three option artifacts, every declared build/explore stage,
raw metrics, memberships, and all algorithm geometry. Deleting a file and its manifest
entry therefore remains an error.

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

The current variants are:

- `trace3_default`: current KNN/Sobol/TRACE3 path;
- `trace3_pythia_skip`: TRACE3 true-label fallback;
- `legacy_svm`: retained legacy TRACE regression.

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

After scientific review, install a verified bundle without flattening or overwriting
paths:

```bash
python -m tools.fixture_provenance install \
  /new/path/reference-bundle tests/fixtures/matlab/current
```

## Current execution status

On 2026-08-18 verified mode completed all three variants under MATLAB R2026a Update 4 from
clean MATLAB `34c0129` and Python generator `b87179f`. The strict verifier accepted 229
manifest-listed files plus `manifest.json`; the reviewed bundle is installed unchanged at
`tests/fixtures/matlab/current/`. The R2024a diagnostic and all historical snapshots retain
their non-oracle trust classes. The initial missing-Financial-Toolbox attempt published no
partial bundle.
