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

The exporter writes to scratch space and publishes atomically. `manifest.json`
records the repositories, exporter hash, MATLAB environment, resolved options,
dataset, file hashes, CSV shapes, semantic roles, and explicit empty artifacts.

Historical files remain classified as `legacy-unknown`, `python-regression`,
`python-synthetic`, or `test-scratch` in `tests/fixture_inventory.json`. They are not
silently promoted to MATLAB references.

## Output layout

```text
<bundle>/
├── manifest.json
├── shared_inputs/<dataset>/
├── build_data/<stage>/<variant>/{inputs,outputs}/
└── explore_data/<stage>/<variant>/{inputs,outputs}/
```

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

On 2026-08-17 the hardened exporter completed all three variants under MATLAB
R2024a and its 196-file, 1.1 MB diagnostic bundle passed the independent hash,
shape, path, toolbox, and file-set verifier. This proves the workflow executes, but
does not close #278: the repository declares R2025a+, and both source repositories
must be clean for a verified run. Consequently #310's committed-fixture migration
remains gated; no unknown-provenance fixture was moved or relabelled.
