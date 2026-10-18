# Test suite

Tests live flat under `tests/`. Training-stage tests use
`test_build_<stage>.py`; fitted inference uses `test_explore_<stage>.py`.
Cross-stage, infrastructure, persistence and plotting tests use descriptive names.
PREPROCESSING and CLOISTER have no explore-time predictive stage.

## Reference trust

| Data | Purpose |
|---|---|
| `fixtures/matlab/current/` | Approved 423-file `reference-export/v2` MATLAB oracle, authenticated before parity readers use it. |
| `fixtures/matlab/geometry/` | Independently hashed stage-level CLOISTER/TRACE geometry cases. |
| `matlab_reference/`, older `pilot_pythia/` data | Unknown-provenance regression snapshots; they cannot establish current MATLAB parity. |
| Synthetic/regression fixtures | Controlled Python behavior and edge cases; classifications are in `fixture_inventory.json`. |

See [fixture provenance](../docs/fixture-provenance.md) for approved source
identities and scientific limits, and [the exporter](matlab_export/README.md) for
generation, candidate review and promotion. Preserve fixture data and pins during
ordinary documentation or dependency work.

## Run checks

Run from the repository root in the locked Python 3.12 development environment:

```sh
poetry run pytest --strict-config --strict-markers -ra -W error \
  --cov=instancespace --cov-report=term-missing --cov-report=xml
poetry run python tools/fixture_provenance.py verify tests/fixtures/matlab/current
poetry run python tools/fixture_provenance.py inventory . tests/fixture_inventory.json
```

The configured coverage gate is 85%, including branches. `poetry run poe test`
also runs Ruff, strict mypy, Black and the documentation build.

For focused checks:

```sh
poetry run pytest tests/test_explore_trace.py -v -s -W error
poetry run pytest tests/test_fixture_provenance.py tests/test_current_matlab_*.py -W error
poetry run pytest tests/test_documentation.py tests/test_bayes_comparison_evidence.py -W error
```

## Scientific assertions

Current MATLAB readers check the manifest and scientific contracts: standard and
SIMPLS projection, viewpoints, geometry/topology, boundary membership, metrics and
rescoring. Assertions state their exact comparisons or local numerical tolerance
and rationale. Keep the rationale with the assertion rather than duplicating a
threshold table that can drift from the tests.

Unknown-provenance snapshots can detect regressions but do not authenticate a
MATLAB result. Diagnostic exports require the explicit opt-in described in the
exporter guide. Never relax a tolerance solely to hide a changed boundary,
rounded summary or optimizer trajectory.

The Bayesian evidence tests validate recorded traces and cross-scores without
rerunning costly stochastic optimization or requiring MATLAB in CI. Documentation
tests execute published examples and check all options, loader aliases, API
anchors and local site links.
