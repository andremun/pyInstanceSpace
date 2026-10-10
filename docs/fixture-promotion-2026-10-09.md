# Approved Linux fixture promotion (9 October 2026)

This promotion replaces the complete 423-artifact reference bundle and its three
verifier pins together. The user authorised promotion after the TRACE boundary
review. It is a reproducible reference for the reviewed changes, not a claim that
all open scientific comparisons or platform differences have been resolved.

## Sources and reproduction

- MATLAB source: `2e2cf7565aee8e60190e788c959405d44c4317f3` (PR #66).
- Python exporter source: `ab72f6bc5c602e0cb03a65041a4677037282e92c` (PR #348).
- Exporter SHA-256: `ebd7917169ca110fd26856ac8cbad43e3c3477b1ce96473e4f5bc9445ff869d6`.
- Artifact content root: `4c4279651bf9f13b20a5e539cd5b5a123beaeb9ebea4a6699b6d0c1a05bc036a`.
- Runtime: MATLAB R2026a Update 5, `26.1.0.3346908`, Linux GLNXA64.
- Two clean checkouts, two independent MATLAB processes, one computation thread,
  parallel execution disabled, exporter seed 42. Both candidate validations passed;
  all 423 artifact hashes match. Only the manifest generation timestamp differs.
- TRACE boundary tolerance is explicitly zero. Coordinate/ring exports retain
  full double precision; positive tolerances are covered separately by the shared
  hand-labelled boundary contracts.

The subsequent MATLAB merge commit `8790f452a3a6178d9768c7793862b4aeeb26d952`
incorporates current master. Its tree is identical to the recorded source tree:
`99ce209823cd2350f91b7bcdbe3d694a82c8eb66`. The sole merge conflict was the generated
documentation search index. No generated fixture identity was rewritten.

Reproduce with the exporter from the pinned Python revision:

```matlab
addpath('/path/to/pinned-python/tests/matlab_export');
maxNumCompThreads(1);
pyis_export_reference_data('/path/to/pinned-matlab', '/new/export-directory', ...
    'generatorRoot', '/path/to/pinned-python');
```

Local repeat exports, logs, changed-path lists, candidate review record and
reproducibility evidence are retained under
`.cache/matlab-review-2026-10-09/boundary-promotion/` and
`.cache/matlab-review-2026-10-09/boundary-repeat-export/`. The committed manifest
records every approved artifact hash and both source identities.

## Numerical review

218 artifact hashes change from the previous approved bundle; the path set is
unchanged, so the inventory classification needs no change. These comprise PRELIM
(2), SIFTED (2), PILOT (70), CLOISTER (3), PYTHIA (35), TRACE (98), and resolved
options (8). See the [earlier numerical audit](matlab-parity-review-2026-10-05.md)
and [TRACE investigation](trace-fixture-investigation-2026-10-09.md).

The new comparison confirms the same 487 old/new TRACE membership changes:
44 default TRACE3, 198 skip-PYTHIA TRACE3, 172 analytic 3D, and 73 legacy SVM.
466 are numerical boundary cases; 21 legacy changes arise from discontinuous
contradiction removal under tiny projection differences. Default-zero tolerance
preserves that behaviour. A supplied positive tolerance does not guarantee stable
legacy trimming, and this promotion does not make that claim.

The current 2D TRACE3 parity tests now require exact Python/MATLAB exploration
membership and summary agreement. The five old serialization exceptions and their
summary/count adjustments have been removed. Build metrics and geometry comparisons
retain their existing tolerances. PLS exploration tests now supply the exported
fitted mean, matching the corrected model contract; projection tolerances are
unchanged. The PILOT scalar error assertion retains its arithmetic allowance and
adds the half-unit (5e-12) of its 15-significant-digit CSV representation at the
exported magnitude. Projection-factor equality remains exact.

## Scope and validation

Promotion completes the reproducible-fixture work in #347; PR #348 closes it on
merge, alongside #344 and #345. Bayesian-search comparison #304 and cross-platform
legacy probability tolerance #341 remain open. Dependency upgrades and the proposed
repository-history restart remain deferred.

Validation: approved-bundle verification passes for 423 files and inventory checks
pass for 771 classified files. Strict mypy passes 92 source files; changed Python
files pass Ruff and Black; documentation builds and whitespace checks pass. The
initial full suite passed 1,118 tests and identified two further stale uncentred
PILOT reader stubs. Both were corrected; the complete 16-test PILOT exploration
file then passed. A final full run is recorded with the PR validation results.
