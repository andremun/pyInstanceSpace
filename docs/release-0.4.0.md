# Release 0.4.0 preparation

Version 0.4.0 is prepared for publication. A merge to main does not publish a
package. The maintainer publishes a GitHub Release only after the release gate
for its exact commit is green. The package continues to require Python 3.12.

## Scope and upgrade notes

This release consolidates the October correctness, MATLAB parity, fixture,
documentation and dead-code audits. See [Release Notes](../RELEASE_NOTES.md)
for the detailed changes. It retains Python's legacy TRACE default and the
existing Bayesian implementation. Neither policy is changed for this release.

- PYTHIA excludes unobserved outcomes from scoring and uses observed-only
  algorithm, Oracle and selector probabilities. Undefined rates are NaN.
- Cost-sensitive models use per-instance regret. Retrain existing weighted
  models to obtain this correction; loading an old model does not refit it.
- PRELIM removes algorithms that are never good in training and recomputes the
  retained portfolio. Fixed PYTHIA parameter rows must describe that portfolio.
- Training records the feature names retained by preprocessing. Exploration
  accepts those columns in a different order and does not need excluded columns.
  Rebuild older models if training removed features; the old metadata fallback
  cannot recover missing fitted feature information.
- Fitted PLS means, constant-coordinate scales, stage options and boundary
  tolerances are persisted. Older models retain documented compatibility fallbacks.
- Direct `PythiaEvaluateInput` construction requires an `observed` mask with shape
  `(n_instances, n_trained_algorithms)`. Direct `PrelimInput` construction requires
  `algo_labels`; `PrelimOutput` includes the retained labels. The high-level
  `InstanceSpace` API supplies these fields.
- TRACE uses an explicit absolute Euclidean boundary tolerance in projection
  units, default zero. MATLAB calls it `boundaryTolerance`; Python uses
  `boundary_tolerance`. Positive tolerance does not buffer area/volume or solve
  discontinuous legacy contradiction trimming.

## Dependency decisions

| Dependency | Release decision |
| --- | --- |
| actions/checkout | v7, merged #353; existing workflows pass the release gate |
| pytest | 9.1.1, merged #354 |
| Black | 26.10.1, repaired lockfile and merged #355 |
| Tornado | Notebook-development dependency patched from 6.5.8 to 6.5.10 |
| mypy | 2.4.0 from #361; correct two Series return annotations without changing function bodies |
| Loguru | 0.7.3 patch from #365 |
| pdoc / PyGAD / joblib | Retain validated 16.0.0 / 3.7.0 / 1.6.0; defer PyGAD 3.8.1 #364 in #358 |
| NumPy | Retain 1.26.4; close #335 as deferred under #358 |
| scikit-learn | Retain 1.5.2; restrict fresh installs to >=1.5.2,<1.6; close #357 under #358 |
| pandas / pandas-stubs | Retain the checked 2.2.3 / 2.2.3.241009 pair; defer #356, #362 and #363 under #359 |

The lockfile pins the tested environment. Wheel requirements express supported
ranges; a fresh pip installation is not an exact copy of the Poetry lock.
Dependabot ignores defer NumPy and pandas major updates, scikit-learn >=1.6,
PyGAD >=3.8 and pandas-stubs >=2.3. Runtime requirements retain the pandas 2.2 and
PyGAD 3.7 patch series; development stubs are pinned to the checked pandas 2.2 pair.
Remove the corresponding restriction when each migration is approved.

Both ecosystems temporarily set `open-pull-requests-limit: 0` while this release
is prepared. This stops the configuration refresh from continually opening new
version-update PRs during final validation; it does not change Dependabot's
separate security-update PR limit. Restore 5 after publishing 0.4.0, retaining
the explicit deferred-version ignores. This freeze is not a permanent update policy.

The initial locked-dependency audit identified three advisories in Tornado 6.5.8:
[GHSA-chx6-46f5-w4vp](https://github.com/tornadoweb/tornado/security/advisories/GHSA-chx6-46f5-w4vp),
[GHSA-c2m8-h5v5-343r](https://github.com/tornadoweb/tornado/security/advisories/GHSA-c2m8-h5v5-343r),
and [GHSA-3hv7-mjh2-fv65](https://github.com/tornadoweb/tornado/security/advisories/GHSA-3hv7-mjh2-fv65).
The patched lockfile audit reports no known vulnerabilities. This is a dated
package-advisory check, not a guarantee about every deployment.

## MATLAB alignment and provenance

The reviewed MATLAB default branch is `ed4bbc26b3381fd97e45d2b32ff93513f7d675c4`.
Its tree equals the approved fixture source
`2e2cf7565aee8e60190e788c959405d44c4317f3`. The 423-artifact approved bundle and
771-file inventory are unchanged. Tests compare controlled scientific contracts;
they do not assert that different optimizers or libraries produce identical
search trajectories on all inputs.

See the [fixture promotion review](fixture-promotion-2026-10-09.md),
[wider parity audit](wider-parity-audit-2026-10-10.md), and
[history restart mapping](history-restart-2026-10-15.md).
The approved exporter's original and rewritten revisions have identical trees
and the same pinned script hash. The manifest keeps its original generator ID.
Do not replace it with a new ID merely to make old links resolve.

## Validation record

Validated locally on 16 October 2026 with Python 3.12.14 on Linux, Poetry
2.3.3 and the release lockfile:

| Gate | Result |
| --- | --- |
| Strict pytest, warnings as errors | 1,128 passed, no skips; 236.29 seconds |
| Coverage | 92.22%, above the configured 85% gate |
| Strict mypy | 93 files pass |
| Ruff / Black | Pass; 97 files unchanged by Black |
| Documentation | Site and API build pass |
| Package | Wheel and source distribution build; lock check and pip check pass |
| Installed wheel | Imports from a separate install directory; real build, explore and saved-model reload pass |
| Approved MATLAB fixtures | All 423 files verify; 771 classified inventory entries pass |
| Locked dependency audit | 80 packages checked; no known vulnerabilities after the Tornado patch |
| History mapping | All 280 mapped commits resolve; approved original/rewritten exporter trees and script hash agree |

The three accepted dependency PRs also passed their GitHub release gates before
merge. The release-preparation PR must pass CI on its exact head before merge;
check main's gate again before publishing a release. No fixture content, verifier
pin, numerical function body or test tolerance changes in this pass. The final
mypy update changes return annotations in PRELIM and PREPROCESSING; an AST
comparison confirms their function bodies are unchanged.
Poetry reports existing metadata-deprecation warnings; lock consistency succeeds.

## Publication procedure

1. Wait for the complete `release_gate` on the exact release commit, and check
   the documentation deployment. Resolve a failing gate before publication.
2. Confirm `pyproject.toml` contains version `0.4.0` and that PyPI does not already
   contain that version. Confirm the PyPI trusted-publisher configuration points
   to `.github/workflows/publish-pypi.yml` and the `pypi` environment. Account and
   environment configuration require the maintainer's access.
3. In GitHub Releases, prepare tag `v0.4.0` at the verified main commit. Use the
   0.4.0 release-notes section and the upgrade notes above for the description.
4. Publish the GitHub Release when ready. This triggers the existing PyPI
   workflow; ordinary pushes, PR merges and tags do not trigger publication.
5. Check the `Publish to PyPI` workflow. Complete any configured environment
   approval. Confirm version 0.4.0 and its Python requirement on PyPI.
6. In a fresh Python 3.12 environment, install `instancespace==0.4.0` and run the
   documented build/explore example. Keep the validated release commit and wheel
   hashes with the release record.
7. Restore `open-pull-requests-limit: 5` for both ecosystems in
   `.github/dependabot.yml`. Retain the version ignores for #358/#359 until each
   numerical/runtime/stub migration is reviewed.

Do not run the publish workflow manually as a pre-release validation step:
`workflow_dispatch` also performs a real PyPI upload. No GitHub Release or PyPI
upload is performed by the release-preparation changes.
