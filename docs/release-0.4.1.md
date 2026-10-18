# Release 0.4.1 preparation

Scope: a unified MATLAB-style Python reference website (#367), complete checked
option documentation, and a controlled equal-budget Bayesian tuning comparison
(#304). User guides describe current behavior. Maintainer records retain current
contracts, release preparation and reproducible scientific evidence.

Compatibility classification: **Additive**. The documentation, comparison tooling
and evidence do not change numerical defaults or the existing runtime API. Any
confirmed runtime defect found during comparison must be recorded separately with
its regression evidence before being included.

## Validation record

Validated on Linux with Python 3.12.14 and Poetry 2.3.3:

| Gate | Result |
|---|---|
| Documentation | 42 HTML pages; all 74 options covered; local links, fragments and duplicate IDs checked |
| Examples | README and guides execute real builds, staged runs, exports, checkpoints, model reload and exploration |
| Visual review | MATLAB comparison, desktop/mobile, keyboard search, drawer dismissal, copy buttons and nested API links checked; [screenshots](reviews/documentation-0.4.1/README.md) |
| Strict typing | Mypy passes 98 source files |
| Formatting and lint | Black checks 99 files; Ruff passes |
| Approved fixtures | All 423 files authenticate; 771 inventory entries classify; no approved data changes |
| Dependencies | All 80 locked package records unchanged; no known vulnerabilities in the lock audit |
| Distributions | Wheel and sdist build; Twine, lock consistency and installed requirements checks pass |
| Installed wheel | Real build, export, checkpoint reload and exploration pass outside the checkout |
| Runtime source | AST comparison confirms 272 function bodies and calling arguments unchanged in 13 edited runtime modules |

The numerical baseline passes all **1,137 tests**, without skips, with
**92.84% branch coverage**, above the 85% requirement. The documentation-only
cleanup additionally passes 36 focused example/evidence/compatibility checks,
skill validation, and a local Markdown-link audit. Runtime and test function
bodies remain unchanged. The release gate must pass on the final PR head.
Poetry's existing metadata-deprecation notices remain; lock consistency succeeds.
The preparation changes do not modify MATLAB source or replace its approved
fixture bundle. Separate research artifacts pin the comparison source and inputs.

## Upgrade

This is an additive documentation and evidence release. Existing fitted models,
checkpoints, option names, schemas and calling conventions remain supported. No
retraining is required solely for this release. Numerical function bodies and
runtime dependency versions are unchanged. The two Series return annotations
are made introspectable while retaining static pandas typing.

The website has a curated entry point; generated API paths and member anchors
remain available. The repository ISA skill supports Python analysis and
interpretation. Superseded plans and stale notebooks are removed; current
architecture, fixture provenance and research evidence remain outside the site.

The default Bayesian budget remains 20 with four initial points and plain EI.
The [controlled comparison](research/bayes-304/report.md) documents the accepted
backend differences and bounded scope. This does not claim arbitrary Python and
MATLAB trajectories or classifiers are numerically identical.

## Publication

1. Require the complete release gate on the exact PR head before merging, then
   confirm the gate and Pages deployment on main.
2. Confirm `pyproject.toml` and citation metadata identify 0.4.1. Confirm PyPI
   does not already contain that version and verify the trusted publisher and
   `pypi` environment still match `.github/workflows/publish-pypi.yml`.
3. Prepare GitHub tag `v0.4.1` at the verified main commit, using the 0.4.1
   release-notes section and upgrade guidance for the release description.
4. Publish the GitHub Release when authorized. This triggers the PyPI workflow;
   a PR, merge or tag alone does not publish. Manual workflow dispatch is also
   a real upload, not a validation operation.
5. Check the publication workflow and install `instancespace==0.4.1` in a fresh
   Python 3.12 environment. Run the Getting Started build/explore example and
   retain the commit and distribution hashes with the release record.
6. Review the version-update freeze in `.github/dependabot.yml` and the
   separately tracked numerical dependency migrations before resuming update PRs.

No release or PyPI upload is performed by this preparation PR.
