# Development branch consolidation (9 October 2026)

Integration branch: `codex/instancespace-consolidation`, based on Python main
`66e65bec30bfad30ef8e8bcf805330dfbda04a83`. The combined changes are maintained locally on this branch.
Remote branch heads were checked with `git ls-remote`; no branches were deleted.

| Branch | Relationship to main / disposition |
| --- | --- |
| `codex/matlab-parity-next-wave` | Fully merged; no unique commits. |
| `codex/validation-serialization-trace3` | Fully merged; no unique commits. |
| `copilot/fix-failing-github-actions-job` | Fully merged; no unique commits. |
| `v0.9.0/development-branch-QSF` | Fully merged; no unique commits. |
| `claude/pyinstancespace-docs-issues-whzkg4` (PR #343) | 14 commits ahead. Curated documentation site, build configuration and contributor documentation integrated. Its #345 scoring fix is superseded by per-observation masking here. Older oracle workflows are superseded by approved/candidate separation; publishing automation is not imported. |
| `matlab-fixture-refresh-2026-09-26` (PR #346) | 6 commits ahead. Per-observation scoring fixes and architecture-documentation corrections adapted and extended here. Replacement fixture bundle has not been promoted; preserve approved pins until independent review. |
| `matlab-oldgold-linux-control-2026-09-26` | One raw export commit; comparison evidence, not a source-code development line. Do not merge over approved fixtures. |
| `matlab-svmfix-run1-2026-09-26` | One raw export commit; comparison evidence, not a source-code development line. |
| `matlab-svmfix-run2-2026-09-26` | One raw export commit; comparison evidence, not a source-code development line. |
| `feature/staged-matilda-support` | Two unique commits on a base 225 commits behind main. Progress-reporting feature requires a separate port/review. Its picklable factory and selected-feature indexing fixes already have counterparts on main. Not merged wholesale. |
| `claude/pyinstancespace-bootstrap-fork-dbhu4t` | One unique assistant-skill documentation commit; separate from package fixes. Not integrated. |

Dependency-update branches are separate upgrades, not missing parity fixes, and
are outside this consolidation.

The MATLAB repository is separate: its `codex/octave-compatibility-foundation`
branch is already included in current remote master `929acfd` (PR #65). The local
MATLAB checkout has been left unchanged. Geometry references were generated from
a clean checkout of that current master with local MATLAB R2026a Update 5.

This is selective consolidation, not a claim that all branch tips have been
merged. The user confirmed that consolidation covers PR #343 and PR #346, while the
three raw export branches remain separate. The old progress-reporting feature
and assistant skill documentation are outside that scope; any full fixture
promotion still requires numerical review. Original branch tips
remain available as evidence. No original branch or PR is closed or deleted by this integration.

Validation: 1,090 broad-suite passes plus the separately passing MATLAB source
check; 196 focused regression passes; strict mypy (88 files), Ruff, changed-file
Black formatting, documentation build/local links and fixture provenance passed.
Original PRs remain available for review history; the combined branch is the
proposed development line, with replacement fixture approval explicitly deferred.
