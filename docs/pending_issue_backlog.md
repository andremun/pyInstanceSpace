# Pending issue backlog

Reviewed against the 17 open GitHub issues on 2026-08-17. This file records local
results only; no issue was closed or edited remotely.

## Ready for maintainer review

| Issue | Status | Recommended action |
|---|---|---|
| [#302](https://github.com/andremun/pyInstanceSpace/issues/302) | Implemented and tested | Close after review. |
| [#314](https://github.com/andremun/pyInstanceSpace/issues/314) | Implemented and tested | Close after review. |
| [#317](https://github.com/andremun/pyInstanceSpace/issues/317) | Implemented and tested | Close after review. |
| [#315](https://github.com/andremun/pyInstanceSpace/issues/315) | Premise disproved | Close as invalid or stale; MATLAB membership is boundary-inclusive. |

## Deferred open issues

| Issue | Priority | Next step |
|---|---:|---|
| [#313](https://github.com/andremun/pyInstanceSpace/issues/313) | High | Design and port TRACE3 with independent MATLAB fixtures. |
| [#262](https://github.com/andremun/pyInstanceSpace/issues/262) | High | Add PILOT dimensions, viewpoints, and 3D parity as a separate feature. |
| [#265](https://github.com/andremun/pyInstanceSpace/issues/265) | Medium | Follow #262 with 3D output and visualization support. |
| [#316](https://github.com/andremun/pyInstanceSpace/issues/316) | Medium | Design a shared train/infer stage contract before changing orchestration. |
| [#278](https://github.com/andremun/pyInstanceSpace/issues/278) | High, blocked | Regenerate and document fixtures in a real MATLAB environment. |
| [#310](https://github.com/andremun/pyInstanceSpace/issues/310) | Blocked | Reorganize fixtures only after #278 establishes provenance. |
| [#305](https://github.com/andremun/pyInstanceSpace/issues/305) | Tracker | Reassess after #278 and #310. |
| [#273](https://github.com/andremun/pyInstanceSpace/issues/273) | Tracker | Reassess after provenance and fixture work. |
| [#304](https://github.com/andremun/pyInstanceSpace/issues/304) | Low | Benchmark Bayesian convergence before tuning defaults. |
| [#297](https://github.com/andremun/pyInstanceSpace/issues/297) | Tracker | Mark #302 addressed, then audit remaining linked work. |
| [#272](https://github.com/andremun/pyInstanceSpace/issues/272) | Medium | Reproduce the alpha-boundary failure before adding retries. |
| [#270](https://github.com/andremun/pyInstanceSpace/issues/270) | Tracker | Reassess after #272. |
| [#260](https://github.com/andremun/pyInstanceSpace/issues/260) | Tracker | Keep open for the major parity features above. |

## New audit backlog

### Data validation

- Require one instance column, numeric and uniquely named feature/algorithm columns,
  and minimum viable dimensions.
- Reject manual feature or algorithm selections that match no columns.
- Validate the remaining active option fields before stage execution.

### Serialization and plotting

- Stop CSV export from mutating model-owned DataFrames.
- Preserve MultiPolygon parts and holes without false connecting edges.
- Sanitize label-derived paths and prevent duplicate flattened ZIP members.
- Guard plot normalization for constant or all-NaN data.
- Replace swallowed save errors with actionable exceptions.

### Quality gate

- Remove NumPy matrix usage in TRACE and reduce the roughly 19,700 warnings in the
  full suite.
- Guard empty PYTHIA metric denominators and clean serializer warnings.
- Resolve the inherited test-suite gate debt: about 340 Ruff findings, six
  Black-formatted test files, and ten non-production strict-mypy findings.
- Repair stale README paths, the invalid documentation conversion target, and the
  `InstanceSpaceOptions.default()` example.

These items are reliability work, but none blocks the corrected build path validated
in this pass.
