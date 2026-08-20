# Pending issue backlog

Reviewed against the 17 open GitHub issues on 2026-08-18. This file records local
results only; no issue was closed or edited remotely.

## Ready for maintainer review

| Issue | Status | Recommended action |
|---|---|---|
| [#302](https://github.com/andremun/pyInstanceSpace/issues/302) | Implemented and tested | Close after review. |
| [#314](https://github.com/andremun/pyInstanceSpace/issues/314) | Implemented and tested | Close after review. |
| [#317](https://github.com/andremun/pyInstanceSpace/issues/317) | Implemented and tested | Close after review. |
| [#315](https://github.com/andremun/pyInstanceSpace/issues/315) | Premise disproved | Close as invalid or stale; MATLAB membership is boundary-inclusive. |
| [#313](https://github.com/andremun/pyInstanceSpace/issues/313) | 2D scope implemented and tested | Review the opt-in port; track 3D and any default switch separately. |
| [#278](https://github.com/andremun/pyInstanceSpace/issues/278) | Implemented and tested locally | Review the R2026a manifest, exporter, verifier, and scientific comparisons; close only after maintainer approval. |
| [#310](https://github.com/andremun/pyInstanceSpace/issues/310) | Implemented and tested locally | Review the installed canonical bundle and current-layout readers; historical snapshots remain separately classified. |

## Deferred open issues

| Issue | Priority | Next step |
|---|---:|---|
| [#262](https://github.com/andremun/pyInstanceSpace/issues/262) | High | Add PILOT dimensions, viewpoints, and 3D parity as a separate feature. |
| [#265](https://github.com/andremun/pyInstanceSpace/issues/265) | Medium | Follow #262 with 3D output and visualization support. |
| [#316](https://github.com/andremun/pyInstanceSpace/issues/316) | Medium | Design a shared train/infer stage contract before changing orchestration. |
| [#305](https://github.com/andremun/pyInstanceSpace/issues/305) | Tracker | Reassess after #278 and #310. |
| [#273](https://github.com/andremun/pyInstanceSpace/issues/273) | Tracker | Reassess after provenance and fixture work. |
| [#304](https://github.com/andremun/pyInstanceSpace/issues/304) | Low | Benchmark Bayesian convergence before tuning defaults. |
| [#297](https://github.com/andremun/pyInstanceSpace/issues/297) | Tracker | Mark #302 addressed, then audit remaining linked work. |
| [#272](https://github.com/andremun/pyInstanceSpace/issues/272) | Medium | Reproduce the alpha-boundary failure before adding retries. |
| [#270](https://github.com/andremun/pyInstanceSpace/issues/270) | Tracker | Reassess after #272. |
| [#260](https://github.com/andremun/pyInstanceSpace/issues/260) | Tracker | Keep open for the major parity features above. |

## New audit backlog

### TRACE3 follow-up

- Add three-dimensional alpha geometry only after #262 and #265 define the projected
  and serialized 3D contracts.
- Keep `method="legacy"` as the Python default until maintainers make an explicit
  compatibility decision using the reviewed R2026a evidence.
- Do not commit or relabel the R2024a diagnostic bundle as a MATLAB oracle.

### Quality gate

- Reduce remaining third-party Sobol, Bayesian-search, and PyGAD warnings without
  changing scientific defaults.
- Repair stale README paths, the invalid documentation conversion target, and the
  `InstanceSpaceOptions.default()` example.

These items do not block the corrected two-dimensional build and explore paths.
