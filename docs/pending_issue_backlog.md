# Pending issue backlog

Reviewed against the branch's tracked GitHub issues on 2026-08-21. This records local
results; no issue was closed or edited remotely in this pass.

## Ready for maintainer review

| Issue | Local status | Recommended action |
|---|---|---|
| [#262](https://github.com/andremun/pyInstanceSpace/issues/262) | Implemented and verified | Review 2D/3D PILOT, SIMPLS, SIFTED propagation, and viewpoints; close after approval. |
| [#265](https://github.com/andremun/pyInstanceSpace/issues/265) | Implemented and verified | Review native 3D output, plots, and mesh schema; close after approval. |
| [#272](https://github.com/andremun/pyInstanceSpace/issues/272) | Premise superseded | Close as stale; R2026a intentionally permits the reproduced multi-region all-points alpha. |
| [#278](https://github.com/andremun/pyInstanceSpace/issues/278) | Implemented and verified | Review the clean R2026a 423-file v2 provenance profile; close after approval. |
| [#310](https://github.com/andremun/pyInstanceSpace/issues/310) | Implemented and verified | Review the atomically installed v2 oracle and current-layout readers; close after approval. |
| [#313](https://github.com/andremun/pyInstanceSpace/issues/313) | Implemented and verified | Review native 2D/3D TRACE3 and explore rescoring; keep any default switch separate. |
| [#316](https://github.com/andremun/pyInstanceSpace/issues/316) | Implemented and verified | Review the typed stage-owned inference contract and compatibility wrappers; close after approval. |
| [#320](https://github.com/andremun/pyInstanceSpace/issues/320) | Implemented and tested | Close after confirming the removed helper was dead code. |
| [#321](https://github.com/andremun/pyInstanceSpace/issues/321) | Implemented and tested | Close after reviewing shared `casefold()` canonicalization and conflict tests. |

Issues #302, #314, #315, and #317 were resolved by the stacked predecessor work and are
already closed upstream.

## Remaining substantive work

| Issue | Priority | Next step |
|---|---:|---|
| [#304](https://github.com/andremun/pyInstanceSpace/issues/304) | Low | Export equal-budget candidate sequences, objective values, fold IDs, and repeated seeds before judging MATLAB/Python Bayesian convergence. Do not tune defaults from rounded legacy metrics. |

## Parent trackers

| Issue | Recommended disposition |
|---|---|
| [#260](https://github.com/andremun/pyInstanceSpace/issues/260) | Reassess for closure after reviewing #262, #265, #313, and the implemented #316 contract. |
| [#270](https://github.com/andremun/pyInstanceSpace/issues/270) | Reassess for closure after #272 is disposed of. |
| [#273](https://github.com/andremun/pyInstanceSpace/issues/273) | Reassess for closure with #305 after #310 is accepted. |
| [#305](https://github.com/andremun/pyInstanceSpace/issues/305) | Close with the completed test-data migration after #310 review. |
| [#297](https://github.com/andremun/pyInstanceSpace/issues/297) | Keep open while #304 still needs acceptable optimizer-trace evidence. |

## Deliberately deferred polish

- Keep legacy TRACE as the Python default until maintainers approve a versioned switch.
- Repair stale examples and low-risk documentation paths independently of the parity
  changes.
- Never promote diagnostic or `legacy-unknown` data into the MATLAB oracle.

These items do not block the verified 2D/3D build and explore paths. The local CI-equivalent
gate passed all 1,046 tests with 92.08% branch coverage and no uncaught warnings
under `-W error`.
