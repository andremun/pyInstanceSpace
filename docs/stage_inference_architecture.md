# Stage-owned inference architecture

## Scope

**Status:** Implemented and verified on 2026-08-21.

Issue [#316](https://github.com/andremun/pyInstanceSpace/issues/316) is a code-ownership
refactor, not a MATLAB parity defect. MATLAB R2026a unifies trained evaluation only inside
PYTHIA and TRACE; PRELIM, SIFTED, and PILOT inference remain orchestrated separately.

The existing `Stage[BuildInput, BuildOutput]` and `StageRunner` remain build-only. Build
outputs are runner-bus records, not persisted trained artifacts, and build-only plugins must
not acquire a new abstract method.

## Contract

Inference-capable stages additionally implement a separate typed contract:

```python
class PredictiveStage(ABC, Generic[PredictInput, Fitted, PredictOutput]):
    @staticmethod
    @abstractmethod
    def predict(inputs: PredictInput, fitted: Fitted) -> PredictOutput: ...
```

`Fitted` is the persisted model artifact: `PrelimOut`, `SiftedOut`, `PilotOut`,
`PythiaOut`, or `TraceOut`. Prediction must not fit, mutate, or deep-copy the artifact.

Stage ownership is:

- PRELIM: trained clipping, Box-Cox/z-score transforms, and the OOD warning.
- SIFTED: application of stored selected-feature indices.
- PILOT: the trained uncentred `X @ A.T` explore projection in 2D or 3D.
- PYTHIA: stored normalization, classifier inference, probability extraction, weighted
  selection, and truth-aware classifier metrics.
- TRACE: 2D/3D dimension checks and inclusive trained-footprint membership.

`TraceStage.rescore()` remains the truth-aware, fixed-geometry evaluation path.
Preprocessing and CLOISTER have no explore-time inference contract.

## Orchestration boundary

`InstanceSpace` retains metadata validation, feature-name alignment, algorithm
reconciliation, lazy stage order, conditional evaluation assembly, and TRACE rescoring.
Scientific implementations move out of its `_explore_*` family. Private forwarding wrappers
remain for one compatibility window.

The observable order remains PRELIM, SIFTED, PILOT, PYTHIA, TRACE, then EVALUATION only when
ground truth is present. Advancing only to a stage must not execute later work.

## Compatibility gates

- StageRunner scheduling, rollback, checkpoints, and plugins are unchanged.
- Existing model/joblib schemas and explore payload layouts are unchanged.
- No fitted classifier, array, polygon, or tetrahedral mesh is mutated.
- Test-only algorithms retain false/zero inference padding and NaN classifier metrics
  because no fitted classifier exists for them.
- A trained algorithm absent from test metadata is scored against the reconciled
  all-false truth column; its metrics are not replaced with NaN.
- All current R2026a `reference-export/v2` readers remain authoritative.

## Verification

- Contract, delegation, no-refit, fitted-state immutability, wrapper compatibility, and
  stage-order tests pass for all five inference-capable stages.
- Collection contains 86 provenance tests and 40 current-MATLAB readers.
- The warning-strict sandbox run passed 1,037 tests; the sole process-pool test passed
  1/1 outside the sandbox, accounting for all 1,038 collected tests.
- Branch coverage was 91.86%, with no uncaught warnings under `-W error`.
