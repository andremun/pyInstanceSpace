# Stage-owned inference architecture

## Scope

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
- New algorithms retain false/zero inference padding and NaN classifier metrics.
- Missing trained-algorithm truth retains NaN metrics.
- All current R2026a `reference-export/v2` readers remain authoritative.

