# Custom Stages

Attach application-specific work with explicit pipeline dependencies

## Description

A plugin subclasses `Stage[Input, Output]` and supplies typed NamedTuple input/output records. `RunAfter` and `RunBefore` markers declare scheduling constraints; matching field types alone do not determine the stage order.

The constructor accepts a complete stage list and an optional `additional_initial_inputs_type`. Preserve the built-in prerequisites required by your custom stage and downstream operations.

## Example

This stage summarizes PYTHIA's precision after classifier training:

```python
import numpy as np
from typing import NamedTuple
from instancespace.stages.stage import Stage, RunAfter
from instancespace.stages.pythia import PythiaStage

class PrecisionInput(NamedTuple):
    precision: list[float]
    run_after: RunAfter[PythiaStage] = RunAfter()

class PrecisionOutput(NamedTuple):
    average_precision: float

class PrecisionStage(Stage[PrecisionInput, PrecisionOutput]):
    @staticmethod
    def _inputs() -> type[NamedTuple]:
        return PrecisionInput

    @staticmethod
    def _outputs() -> type[NamedTuple]:
        return PrecisionOutput

    @staticmethod
    def _run(inputs: PrecisionInput) -> PrecisionOutput:
        return PrecisionOutput(float(np.nanmean(inputs.precision)))
```

Include `PrecisionStage` alongside the complete built-in stage list when constructing an `InstanceSpace`. Read the resulting extra field from `run_until_stage`'s output dictionary. A plugin must not silently replace a built-in field with incompatible semantics.

Using the metadata and options from [Getting Started](getting-started.html):

```python
from instancespace.stages.preprocessing import PreprocessingStage
from instancespace.stages.prelim import PrelimStage
from instancespace.stages.sifted import SiftedStage
from instancespace.stages.pilot import PilotStage
from instancespace.stages.cloister import CloisterStage
from instancespace.stages.trace import TraceStage

custom = InstanceSpace(metadata, options, stages=[
    PreprocessingStage, PrelimStage, SiftedStage, PilotStage,
    PythiaStage, CloisterStage, TraceStage, PrecisionStage,
])
raw = custom.run_until_stage(PrecisionStage)
print(raw["average_precision"])
custom.close()
```

## Parameters and Returns

Stage input records must carry concrete runtime type annotations. This example does not use `from __future__ import annotations`; unresolved string annotations cannot declare scheduling markers.

Named input fields must be available before the plugin runs. Output fields become named stage arguments for later stages. Use unique, meaningful names and explicit return types.

## See Also

[Stage API](api/instancespace/stages/stage.html) · [Stage Runner API](api/instancespace/stage_runner.html) · [InstanceSpace](instance-space.html)
