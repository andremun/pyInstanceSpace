# Build and Explore

Run the pipeline, inspect stages and apply a fitted model

Use the metadata and options from [Getting Started](getting-started.html). Building fits scientific artifacts; exploration applies those artifacts to new instances.

## Build a Complete Model

```python
space = InstanceSpace(metadata, options)
model = space.build()
```

The built-in order is PREPROCESSING → PRELIM → SIFTED → PILOT, followed by PYTHIA and CLOISTER, then TRACE. The stage runner enforces dependencies and invalidates downstream artifacts when a prerequisite is rerun.

## Inspect a Projection

Use a fresh instance to stop at PILOT:

```python
from instancespace.stages.pilot import PilotStage

staged = InstanceSpace(metadata, options)
outputs = staged.run_until_stage(PilotStage)
print(outputs["z"])
```

`run_stage(StageClass, **arguments)` runs a stage whose prerequisites are available. `run_until_stage` runs prerequisites as well. Input overrides are named stage inputs, not arbitrary JSON option keys. `run_stage` returns a typed stage record, while `run_until_stage` returns a dictionary. A partial pipeline has stage outputs; a complete model is available only when its required stages have finished.

## Observe Stages

```python
observed = InstanceSpace(metadata, options)
for step in observed.run_iter():
    print(step)
model = observed.model
```

Consume the iterator completely to finalize the model. An interrupted run does not expose a completed model. Progress reporters provide another way to integrate stage notifications with an application; see the [InstanceSpace API](api/instancespace/instance_space.html#InstanceSpace).

## Explore New Data

```python
result = space.explore(new_metadata, dataset_id="validation")
print(result.z)
print(result.selection0)
```

The returned `ExploreResult` contains projected coordinates, predictions, selections, optional observed-outcome metrics and footprint membership. `selection0` is the abstaining selection and `selection1` includes the saved training fallback. Python's selector indices are zero-based; `-1` indicates no selected algorithm.

`explore_stage_iter(new_metadata)` exposes intermediate prediction stages. New outcome labels are used for evaluation only, not to choose recommendations.

## Save and Resume

```python
space.save("analysis.checkpoint")
restored = InstanceSpace.load("analysis.checkpoint")
result = restored.explore(new_metadata)
```

An InstanceSpace checkpoint preserves runner state for resuming and exploration. A standalone `Model.save` stores fitted results; these are distinct persistence interfaces. Only load files from a trusted producer. Optional HMAC authentication is supported by the save/load APIs.

## See Also

[InstanceSpace](instance-space.html) · [Model](model.html) · [Outputs](outputs.html)
