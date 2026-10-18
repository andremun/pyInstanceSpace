# InstanceSpace

Build, inspect, persist and explore an instance-space pipeline

## Syntax

```python
space = InstanceSpace(metadata, options)
model = space.build()
result = space.explore(test_metadata, dataset_id=None)
space.save(path, secret_key=None)
space = InstanceSpace.load(path, secret_key=None)
```

See [exact signatures and return types](api/instancespace/instance_space.html#InstanceSpace) for all optional constructor and method arguments.

## Description

`InstanceSpace` owns the training metadata, immutable configuration, stage runner, fitted model and exploration results. Its default stage set is the complete seven-stage pipeline. Additional stages declare their ordering explicitly with `RunBefore` and `RunAfter`.

`build()` returns a `Model`. `model` raises when a complete valid model is unavailable. `run_until_stage` returns a dictionary of available outputs; `run_stage` returns that stage's typed output record. A complete model requires the full pipeline. `explore()` applies trained artifacts and retains results in `explore_results`.

## Examples

Follow the runnable [Getting Started](getting-started.html) example for construction, fitting, plotting and exploration. [Build and Explore](workflows.html) shows stage inspection and checkpointing.

## Parameters

| Parameter | Meaning |
|---|---|
| `metadata` | Validated `Metadata` object containing training features and algorithm outcomes. |
| `options` | Validated `InstanceSpaceOptions`. Use `.default()` or `.from_dict()`. |
| `stages` | Optional stage classes. The default is the complete built-in pipeline. A subset must retain the prerequisites needed by the operations you request. |
| `additional_initial_inputs_type` | Optional NamedTuple type declaring extra plugin inputs. |
| `progress_reporter` | Optional progress reporter. The default reporter performs no external notifications. |

## Returns

`build()` returns `Model`; `explore()` returns `ExploreResult`. `run_iter` and `explore_stage_iter` yield intermediate stage records. File-loading convenience functions can return `None` on load failure; check before calling methods.

## Resource Management

Call `close()` when an instance's executor is no longer needed. Keep outer application parallelism and per-model worker counts within the available machine resources.

## See Also

[Model](model.html) · [Custom Stages](plugins.html) · [Options Reference](options-reference.html)
