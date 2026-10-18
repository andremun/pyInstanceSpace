# Model and Persistence

Inspect fitted results, save a model and export scientific outputs

## Syntax

```python
model = space.build()
model.save(path, secret_key=None)
model = Model.load(path, secret_key=None)
model.save_to_csv(directory)
model.save_graphs(directory)
model.save_for_web(directory)
model.save_to_mat(directory)
model.save_zip(filename, directory)
```

## Description

A `Model` stores training data, selected features, preprocessing, projection, classifier and footprint results, effective options and stage provenance. It preserves fitted feature order, PLS centering and the footprint boundary tolerance.

`Model.save` and `Model.load` use native Python persistence for fitted objects. Use `InstanceSpace.save` and `InstanceSpace.load` when you need a checkpoint containing the stage runner and exploration interface. Export directories must already exist. MAT export is a data export, not a MATLAB classifier-loading format.

## Examples

Using the fitted model from [Getting Started](getting-started.html):

```python
from instancespace.model import Model

model.save("fitted-model.joblib")
loaded = Model.load("fitted-model.joblib")
print(loaded.data.algo_labels)
print(loaded.pythia.summary)
print(loaded.trace.summary)
```

## Fitted Results

| Field | Purpose |
|---|---|
| `data` | Fitted instance data and retained feature/algorithm labels. |
| `feat_sel`, `sifted` | Feature selection outputs and diagnostics. |
| `prelim` | Fitted bounds and normalization parameters. |
| `pilot` | Projection matrix, coordinates, reconstruction and viewpoints. |
| `cloister` | Empirical boundary coordinates and mesh information. |
| `pythia` | Classifiers, cross-validation results and selection summary. |
| `trace` | Footprints, membership summaries and boundary tolerance. |
| `opts`, `stage_options` | Effective configuration and consumed stage settings. |
| `preprocessing_features` | Retained input feature order used for exploration. |

Exact fields and types are in the [Model API](api/instancespace/model.html#Model) and [result dataclasses](api/instancespace/data/model.html).

## Safe Loading

Native persistence can execute code while loading. Load only trusted artifacts. An optional shared `secret_key` authenticates the serialized bytes with HMAC-SHA256; use the same key when saving and loading. Authentication does not make an untrusted producer safe.

## See Also

[Build and Explore](workflows.html) · [Outputs and Plotting](outputs.html)
