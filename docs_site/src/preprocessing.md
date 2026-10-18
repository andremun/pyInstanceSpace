# PREPROCESSING

Select metadata columns and prepare viable training inputs

## Syntax

```python
from instancespace.stages.preprocessing import PreprocessingStage

outputs = space.run_until_stage(PreprocessingStage)
```

## Description

PREPROCESSING selects requested features and algorithms, filters missing input data and records retained feature order. Metadata column roles and names are validated before subsequent scientific stages run.

Names in `selvars.feats` and `selvars.algos` can include their CSV prefixes. Selections are case-insensitive. If some names match, unknown names warn and are ignored; a request with no matches raises an error. At least three input features and one training algorithm must remain viable.

## Parameters

Inputs include feature and performance matrices, instance/feature/algorithm labels, optional source labels, `selvars` settings and the composed PRELIM configuration. All rows and labels must align.

## Returns

Raw and working feature/performance matrices, filtered labels, sources and a retained-feature record. Full typed contracts are in [PreprocessingInput and PreprocessingOutput](api/instancespace/stages/preprocessing.html).

## Examples

Prepare data as in [Getting Started](getting-started.html). To restrict the analysis before building:

```python
from dataclasses import replace
from instancespace.data.options import SelvarsOptions

selected_options = replace(options, selvars=SelvarsOptions.default(
    feats=["size", "shape", "density"], algos=["left", "right"],
))
selected = InstanceSpace(metadata, selected_options)
outputs = selected.run_until_stage(PreprocessingStage)
```

## See Also

[Metadata Format](metadata-format.html) · [PRELIM](prelim.html) · [Selection Options](options-reference.html#options-selvars)
