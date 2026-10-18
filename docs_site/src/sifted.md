# SIFTED

Select informative features through correlation and genetic search

## Syntax

```python
from instancespace.stages.sifted import SiftedStage

outputs = space.run_until_stage(SiftedStage)
```

## Description

SIFTED screens features by their correlation with algorithm performance, clusters candidate features using correlation distance, and searches for representative features with PyGAD. Candidate fitness uses a PILOT projection and KNN cross-validation. It does not use PCA or Random Forest fitness.

The cache belongs to the current genetic-search run. Selected feature indices are mapped back through the correlation-screened columns. Exploration reuses the fitted selection instead of running the search again.

## Parameters

`sifted.flag` enables selection. `rho` and `pval` control screening; `k`, `max_iter` and `replicates` control clustering. Genetic-search fields configure population, parents, elitism, crossover, mutation and stopping. The [Options Reference](options-reference.html#options-sifted) documents every field and accepted choice.

The full pipeline derives candidate projection dimensionality from `pilot.dims`. The direct-call `sifted.dims` default is 2. A stage seed of `None` inherits `general.seed`. `n_trees` is a compatibility field with no effect on current fitness.

## Returns

Selected feature indices and labels, fitted selected data and search diagnostics. See [SiftedInput and SiftedOutput](api/instancespace/stages/sifted.html).

## Examples

```python
from dataclasses import replace
from instancespace.data.options import SiftedOptions

search_options = replace(options, sifted=SiftedOptions.default(
    flag=True, k=3, num_generations=20, sol_per_pop=12,
))
```

These reduce the example search budget. Increase it only when the data and validation justify more search.

## See Also

[PILOT](pilot.html) · [Options Reference](options-reference.html#options-sifted)
