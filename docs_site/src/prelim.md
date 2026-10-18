# PRELIM

Define good performance, difficulty and fitted data transformations

## Syntax

```python
from instancespace.stages.prelim import PrelimStage

outputs = space.run_until_stage(PrelimStage)
```

## Description

PRELIM computes good/bad algorithm labels, best performance, winner indices, good-algorithm counts and beta-easy labels. It bounds feature outliers and fits Box-Cox/Z-score transforms when enabled. Exploration applies saved transforms.

The full pipeline removes algorithms never good in training and recomputes from the retained raw portfolio. An empty retained portfolio raises an error. Direct `PrelimStage.prelim()` calls perform numerical preprocessing without pipeline portfolio pruning.

## Performance Rules

For a nonnegative raw outcome `y` and best outcome `best`:

| Setting | Good-performance rule |
|---|---|
| Absolute, minimize | `y <= epsilon` |
| Absolute, maximize | `y >= epsilon` |
| Relative, minimize | `y / best - 1 <= epsilon` |
| Relative, maximize | `1 - y / best <= epsilon` |

The relative calculation substitutes machine epsilon for exact zeros. This finite surrogate is unit-sensitive; consider an absolute threshold when zero-best values are common. An instance is beta-easy only when its good-algorithm count is strictly greater than `beta_threshold * n_algorithms`.

## Parameters

Use `perf`, `auto`, `bound`, `norm`, `prelim` and `selvars` options. Feature clipping uses `median ± iqr_multiplier * IQR`; it does not clip raw algorithm outcomes. Constant columns use a unit normalization divisor.

## Returns

Processed and raw matrices, performance labels, selected rows and fitted transformation parameters. See the [typed PRELIM API](api/instancespace/stages/prelim.html).

## Examples

```python
from dataclasses import replace
from instancespace.data.options import PerformanceOptions

cost_options = replace(options, perf=PerformanceOptions.default(
    max_perf=False, abs_perf=True, epsilon=0.20,
))
```

## See Also

[PREPROCESSING](preprocessing.html) · [SIFTED](sifted.html) · [Options Reference](options-reference.html)
