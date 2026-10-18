# PILOT

Project retained features into a 2D or 3D instance space

## Syntax

```python
from instancespace.stages.pilot import PilotStage

outputs = space.run_until_stage(PilotStage)
```

## Description

Standard PILOT learns a linear projection while reconstructing features and algorithm performance. `cost_weight` controls the performance-reconstruction block. Use an analytic solution when requested and viable; otherwise the numerical path evaluates multiple starts and selects by distance preservation.

The PLS alternative uses SIMPLS. It stores the fitted feature mean and applies `(X - x_mean) @ A.T` to new instances. Standard projections apply `X @ A.T` without that PLS mean subtraction.

## Parameters

`method` is `standard` or `pls`; `dims` is 2 or 3. `analytic` is used only for standard projection. `n_tries` controls generated numerical starts. A supplied `x0` matrix provides one start per column. A valid `precalc_alpha` bypasses the numerical solve.

For `p` features, `q` algorithm columns and dimension `d`, numerical solution vectors have `d * (2*p + q)` values. See [PILOT options](options-reference.html#options-pilot) for all fields and interactions.

## Viewpoints and Rotation

Three-dimensional projections store an optimized global camera view or one view per `view_groups` entry. Groups use zero-based algorithm indices. `adjust_rotation=True` is a 2D rigid rotation for consistent visual orientation; it cannot be used with 3D.

Seeds control starts, not numerical-library arithmetic. Optimizer trajectories and restart winners can differ across environments. Do not rotate coordinates independently of the fitted model's classifiers or geometry.

## Returns

Projection matrix, coordinates, reconstruction parameters and error, fitted PLS mean when applicable, numerical restart diagnostics and viewpoints. See the [PILOT API](api/instancespace/stages/pilot.html).

## Examples

```python
from dataclasses import replace
from instancespace.data.options import PilotOptions

projection_options = replace(options, pilot=PilotOptions.default(
    method="pls", dims=3,
))
```

## See Also

[CLOISTER](cloister.html) · [PYTHIA](pythia.html) · [Outputs](outputs.html)
