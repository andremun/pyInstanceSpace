# CLOISTER

Estimate empirical full and correlation-constrained instance-space bounds

## Syntax

```python
from instancespace.stages.cloister import CloisterStage

outputs = space.run_until_stage(CloisterStage)
```

## Description

CLOISTER constructs feature-range corners, removes combinations contradicted by strong significant feature correlations, projects the permitted corners with the fitted PILOT model and computes convex hulls. Large feature counts use the observed-data fallback specified by `max_features`.

PLS bounds use the fitted feature mean. Full-dimensional 3D hulls include surface triangulation; coplanar 3D boundaries use a flat triangulated polygon. Collinear inputs cannot define a 2D/3D hull and raise an actionable error.

## Parameters

`p_val` and `c_thres` control correlation significance and strength. `max_features` bounds corner enumeration. `hull_dims="all"` uses the projected dimensionality; an integer selects a leading-coordinate hull. See [CLOISTER options](options-reference.html#options-cloister).

## Returns

Full and correlation-constrained boundary coordinates plus derived mesh information for persistence, plotting and CSV output. The low-level stage retains its two-array return contract. See the [CLOISTER API](api/instancespace/stages/cloister.html).

## Examples

```python
from dataclasses import replace
from instancespace.data.options import CloisterOptions

boundary_options = replace(options, cloister=CloisterOptions.default(
    hull_dims="all", max_features=16,
))
```

## See Also

[PILOT](pilot.html) · [Outputs and Plotting](outputs.html)
