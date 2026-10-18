# TRACE

Construct algorithm footprints and evaluate their membership

## Syntax

```python
from instancespace.stages.trace import TraceStage

model = space.build()
print(model.trace.summary)
```

## Description

TRACE constructs good-performance, best-algorithm and hard-instance footprints. It reports geometry, area/volume, counts, density and purity. Exploration reuses trained geometry and the training-space normalization; it does not fit a new hull from the test batch.

Python defaults to `method="legacy"`, a 2D DBSCAN/polygon method with optional contradiction removal. Select `method="trace3"` for 2D/3D alpha-shape footprints. A 3D projection configured with legacy warns and dispatches to TRACE3. Set the method explicitly when selecting a purity policy.

## Parameters

The omitted purity is 0.55 for legacy and 0.60 for TRACE3. `use_sim` and `contra` control legacy prediction use and contradiction removal. `min_instances` and `min_area_frac` constrain TRACE3 support and measure. See every field in [TRACE options](options-reference.html#options-trace).

With `pythia.skip=True`, legacy TRACE requires `use_sim=False`. TRACE3 has a skip path using observed labels.

## Boundary Membership

Footprints include their boundaries. `boundary_tolerance=0.0` is exact closed-boundary membership. A positive tolerance also includes points within that Euclidean distance of an edge or face, including hole boundaries. More distant hole interiors remain outside.

The tolerance is in projected-coordinate units, independent of the query batch. It is saved with the model and reused during exploration. It does not buffer geometry or change area/volume. It also does not guarantee that sequential legacy contradiction trimming is stable under small coordinate perturbations.

## Returns

A `TraceOut` contains the good, best and hard footprints and a summary. TRACE3 supports holes, disconnected components and native 3D tetrahedral geometry. A retained TRACE3 candidate can remain below the requested purity when the alpha search ends. Inspect its reported purity; Python does not currently persist MATLAB's acceptance/search-termination fields. Exact records are in the [TRACE API](api/instancespace/stages/trace.html) and [result dataclasses](api/instancespace/data/model.html#TraceOut).

## Examples

```python
from dataclasses import replace
from instancespace.data.options import TraceOptions

footprint_options = replace(options, trace=TraceOptions.default(
    method="trace3", boundary_tolerance=0.0,
))
```

## See Also

[PYTHIA](pythia.html) · [Outputs](outputs.html) · [MATLAB Compatibility](matlab-compatibility.html)
