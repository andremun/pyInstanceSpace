# Outputs and Plotting

Write fitted results and visualize algorithm performance

## Explicit Exports

`build()` fits a model. It does not automatically write output files from the `outputs` flags. Call the required method:

```python
from pathlib import Path

Path("output").mkdir(exist_ok=True)
model.save_to_csv("output")
model.save_graphs("output")
model.save_for_web("output")
model.save_to_mat("output")
model.save_zip("analysis.zip", "output")
```

Create the output directory first. `save_zip` archives the existing directory after the selected exports have been written.

The `outputs.csv`, `outputs.png` and `outputs.web` fields are preferences for callers and orchestration. They remain valid configuration fields even though built-in file creation is explicit.

## CSV and Geometry

CSV output includes coordinates, processed data, algorithm-selection summaries and footprints. Two-dimensional rings retain holes and disconnected components. Three-dimensional exports include native mesh geometry and a versioned manifest.

Read each mesh manifest's index-base and schema declarations instead of assuming every export uses the same convention. Python's TRACE mesh and CLOISTER mesh schemas have different index conventions. Do not interchange their face tables without interpreting the manifest.

## Plotting

```python
import matplotlib.pyplot as plt

space.plot_sources()
space.plot_portfolio()
space.plot_good(0)
space.plot_footprint(0)
plt.show()
```

Algorithm arguments accept a retained algorithm name or a zero-based index. These helpers return matplotlib `Axes` and accept an existing axes where supported. Plots use 2D scatter/polygons or 3D axes/meshes according to the fitted projection. Source plotting requires source labels.

## Persistence Versus Export

CSV, PNG, web and MAT files are presentation/interchange outputs. Native `Model.save` or an `InstanceSpace` checkpoint preserves the trained objects needed for continued Python use. A MAT export is not a portable MATLAB classifier archive.

## See Also

[Model](model.html) · [Plotting API](api/instancespace/plotting.html) · [Serialization API](api/instancespace/model.html)
