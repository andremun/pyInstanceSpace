# Getting Started

Build an instance space, inspect its fitted results and evaluate new instances

This example creates its own metadata and runs a complete analysis. Its two algorithms have lower-is-better costs. A cost at or below 0.20 is good. Replace the generated table with your own [metadata](metadata-format.html) for a real analysis.

## Install

Use Python 3.12 in a virtual environment:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install instancespace
```

On Windows, activate the environment with `.\.venv\Scripts\Activate.ps1` in PowerShell. On a machine without a graphical display, set `MPLBACKEND=Agg` before running Python. Save figures with `model.save_graphs("output")` after creating the output directory.

Run the following Python blocks in order in one script or session.

## Prepare Metadata

```python
import numpy as np
import pandas as pd

from instancespace.data.metadata import Metadata
from instancespace.data.options import (
    InstanceSpaceOptions, PilotOptions, PythiaOptions,
    SiftedOptions, TraceOptions,
)
from instancespace.instance_space import InstanceSpace

x = np.linspace(1.0, 4.0, 24)
table = pd.DataFrame({
    "instances": [f"instance_{i}" for i in range(len(x))],
    "source": ["example"] * len(x),
    "feature_size": x,
    "feature_shape": np.sin(x),
    "feature_density": np.cos(x),
    "algo_left": np.where(np.sin(x) > 0, 0.10, 0.30),
    "algo_right": np.where(np.sin(x) > 0, 0.30, 0.10),
})
metadata = Metadata.from_data_frame(table)
```

## Build

This compact example keeps all features, uses an analytic projection and a small KNN tuning budget. These are explicit example settings, not changes to the package defaults.

```python
options = InstanceSpaceOptions.default(
    sifted=SiftedOptions.default(flag=False),
    pilot=PilotOptions.default(analytic=True),
    pythia=PythiaOptions.default(
        classifier="knn", cv_folds=3, n_tuning_iter=4,
    ),
    trace=TraceOptions.default(method="trace3"),
)
space = InstanceSpace(metadata, options)
model = space.build()
print(model.pythia.summary)
print(model.trace.summary)
```

`build()` returns a [Model](model.html). The default performance settings are absolute, lower-is-better and epsilon 0.20. Set these deliberately for your application. Algorithms never good in training are removed, and downstream results use the retained portfolio.

## Plot and Save

```python
import matplotlib.pyplot as plt

space.plot_portfolio()
plt.show()
from pathlib import Path
Path("output").mkdir(exist_ok=True)
model.save_to_csv("output")
space.save("instance-space.checkpoint")
```

Exports are explicit calls. The `outputs.csv`, `outputs.png` and `outputs.web` preferences do not automatically create files during `build()`.

## Explore New Instances

New metadata must provide the retained training features. It can omit algorithm performance when no ground truth is available.

```python
new_table = table.iloc[:5].drop(columns=["algo_left", "algo_right"]).copy()
new_table["instances"] = [f"new_{i}" for i in range(len(new_table))]
new_metadata = Metadata.from_data_frame(new_table)
result = space.explore(new_metadata)
print(result.z)
print(result.selection0)
```

Exploration applies saved preprocessing, feature selection, projection, classifiers and geometry. It does not refit them. Missing outcomes are excluded from evaluation metrics.

## Interpret the Results

`model.pythia.summary` reports classifier and algorithm-selection results. `model.trace.summary` reports footprint support, purity, and area or volume. Inspect these summaries together with the plots. A large footprint alone does not establish high predictive accuracy or adequate benchmark coverage.

The example uses training data to fit preprocessing, projection, and classifiers. It keeps all three features. Its classifier CV summaries are conditional on those fitted steps. Use independent evaluation data to assess generalization. The feature-only exploration above produces recommendations without an observed-outcome accuracy score.

Call `space.close()` after the final operation to release its executor.

## See Also

[Build and Explore](workflows.html) · [Metadata Format](metadata-format.html) · [Options Reference](options-reference.html)
