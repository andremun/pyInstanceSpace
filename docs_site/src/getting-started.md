# Getting Started

Build an instance space, inspect its fitted results and evaluate new instances

Use Python 3.12 and install `instancespace`. This example is self-contained. Its two algorithms have lower-is-better costs, and a cost at or below 0.20 is good. Replace the generated table with your own [metadata](metadata-format.html) for a real analysis.

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

## See Also

[Build and Explore](workflows.html) · [Metadata Format](metadata-format.html) · [Options Reference](options-reference.html)
