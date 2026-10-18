# Metadata Format

Describe instances, features and algorithm performance in a table

Each row is one instance. Training needs at least three input features and one algorithm. Choose a performance measure whose observed values are nonnegative and whose direction is known.

## Columns

| Column | Meaning |
|---|---|
| `instances` | Instance identifier. Required. Use distinct, meaningful identifiers. |
| `feature_name` | Numeric structural feature. The `feature_` prefix marks feature columns. |
| `algo_name` | Numeric algorithm outcome. The `algo_` prefix marks performance columns. |
| `source` | Optional dataset/source label. |

Column roles are recognized case-insensitively. Prefixes are removed from the feature and algorithm names stored in `Metadata`. Header names must be unambiguous. Do not put spreadsheet error strings into numeric columns.

```csv
instances,feature_size,feature_density,feature_shape,algo_fast,algo_robust,source
instance_a,10,0.2,1.1,0.12,0.18,collection_a
instance_b,20,0.4,0.9,0.25,0.10,collection_a
```

These two rows illustrate the format; use enough training instances in each good/bad class for the configured cross-validation folds.

## Read Data

```python
from pathlib import Path
from instancespace.data.metadata import from_csv_file

metadata = from_csv_file(Path("metadata.csv"))
if metadata is None:
    raise ValueError("Could not load metadata.csv")
```

For an in-memory pandas table, use `Metadata.from_data_frame(table)` as in [Getting Started](getting-started.html).

## Missing Values

Empty CSV cells and `NaN` denote missing numeric values. Build-time ingestion removes wholly missing rows and feature columns at the configured missing-value threshold. Remaining training features and outcomes must be complete. Resolve missing values before fitting; the pipeline does not infer unobserved training outcomes.

Exploration requires the retained training features, but outcomes are optional. Unobserved algorithm outcomes are excluded from metrics, not counted as observed bad performance. Undefined rates are `NaN`.

## Training and Exploration Columns

`selvars.feats` and `selvars.algos` select columns before preprocessing. Training records the retained feature order and later reorders exploration columns to that fitted order. Excluded columns do not have to appear in new data.

Algorithms never good in training are removed from the fitted portfolio. New algorithms in exploration can be evaluated as available ground truth, but the model has no classifier for them and cannot recommend them. Fixed PYTHIA parameter rows must describe the retained training portfolio.

## See Also

[PREPROCESSING](preprocessing.html) · [PRELIM](prelim.html) · [Build and Explore](workflows.html)
