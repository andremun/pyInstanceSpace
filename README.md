# Instance Space Analysis for Python

[![Tests](https://github.com/andremun/pyInstanceSpace/actions/workflows/validation-tests.yml/badge.svg)](https://github.com/andremun/pyInstanceSpace/actions/workflows/validation-tests.yml)
[![Documentation](https://img.shields.io/badge/docs-reference-blue)](https://andremun.github.io/pyInstanceSpace/)
[![PyPI](https://img.shields.io/pypi/v/instancespace.svg)](https://pypi.org/project/instancespace/)
[![Python](https://img.shields.io/pypi/pyversions/instancespace.svg)](https://pypi.org/project/instancespace/)
[![DOI](https://zenodo.org/badge/770130753.svg)](https://doi.org/10.5281/zenodo.15562567)

InstanceSpace relates the structural features of problem instances to the performance of a portfolio of algorithms. It creates a 2D or 3D instance space, trains algorithm selectors, and constructs footprints showing where each algorithm performs well.

Use it to compare algorithm strengths, inspect benchmark diversity, find difficult regions and recommend algorithms for new instances.

## Documentation

The [reference website](https://andremun.github.io/pyInstanceSpace/) provides:

- [Getting Started](https://andremun.github.io/pyInstanceSpace/getting-started.html): a self-contained runnable analysis.
- [Metadata Format](https://andremun.github.io/pyInstanceSpace/metadata-format.html): features, performance and missing values.
- [Build and Explore](https://andremun.github.io/pyInstanceSpace/workflows.html): training, stages, checkpoints and inference.
- [Options Reference](https://andremun.github.io/pyInstanceSpace/options-reference.html): every configuration field, default and accepted JSON alias.
- [Classifiers and Tuning](https://andremun.github.io/pyInstanceSpace/tuning.html): learners, parameter units, Sobol and Bayesian search.
- [Outputs and Plotting](https://andremun.github.io/pyInstanceSpace/outputs.html): explicit CSV, image, web and model exports.
- [MATLAB Compatibility](https://andremun.github.io/pyInstanceSpace/matlab-compatibility.html): shared contracts and implementation differences.
- [API Reference](https://andremun.github.io/pyInstanceSpace/api-reference.html): exact signatures, parameters and result fields.

## Install

Use Python 3.12:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install instancespace
```

## Prepare Input Files

`metadata.csv` contains one instance per row: an `instances` identifier, at least three numeric `feature_` columns and one or more numeric `algo_` columns. An optional `source` column identifies collections.

Training outcomes must be nonnegative and complete after ingestion filtering. Zero is allowed. Exploration can omit algorithm outcomes; unobserved values are excluded from metrics. Supply enough examples of each good/bad class for the requested folds.

Create `options.json` for your performance measure. This example uses a lower-is-better cost, with values at or below 0.20 considered good:

```json
{
  "perf": {"max_perf": false, "abs_perf": true, "epsilon": 0.20},
  "sifted": {"flag": false},
  "pilot": {"analytic": true},
  "pythia": {"classifier": "knn", "cv_folds": 3, "n_tuning_iter": 4},
  "trace": {"method": "trace3"}
}
```

These are compact example settings, not the package defaults. Configure the scientific threshold and search budgets for your own data.

## Build, Inspect and Save

```python
from pathlib import Path
from instancespace.instance_space import instance_space_from_files

space = instance_space_from_files(Path("metadata.csv"), Path("options.json"))
if space is None:
    raise ValueError("Could not load metadata or options")
model = space.build()
print(model.pythia.summary)
print(model.trace.summary)
Path("output").mkdir(exist_ok=True)
model.save_to_csv("output")
space.save("analysis.checkpoint")
```

Exports are explicit calls. `outputs.csv`, `outputs.png` and `outputs.web` are caller preferences and do not automatically write files during `build()`.

Training removes algorithms that are never good and recomputes results from the retained portfolio. Exploration applies saved feature order, preprocessing, projection, classifiers and footprints without refitting. Test outcomes score predictions; they do not determine recommendations.

## Configure Python Objects

Use immutable dataclasses to configure an in-memory workflow:

```python
from instancespace.data.options import InstanceSpaceOptions, PilotOptions

options = InstanceSpaceOptions.default(pilot=PilotOptions.default(dims=3))
```

Use `dataclasses.replace` to create an updated configuration. The complete options reference is checked against the source schema when the website builds.

## Develop

Install the locked development environment with Poetry, then run:

```sh
poetry install
poetry run poe docs
poetry run pytest
```

The documentation build writes `site/`. It checks option coverage and local files/anchors, and gives curated guides and generated API pages one shared layout. Serve the directory locally to preview it. See [CONTRIBUTING.md](CONTRIBUTING.md) for development checks and reporting guidance.

## Cite

For the methodology:

> K. Smith-Miles and M. A. Muñoz. *Instance Space Analysis for Algorithm Testing: Methodology and Software Tools*. ACM Computing Surveys, 55(12), Article 255, 2023. [DOI](https://doi.org/10.1145/3572895).

For the Python software:

> Y. B. Güzel, K. Khare, N. Harvey, K. Dsouza, D. H. Jang, J. Chen, C. Z. Lam and M. A. Muñoz. *instancespace: A Python package for insightful algorithm testing through Instance Space Analysis*. SoftwareX, 31:102246, 2025. [DOI](https://doi.org/10.1016/j.softx.2025.102246).

> M. A. Muñoz and K. Smith-Miles. *Instance Space Analysis: A Python toolkit for the assessment of algorithmic power*. [Zenodo](https://doi.org/10.5281/zenodo.15562567).

For 3D Instance Space Analysis, also cite Simpson et al. (2025), *ISA3: A 3-dimensional expansion of Instance Space Analysis*, Machine Learning, 114, 240. [DOI](https://doi.org/10.1007/s10994-025-06871-5).

## License and Support

[PolyForm Noncommercial License 1.0.0](LICENSE). See [SECURITY.md](SECURITY.md) for security reporting and [GitHub Issues](https://github.com/andremun/pyInstanceSpace/issues) for support and defects.
