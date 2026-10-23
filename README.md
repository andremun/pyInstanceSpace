# Instance Space Analysis: A Python toolkit for the assessment of algorithmic power

[![Tests](https://github.com/andremun/pyInstanceSpace/actions/workflows/validation-tests.yml/badge.svg)](https://github.com/andremun/pyInstanceSpace/actions/workflows/validation-tests.yml)
[![codecov](https://codecov.io/gh/andremun/pyInstanceSpace/graph/badge.svg)](https://codecov.io/gh/andremun/pyInstanceSpace)
[![Documentation](https://github.com/andremun/pyInstanceSpace/actions/workflows/docs-pages.yml/badge.svg)](https://andremun.github.io/pyInstanceSpace/)
[![PyPI](https://img.shields.io/pypi/v/instancespace.svg)](https://pypi.org/project/instancespace/)
[![PyPI downloads](https://img.shields.io/pypi/dm/instancespace.svg)](https://pypi.org/project/instancespace/)
[![Python](https://img.shields.io/pypi/pyversions/instancespace.svg)](https://pypi.org/project/instancespace/)
[![DOI](https://zenodo.org/badge/770130753.svg)](https://doi.org/10.5281/zenodo.15562567)
[![Citations](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2FDOI%3A10.1145%2F3572895%3Ffields%3DcitationCount&query=%24.citationCount&label=citations&color=blue)](https://doi.org/10.1145/3572895)

Instance Space Analysis (ISA) relates the structural features of problem instances to the performance of a portfolio of algorithms. This Python package projects instances into a 2D or 3D instance space, trains algorithm selectors, and constructs algorithm **footprints**. A footprint describes a region where an algorithm is expected to perform well under a chosen performance threshold.

Use the analysis to:

- Compare algorithm strengths through footprint support, purity, and area or volume.
- Examine relationships between instance features and algorithm performance.
- Inspect the distribution and diversity of benchmark collections.
- Assess whether the selected features help explain performance variation.
- Recommend algorithms for new instances.
- Identify gaps where additional instances could improve coverage.

An instance-space plot shows regional performance differences that an average score can hide. Its conclusions depend on the features, performance measure, and collected instances. A footprint does not guarantee good performance on every new instance within its boundary.

The package implements the ISA methodology and its ISA3 extension. See [Cite](#cite) for the methodology and Python software references. The [MATLAB toolkit](https://github.com/andremun/InstanceSpace) provides a separate implementation of the same methodology.

## Repository Layout

```text
instancespace/
  instance_space.py          InstanceSpace: build, inspect stages, and explore
  model.py                  fitted results, persistence, and exports
  data/                     metadata, options, defaults, and result records
  stages/                   PREPROCESSING, PRELIM, SIFTED, PILOT,
                            CLOISTER, PYTHIA, and TRACE
  utils/                    classifier registry, filtering, and geometry
  plotting.py               2D and 3D plots
  stage_runner.py           stage dependencies and execution
  progress_reporter.py      progress notifications
docs_site/                  reference website sources and builder
docs/                       maintainer contracts and reproducibility evidence
tests/                      regression tests and classified reference fixtures
.claude/skills/              Python ISA analysis guidance for AI assistants
pyproject.toml, poetry.lock  package metadata and locked dependencies
CITATION.cff                machine-readable software citation
```

## Documentation

The [reference website](https://andremun.github.io/pyInstanceSpace/) provides guides, stage descriptions, and API signatures:

- [Getting Started](https://andremun.github.io/pyInstanceSpace/getting-started.html): a self-contained, runnable analysis.
- [Metadata Format](https://andremun.github.io/pyInstanceSpace/metadata-format.html): features, performance, and missing values.
- [Build and Explore](https://andremun.github.io/pyInstanceSpace/workflows.html): training, stage inspection, checkpoints, and inference.
- [Options Reference](https://andremun.github.io/pyInstanceSpace/options-reference.html): every configuration field, default, and accepted JSON alias.
- [Classifiers and Tuning](https://andremun.github.io/pyInstanceSpace/tuning.html): learners, parameter units, and search strategies.
- [Outputs and Plotting](https://andremun.github.io/pyInstanceSpace/outputs.html): CSV, image, web, geometry, and model exports.
- [MATLAB Compatibility](https://andremun.github.io/pyInstanceSpace/matlab-compatibility.html): shared contracts and implementation differences.
- [API Reference](https://andremun.github.io/pyInstanceSpace/api-reference.html): exact signatures, parameters, and result fields.

## Installation Instructions

The package requires Python 3.12. Install it in a virtual environment:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install instancespace
```

On Windows, activate the environment with `.\.venv\Scripts\Activate.ps1` in PowerShell. The distribution and import names are both `instancespace`.

The installer installs the required numerical, machine-learning, geometry, and plotting libraries. Python workflows do not require MATLAB, GNU Octave, or MATLAB toolboxes. For the locked source environment, see [Develop](#develop).

## The Metadata File

Prepare `metadata.csv` before fitting. Each row describes one problem instance:

| Column | Content |
|---|---|
| `instances` | Required instance identifier. Use distinct, meaningful identifiers. |
| `feature_name` | Numeric structural feature, such as `feature_density`. Supply at least three viable input features. |
| `algo_name` | Nonnegative numeric algorithm performance, such as `algo_greedy`. Supply at least one training algorithm. |
| `source` | Optional collection label for source comparisons and plots. |

For example:

```csv
instances,feature_size,feature_density,feature_shape,algo_fast,algo_robust,source
instance_a,10,0.2,1.1,0.12,0.18,collection_a
instance_b,20,0.4,0.9,0.25,0.10,collection_b
```

These two rows show the format. Training needs enough examples of each good/bad class for the requested cross-validation folds.

Use empty cells or `NaN` for missing numeric values. Remove spreadsheet error strings from numeric columns. Training filters wholly missing rows and feature columns at the configured missing-value threshold. Remaining training features and outcomes must be complete. Zero performance values are allowed.

Exploration requires the retained training features and can omit algorithm outcomes. The model excludes unobserved outcomes from evaluation metrics. See [Metadata Format](https://andremun.github.io/pyInstanceSpace/metadata-format.html) for selection, name matching, and missing-value rules.

## Working with the Code

Start with the [Getting Started example](https://andremun.github.io/pyInstanceSpace/getting-started.html). It creates synthetic metadata, builds a model, plots results, saves outputs, and explores new instances. For your own data, prepare the metadata and choose the performance settings before fitting.

Create `options.json` for your performance measure. The following settings use a lower-is-better cost and define values at or below 0.20 as good:

```json
{
  "general": {"seed": 0},
  "perf": {"max_perf": false, "abs_perf": true, "epsilon": 0.20},
  "sifted": {"flag": false},
  "pilot": {"analytic": true},
  "pythia": {"classifier": "knn", "cv_folds": 3, "n_tuning_iter": 4},
  "trace": {"method": "trace3"}
}
```

These are compact example settings. They keep all features and reduce the search budget. Choose the scientific threshold and search budget for your data. Python reads this JSON configuration without overwriting it.

### The InstanceSpace Class

`InstanceSpace` runs the pipeline and stores fitted results. The file-loading helper reads the metadata and configuration:

```python
from pathlib import Path
from instancespace.instance_space import instance_space_from_files

space = instance_space_from_files(Path("metadata.csv"), Path("options.json"))
if space is None:
    raise ValueError("Could not load metadata or options")
model = space.build()
print(model.pythia.summary)
print(model.trace.summary)
Path("output").mkdir(parents=True, exist_ok=True)
model.save_to_csv("output")
space.save("analysis.checkpoint")
space.close()
```

The pipeline prepares data with PREPROCESSING and PRELIM, selects features with SIFTED, and projects them with PILOT. CLOISTER estimates empirical bounds, PYTHIA trains algorithm selectors, and TRACE constructs footprints.

Training removes algorithms that are never good and recomputes results from the retained portfolio. `space.explore(new_metadata)` applies the fitted preprocessing, selected features, projection, classifiers, and footprints without refitting. Test outcomes score predictions. They do not determine recommendations.

Use a new or loaded `InstanceSpace` for exploration after calling `close()`. For partial runs, stage iteration, and progress notifications, see [Build and Explore](https://andremun.github.io/pyInstanceSpace/workflows.html).

### Outputs

Exports require explicit method calls. The `outputs.csv`, `outputs.png`, and `outputs.web` flags do not automatically write files during `build()`.

| Method | Output |
|---|---|
| `model.save_to_csv(directory)` | CSV tables for coordinates, processed data, summaries, and footprint geometry. |
| `model.save_graphs(directory)` | PNG figures for the fitted analysis. |
| `model.save_for_web(directory)` | Web presentation data. |
| `model.save_to_mat(directory)` | MATLAB-format data export. This does not transfer fitted classifiers between implementations. |
| `model.save(path)` | Native fitted-model archive for Python. |
| `space.save(path)` | Checkpoint with runner state and the exploration interface. |

Create export directories before writing files. A 2D projection produces two coordinate columns. A 3D projection produces three and supports native footprint meshes. Read geometry manifests for their schema and index conventions.

Only load native archives and checkpoints from a trusted producer. See [Model and Persistence](https://andremun.github.io/pyInstanceSpace/model.html) for loading and authentication.

## Options

Configure an in-memory workflow with immutable dataclasses:

```python
from instancespace.data.options import InstanceSpaceOptions, PilotOptions

options = InstanceSpaceOptions.default(pilot=PilotOptions.default(dims=3))
```

Use `dataclasses.replace` to create an updated configuration. A nested JSON object uses the same groups and canonical field names. Omitted fields take their defaults. The [Options Reference](https://andremun.github.io/pyInstanceSpace/options-reference.html) describes all 74 fields and accepted aliases.

The following table shows frequently used **Python defaults**:

| Field | Default | Meaning |
|---|---|---|
| `perf.max_perf` | `False` | Larger performance values are better when `True`. |
| `perf.abs_perf` | `True` | Use an absolute good-performance threshold. `False` uses performance relative to the best algorithm on each instance. |
| `perf.epsilon` | `0.20` | Good-performance threshold. Its meaning depends on direction and absolute/relative mode. |
| `perf.beta_threshold` | `0.55` | An instance is easy when its fraction of good algorithms is strictly greater than this value. |
| `general.seed` | `0` | General random seed. Stage seeds of `None` inherit this value. |
| `parallel.flag` | `False` | Enable supported parallel work. |
| `parallel.n_cores` | `2` | Worker count when parallel work is enabled. |
| `selvars.feats`, `selvars.algos` | `None` | Select all columns. Lists restrict the feature or algorithm names. |
| `sifted.flag` | `True` | Enable automatic feature selection. |
| `sifted.k` | `6` | Number of feature clusters for SIFTED selection. |
| `pilot.dims` | `2` | Projection dimension, either 2 or 3. |
| `pilot.method` | `"standard"` | Standard PILOT projection. `"pls"` selects Partial Least Squares. |
| `pythia.classifier` | `"svm"` | Classifier: `svm`, `knn`, `tree`, `nb`, `linear`, or `ensemble`. |
| `pythia.tuning` | `"sobol"` | Hyperparameter search: `sobol`, `bayes`, or `none`. |
| `pythia.n_tuning_iter` | `20` | Candidate budget per retained algorithm, including Bayesian initialization. |
| `trace.method` | `"legacy"` | Footprint method. Select `"trace3"` explicitly for 2D/3D alpha shapes. |
| `trace.purity` | `0.55` | Legacy footprint purity target. Omitted TRACE3 purity defaults to `0.60`. |
| `trace.boundary_tolerance` | `0.0` | Closed-boundary membership tolerance in projection units. |
| `outputs.csv`, `outputs.png` | `True` | Caller preferences. File creation still requires explicit save methods. |

The website checks option coverage against the source schema. Python and MATLAB have different defaults, option names, and output behavior. Set performance and footprint settings explicitly when comparing them. See [MATLAB Compatibility](https://andremun.github.io/pyInstanceSpace/matlab-compatibility.html).

## Reproducibility and Platform Notes

Keep the input metadata, complete options, package version, and dependency versions with each analysis. Record the retained feature and algorithm order when comparing runs. Use explicit seeds and a fixed environment for repeated experiments.

A seed does not guarantee identical floating-point results, optimizer paths, or classifier outputs across platforms or implementations. Cross-validation summaries depend on the fitted preprocessing, feature selection, projection, and tuning. Use independent evaluation data to assess the complete analysis.

The validation workflow uses Linux and Python 3.12. On a machine without a graphical display, set `MPLBACKEND=Agg` before running Python to save figures. Coordinate application workers and numerical-library threads with the available CPU resources. CLOISTER bounds corner enumeration with `cloister.max_features` and uses observed-data bounds above that limit.

## AI-Assisted Analysis

The repository includes a [Python ISA skill](.claude/skills/instance-space-analysis/SKILL.md) for AI assistants. It covers data checks, option choices, stage interpretation, evaluation limits, and Python workflows. Its implementation guidance applies to this package. Verify MATLAB option names and behavior against the MATLAB toolkit when using that implementation.

## Develop

Install the locked development environment with Python 3.12 and Poetry 2.3.3:

```sh
poetry install
poetry run poe docs
poetry run pytest
```

The documentation build writes `site/` and checks option coverage, local files, and anchors. It gives guides and generated API pages a shared layout. Preview the site with `python -m http.server --directory site`. See [CONTRIBUTING.md](CONTRIBUTING.md) for the complete development checks.

## Cite

For the methodology:

> K. Smith-Miles and M. A. Muñoz. *Instance Space Analysis for Algorithm Testing: Methodology and Software Tools*. ACM Computing Surveys, 55(12), Article 255, 2023. [DOI](https://doi.org/10.1145/3572895).

For the Python software:

> Y. B. Güzel, K. Khare, N. Harvey, K. Dsouza, D. H. Jang, J. Chen, C. Z. Lam and M. A. Muñoz. *instancespace: A Python package for insightful algorithm testing through Instance Space Analysis*. SoftwareX, 31:102246, 2025. [DOI](https://doi.org/10.1016/j.softx.2025.102246).

> M. A. Muñoz and K. Smith-Miles. *Instance Space Analysis: A Python toolkit for the assessment of algorithmic power*. [Zenodo](https://doi.org/10.5281/zenodo.15562567).

For 3D Instance Space Analysis, also cite Simpson et al. (2025), *ISA3: A 3-dimensional expansion of Instance Space Analysis*, Machine Learning, 114, 240. [DOI](https://doi.org/10.1007/s10994-025-06871-5).

## License and Support

The package uses the [PolyForm Noncommercial License 1.0.0](LICENSE). Report defects and request features through [GitHub Issues](https://github.com/andremun/pyInstanceSpace/issues). See [CONTRIBUTING.md](CONTRIBUTING.md) before submitting changes and [SECURITY.md](SECURITY.md) for private security reporting.
