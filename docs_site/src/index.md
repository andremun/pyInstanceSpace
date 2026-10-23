# Instance Space Analysis for Python

Understand algorithm performance across the instance space of a problem

Instance Space Analysis (ISA) relates the structural features of problem instances to the performance of a portfolio of algorithms. This Python package projects instances into a 2D or 3D space, fits algorithm selectors, and constructs algorithm footprints. A footprint describes a region where an algorithm is expected to perform well under a chosen performance threshold.

Use ISA to compare algorithm strengths, examine feature and performance relationships, inspect benchmark diversity, and recommend algorithms for new instances. Gaps in the observed space can help identify where additional instances would improve coverage. The conclusions depend on the features, performance measure, and collected instances. Footprints do not guarantee good performance for every new instance within their boundaries.

<div class="figure">
<svg class="pipeline-diagram" viewBox="0 0 760 145" role="group" aria-label="Pipeline stages: PREPROCESSING, PRELIM, SIFTED, PILOT, then CLOISTER and PYTHIA, followed by TRACE">
<defs><marker id="pipeline-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="currentColor"/></marker></defs>
<g fill="none" stroke="currentColor" stroke-width="1.2" marker-end="url(#pipeline-arrow)"><path d="M150 70 H168"/><path d="M252 70 H270"/><path d="M354 70 H372"/><path d="M456 70 H489 V31 H520"/><path d="M456 70 H489 V109 H520"/><path d="M624 109 H658"/></g>
<a href="preprocessing.html"><rect x="8" y="52" width="142" height="36" rx="5"/><text x="79" y="75" text-anchor="middle">PREPROCESSING</text></a>
<a href="prelim.html"><rect x="170" y="52" width="82" height="36" rx="5"/><text x="211" y="75" text-anchor="middle">PRELIM</text></a>
<a href="sifted.html"><rect x="272" y="52" width="82" height="36" rx="5"/><text x="313" y="75" text-anchor="middle">SIFTED</text></a>
<a href="pilot.html"><rect x="374" y="52" width="82" height="36" rx="5"/><text x="415" y="75" text-anchor="middle">PILOT</text></a>
<a href="cloister.html"><rect x="522" y="13" width="102" height="36" rx="5"/><text x="573" y="36" text-anchor="middle">CLOISTER</text></a>
<a href="pythia.html"><rect x="522" y="91" width="102" height="36" rx="5"/><text x="573" y="114" text-anchor="middle">PYTHIA</text></a>
<a href="trace.html"><rect x="660" y="91" width="92" height="36" rx="5"/><text x="706" y="114" text-anchor="middle">TRACE</text></a>
</svg>
<p class="caption">PILOT supplies the projected space to CLOISTER and PYTHIA. TRACE uses the trained labels and coordinates.</p>
</div>

## Get Started

<div class="tiles">
<div class="tile"><h3><a href="getting-started.html">Getting Started</a></h3><p>Install the package and build a complete instance space from a runnable example.</p></div>
<div class="tile"><h3><a href="metadata-format.html">Metadata Format</a></h3><p>Prepare features, algorithm outcomes and instance identifiers.</p></div>
<div class="tile"><h3><a href="workflows.html">Build and Explore</a></h3><p>Inspect stages, save a checkpoint and evaluate new instances.</p></div>
<div class="tile"><h3><a href="options-reference.html">Options Reference</a></h3><p>Every configuration field, its default, accepted JSON aliases and effect.</p></div>
</div>

## Classes and Reference

Start with [InstanceSpace](instance-space.html) to build and explore an analysis. Use [Model](model.html) to inspect fitted results, save an archive, or export files.

Each stage page provides syntax, a description, parameters, results, and examples:

| Stage | Purpose |
|---|---|
| [PREPROCESSING](preprocessing.html) | Select metadata columns and filter missing inputs. |
| [PRELIM](prelim.html) | Define good performance and fit data transformations. |
| [SIFTED](sifted.html) | Select informative features. |
| [PILOT](pilot.html) | Project features into a 2D or 3D instance space. |
| [CLOISTER](cloister.html) | Estimate empirical bounds. |
| [PYTHIA](pythia.html) | Train algorithm selectors and score observed outcomes. |
| [TRACE](trace.html) | Construct footprints and evaluate membership. |

[Classifiers and Tuning](tuning.html) explains the learners and search strategies. [Outputs and Plotting](outputs.html) describes file formats, geometry, and figures. [Custom Stages](plugins.html) explains pipeline extensions.

The [API Reference](api-reference.html) supplies exact Python signatures and result fields. [MATLAB Compatibility](matlab-compatibility.html) describes shared contracts and current differences. Use [Troubleshooting](troubleshooting.html) for input, configuration, and workflow errors.

## Installation

Use Python 3.12 in a virtual environment:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install instancespace
```

On Windows, activate the environment with `.\.venv\Scripts\Activate.ps1` in PowerShell. The distribution and import names are both `instancespace`. Python workflows do not require MATLAB, GNU Octave, or MATLAB toolboxes.

## Reproducibility

Keep the input metadata, complete options, package version, and dependency versions with each analysis. Choose performance direction, threshold mode, epsilon, projection dimensions, and footprint method explicitly. Stage seeds of `None` inherit `general.seed`, whose Python default is 0.

A fixed seed does not guarantee identical optimizer paths or floating-point results across platforms or implementations. Use independent evaluation data to assess the complete analysis. PYTHIA cross-validation summaries remain conditional on the fitted preprocessing, feature selection, projection, and tuning.

## AI-Assisted Analysis

The repository includes a [Python ISA skill](https://github.com/andremun/pyInstanceSpace/blob/main/.claude/skills/instance-space-analysis/SKILL.md) for AI assistants. It covers data checks, option choices, stage interpretation, and evaluation limits. Its implementation guidance applies to this Python package.

## Methodology

- Smith-Miles, K. and Muñoz, M. A. (2023). Instance Space Analysis for Algorithm Testing: Methodology and Software Tools. *ACM Computing Surveys*, 55(12), Article 255. [DOI](https://doi.org/10.1145/3572895).
- Simpson, C., Muñoz, M. A., Kandanaarachchi, S. and Campello, R. J. G. B. (2025). ISA3: A 3-dimensional expansion of Instance Space Analysis. *Machine Learning*, 114, 240. [DOI](https://doi.org/10.1007/s10994-025-06871-5).

For the Python software, cite:

- Güzel, Y. B. et al. (2025). instancespace: A Python package for insightful algorithm testing through Instance Space Analysis. *SoftwareX*, 31, 102246. [DOI](https://doi.org/10.1016/j.softx.2025.102246).
- Muñoz, M. A. and Smith-Miles, K. Instance Space Analysis: A Python toolkit for the assessment of algorithmic power. [Zenodo](https://doi.org/10.5281/zenodo.15562567).

## License and Support

The package uses the [PolyForm Noncommercial License 1.0.0](https://polyformproject.org/licenses/noncommercial/1.0.0/). Report defects and request features through [GitHub Issues](https://github.com/andremun/pyInstanceSpace/issues). See [Contributing](https://github.com/andremun/pyInstanceSpace/blob/main/CONTRIBUTING.md) for development checks and [Security](https://github.com/andremun/pyInstanceSpace/blob/main/SECURITY.md) for private security reporting.
