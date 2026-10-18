# Instance Space Analysis for Python

Understand algorithm performance across the instance space of a problem

InstanceSpace relates instance features to a portfolio's algorithm performance. It creates a 2D or 3D projection, fits algorithm selectors, and constructs footprints showing where algorithms perform well. The package implements the Instance Space Analysis methodology with Python's numerical and machine-learning libraries.

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

Start with [InstanceSpace](instance-space.html) for pipeline execution and [Model](model.html) for fitted results and exports. The stage pages describe each scientific step with syntax, inputs, outputs and examples. [Classifiers and Tuning](tuning.html) explains Sobol, Bayesian and fixed-parameter fitting.

The [API Reference](api-reference.html) supplies exact Python signatures and result fields. [MATLAB Compatibility](matlab-compatibility.html) records current differences between implementations.

## Installation

Use Python 3.12 in a virtual environment:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install instancespace
```

The distribution is named `instancespace`; import it as `instancespace`.

## Methodology

- Smith-Miles, K. and Muñoz, M. A. (2023). Instance Space Analysis for Algorithm Testing: Methodology and Software Tools. *ACM Computing Surveys*, 55(12), Article 255. [DOI](https://doi.org/10.1145/3572895).
- Simpson, C., Muñoz, M. A., Kandanaarachchi, S. and Campello, R. J. G. B. (2025). ISA3: A 3-dimensional expansion of Instance Space Analysis. *Machine Learning*, 114, 240. [DOI](https://doi.org/10.1007/s10994-025-06871-5).

The package is distributed under the [PolyForm Noncommercial License 1.0.0](https://polyformproject.org/licenses/noncommercial/1.0.0/).
