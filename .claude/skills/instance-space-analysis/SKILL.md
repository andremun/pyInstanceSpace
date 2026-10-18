---
name: instance-space-analysis
description: >
  Run, interpret or advise on Instance Space Analysis with the Python
  instancespace package. Use for feature/performance metadata, build/explore
  workflows, PILOT projections, algorithm selection, TRACE footprints and
  benchmark coverage, including comparisons with the MATLAB implementation.
---

# Instance Space Analysis with Python

Help the user connect problem-instance features to algorithm performance,
selection and geometric footprints. Work within the requested analysis or code
change. This skill supplies methodology and Python-specific boundaries; it does
not authorize publication, fixture promotion or changes to the MATLAB repository.

## Start with the question and data

Identify the portfolio, performance measure, whether larger or smaller is better,
and what “good” means for the application. Make the absolute/relative mode and
threshold explicit. Do not change them merely to make all algorithms survive or
to improve a picture.

Metadata rows are instances; `feature_` columns describe structure and `algo_`
columns hold nonnegative observed performance. Require an `instances` identifier,
at least three viable input features and one training algorithm. Source labels
are optional. Training data must be complete after ingestion filtering;
exploration may omit outcomes. Check feature units, alignment, class balance and
whether the configured folds are viable before fitting.

Assess whether the features and collected instances represent the domain of
interest. Source separation or an attractive projection alone does not establish
benchmark diversity, model accuracy or a causal explanation of difficulty.

## Choose and run the workflow

Use the [Getting Started example](../../../docs_site/src/getting-started.md) for a
self-contained build, plotting, exports, checkpoint and feature-only exploration.
Use [Build and Explore](../../../docs_site/src/workflows.md) for partial stages and
iteration, and [Custom Stages](../../../docs_site/src/plugins.md) for extensions.
Read [Metadata Format](../../../docs_site/src/metadata-format.md) when loading new
application data.

Read defaults and validation from `InstanceSpaceOptions.default()`,
`instancespace/data/options.py` and the
[Options Reference](../../../docs_site/src/options-reference.md). Do not copy
MATLAB defaults into an unspecified Python configuration. Options are frozen;
use `dataclasses.replace` for changes. Some fields are compatibility preferences,
not active numerical controls, as documented in the catalog.

The pipeline performs PREPROCESSING, PRELIM, SIFTED and PILOT, then PYTHIA and
CLOISTER, followed by TRACE. To explain results:

- **PRELIM:** distinguish raw outcomes, relative regret, transformed performance
  and binary good/bad labels. Pipeline training prunes never-good algorithms and
  recomputes from retained raw data; use the retained portfolio for downstream
  parameter rows and interpretation. Zero-best relative comparisons use a finite
  epsilon surrogate and can be sensitive to performance units.
- **SIFTED:** correlation screening/clustering and genetic PILOT/KNN fitness
  choose features. `n_trees` does not control a Random Forest in current fitness.
  Feature selection on all data also conditions subsequent CV statistics.
- **PILOT:** choose standard or PLS projection and 2D/3D deliberately. Interpret
  reconstruction and distance preservation before drawing conclusions from
  distances or apparent separation. PLS inference reuses its fitted mean;
  standard inference uses the stored linear projection directly.
- **CLOISTER:** distinguish projected feature-range/correlation bounds from the
  observed-data fallback above the feature cap. Neither certifies that every
  unobserved interior point is physically feasible. Domain knowledge is needed
  when interpreting gaps or proposing new instances.
- **PYTHIA:** a binary learner predicts good performance for each algorithm;
  the saved training policy combines those predictions into recommendations.
  A classifier probability, an Oracle summary and a recommended algorithm are
  different outputs. Python selectors are zero-based, with `-1` for abstention;
  PRELIM's best-algorithm index is one-based.
- **TRACE:** report footprint support, purity and area/volume with the geometry.
  Python defaults to legacy TRACE; request TRACE3 explicitly when desired.
  Reaching the end of TRACE3's search does not prove the purity target was met.
  Python does not persist MATLAB's acceptance/termination fields.

## Evaluate and interpret honestly

Apply a fitted model to new instances without refitting preprocessing, selected
features, projection, classifiers or footprints. Replay the retained feature
order. Outcomes may score predictions but must not determine recommendations.
Unknown or absent outcomes are excluded from metrics, not treated as observed
bad performance. New test algorithms lack trained classifiers and cannot be
recommended merely because their outcomes are available.

CV summaries are conditional on fitted preprocessing, feature selection,
projection and tuning. They are not an unbiased outer-CV estimate of the full
analysis. Use independent evaluation or a suitably designed outer validation
when the user's question requires generalization evidence. Report sampling and
source-coverage limits and undefined metrics rather than replacing `NaN` with a
favorable value.

Footprint membership includes boundaries. The explicit distance tolerance is in
projection units, defaults to zero, and is saved with the fitted model. A positive
value changes membership rather than geometry or measure; it does not guarantee
stable sequential legacy contradiction trimming. Do not adjust geometry or a
query-batch-dependent tolerance to manufacture agreement.

Exports require explicit `Model` save methods and existing output directories.
Native `Model` persistence and `InstanceSpace` checkpoints serve different
purposes. Load only trusted producers' native files. CSV/MAT exports do not make
fitted Python and MATLAB classifiers interchangeable.

## Compare implementations and maintain evidence

Consult [MATLAB Compatibility](../../../docs_site/src/matlab-compatibility.md) and
[the architecture](../../../docs/architecture.md) before asserting parity.
Match performance mode, threshold, portfolio, retained features, projection,
TRACE settings, parameter units and evaluation protocol. Shared seeds alone do
not imply shared folds, starting points or optimizer trajectories.

For search comparisons, record inputs, folds, initial design, budgets, seeds,
candidate/objective traces and cross-evaluations. Python uses plain EI with four
initial points and a default budget of 20; MATLAB uses EI-plus. The
[controlled SVM comparison](../../../docs/research/bayes-304/report.md) supports
retaining those Python settings, within its stated dataset and classifier scope.
Do not infer an optimizer defect from rounded final metrics alone.

Use the [fixture workflow](../../../tests/matlab_export/README.md) and
[provenance contract](../../../docs/fixture-provenance.md) for scientific
references. Preserve raw evidence and immutable identities; diagnostic,
unknown-provenance and Python-generated results are not approved MATLAB oracles.
For code or documentation changes, follow
[CONTRIBUTING.md](../../../CONTRIBUTING.md), keep every option documented, and
verify the affected contracts. Use GitHub Issues for active work rather than
reintroducing completed plans. Methodology citations are in
[README.md](../../../README.md#cite) and [CITATION.cff](../../../CITATION.cff).
