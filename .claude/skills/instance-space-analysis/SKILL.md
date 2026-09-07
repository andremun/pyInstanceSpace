---
name: instance-space-analysis
description: >
  Perform, interpret, or advise on Instance Space Analysis (ISA) -- the
  methodology for objective algorithm benchmarking via a 2D/3D projection
  of problem instances (Smith-Miles & Munoz, ACM Comput. Surv. 55(12),
  2023). Use this skill whenever the user wants to: build an instance
  space from feature/performance meta-data; interpret PRELIM, SIFTED,
  PILOT, CLOISTER, PYTHIA, or TRACE/TRACE3 output; design a new ISA
  application for a problem domain; debug the MATLAB InstanceSpace
  toolkit or the pyInstanceSpace Python package; or assess whether a
  benchmark suite or instance space is adequate. Trigger on "instance
  space", "algorithm footprint", "PILOT projection", "TRACE3",
  "MATILDA", "ISA toolkit", or a request to run/interpret this pipeline
  on new meta-data. See references/matlab-toolkit.md and
  references/python-toolkit.md for operational (file-format, CLI,
  config-schema) detail; this file covers methodology, decisions, and
  pitfalls.
---

# Instance Space Analysis (ISA)

ISA extends Rice's Algorithm Selection Problem (1976) with a geometric
layer: instead of learning a selection mapping directly from features to
algorithms, it projects instances into a 2D (or 3D) plane and identifies
contiguous regions -- **footprints** -- where each algorithm performs
well. The result is not just a predictor; it is a visual, falsifiable
account of *why* an algorithm wins where it wins, and whether the
benchmark suite used to draw that conclusion was diverse enough to trust.

Primary source: Smith-Miles & Munoz, "Instance Space Analysis for
Algorithm Testing: Methodology and Software Tools," *ACM Computing
Surveys* 55(12), Article 255, 2023 (the tutorial this skill is built
from). Supersedes/extends: Munoz, Villanova, Baatar & Smith-Miles,
*Machine Learning* 107(1), 2018 (PILOT); Simpson, Munoz, Kandanaarachchi
& Campello, "ISA3: a 3-dimensional expansion of instance space analysis,"
*Machine Learning* 114:240, 2025 (3D projection, TRACE3).

Everything in this file that describes *current code behaviour* was
verified against `andremun/InstanceSpace` (MATLAB, v0.9.1,
`98a01ac0513c0dd0f8a9bd91ed2926c871334d7b`) and `andremun/pyInstanceSpace`
(Python, `instancespace` package version 0.3.0) as of 2026-09-07, not
reconstructed from the papers. Where the shipped code has moved past
what a paper describes, that is flagged explicitly -- treat the code as
ground truth per the source-of-truth discipline in the persona/
scientific-computing skills, and re-verify against the live repository
before trusting a detail below if it might have changed since (both
projects move quickly; check version numbers/commits again first).

---

## 1. The conceptual framework -- six spaces

| Space | Symbol | Meaning |
|---|---|---|
| Problem space | P | All relevant instances of the problem, most of which are never observed |
| Instance subset | I ⊂ P | The instances actually collected, with meta-data |
| Feature space | F | Each instance x is a vector f(x) characterising its structural difficulty |
| Algorithm space | A | The portfolio of algorithms compared |
| Performance space | Y | A user-defined "goodness" measure y(alpha, x) per (algorithm, instance) pair |
| Instance space | Z ⊂ R^2 (or R^3) | The projected 2D/3D coordinates the whole analysis is read from |

Meta-data is two matrices: F ∈ R^(m×n) (m features, n instances) and
Y ∈ R^(a×n) (a algorithms). Rice's original framework only learns
S(f(x), y) → alpha*; ISA additionally learns a projection g(f(x), y) → z,
so that *why* is answered geometrically, not just *which*.

---

## 2. The six-step iterative methodology

1. **Collect meta-data**: instances I, features F, performance Y for a
   portfolio A. This step is the one most likely to determine whether
   later insights are trustworthy -- see Section 6.
2. **Construct the instance space**: feature selection, projection, and
   theoretical boundary (PRELIM -> SIFTED -> PILOT -> CLOISTER; Section 3).
3. **Automated algorithm selection**: learn alpha* from instance
   coordinates (PYTHIA; Section 3).
4. **Generate footprints and metrics** (TRACE / TRACE3; Section 3).
5. **Analyse**: distribution of sources, SVM/classifier accuracy,
   footprints, feature distributions (Section 4).
6. **Augment**: generate new instances, algorithms, or features to close
   gaps identified in Step 5; return to Step 2, or stop if the space has
   converged (no new features selected, no gaps in the boundary).

Convergence is a real stopping criterion, not a formality: ISA is
finished for a given algorithm set when the boundary's interior is
densely filled by real or generated instances and the feature set is
stable across an added iteration. Treat "we ran ISA once" as an
interim result, not a final one, unless the domain genuinely has no route
to generate more instances.

---

## 3. The five core methods

### PRELIM -- Preparation for Learning of Instance Meta-Data

Two jobs: (a) define binary "good performance," and (b) bound and
normalise the meta-data.

- **Good performance** is user-defined and consequential, not a detail:
  *absolute* (performance passes a fixed threshold) or *relative*
  (within a margin of the best algorithm on that instance). The paper
  calls this choice "somewhat arbitrary but exerting significant
  influence on the final results" -- do not let it be an unexamined
  default; state and justify it the way you would a p-value threshold.
- Bounding: each feature clipped to median +/- 5*IQR (robust to
  outliers by construction).
- Normalisation: one-parameter Box-Cox (lambda fit by maximum
  log-likelihood) then z-score, applied to both features and
  performance.
- **Code detail beyond the paper**: the shipped `PRELIM.m` also converts
  performance into a *relative deficit* from the best algorithm on each
  instance (`Y = 1 - Y/Ybest` maximising, `Y/Ybest - 1` minimising) before
  normalisation -- not the raw performance values. It also derives a
  per-instance `beta` (easy/hard) flag from a user `betaThreshold`
  fraction of algorithms achieving good performance, used later by
  TRACE's beta-hard footprint. Ties in "best algorithm" are broken at
  random per instance, logged as a percentage.
- **Fabricated-truth trap for algorithms trained but not covered by a
  test set** (verified 2026-09; filed as `andremun/InstanceSpace#58`):
  `INIT.m` seeds unreconciled algorithm columns with `NaN`, and PRELIM's
  binarisation (`Ybin = Yaux >= opts.epsilon`) turns every `NaN`
  comparison into `False` -- i.e. an algorithm the model was trained on
  but that the test set never exercised gets a fabricated all-bad truth
  column, not a "no data" marker. PYTHIA's `PYTHIAevalMode` (see below)
  then unconditionally scores against it. Read per-algorithm PYTHIA
  precision/recall/accuracy only for algorithms the test set actually
  covers; a suspiciously bad score for a trained-but-uncovered algorithm
  is this artefact, not a real finding.

### SIFTED -- Selection of Instance Features to Explain Difficulty

Goal: a small, uncorrelated feature subset that explains algorithm
performance.

- Textbook version (2023 tutorial, Algorithm 2): correlation filter
  (keep the top feature per algorithm, plus any feature with |correlation|
  >= 0.3) -> k-means clustering on 1-|correlation| dissimilarity (K
  clusters, default 10) -> one feature per cluster chosen by lowest
  out-of-bag error of a Random Forest on a temporary PCA 2D projection.
- **Code has moved past this description.** The shipped `SIFTED2.m` (and
  its Python port `stages/sifted.py`) still does the correlation filter
  (default `rho = 0.1` minimum correlation, `pval = 0.05` significance,
  not the paper's 0.3 threshold), and still clusters on 1-|correlation|
  with k-means (default K = 10), but replaces the PCA+RandomForest
  representative-selection step with a **genetic algorithm** whose
  fitness is the k-fold cross-validated loss of a k-NN classifier (k=3)
  built on a *quick PILOT projection* of the candidate feature
  combination. This is a materially different (and more expensive)
  search than the paper describes. If reproducing a published number
  or a paper's exact SIFTED description, verify which version and which
  commit produced it.
- Edge cases handled explicitly in code: <=1 feature is a hard error;
  <=3 features skips clustering entirely; fewer features than K skips
  clustering. A silhouette-score advisory suggests a better K if the
  chosen K clusters poorly (below 0.5) -- read this warning, since a bad
  K silently degrades the whole downstream projection.

### PILOT -- Projecting Instances with Linearly Observable Trends

The most important algorithm in ISA (its own description). Finds
A_r ∈ R^(2×q), B_r ∈ R^(q×2), C_r ∈ R^(a×2) minimising the joint
reconstruction error of features and performance:

```
min  ||F~ - B_r Z||_F^2 + ||Y - C_r Z||_F^2   s.t.  Z = A_r F~
```

- A **global optimum exists but is not unique** (proved in Munoz et al.
  2018, Theorem 1/2); the practical solver is BFGS from `Ntry` random
  restarts (paper default 30; **shipped MATLAB/Python default is 10**
  via `opts.pilot.ntries` / `PilotOptions.n_tries`, matching between the
  two toolkits since Python's default was raised from 5 to 10), keeping
  the restart with the highest topological preservation (Pearson
  correlation between feature-space and instance-space pairwise
  distances).
- An **analytic** fallback exists (`opts.pilot.analytic` /
  `PilotOptions.analytic`, only consulted when `method='standard'`) via
  the top-2 eigenvectors of [F~; Y][F~; Y]', valid only when F~F~' is
  invertible; falls back to numerical automatically if the feature
  matrix is rank-deficient. A separate `'pls'` method (Partial Least
  Squares) also exists in the Python port and ignores `analytic`
  entirely.
- PCA is a *provably suboptimal* solution to this same objective
  (Munoz et al. 2018) -- if someone asks "why not just use PCA," this is
  the citable answer, not a stylistic preference.
- **3D extension (ISA3, Simpson et al. 2025)**: redefine Z ∈ R^(n×3) and
  solve the same problem with 3-column A_r/B_r/C_r; then solve a second,
  small optimisation for an optimal *viewing rotation* V (2 columns) that
  flattens the 3D space for a specific feature/algorithm subset, with an
  orthogonality penalty (default scaling lambda=0.2, low-sensitivity
  below 0.5). 3D retains more information and separates good/bad
  instances better (validated on 56 anomaly-detection variants; F-score
  improved for 12 of 16 checked cases moving 2D TRACE3 -> 3D TRACE3), but
  costs a manual/optimised viewpoint choice per subset -- it is not a
  free upgrade for every figure, only where crowding in 2D is the actual
  problem. Present in both toolkits: MATLAB's `opts.ISA3D`/`ISA3D`
  branches in `PILOT.m`, and the Python port's `PilotOptions.dims`
  (2 or 3, default 2) with `PilotOptions.view_groups` selecting one
  2D viewpoint per algorithm group when `dims=3` (see
  `references/python-toolkit.md`).

### CLOISTER -- Correlated Limits of the Instance Space's Theoretical Regions

Projects the *theoretical* boundary of all feasible instances (not just
the observed ones) by combinatorially enumerating feature-bound vertices
(each feature at its min or max), pruning combinations that violate
observed feature correlations (a vertex cannot pair f_i at its upper
bound with f_j at its lower bound if rho_ij is strongly positive, and the
symmetric rule for strongly negative rho), then taking the convex hull of
the surviving vertices' projections.

- Default correlation-collinearity threshold and significance:
  `cthres = 0.7`, `pval = 0.05` in the shipped defaults (matches the
  paper's epsilon/p notation).
- **Code guard beyond the paper**: enumeration is 2^(number of features),
  so the shipped MATLAB code hard-caps at `MAX_FEATS = 20` and silently
  falls back to a plain convex hull of the *observed* projected instances
  if exceeded -- at that point CLOISTER is no longer estimating a
  theoretical boundary, it is just re-describing the empirical one. Do
  not interpret a >20-feature CLOISTER boundary as theoretical without
  checking which branch ran.
- The boundary is the tool for assessing benchmark diversity: if real
  instances hug a small region far from the boundary's interior, the
  benchmark suite is not representative of the theoretically possible
  problem space.

### PYTHIA -- Automated Algorithm Selection

Trains one classifier per algorithm on the (z-scored) 2D/3D instance
coordinates, predicting the PRELIM binary good/bad label. MATLAB
self-tunes an SVM's (C, gamma) via Bayesian optimisation (MATLAB native)
or random search on a Latin hypercube (LIBSVM), both bounded to
[2^-10, 2^4] with k-fold CV, 30 iterations, Gaussian kernel by default
(the code recommends switching to polynomial above ~1000 instances). The
Python port keeps SVM as the default (`PythiaOptions.classifier = "svm"`)
but exposes a registry of six selectable classifiers -- `svm`, `knn`,
`tree`, `nb` (naive Bayes), `linear`, `ensemble` -- via
`PythiaOptions.classifier`; check `references/python-toolkit.md` for the
current tuning options per classifier. Ties among algorithms predicted
"good" are broken by higher model precision; if no classifier predicts
good, it falls back to the algorithm with the highest average performance
and flags the instance as having no confident recommendation.

**`PythiaOptions.skip` / MATLAB's `opts.pythia.skip`** bypasses
classifier training entirely (e.g. when only footprint construction is
wanted, or predictions are supplied precomputed). Legacy TRACE requires
true-label footprints when this is set, since there is no PYTHIA `Yhat`
to build footprints from; TRACE3 can fall back to true labels too without
needing `trace.use_sim` changed.

**Precision matters more than accuracy here**: a high-precision,
lower-recall classifier is exactly what you want, because the claim being
made is "when I recommend this algorithm, trust it," not "I never miss a
good instance." Read PYTHIA's precision column before its accuracy
column.

**`PYTHIAevalMode` fabricated-truth trap**: see the PRELIM pitfall above
(`andremun/InstanceSpace#58`) -- a trained algorithm absent from the test
set is scored against an all-`False` truth column, not skipped. This
applies to both `explore()`-time evaluation in the Python port and
MATLAB's equivalent evaluation path.

### TRACE / TRACE3 -- Algorithm Footprints

A footprint is characterised by three numbers, all normalised against
the convex-hull baseline of the full instance space: **area** (percentage
of the space -- a robustness proxy), **density** (instances per unit
area relative to the space's overall density -- strength of evidence),
and **purity** (fraction of enclosed instances that are actually good --
absence of contradicting evidence).

- **Legacy TRACE** (Munoz & Smith-Miles 2017): DBSCAN to find dense
  clusters of good instances (k and epsilon auto-set from instance
  count via a density-based heuristic), alpha-shapes around each
  cluster, then pairwise overlap resolution between algorithms' best-
  footprints by removing the lower-purity side of any overlap (ties
  kept, as evidence is insufficient to prefer either).
- **TRACE3** (Simpson et al. 2025; the default method in the MATLAB
  toolkit's `options.json`, `opts.trace.method = "trace3"`):
  replaces DBSCAN with a k-NN classifier per algorithm (paper default
  k=50, prior [0.6, 0.4] -- meaning a majority-of-neighbourhood rule
  biased toward the "bad" class, i.e. conservative), builds an
  alpha-shape only from instances where the true and predicted labels
  agree, then iteratively shrinks the alpha-radius until a purity
  threshold is met (default 0.6) or the shape collapses -- trimming a
  fixed fraction (5% per step, i.e. a "region threshold" of area/20) of
  residual area each iteration to discard outlier-driven offshoots.
  Works natively in 3D (volumes, not just areas).
- **`pyInstanceSpace` now implements TRACE3 as well as legacy TRACE**
  (verified 2026-09 -- this changed since the 2025 revision of this
  skill, which flagged TRACE3 as Python-side unimplemented; that is no
  longer true). Selectable via `TraceOptions.method` (`"legacy"`
  default, or `"trace3"`); the purity default is method-aware
  (`DEFAULT_TRACE_PURITY = 0.55` for legacy, `DEFAULT_TRACE3_PURITY =
  0.60` for trace3, matching the paper's default). Native 3D volumes are
  supported by pairing `trace.method="trace3"` with `pilot.dims=3`.
  `TraceOptions` also gained `min_instances` and `min_area_frac` floor
  parameters not present in the original description. Check
  `references/python-toolkit.md` for the exact current field list before
  relying on this from code, since both toolkits continue to evolve.
- **MATLAB `nn`/`prior` dispatch -- re-verify before trusting**: as of an
  earlier commit of the MATLAB toolkit, `TRACE.m`'s TRACE3 dispatch
  reused PYTHIA's `Yhat` directly as the "predicted good" label rather
  than building its own k-NN classifier from `nn`/`prior`, falling back
  to true labels (`Ybin`) only if `Yhat` was empty -- meaning
  `options.json`'s `"trace": {"nn": 50, "prior": [0.6, 0.4]}` fields did
  nothing in that code path. The MATLAB toolkit has since advanced to
  v0.9.1; this specific claim was not re-checked against that commit as
  part of this update -- verify against the exact commit in use before
  repeating it, rather than trusting this note indefinitely.
- TRACE3 vs. legacy, per the ISA3 paper's own validation: TRACE3 produces
  a footprint (possibly small) in almost every case where legacy DBSCAN
  produces none; TRACE3 footprints have normalised density > 1 far more
  often (denser-than-average regions actually found); legacy sometimes
  reports a higher raw purity, but by covering a much smaller, weak-
  evidence area (excluding real bad instances rather than finding
  where good instances concentrate). Treat legacy TRACE's "no footprint"
  outcome for an algorithm as ambiguous, not as proof of a weak
  algorithm -- this exact failure mode (LOF) contradicted a prior
  published finding and was traced to the footprint algorithm, not to
  LOF actually being weak.
- k, prior, and purity threshold are all described in the ISA3 paper as
  low-sensitivity in a wide range (k=50 default; purity threshold 0.6
  default) but each has a documented failure mode at the extremes:
  k too small overfits small good-instance pockets into spurious
  footprints; k too large misses genuinely good but sparse regions;
  prior too permissive (e.g. [0.5,0.5]) admits low-density noise;
  purity threshold below the prior's "good" weight has little to no
  effect (must be raised together with a more conservative prior, not
  independently).

---

## 4. Reading the results (Step 5 of the methodology)

Work through these in order; each answers a different adequacy question.

1. **Distribution of instance sources**: are real-world instances
   distinct from synthetic ones? Do randomly generated instances cluster
   near the space's centre (typical for naive random generators) while
   structured/real instances occupy distinctive regions? Are there
   visible holes -- regions inside the theoretical boundary with no
   instances at all?
2. **Algorithm-selection accuracy** (PYTHIA precision/recall per
   algorithm): if precision is high and the predicted-good regions
   visually match the empirical good-performance scatter, the selected
   features are adequate. If not, the feature set -- not necessarily the
   algorithms -- is the thing to revisit. Only trust this for algorithms
   the test set actually covers (see the fabricated-truth pitfall above).
3. **Footprints**: which algorithms have unique, dense, pure regions?
   Overlap and shared strength are real, informative findings, not
   failures of the method -- two strong, broadly competitive algorithms
   with heavily overlapping footprints is itself the finding.
4. **Feature distributions across the space**: for each axis-defining
   feature, does its gradient across the space explain why the easy/hard
   or algorithm-A/algorithm-B regions fall where they do? This is where
   the "why," not just the "where," comes from -- an instance space with
   good footprints but uninterpretable feature gradients has captured
   discriminative power without captured mechanism.
5. **Insights**: synthesise 1-4 into statements a domain expert could
   falsify (e.g. "algorithm X wins when slack is high and teacher-
   conflict degree is low"), not just "algorithm X has R^2 = 0.8."

---

## 5. When to augment vs. stop (Step 6)

Augment the meta-data when: instances are not diverse/dense enough to
fill the theoretical boundary; footprints are near-identical across a
supposedly diverse algorithm portfolio (may indicate the algorithms are
mechanistically similar, or that the instances fail to expose real
differences -- distinguish these two explanations before concluding
either); or PYTHIA's accuracy is poor for specific algorithms (usually a
feature-inadequacy problem, not an algorithm problem, unless it is the
fabricated-truth artefact above). New instances, new algorithms, and new
features are three independent levers -- identify which one the Step-5
analysis actually implicates before spending effort generating more of
the wrong thing.

Stop when the boundary's interior is adequately filled and an added
iteration selects the same features. Do not treat a single ISA pass as
final by default; say so explicitly if scope or time genuinely limits the
work to one iteration.

---

## 6. Pitfalls (checked against the papers and the live code)

- **The "good performance" threshold is a hidden researcher degree of
  freedom.** State it, justify it, and consider reporting sensitivity to
  it, exactly as you would a significance level.
- **Small feature counts silently change the pipeline.** SIFTED skips
  clustering below 3 or below K features; CLOISTER silently degrades to
  an empirical convex hull above 20 features. Know which branch executed
  before interpreting the result as the textbook algorithm.
- **PILOT's global optimum is not unique.** Different valid solutions can
  give visually different (but equally optimal by the objective) axes;
  the topological-preservation tiebreak is what actually selects the
  reported figure, and it depends on `Ntry` random restarts (shipped
  default 10 in both toolkits, not the paper's 30) -- a low `Ntry` on a
  hard instance can under-explore the optimum landscape.
- **A trained-but-test-uncovered algorithm gets a fabricated "always
  bad" truth column, not a "no data" marker.** PRELIM's `NaN`-comparison
  binarisation plus PYTHIA's unconditional `PYTHIAevalMode` scoring means
  a suspiciously poor precision/recall/accuracy for one algorithm can be
  this artefact rather than a real finding -- filed as
  `andremun/InstanceSpace#58`. Check test-set algorithm coverage before
  trusting a single algorithm's PYTHIA scores.
- **TRACE's MATLAB `nn`/`prior` dispatch may not do what the config
  implies** -- see the verified-at-an-earlier-commit discrepancy above.
  Do not assume `options.json` fully determines footprint behaviour
  without checking which TRACE branch actually consumes those fields in
  the commit you are using; this was last checked before the MATLAB
  toolkit's v0.9.1, so re-verify.
- **A missing footprint is not evidence of a weak algorithm** -- it may
  be a footprint-algorithm artefact (documented LOF/legacy-TRACE case).
  Cross-check against a second footprint method (legacy vs TRACE3, or 2D
  vs 3D) before concluding an algorithm is uncompetitive.
- **CSV formatting failures are the most common practical error**:
  `NA` instead of `NaN`, Excel error codes (`#REF!`, `#DIV/0!`), or empty
  rows all corrupt PRELIM silently or loudly depending on downstream
  step. Validate the raw CSV before debugging the pipeline.
- **Ground-truth features are legitimate in ISA** (unlike in meta-
  learning/algorithm selection proper) precisely because ISA's goal is
  explanation, not deployable prediction -- do not reflexively flag their
  use as a leakage bug; check which goal the analysis is actually serving
  first.
- **A published ISA figure and a rerun of the current toolkit can
  legitimately differ** even on the same nominal method name (SIFTED is
  the clearest example here). When reproducing a specific paper's
  numbers, pin the toolkit commit, not just the method name.
- **Both toolkits move quickly.** This file's TRACE3-in-Python correction
  is itself evidence: a claim that was true in an earlier revision of
  this skill (Python lacking TRACE3) became false within the same
  project's lifetime. Re-check package version / commit before repeating
  any "X is/isn't implemented" claim from this file verbatim.

---

See `references/matlab-toolkit.md` for the MATLAB `InstanceSpace` file
format, `options.json` schema with verified current defaults, and
pipeline invocation; see `references/python-toolkit.md` for the
`pyInstanceSpace` package API and stage-based architecture.
