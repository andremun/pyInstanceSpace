# Release notes

## 0.4.1 — release preparation

- Curated MATLAB-style reference pages cover metadata, workflows, stages,
  classifiers, tuning, exports, persistence and compatibility. Generated API
  signatures retain their URLs and member anchors within the shared layout.
- Every configuration field has source and website documentation. The builder
  checks schema coverage, defaults, JSON aliases and local links; tests execute
  the published workflows. PRs build documentation previews; main deploys Pages.
- A repository ISA skill supports Python analysis and interpretation. Maintainer
  guidance describes current contracts; superseded plans, audit logs and stale
  notebooks are removed. Fixture provenance and reproducible numerical evidence
  remain available.
- The controlled Bayesian SVM comparison uses shared folds and initial points,
  matched budgets and cross-scored parameters. Results include classifier and
  objective differences; retain 20 evaluations, four starts and plain EI.
- Numerical defaults, approved fixtures, runtime function bodies, calling
  arguments and serialization schemas remain unchanged. Python 3.12 is required.
  Runtime dependencies are unchanged; markdown2 is an explicit documentation
  dependency at its existing locked version.

See [release preparation](docs/release-0.4.1.md) for validation, upgrade guidance
and the publication procedure. A PR, merge or tag alone does not upload to PyPI.

## 0.4.0 — upgrade compatibility

- Observed-only evaluation, Oracle/selector statistics and regret weighting
  apply to algorithm-selection results. Retrain older cost-sensitive models to
  obtain the current weighting behavior.
- Training prunes never-good algorithms and refits preprocessing from retained
  raw data. Rebuild models whose training removed features to obtain the fitted
  retained-feature replay record; older persisted models retain compatibility
  fallbacks.
- Persisted PLS centering, constant-coordinate scales and stage options support
  fitted inference. TRACE uses closed-boundary membership and an explicit saved
  distance tolerance, default zero. Native 3D geometry and exports include holes,
  disconnected regions and coplanar CLOISTER boundaries.
- Direct `PythiaEvaluateInput` callers supply `observed`. `PrelimInput` requires
  `algo_labels`; `PrelimOutput` includes the retained algorithm labels.
- The validated dependency series remain NumPy 1.26, scikit-learn 1.5, pandas 2.2
  and PyGAD 3.7. Check `pyproject.toml` and the lockfile for exact requirements;
  broader migrations are tracked in GitHub Issues.
