# Release Notes

Mirrors MATLAB `InstanceSpace`'s release-notes convention: entries are grouped into
*New functionality*, *Better engineering*, *Bug fixes*, and *Licence*. Every PR that
changes behaviour gets an entry here before merge.

## 0.2.1 (baseline)

Seeded as the starting point for this convention — describes the current state of the
toolkit rather than a single change.

### New functionality

- Stage architecture (`preprocessing → prelim → sifted → pilot → pythia → cloister →
  trace`) with an `InstanceSpace` class exposing `build()`, `explore()`, and
  `explore_stage_iter()`.
- `explore()`/`explore_stage_iter()` apply a previously trained model to unseen instances,
  mirroring MATLAB's `exploreIS.m`, by calling the `build()`-trained scikit-learn
  objects (e.g. `SVC.predict`/`predict_proba`) directly. (This originally went through
  `build_explore_adapter.py`, converting the trained model into the flattened form
  MATLAB-exported models use; that module has since been deleted — see the *Better
  engineering* entry below.)

### Better engineering

- Frozen dataclasses for options and metadata give real immutability MATLAB structs
  don't have.
- `stage_runner.py` executes a hardcoded, explicit 7-stage order (see the *Better
  engineering* entry below); `build_stage_runner()` attaches any extra/plugin stage to
  it via an explicit `RunBefore`/`RunAfter` declaration.
- `tests/matlab_reference/` provides historical cross-implementation regression data.
  Its generator provenance is unknown, so it is not treated as a MATLAB oracle; the
  verified replacement is recorded under *Unreleased* below.

### Bug fixes

(None catalogued retroactively — the list starts from here forward.)

### Licence

PolyForm Noncommercial 1.0.0, matching the MATLAB `InstanceSpace` toolkit.

---

## Unreleased

### New functionality

- `explore()` now logs a warning when more than 5% of test instances have a feature
  outside the training PRELIM bounds and get clipped to them, matching MATLAB's
  equivalent out-of-distribution check (#250).
- Added `InstanceSpace.plot_sources()`/`plot_portfolio()`/`plot_good()`/
  `plot_footprint()` convenience methods (`instancespace/plotting.py`), thin
  matplotlib wrappers mirroring MATLAB's `InstanceSpace.plot()` (#255).
- `PrelimStage` now warns when more than 5% of instances have a best-algorithm
  performance of exactly zero, matching MATLAB's `ISA:PRELIM:manyZeroBest` diagnostic —
  the relative-performance matrix becomes uninformative (close to 1 everywhere) for
  those instances once the existing `eps`-substitution kicks in (F14, #291).
- Added `PilotOptions.adjust_rotation` (default `False`), ported from PyISpace's
  `adjust_rotation()` (`gitlab.com/ita-ml/pyispace`): rotates PILOT's trained 2D
  projection so instances poorly solved by every algorithm face a consistent direction
  (135°, upper-left), making similar datasets easier to visually compare across runs.
  Rotation is a rigid transform — pairwise distances, error, R², and footprint areas are
  unchanged either way (R1).
- PILOT now supports `dims=2` and `dims=3` for the standard analytic/numerical solvers
  and MATLAB R2026a-compatible SIMPLS (`method='pls'`). Three-dimensional builds also
  optimize and persist a global or grouped 2D camera viewpoint. `cost_weight`, `x0`,
  `precalc_alpha`, restart selection, and public explore projection follow the MATLAB
  option and dispatch contracts (#262).
- **Behavior-changing:** the default numerical PILOT restart count changed from the
  former Python value of 5 to MATLAB R2026a's `opts.pilot.ntries = 10`. This doubles the
  default multi-start budget and can select a different fitted projection. Callers that
  need the earlier Python behavior can set `PilotOptions.default(n_tries=5)` explicitly.
  The new default is pinned against `ISAdefaults.m`, the verified R2026a fixtures, and
  the release validation suite (#262).
- **Behavior-changing:** MATLAB InstanceSpace v0.9.1 added stage-local
  `sifted.seed` and `pilot.seed` options. Python now exposes matching fields, inherits
  `GeneralOptions.seed` when they are omitted, and uses the selected PILOT seed for
  numerical and viewpoint restarts.
- `TraceOptions.method='trace3'` now implements MATLAB's current alpha-shape TRACE3
  algorithm with 2D polygons and native 3D tetrahedral meshes. Python keeps
  `method='legacy'` as its compatibility default; a 3D projection configured with that
  default warns and dispatches to TRACE3 because legacy TRACE is 2D-only.
  `TraceOptions.contra` continues to gate legacy contradiction removal (#313).
- CSV and plotting output now support 3D projections. Mesh output uses the versioned
  `pyinstancespace.trace-mesh/v1` manifest with one-based vertices, tetrahedra, and
  outward boundary faces; plotting uses native matplotlib 3D axes, mesh surfaces, and
  stored global/per-algorithm viewpoints (#265).
- **Behavior-changing:** `PythiaOptions.tuning` (`'sobol'`/`'bayes'`/`'none'`, default
  `'sobol'`) selects PYTHIA's SVM hyperparameter search strategy, matching MATLAB's own
  default. `'sobol'` evaluates `PythiaOptions.n_tuning_iter` (default 20) scrambled Sobol
  quasi-random candidates via cross-validation and keeps the best — a direct, lighter-weight
  port of MATLAB's `sobolSearch`, replacing the previous Bayes-only search as PYTHIA's
  default behaviour. Set `tuning='bayes'` to keep the exact prior behaviour. `'none'` skips
  tuning and requires `PythiaOptions.params` (F10).
- **Breaking:** `PythiaOptions.use_grid_search` has been removed — Sobol tuning (above)
  supersedes the `RandomizedSearchCV` "grid search" it used to select, matching MATLAB
  (which has no grid-search tuning mode at all). Callers setting this field, or JSON option
  files with a `usegridsearch`/`use_grid_search` key, need to switch to `tuning`. The
  legacy `uselibsvm` JSON key (previously silently aliased onto `use_grid_search`) is now
  genuinely ignored, matching its actual deprecated status (F10).

### Bug fixes

- PRELIM now reproduces MATLAB's seeded random selection among exactly tied best
  algorithms, including zero-valued ties, instead of always selecting the first tied
  algorithm. The refreshed v0.9.1 oracle checks the choices exactly.
- JSON option validation and loading now share one Unicode `casefold()` key
  canonicalizer, so accepted spellings load consistently and equivalent duplicates are
  rejected (#321). The unused legacy polygon-region filter reported in #320 was removed;
  active TRACE3 simplex filtering is unchanged.
- PYTHIA now derives cross-validation, classifier, and search randomness from MATLAB's
  per-algorithm `seed + i` boundary. KNN retains the requested 1--25 parameter for
  reporting while capping its effective neighbour count independently for each fold or
  final fit. Bayesian tuning remains skopt EI, the closest available base analogue to
  MATLAB's EI-plus; no convergence or default change was inferred from legacy data
  (#304).
- Graph labels, exported CSV headers, and filenames no longer leak the metadata.csv
  `feature_`/`algo_` column-naming prefix (e.g. `algo_CART` is now `CART`), and the
  `z_1`/`z_2` axis labels on generated plots now actually render as subscripts instead
  of literal text `z_{1}`/`z_{2}` (#222).
- `PythiaStage` no longer mutates the caller's `y_raw` array in place while generating
  its summary table; it now copies before mutating, matching the pattern already used
  for the same array's other derived copies (#229).
- `SiftedStage` no longer returns a stale, un-narrowed `idx` (computed once from the
  pre-selection feature count) alongside the correctly-narrowed `selvars`. Previously
  this caused `Model.save_to_csv`/`save_instance_space_for_web` to crash or misalign
  columns whenever SIFTED reduced the feature set; the redundant `idx` field has since
  been removed from `SiftedOutput`/`SiftedOut` entirely (`FeatSel.idx` is unaffected and
  now reads `selvars` directly).
- `build_explore_adapter._svc_to_artifact` no longer raises `NotImplementedError` for
  polynomial-kernel PYTHIA SVMs — `explore()` after a `build()` trained with
  `opts.pythia.is_poly_krnl = True` now works. The support vectors are pre-scaled by the
  trained `gamma` so `_explore_pythia`'s existing polynomial-kernel formula (which
  assumes `gamma=1`, matching its hardcoded `coef0=1`) reproduces the actual trained
  decision function. (`build_explore_adapter.py` itself has since been deleted — see the
  *Better engineering* entry below; `explore()` now calls `SVC.predict_proba` directly
  for every kernel type, so this specific formula no longer exists either.)
- `StageRunner.run_stage()` no longer crashes with `TypeError: cannot pickle
  '_queue.SimpleQueue' object` when a stage's inputs carry a live `ThreadPoolExecutor`
  (introduced by the Q6 pool-reuse change below, caught by the new end-to-end build test
  before ever shipping) — its unconditional `deepcopy(inputs)` now exempts live
  `ThreadPoolExecutor` references, passing them through by reference instead of
  attempting to copy them, which both isn't deepcopy-safe and would have silently
  defeated Q6's pool-reuse purpose even if it somehow succeeded.
- TRACE3 build membership and `explore()` rescoring now use MATLAB's inclusive boundary
  semantics in both dimensions. Three-dimensional points near a tetrahedron face use an
  exact orientation fallback, removing floating-point tolerance shells that could admit
  exterior instances and change purity or selection results (#313).

### Better engineering

- Removed the unused `_validate_explore_trace_dimensions()` compatibility hook. TRACE
  dimension validation remains stage-owned in `TraceStage.predict()` and runs exactly
  when a lazy `explore_stage_iter()` advances to TRACE; PYTHIA remains inspectable first.
  This cleanup changes neither outputs nor stage order (#316).
- Dependency security bumps: `pillow` (12.2.0 → 12.3.0), `tornado` (6.5.5 → 6.5.7),
  `click` (8.1.7 → 8.4.2), `jupyter-core` (5.7.2 → 5.9.1) — resolves all known CVEs in
  the locked dependency set at the time of writing.
- Added `.github/dependabot.yml` (weekly, pip ecosystem) so future dependency drift is
  caught automatically.
- Added a non-blocking `pip-audit` CI step to `validation-tests.yml`.
- Added `CITATION.cff` and resolved the README's citation placeholder.
- Replaced inferred 3D parity claims with a verified 423-file `reference-export/v2`
  oracle generated from clean MATLAB R2026a Update 4 and the pinned MATLAB source.
  The manifest records resolved options, inputs, solver evidence, 2D/3D geometry,
  alpha spectra, topology, metrics, membership, rescoring, and file hashes; independent
  verification recomputes the scientific contracts before installation (#262, #265).
- README: Contact section now points at this repository's own issue tracker rather than
  the MATLAB repository's; PYTHIA options section rewritten to describe the actual
  scikit-learn-backed implementation instead of MATLAB's Statistics and Machine Learning
  Toolbox/LIBSVM; added *Repository layout* and *Working with the code* sections.
- `instance_space_from_files`' options listing now recurses into nested option
  dataclasses (`instancespace/utils/print_options.py`), printing one line per leaf
  field instead of one line per top-level group with a raw nested-dataclass repr (#252).
- Documented and added a regression test for the already-decided permissive
  feature-order behaviour in `explore()`: test metadata's feature columns are matched
  by name, not position (#253).
- Added `SECURITY.md` and `CONTRIBUTING.md` (#258).
- `explore()` now calls the `build()`-trained scikit-learn objects (`SVC.predict`/
  `predict_proba`) directly instead of converting them into a flattened, MATLAB-artifact
  shaped representation first. `instance_space.py`'s model-shape detection branch (which
  handled only that flattened shape as a second, never-actually-reachable case) has been
  removed along with it (S1, #282). `build_explore_adapter.py` and its test file, which
  existed solely to produce that flattened shape, have been deleted in full — nothing
  called them once S1 landed (S3, #284).
- Removed a dead, commented-out CI lint/format-check step from `validation-tests.yml`
  rather than leaving it neither enabled nor deleted (#259).
- Replaced `stage_builder.py`'s type-matching DAG auto-resolution (ambiguous-ordering
  detection, mutating-stage special-casing, iterative wave computation) with a hardcoded
  explicit order for the built-in 7 stages, verified to resolve to the identical schedule
  the auto-resolver produced before the change. Any extra/plugin stage now attaches via
  an explicit `RunBefore[X]`/`RunAfter[X]` declaration instead of relying on its
  input/output types being matched against the rest of the pipeline — `example_plugin.py`
  updated accordingly. The remaining attachment logic was subsequently folded directly
  into `stage_runner.py` as `build_stage_runner()`, and `stage_builder.py` deleted
  entirely, since it had shrunk to a single call site with no further reason to be a
  separate module (S2).
- Added a `classifier` option to `PythiaOptions` (default `'svm'`, matching prior
  behaviour exactly) and a registry (`instancespace/utils/get_classifier_fcn.py`)
  dispatching PYTHIA's training to one of six scikit-learn classifiers (`svm`, `knn`,
  `tree`, `nb`, `linear`, `ensemble`), mirroring MATLAB's `ISAgetClassifierFcn.m`
  structurally. Only `svm` is tuned via PYTHIA's existing `C`/`gamma` search; the other
  five are fit with scikit-learn's own default hyperparameters — registering a classifier
  here is not a claim of MATLAB-verified tuning parity for it (F1).
- `InstanceSpace` now reuses a single `ThreadPoolExecutor` across staged calls instead of
  creating and tearing one down on every `TraceStage` run, mirroring MATLAB's
  `ensurePool()`. The pool is created lazily, recreated only if the requested worker
  count changes, and released via a new explicit `InstanceSpace.close()` method (Q6).
- Added `Model.save()`/`Model.load()` (`instancespace/model.py`): a signed
  `joblib`-based persistence round-trip, matching MATLAB's model save/load. Signing via
  an HMAC-SHA256 `secret_key` is optional — omitted for local/desktop use (identical
  trust caveat to any other unsigned `pickle`/`joblib` file), required and verified
  *before* deserialising for the production/server path. A signed file can never be
  loaded unverified by omitting the key (the downgrade-attack case is refused, not
  silently allowed) (F7).
- Added `tests/test_build_integration.py`, the repo's first genuine end-to-end
  `.build()` test — real metadata/options through all 7 stages, asserting every
  stage's output actually lands on the resulting `Model` (T2). Also verified, against
  that same real build, that rerunning `CloisterStage` neither wrongly invalidates
  `PythiaStage`'s already-computed output nor blocks a subsequent `TraceStage` run — a
  negative result for a previously-flagged concern about `_rollback_to_schedule_index()`
  over-invalidating by schedule-wave position rather than real dependency, scoped to the
  current built-in 7-stage order (Q8).
- Renamed `InstanceSpace.explore_iter()` to `explore_stage_iter()` and gave it a proper
  `AnnotatedExploreOutput` (`stage: ExploreStage, output: Any`) return type in place of a
  bare `(str, Any)` tuple, so it mirrors `run_iter()`'s `AnnotatedStageOutput` shape —
  build's and explore's incremental-iteration APIs now use the same vocabulary. Naming
  only: no computation changed. (Real code-sharing between build- and explore-time stage
  execution is separately tracked as F8, not done here.)
- Added `pytest-cov`; CI and `poe test` now run with `--cov=instancespace
  --cov-report=term-missing`, gated by a `fail_under = 75` threshold in `pyproject.toml`'s
  `[tool.coverage.report]` — set a few points below a measured real baseline (79%, full
  suite) rather than guessed (T1).
