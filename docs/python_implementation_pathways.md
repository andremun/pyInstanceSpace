# pyInstanceSpace — Implementation Pathways

**Companion to:** `pyIS_docs_quality_roadmap.md` (v1.12)
**Purpose:** every task in that roadmap, expanded to implementation-ready detail — files to
touch, concrete steps, and every open decision flagged explicitly with a recommended default.
Organised in the same P/Q/F/R/T structure. Nothing here has been implemented; this is the plan
for each plan item.

**Production context (added alongside roadmap v1.12):** this codebase is going into production
behind a web server, Claude Code sessions will have write access, and work is delegated —
recommended defaults in this document were originally reasoned on general-purpose-library
grounds and need re-checking against "does this change output for an existing caller." One
correction already made: Q9's seed default, below.

---

## Phase P — documentation & quality

### P0 — Dependency security hygiene
**Files:** `pyproject.toml`, `poetry.lock`, new `.github/dependabot.yml`
**Pathway:**
1. `poetry add pillow@^12.3.0 click@^8.3.3` (let poetry resolve transitively for `tornado`/
   `jupyter-core` via `ipykernel`) — check `poetry.lock` actually lands on patched versions
   for all four; if `ipykernel`'s pin caps `tornado` below 6.5.7, that needs its own bump or an
   override.
2. Run full `pytest` suite after the bump — pure dependency change, should be a no-op for
   behaviour.
3. Add `.github/dependabot.yml`:
   ```yaml
   version: 2
   updates:
     - package-ecosystem: "pip"
       directory: "/"
       schedule:
         interval: "weekly"
   ```
4. Optional: add `pip-audit` (or `poetry run pip-audit`) as a CI step in
   `validation-tests.yml`, non-blocking at first (a warning, not a failure) until confidence
   builds that it doesn't flag dev-only/unreachable findings noisily.
**Decision needed:** none — this is mechanical.

### P1 — Baseline hygiene
**Files:** `README.md`, new `CITATION.cff`
**Pathway:**
1. Contact section: replace the `andremun/InstanceSpace` issues link with
   `https://github.com/aoxiangx/pyInstanceSpace/issues` (or `andremun/pyInstanceSpace`'s, if
   that's intended as the long-term canonical home — see decision below).
2. Citation placeholder: either supply a real citation (if one exists/is planned) or explicitly
   state "citation forthcoming" rather than the bare `TBD`.
3. `CITATION.cff`: copy MATLAB's schema (`title`, `version`, `date-released`, `doi`, `authors`,
   `license`, `repository-code`), point `repository-code` at the Python repo, reuse the existing
   Zenodo concept DOI (`10.5281/zenodo.15562567`) unless a version-specific DOI is preferred.
4. PYTHIA options section rewrite: replace the MATLAB/LIBSVM-toolbox language with a description
   of the actual scikit-learn-backed implementation and current option names
   (`cv_folds`, `is_poly_krnl`, `use_weights`, `use_grid_search`, `params`) — this section will
   need a second pass once F1 (classifier registry) lands, since the option surface changes.
**Decision needed:** is `aoxiangx/pyInstanceSpace` the long-term canonical repo, or does this
eventually merge back into `andremun/pyInstanceSpace`? Affects where Contact/`repository-code`
should point. Recommended default: point at whichever repo is currently authoritative for
issues today (`aoxiangx`'s, since that's where the active branch lives) and revisit if/when the
fork merges upstream.

### P2 — Notebook parity
**Files:** `liveDemoExploreIS.ipynb`, `docs/explore_validation.ipynb`
**Pathway:**
1. Cell-by-cell pass: for each of the 9 markdown cells, add one short paragraph answering "why
   does this stage matter" and "what should the output look like / how do you know it worked" —
   MATLAB's manual answers both; the Python notebook currently answers only "how do you call it."
2. Add a short markdown cell near the top distinguishing this notebook (user-facing walkthrough)
   from `docs/explore_validation.ipynb` (CI/validation artefact) — one paragraph, cross-linking
   both directions.
3. If Q7 (`plot()` methods) lands first, simplify the notebook's inline matplotlib cells to call
   it instead — worth sequencing P2 *after* Q7 for that reason.
**Decision needed:** none — purely additive prose/notebook edits.

### P3 — README structural parity
**Files:** `README.md`
**Pathway:** add three new `##` sections mirroring MATLAB's structure:
1. **Repository layout** — a short tree + one-line description per top-level file/folder
   (`instancespace/`, `tests/`, `integration_demo.py`, `example_plugin.py`, `CLIDocs.txt`).
2. **Working with the code** — a minimal `InstanceSpace` walkthrough (construct → `build()` →
   `save_to_csv()`/`save_graphs()`), cross-referencing `integration_demo.py` as the runnable
   version and `example_plugin.py` for extension points.
3. **The metadata file** — document the `Instances`/`feature_*`/`algo_*`/`source` column
   convention `from_csv_file` expects (currently only discoverable by reading
   `data/metadata.py`).
**Decision needed:** defer *AI-assisted analysis* section until a Claude Code skill exists for
this repo (already noted in the roadmap) — no new decision here.

### P4 — Release notes discipline
**Files:** new `RELEASE_NOTES.md`
**Pathway:**
1. Seed with one entry titled after the current version (`0.2.1` per `pyproject.toml`),
   structured like MATLAB's: *New functionality* (stage architecture, `build()`/`explore()`/
   `explore_iter()`), *Better engineering* (frozen dataclasses, DAG scheduler), *Bug fixes*
   (none yet catalogued — start the list from here forward), *Licence* (PolyForm Noncommercial
   1.0.0, already in place).
2. Process rule: every PR that changes behaviour gets a `RELEASE_NOTES.md` entry before merge —
   enforce lightly at first (a PR template checkbox, not a CI gate) since retrofitting is
   friction-free but a hard CI gate on prose content is brittle.
**Decision needed:** none.

### P5 — Docs CI honesty
**Files:** `.github/workflows/validation-tests.yml` (or new workflow), `README.md`
**Pathway — two options, pick one:**
- **(a)** Add a `pdoc instancespace -o docs/` step to CI, publish to GitHub Pages, badge points
  at the real Pages deployment status.
- **(b)** Replace the `docs-passing` badge with plain text ("API docs: run `pdoc instancespace`
  locally") until (a) is worth the setup cost.
**Decision needed:** (a) vs (b) — recommended default: (b) now, low-effort and honest
immediately; revisit (a) once the repo has more external users who'd benefit from hosted docs.

---

## Phase Q — MATLAB-derived quality ideas

### Q1 — Fix build→explore adapter's missing polynomial-kernel branch
**Files:** `instancespace/build_explore_adapter.py`
**Pathway:**
1. In `_svc_to_artifact()`, add an `elif svc.kernel == "poly":` branch. MATLAB's PYTHIA doesn't
   expose a `poly` *degree* option in the current registry, but scikit-learn's default degree
   is 3 — decide whether to read `svc.degree` and pass it through, or hardcode 3 to match
   MATLAB's implicit behaviour today (MATLAB's `ispolykrnl` doesn't parameterise degree either).
2. `kernel_param` for poly should carry whatever the artifact format's poly-kernel scoring
   function expects — check `_explore_pythia`'s existing poly-handling branch (it already has
   one: `elif kernel_fn == "polynomial": k = (z_norm @ svs.T + 1.0) ** order`) — so `kernel_fn`
   should be the string `"polynomial"` (not `"poly"`) to match what `_explore_pythia` already
   checks for, and `kernel_param` should carry the degree as `order`.
3. Aside worth noting while in this function: the existing `"linear"` branch appears to be dead
   code today — `stages/pythia.py`'s only training call is `kernel = "poly" if is_poly_kernel
   else "rbf"`, never `"linear"` — so `build()` can never actually produce a linear-kernel SVC
   for this branch to handle. Not a bug, just worth a comment noting it's defensive/future-proofing
   rather than reachable today.
4. Generate the missing MATLAB reference fixture (T-phase dependency, §7.3 of the roadmap)
   before writing the round-trip validation test — without it there's nothing to check the
   fix's numerical output against.
**Test:** build with `is_poly_krnl=True` → `.explore()` → assert no exception and predictions
are finite/sane; once the new reference fixture exists, validate against it with the same
tolerance convention as the other `_validation.py` tests.
**Decision needed:** none blocking — the degree-handling question above is a one-line choice,
recommended default: hardcode degree 3 to match MATLAB's current (also-hardcoded) behaviour.

### Q2 — Out-of-distribution warning in `explore()`
**Files:** `instancespace/instance_space.py` (`_explore_prelim`)
**Pathway:**
1. Before/alongside the existing `np.clip` calls, compute `frac_clipped = np.mean(np.any((x <
   prelim.lo_bound) | (x > prelim.hi_bound), axis=1))`.
2. If `frac_clipped > 0.05`, emit via `loguru`'s `logger.warning(...)` (not `print`, per Q3) with
   the same message shape as MATLAB's: percentage clipped + a suggestion to retrain with a
   combined dataset.
3. Threshold (5%) should become a named constant near the top of the module, not a bare literal
   — small thing, but avoids a second "magic number" finding down the line.
**Test:** two cases — a synthetic test set engineered to clip >5% of instances (assert warning
fires) and one that clips none (assert it doesn't).
**Decision needed:** should the 5% threshold be configurable via options, or fixed like MATLAB's?
Recommended default: fixed, matching MATLAB — it's paper-calibrated, not something users have
asked to tune.

### Q3 — Standardise console output + `general.verbose` option
**Files:** `instancespace/data/options.py` (new `GeneralOptions` or add to
`InstanceSpaceOptions` directly), all `instancespace/stages/*.py`, `instancespace/
instance_space.py`
**Pathway:**
1. Add a `general` options group (new `GeneralOptions` frozen dataclass: `verbose: bool = True`,
   and — see Q9 — `seed: int | None = 0`) mirroring MATLAB's `opts.general.*` namespace,
   rather than bolting `verbose` onto an existing unrelated dataclass.
2. Replace the 121 `print()` calls in `stages/*.py` with `logger.info(...)`/`logger.debug(...)`
   calls (loguru is already a dependency, barely used — 10 call sites today).
3. Adopt MATLAB's `[STAGE] message` prefix convention for anything user-facing at the default
   verbosity; gate per-trial/per-iteration detail (e.g. PYTHIA's per-classifier tuning-iteration
   prints) behind `if opts.general.verbose:` or a loguru level filter.
4. This is the single largest mechanical change in Phase Q by line count (121 call sites) —
   worth doing as its own PR, not bundled with anything else.
**Test:** a smoke test capturing log output at default vs. `verbose=False`, asserting per-trial
detail only appears in the former.
**Decision needed:** where does `verbose` (and `seed`) live structurally — a new top-level
`GeneralOptions` group (matches MATLAB's namespacing exactly) or fields directly on
`InstanceSpaceOptions`? Recommended default: new `GeneralOptions` group, for direct
option-name parity with MATLAB's `opts.general.*` and to avoid overloading an unrelated
dataclass.

### Q4 — Recursive, compact options printer
**Files:** `instancespace/instance_space.py` (`instance_space_from_files`), possibly a new
`instancespace/utils/print_options.py`
**Pathway:**
1. Write a small recursive function: for each field on a dataclass, if the value is itself a
   dataclass, recurse with an extended prefix (`"parallel.flag"`, `"parallel.n_cores"`); else
   print `f"  {prefix:<28} {value!r}"` — direct port of MATLAB's `printOptions`/
   `formatOptionValue` logic, adapted for Python's `dataclasses.fields()`.
2. Swap the current flat loop in `instance_space_from_files` for a call to this function.
**Test:** snapshot test asserting a known `InstanceSpaceOptions` instance prints one line per
leaf field, not one line per top-level group.
**Decision needed:** none.

### Q5 — Feature-order handling: confirmed permissive
**Decision made: keep auto-reorder-by-name.** Not a bug, not accidental — confirmed as the
intended, permanent behaviour.
**Files:** `instancespace/instance_space.py` (`_extract_features`, `explore()`'s docstring)
**Pathway:**
1. Add one sentence to `explore()`'s docstring stating this explicitly: test metadata's feature
   columns may be supplied in any order; they're matched by name, not position.
2. Add the regression test already scoped in Phase Q's checkpoint: construct a test metadata
   frame with columns in a different order than training, assert `explore()` still produces
   correct results (matching what it would with the original order).
**No further decision needed — this item is now a documentation + test task, not a design
question.**

### Q6 — Reuse thread/process pools across staged calls
**Files:** `instancespace/instance_space.py` or `stage_runner.py` (wherever a pool-holder makes
most sense), `stages/trace.py`, `stages/pythia.py`, `stages/sifted.py`
**Pathway:**
1. Add a pool/executor cache on `InstanceSpace` (or `StageRunner`) — e.g. `self._executor:
   ThreadPoolExecutor | None = None` — created lazily on first use, matching MATLAB's
   `ensurePool()`'s "rightSize" check (recreate only if worker count changed).
2. Thread it through to TRACE's `ThreadPoolExecutor` usage and wherever `n_jobs` is currently
   passed directly to scikit-learn calls (those already reuse joblib's own backend pool by
   default when called repeatedly in-process, so the main win is TRACE's explicit
   `ThreadPoolExecutor`).
3. Close/dispose on `InstanceSpace.__del__` or an explicit `close()` method — decide which.
**Test:** assert no new `ThreadPoolExecutor` is constructed on a second `run_stage(TraceStage)`
call with the same `n_cores`; assert one *is* constructed if `n_cores` changes between calls.
**Decision needed:** explicit `close()` method the user must call, or rely on `__del__`
(implicit, less reliable in Python but less API surface)? Recommended default: explicit
`close()`, since relying on `__del__` for resource cleanup is a known Python anti-pattern
(non-deterministic timing, exceptions during interpreter shutdown are swallowed).

### Q7 — Add `plot()` convenience methods
**Files:** new `instancespace/plotting.py` (or methods directly on `InstanceSpace`)
**Pathway:**
1. Four thin wrappers mirroring MATLAB's four views: `sources` (needs `data.s`, currently only
   populated if a `source` column exists in metadata — check `Data`'s field for this), `portfolio`
   (best-algorithm-per-instance scatter), `good`/`footprint` (per-algorithm, need an `algo_idx`
   or algorithm-name parameter).
2. Each wrapper: pull the relevant arrays off `self.model`, call the matching matplotlib
   primitive (`scatter`, polygon patches for footprints via `shapely`→`matplotlib.patches`), draw
   to the *current* axes (matching MATLAB's `plot()` drawing to the current figure) rather than
   creating a new figure each call, so it composes with notebook cells the way MATLAB's does.
**Decision needed:** API shape — one dispatch method `plot(view_name: str, algo: str | None =
None)` mirroring MATLAB's single-method-plus-string-argument design exactly, or four separate
idiomatic methods (`plot_sources()`, `plot_portfolio()`, `plot_good(algo)`,
`plot_footprint(algo)`)? Recommended default: **four separate methods** — more idiomatic Python
(better autocomplete/type-checking, no string-typo failure mode), at the small cost of not
mirroring MATLAB's exact call signature. Asked as a direct question below since it affects every
call site P2/notebook work would use.

### Q8 — Regression test for stage-rerun invalidation
**Files:** new test in `tests/` (exact location depends on T7's consolidation decision), no
production code changes yet — this is verification-only per the roadmap.
**Pathway:**
1. Requires T2 (real end-to-end `build()` test) to exist first, or at minimum a fixture that
   constructs a real `InstanceSpace` with all 7 stages — the current synthetic 2-stage
   `test_stage_builder_runner.py` setup can't exercise this at all.
2. Build fully, capture `pythia`'s output object identity/values, call
   `space.run_stage(CloisterStage)` again, then check whether `pythia`'s output changed identity
   or got marked for re-run despite not depending on `cloister`.
3. If it reveals over-invalidation: the fix belongs in `stage_runner.py`'s
   `_rollback_to_schedule_index`, replacing schedule-index comparison with a real dependency-graph
   walk (mirroring MATLAB's BFS in `invalidateDownstream`) — that fix is F-phase work per the
   roadmap, not part of Q8 itself.
**Decision needed:** none for Q8 itself (it's a test, not a design choice) — but note the
dependency on T2 existing first, which affects sequencing.

### Q9 — Centralise RNG seeding via a `general.seed` option
**[Behavior-changing if defaulted wrong — corrected]** Every current call is implicitly
deterministic (hardcoded `0` everywhere). Original recommendation below (`None`) was reasoned on
library-hygiene grounds without accounting for production callers — corrected to **default `0`**,
exactly matching today's behaviour, now that this is confirmed to be shipping behind a
production web server.
**Files:** `instancespace/data/options.py` (new `GeneralOptions.seed`, see Q3), `stages/pilot.py`,
`stages/sifted.py` (×3 sites), `stages/prelim.py`, `stages/pythia.py` (×4 sites)
**Pathway:**
1. Add `seed: int | None = 0` to the new `GeneralOptions` group (see Q3 — do these two together,
   they touch the same new dataclass). `None` remains a valid explicit opt-in for
   non-deterministic runs; it just isn't the default.
2. Thread `opts.general.seed` down to each stage's `_inputs()` NamedTuple (all stages already
   receive an options object of some kind; add `general_options: GeneralOptions` alongside the
   existing per-stage options where randomness is used: pilot, sifted, prelim, pythia).
3. Replace each hardcoded `np.random.default_rng(seed=0)` with `np.random.default_rng(seed=
   general_options.seed)`, and each `random_state=0` passed to scikit-learn with
   `random_state=general_options.seed`.
4. Follow MATLAB's per-fold/per-trial reseeding discipline where randomness is evaluated
   repeatedly in a loop being compared against itself (PYTHIA's tuning loop, SIFTED's GA fitness
   evaluations) — i.e. derive a per-iteration seed like MATLAB's `foldSeed = baseSeed*1e5 +
   fold*1e3` rather than reusing one global `rng` object across all iterations, so that
   "common random numbers" holds for fair candidate comparison the same way MATLAB's PYTHIA.m
   comments explain.
**Test:** run PILOT/SIFTED/PYTHIA twice with the same seed → assert identical output; twice with
different seeds → assert different output (at least one of the two, to catch a seed silently not
being threaded through somewhere). Add one more, given the production context: build with no
`general.seed` specified at all → assert output is bit-identical to today's (pre-this-change)
output, proving the new option is truly a no-op by default.
**Decision needed:** ~~default value — match MATLAB's `42`, or use `None`~~ — **resolved: `0`**,
for the production/backward-compatibility reason stated above. `42` was never seriously in the
running (no reason to inherit MATLAB's specific number); the real choice was `0` vs. `None`, and
production callers settle it in favour of `0`.

### Q10 — Add `SECURITY.md` and `CONTRIBUTING.md`
**Files:** new `SECURITY.md`, new `CONTRIBUTING.md`
**Pathway:**
1. `SECURITY.md`: one short paragraph — how to report a vulnerability (GitHub private
   vulnerability reporting, or an email address), no formal SLA needed for research software.
2. `CONTRIBUTING.md`: point at `README.md`'s existing "Development Environment Setup Guide"
   rather than duplicating it; add "run `poetry run pytest` and `poe test` before opening a PR"
   (once T4 fixes `poe test` to actually include pytest) and a one-line code-style note (ruff +
   mypy strict + black, all already configured).
**Decision needed:** none — content is templated/standard.

---

## Phase F — functionality parity (long-term)

### F1 — PYTHIA classifier registry
**Verified starting point:** `PythiaOptions` has no `classifier` field at all; `stages/
pythia.py` has exactly one `SVC(...)` call, switched only between `poly`/`rbf` kernel via
`is_poly_krnl`. This is a from-scratch build, not an extension — MATLAB's registry (`knn`, `svm`,
`tree`, `nb`, `linear`, `ensemble`, resolved via `ISAgetClassifierFcn`) has no Python counterpart
yet at all.
**Files:** `instancespace/data/options.py` (`PythiaOptions` — add `classifier: str`), `stages/
pythia.py` (the bulk of the work), possibly a new `instancespace/utils/get_classifier_fcn.py`
mirroring `ISAgetClassifierFcn.m`'s table.
**Pathway:**
1. Build the registry table, scikit-learn side:

   | `classifier` | scikit-learn class | Notes |
   |---|---|---|
   | `'knn'` | `KNeighborsClassifier` | MATLAB's default |
   | `'svm'` | `SVC` | Python's current (only) behaviour |
   | `'tree'` | `DecisionTreeClassifier` | |
   | `'nb'` | `GaussianNB` | |
   | `'linear'` | `LogisticRegression` | MATLAB uses `fitclinear`; closest scikit-learn analogue |
   | `'ensemble'` | one of `RandomForestClassifier`/`AdaBoostClassifier`/`GradientBoostingClassifier` | MATLAB's `ensembleMethod` sub-option picks among equivalents — needs its own small sub-registry |

2. Write `get_classifier_fcn(name)` returning `(estimator_class, hyperparameter_ranges)` —
   direct structural port of `ISAgetClassifierFcn.m`'s table.
3. Refactor `stages/pythia.py`'s training call to dispatch through this registry instead of the
   hardcoded `SVC(...)`. The existing tuning machinery (Sobol/Bayes search over `p1`/`p2`) is
   already generic over two numeric hyperparameters per the code I read earlier — check it
   isn't accidentally SVM-kernel-specific anywhere before assuming it "just works" for other
   classifiers.
4. **This interacts directly with the build→explore adapter (Q1/`build_explore_adapter.py`)** —
   that adapter currently only knows how to flatten an `SVC` into the artifact format. Adding
   `knn`/`tree`/`nb`/`linear`/`ensemble` as trainable classifiers means `explore()` needs an
   equivalent artifact-flattening (or a different mechanism entirely) for each new classifier
   type, or `explore()` needs to learn to consume whichever classifier was actually trained
   directly rather than only ever going through the flattened-artifact path. This is a real
   scope dependency between F1 and F8 (unifying build/explore) worth resolving together rather
   than separately.
**Decision needed:** what should the *default* `classifier` value be once this exists — `'svm'`
(preserves today's Python behaviour exactly) or `'knn'` (matches MATLAB's default, for
cross-implementation consistency)? Recommended default: **`'svm'`**, to avoid silently changing
existing users' default output the moment this option is introduced — `'knn'` remains one
config change away.

### F2 — PILOT 3D / viewpoint / PLS alternative
**Verified starting point:** no `dims`, `viewGroups`, or `method` handling found in `stages/
pilot.py` at all — Python's PILOT is 2D-only, single-method, with none of MATLAB's Phase 5
surface.
**Files:** `data/options.py` (`PilotOptions` — add `dims: int`, `view_groups: list[list[int]] |
None`, `method: str`), `stages/pilot.py` (extend the analytic/numeric solvers to `n×3` where
`dims=3`), new `stages/pilot_viewpoint.py` (direct port of `PILOTviewpoint.m`).
**Pathway:**
1. Extend the existing 2D projection math to general `dims` (2 or 3) — check whether the
   analytic eigen-solution path and the BFGS numeric path both generalise cleanly to 3 output
   dimensions or whether either has a 2D-specific assumption baked in (e.g. array shapes assuming
   exactly 2 columns).
2. Port `PILOTviewpoint`: solve `min_V ||Y - VZ'||_F^2 + λ⟨v1,v2⟩` s.t. `V ∈ R^{3×2}`, `V'V ≈ I`,
   with `λ = 0.2` fixed (paper-calibrated, not user-exposed — keep it that way rather than adding
   a knob nobody's asked for). Returns azimuth/elevation from `v1 × v2`.
3. `view_groups` format: list of lists of 0-based algorithm indices (adjust for Python's 0-based
   vs. MATLAB's 1-based convention — this is an easy off-by-one to get wrong when porting the
   worked example `{[1 2 3], [4 5 6]}` directly).
4. PLS alternative (`method='pls'`): `sklearn.cross_decomposition.PLSRegression` gives the
   weight/loading matrices analogous to MATLAB's `plsregress` output — map its `x_weights_`/
   `x_loadings_`/`y_loadings_` onto the `A`/`B`/`C` triple the rest of the pipeline expects.
**Decision needed:** should R1 (rotation canonicalisation, already scoped separately) be applied
before or after `dims=3` lands? Recommended default: **after** — R1's centroid-angle math as
written assumes 2D; generalising it to 3D orientation (which needs an axis choice, not just an
angle) is a bigger question than R1's original scope intended, so land 2D rotation first, revisit
3D rotation as its own follow-on once F2 exists.

### F3 — SIFTED refinements
**Verified starting point:** MATLAB's "SIFTED promotion" specifically meant promoting an
existing `SIFTED2.m` to canonical status plus four fixes: a thread-unsafe global `containers.Map`
→ persistent variable; a nested-`parfor`-inside-GA bug; an `rng('default')` reset inside the
per-candidate cost function (silently discarding any user seed); and vectorising the
correlation-selection loop. Python has no `sifted.py`/`sifted2.py` duality, so this isn't a
literal port — it's a checklist to audit Python's `sifted.py` against, since analogous smells are
possible even in different form. Spot-checked one thing already: the module-level `rng =
np.random.default_rng(seed=0)` I found is created once per stage call, not reset inside a
per-candidate loop — so the MATLAB bug's exact shape doesn't appear to be present, but this was
a single spot-check, not a full audit.
**Files:** `stages/sifted.py`
**Pathway:**
1. Audit for MATLAB's four issues' Python-shape equivalents:
   - Shared mutable cache: does anything cache GA fitness evaluations in a module-level or
     class-level mutable structure (Python's GIL makes this less dangerous than MATLAB's
     `parfor`, but a joblib/multiprocessing backend reintroduces the same class of hazard)?
   - Nested parallelism: does `pygad`'s own parallelism option (if enabled) ever wrap a call that
     also sets `n_jobs`/`ProcessPoolExecutor` internally?
   - RNG reset inside a hot loop: confirmed not present at the one call site checked; check the
     GA fitness function itself and any k-means/PCA calls inside SIFTED's per-candidate
     evaluation path.
   - Vectorisation: check whether the correlation-selection step is already vectorised (NumPy
     code is more naturally vectorised by default than MATLAB loops, so this specific item may
     already be moot in Python) or still loop-based.
2. Fix whatever the audit actually finds — this step can't be scoped further until the audit
   runs.
**Decision needed:** none yet — this *is* the audit-first item the roadmap already flagged F3 as
needing.

### F4 — `InstanceSpace` class & build/explore robustness
Already has its own detailed audit (roadmap §5.1, 7 findings) — see F7/F8/F9 below, which are
the concrete work items this audit produced. No further pathway needed here; F4 itself is now
"audited," not "not started."

### F5 — Output consolidation / 3D visualisation parity
**Files:** `instancespace/scripting/script_fcn.py`, `instancespace/scripting/script_disc.py`
**Pathway:**
1. Audit-first (like F1/F2): check whether `script_fcn.py` already branches on projection
   dimensionality anywhere, or is 2D-only throughout — not yet checked in this pass.
2. Once F2 (3D PILOT) exists, this becomes the consumer: 3D scatter/footprint rendering
   (matplotlib's `mplot3d`, or a 2D camera-angle-projected render using F2's viewpoint output —
   decide which matches the "spirit" of MATLAB's `scriptpng` 3D handling more usefully for a
   Python audience).
3. This is the natural home for R1's rotation output too, once both exist — the 3D/2D rendering
   layer is where a canonicalised orientation actually gets *used* to make plots comparable.
**Decision needed:** genuinely blocked on F2 landing first — no further detail useful until
then.

### F6 — Namespace & per-file licence headers
**Files:** every `instancespace/**/*.py`
**Pathway:** mechanical — add a short header comment (SPDX identifier + copyright line) matching
MATLAB's `SPDX-License-Identifier: LicenseRef-PolyForm-Noncommercial-1.0.0` convention, adapted
to Python comment syntax. A one-off script (or a pre-commit hook, if Q10-adjacent tooling grows)
can insert this across all files in one pass.
**Decision needed:** exact copyright line/year range and author list — same question as
`CITATION.cff`'s `authors` field in P1; worth deciding both together rather than twice.

### F7 — Model save/load round-trip
**Decision made: HDF5 via `h5py`.**
**Files:** `instancespace/model.py` (new `Model.save()`/`Model.load()`), `pyproject.toml` (new
`h5py` dependency), `instancespace/data/model.py` (for schema versioning)
**Pathway:**
1. `poetry add h5py`. Since this is a new dependency, it goes through the same P0-style
   scrutiny as anything else in `pyproject.toml` — check its own transitive dependency tree
   before merging, not just after (this repo's own P0 audit found problems that had sat
   unnoticed in existing transitive dependencies; no reason a new one gets a pass).
2. `save()`: open an `h5py.File` in write mode; one HDF5 group per top-level `Model` field
   (`data`, `prelim`, `sifted`, `pilot`, `cloister`, `pythia`, `trace`, `opts`); numpy arrays
   become HDF5 datasets directly (h5py's native strength — no manual flattening needed, unlike
   the JSON+`.npz` option); scalars/strings become group attributes; nested dataclasses become
   nested HDF5 groups. Record a `schema_version` attribute at the file root.
3. `load()`: reverse the process, reconstructing each frozen dataclass from its group's
   attributes/datasets; raise a clear error if `schema_version` doesn't match a version this
   `load()` knows how to read (don't attempt a silent best-effort partial load).
4. Non-trivial parts worth flagging up front, not discovered mid-implementation:
   - `pythia.svm` is a list of per-algorithm SVM objects (either fitted scikit-learn `SVC`s from
     `build()`, or the flattened MATLAB-artifact `SimpleNamespace` form from `explore()`'s
     adapter — see F8) — these aren't numpy arrays or simple scalars, so they need an explicit
     serialisation shape decision: store each `SVC`'s constituent arrays (`support_vectors_`,
     `dual_coef_`, `intercept_`, `probA_`/`probB_`) directly as HDF5 datasets per algorithm
     (recommended — keeps the file free of anything resembling object serialisation), rather
     than trying to round-trip the `SVC` object itself.
   - `trace.good`/`trace.best` are `shapely` `Polygon`/`MultiPolygon` objects — serialise as
     vertex arrays (the same vertex-list-with-NaN-separator convention already used for CSV
     export, per `_serialisers.py`) rather than trying to store shapely objects directly.
   - Neither of these needs anything unsafe — both are "flatten a Python object into arrays we
     already know how to write," not "serialise arbitrary Python state," so the F7 design
     constraint (no unsafe deserialisation) holds throughout.
5. Round-trip test: save then load, assert deep equality — `np.array_equal` for arrays,
   value-equality for the reconstructed SVM/polygon objects (compare their constituent arrays,
   not object identity).
6. Adversarial test: truncate/corrupt a saved `.h5` file, assert `load()` raises a clear `h5py`-
   or schema-level error rather than partially succeeding or executing anything.
**Decision needed:** none remaining — format is chosen; the SVM/polygon flattening shape above
is a design detail worth a second look during implementation, not a blocking decision now.

### F8 — Unify `explore()` with build-time stage code
**Files:** `instancespace/stages/stage.py` (possibly extending the `Stage` contract),
`instancespace/instance_space.py` (`_explore_pythia`/`_explore_trace`/etc. would be replaced or
rewritten to call into the stage classes)
**Pathway — two ambition levels, pick one:**
- **Lighter:** extract the *numerical core* of each stage's train-time logic into a shared
  pure function that both the stage's `_run()` and `explore()`'s corresponding `_explore_*`
  method call — no change to the `Stage`/`StageRunner` architecture itself, just de-duplicating
  the math underneath it. Lower risk, doesn't touch the DAG scheduler.
- **Fuller:** extend the `Stage[IN, OUT]` contract with a second entry point (e.g. `_predict()`
  alongside `_run()`) so `PythiaStage`/`TraceStage` themselves know how to run in inference mode,
  and `explore()` dispatches to `_predict()` directly instead of maintaining separate
  `_explore_*` methods at all. Higher risk (changes a core abstraction every stage implements),
  but closes the drift risk more completely — a bug fixed in `_run()` is structurally guaranteed
  to also be fixed in `_predict()` if they share the surrounding class, not just a shared helper
  function.
**Test (either way):** the drift-detection test already scoped in the roadmap's Phase T —
deliberately break something in `PythiaStage`'s training logic, assert both the build-path test
*and* the explore-path test fail, proving they can no longer silently diverge.
**Decision needed:** lighter (shared function) or fuller (extended `Stage` contract)?
Recommended default: **lighter**, as a first step — it captures most of the "no more silent
drift" benefit for much less architectural risk, and the fuller redesign remains available later
if the lighter version proves insufficient in practice.

### F9 — Expand `explore()` to full evaluation scope
**Decision made: Option 1 — extend `explore()` itself** (not a new method; silent branching
based on whether ground truth is present in the input).
**Files:** `instancespace/instance_space.py` (`explore()`, `explore_iter()`), `instancespace/
data/model.py` (`ExploreResult` — new optional fields), `instancespace/stages/prelim.py`
(extract shared binary-performance logic)
**Pathway:**
1. **Detecting ground truth is free — already checked.** `Metadata.from_csv_file` parses
   `algo_*` columns unconditionally whenever present and doesn't require them (confirmed:
   no validation forces `algo_` columns to exist). So `test_metadata.algorithm_names`/
   `.algorithms` are already populated whenever `metadata_test.csv` has them — `explore()`
   just needs to start looking at them. Branch condition: `len(test_metadata.algorithm_names)
   > 0`.
2. **Extract the shared logic first, to serve F8's goal at the same time.** `PrelimStage`'s
   private `_prelim()` method already contains the exact binary-performance computation this
   needs (`y_bin`/`y_best`/`p`/`beta` from raw `Y` + `PerformanceOptions` — the `max_perf`/
   `abs_perf`/`epsilon` branching, ~lines 619–671 of `prelim.py`). Pull this into a standalone
   function (e.g. `compute_binary_performance(y_raw, perf_options) -> BinaryPerformance`, a
   small NamedTuple). `PrelimStage._prelim()` calls it for training; `explore()`'s new
   evaluation path calls the *same* function for the test set. One implementation, not a second
   one written to match it by hand — directly serves F8's "no drift between build and explore"
   goal, not just F9's.
3. **Algorithm reconciliation:** match `test_metadata.algorithm_names` against the trained
   model's algorithm names, case-insensitively (mirrors MATLAB's `strcmpi`). Algorithms in
   training but absent from the test set: simply not evaluated. Algorithms in the test set
   absent from training ("new" algorithms, per MATLAB's `autoNormalize` handling): see decision
   below.
4. **Extend `ExploreResult`** with new fields, all `| None`, populated only when ground truth is
   present: `y_actual: NDArray[np.bool_] | None`, `y_best_actual: NDArray[np.double] | None`,
   `p_actual: NDArray[np.int_] | None`, `beta_actual: NDArray[np.bool_] | None`. `None` in the
   feature-only case preserves today's behaviour exactly — existing callers see no change.
5. **Make the silent branch visible, even though it's automatic.** Since this is Option 1 (not
   an explicit separate call), log an info message when ground truth is detected and evaluation
   fields get populated (ties to Q3's logging work) — mirrors MATLAB's own "[EXPLORE]
   Calculating the binary measure of performance" console line, so the mode switch is
   observable, not a silent surprise, even though it's inferred from input shape rather than an
   explicit flag.
6. **`explore_iter()` needs the same treatment:** yield a 6th `("evaluation", ...)` item after
   `"trace"` when ground truth is present; omit it entirely (yield only the original 5) when it
   isn't. `explore_iter()`'s yielded-stage count becomes conditional on input shape, matching
   `explore()`'s own conditional field population.
**Decision needed (smaller, now that the main structural choice is made):** implement the
"new algorithm absent from training" edge case now, or defer it as a documented limitation?
Recommended default: **defer** — evaluating against the *same* algorithm portfolio used in
training is the common case and worth shipping first; the unseen-algorithm path is a real but
rarer need that can follow once someone actually hits it.
**Test:** `tests/matlab_reference/`'s existing fixtures already carry ground truth for the
235-instance test set (per that fixture set's own README) — a ready-made validation case.
Worth confirming during implementation whether `explore_outputs/step4_pythia_predictions.csv`
is PYTHIA's *predictions* (already used to validate `y_hat` today) or could double as the
ground-truth comparison target — likely a new reference export is needed either way, since
predictions and ground truth are conceptually different things even if this fixture set's
current files don't cleanly distinguish them.

---

## Phase R — ideas from PyISpace/PyHard

### R1 — Canonicalise PILOT's 2D projection orientation
**Files:** `stages/pilot.py`
**Pathway:**
1. Direct port of PyISpace's `adjust_rotation()`: compute the centroid of a reference group in
   `Z`-space (need to decide which group — PyISpace used "bad-performing instances"; check
   whether Python's `Data`/`PrelimOut` already has an equivalent "is this instance bad for this
   algorithm" boolean array to reuse, likely `Ybin` inverted, or `beta`).
2. Rotate: `theta = radians(135) - arctan2(*centroid[::-1])`, build the 2×2 rotation matrix,
   apply to `Z`, store the rotation matrix on `PilotOut` for anyone who needs to invert it.
3. Make this opt-in via a new `PilotOptions` field (e.g. `adjust_rotation: bool = False`) rather
   than always-on — changes existing output for anyone relying on today's (arbitrary)
   orientation, so shouldn't be a silent default change.
**Test:** pairwise-distance invariance (already scoped in the roadmap) *and* the centroid-angle
assertion (also already scoped) — both needed, not either/or, per the Phase T cross-reference.
**Decision needed:** which group's centroid is the rotation target — confirm "bad/not-good
across all algorithms" matches PyISpace's actual definition precisely (worth re-checking their
`Ybad` construction directly if this gets implemented, rather than assuming from the function
signature alone).

### R2 — Alpha-shape auto-retry for TRACE
**Files:** `stages/trace.py`
**Pathway:**
1. Wherever `alphashape.alphashape(points, alpha)` (or equivalent) is called, check the result
   type — if it's a `MultiPolygon` where a single `Polygon` was expected (the multi-region
   failure mode), retry with `alphashape.optimizealpha(points)` before falling back to exporting
   a partial/flagged result.
2. Needs a constructed test case (per Phase T's R2 entry) — a synthetic point cloud engineered
   to produce a `MultiPolygon` at a naive alpha, to actually exercise this path in tests rather
   than relying on incidentally hitting it with real data.
**Decision needed:** none — the pattern is concrete enough to implement directly once R2 is
prioritised.

### R3 — Small CLI ideas
Not scoped further — the roadmap already marks this "lower priority... not urgent, no
corresponding issue drafted yet." No pathway needed until the Python port's CLI work is actually
picked back up.

---

## Phase T — testing infrastructure

### T1 — `pytest-cov` + coverage threshold
**Files:** `pyproject.toml` (`[tool.poetry.group.dev.dependencies]`), CI workflow
**Pathway:** `poetry add --group dev pytest-cov`; add `--cov=instancespace --cov-report=term-
missing` to the CI pytest invocation; establish a baseline number first (don't pick a threshold
blind), then decide a minimum to enforce going forward.
**Decision needed:** what threshold, once a baseline is known? Can't be answered before T1 is
actually run once — defer.

### T2 — Real end-to-end `build()` integration test
**Files:** new `tests/test_integration.py` (or similar), likely reusing
`tests/matlab_reference/input/metadata.csv` as real input
**Pathway:** construct a real `InstanceSpace(metadata, InstanceSpaceOptions.default())`, call
`.build()`, assert it completes without exception and that `.model` has every expected stage
output populated (`prelim`, `sifted`, `pilot`, `pythia`, `cloister`, `trace` all non-`None`).
This single test is also the prerequisite for Q8 and several Phase T items — sequence it early.
**Decision needed:** none.

### T3 — `conftest.py`
**Files:** new `tests/conftest.py`
**Pathway:** extract the repeated "load reference metadata"/"build a Mock `InstanceSpace`" setup
scattered across the `exploreIS/` validation tests into shared fixtures; audit for other
duplicated setup while there.
**Decision needed:** none.

### T4 — Fix `poe test`
**Files:** `pyproject.toml`
**Pathway:** one-line change —
```toml
test.sequence = ["test_ruff", "check_mypy", "test_black", "test_pytest"]
```
(defining `test_pytest = "pytest"` alongside the other `test_*` tasks).
**Decision needed:** none.

### T5 — Version-pin `tests/matlab_reference/` provenance
Covered in the roadmap's §7.3 — implementation lives on the MATLAB side (the export-script
issue in the MATLAB batch), not the Python side. Python-side follow-up once that exists: add a
CI step comparing the committed fixtures' provenance stamp against the latest available MATLAB
release, failing (or warning) if they've diverged.
**Decision needed:** warn or fail on divergence? Recommended default: warn at first — failing CI
on a fixture-staleness signal before the MATLAB-side export tooling is mature risks being noisy.

### T6 — DAG-resolver edge-case tests
**Files:** `tests/test_stage_builder_runner.py` (or split into a new file)
**Pathway:** add cases exercising: a genuine mutating stage (input name == output name), an
explicit `RunBefore`/`RunAfter` pair, and two stages producing the same output at the same
resolved schedule step (asserting the `StageResolutionError` fires with a useful message) — using
either slightly-less-trivial synthetic stages than today's `StageA`/`StageB`, or (once T2 exists)
the real 7-stage pipeline for the parts that need real branching (cloister/pythia sibling
structure) to mean anything.
**Decision needed:** none.

### T7 — Consolidate fragmented per-stage test files
**Files:** `tests/test_pilot.py`, `tests/exploreIS/pilot/test_pilot_unit.py`, `tests/exploreIS/
pilot/test_pilot_validation.py` (and the equivalent trio for sifted/trace/pythia/prelim)
**Pathway:** decide a rule (see below), then apply it consistently — likely: top-level
`test_<stage>.py` owns build-time unit tests, `exploreIS/<stage>/test_<stage>_unit.py` owns
explore-time orchestration unit tests (stubbed dependencies), `..._validation.py` owns
MATLAB-reference numerical validation — then merge anything that violates that rule into the
right file, and delete now-empty files.
**Decision needed:** confirm the three-way split rule above actually matches intent, or would
two files (build-time / explore-time) be simpler than three? Lower stakes than the others in
this document — noting it rather than escalating to the top-3 question list.

---

## Summary: decisions requiring your input

**Resolved:** F7 (HDF5 via `h5py`), Q5 (keep permissive feature-order auto-reorder), F9
(Option 1 — extend `explore()` itself, with the "new algorithm" edge case deferred by default),
Q9 (seed default — `0`, not `None`; corrected once the production/backward-compatibility
context was confirmed — see the note at the top of this document).

**Still open, each with a recommended default stated in place above:** Q7 (API shape — four
idiomatic methods vs. one MATLAB-style dispatch method),
F1 (default classifier once the registry exists — `'svm'` vs. `'knn'`), F8 (ambition level —
shared function vs. extended `Stage` contract), T1 (coverage threshold, can't be set before a
baseline exists), T5 (warn vs. fail on fixture staleness), T7 (three-way vs. two-way test-file
split). Flag any of these you'd like to override and I'll update this document and the roadmap
accordingly.
