# pyInstanceSpace — Implementation Pathways

> **Historical planning record.** Status labels and “not implemented” statements below
> describe the dated planning baseline, not the current branch. In particular, PILOT
> 3D/viewpoints/SIMPLS, TRACE3 3D, and 3D output are now implemented and verified against
> the installed 423-file MATLAB R2026a Update 4 v2 oracle. Use
> `docs/architecture.md`, `docs/implemented_fixes.md`, and
> `docs/pending_issue_backlog.md` for current decisions.

**Companion to:** `pyIS_docs_quality_roadmap.md` (v1.15)
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
   `explore_stage_iter()`), *Better engineering* (frozen dataclasses, DAG scheduler), *Bug fixes*
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

### Q1 — RETIRED, superseded by S3
This pathway (adding a `poly` branch to `_svc_to_artifact()`) is no longer the plan. S1 makes
`explore()` operate on native `SVC` objects, which handle poly kernels via `.predict_proba()`
with no special-casing needed at all — and S3 retires `build_explore_adapter.py` entirely once
that lands, since nothing calls it anymore. See S3's pathway below instead. (The aside in the
original pathway about `_svc_to_artifact()`'s `"linear"` branch being unreachable is now moot
too — the whole function is being deleted, not patched.)

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
4. **Pickle-exclusion (added alongside roadmap v1.18, required once F7 exists in any form):**
   add `__getstate__`/`__setstate__` to whichever class holds the pool attribute, dropping it
   from the returned state dict on `__getstate__` and setting it back to `None` on
   `__setstate__`. Without this, `Model.save()`/`InstanceSpace.save()` (F7) either crashes
   outright when a pool is live (a `ThreadPoolExecutor` holds `threading.Lock`/`Thread` objects
   pickle can't serialise) or only "works" because the caller happened to call `close()` first —
   a scenario-dependent failure that would surface as an intermittent F7 test failure rather than
   an obvious gap here. This is not optional cleanup, it's a correctness requirement for
   anything downstream that pickles the object holding the pool.
**Test:** assert no new `ThreadPoolExecutor` is constructed on a second `run_stage(TraceStage)`
call with the same `n_cores`; assert one *is* constructed if `n_cores` changes between calls.
Also assert `pickle.dumps()`/`save()` succeeds with a live (unclosed) pool and that the pool
attribute comes back unset after `load()`, with the next `run_stage()` call recreating it lazily
— this is the test that actually proves the pickle-exclusion works, not just an assumption.
**Decision needed:** explicit `close()` method the user must call, or rely on `__del__`
(implicit, less reliable in Python but less API surface)? Recommended default: explicit
`close()`, since relying on `__del__` for resource cleanup is a known Python anti-pattern
(non-deterministic timing, exceptions during interpreter shutdown are swallowed). Note this
decision only governs live-session cleanup — the pickle-exclusion above is required regardless
of which option is chosen here.

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
   constructs a real `InstanceSpace` with all 7 stages — the synthetic 2-stage
   `test_stage_runner.py` setup (renamed from `test_stage_builder_runner.py` when S2
   folded `StageBuilder` into `stage_runner.py`, v1.22) can't exercise this at all.
2. Build fully, capture `pythia`'s output object identity/values, call
   `space.run_stage(CloisterStage)` again, then check whether `pythia`'s output changed identity
   or got marked for re-run despite not depending on `cloister`.
3. If it reveals over-invalidation: the fix belongs in `stage_runner.py`'s
   `_rollback_to_schedule_index`, replacing schedule-index comparison with a real dependency-graph
   walk (mirroring MATLAB's BFS in `invalidateDownstream`) — that fix is F-phase work per the
   roadmap, not part of Q8 itself.
**Decision needed:** none for Q8 itself (it's a test, not a design choice) — but note the
dependency on T2 existing first, which affects sequencing.
**Sequencing with S2 (added alongside roadmap v1.19, sharpened v1.20):** run this *after* S2,
not before. `_rollback_to_schedule_index` (step 3 above) operates on `self._stage_order`, the
wave-grouped schedule list — verified in `stage_runner.py:256-267`, it invalidates everything in
`_stage_order[index+1:]` by position. S2 removes wave computation entirely, replacing it with an
explicit flat order. Fixing Q8's invalidation property against the pre-S2 structure means
writing the dependency-graph walk once against a data structure S2 then deletes, and S2 having
to re-derive the equivalent walk against its own new structure regardless — wasted
*implementation*, not just a wasted test. This differs from T6's version of the same sequencing
note: T6 tests the resolution *algorithm itself*, which S2 deletes outright, so T6 may have no
remaining subject matter post-S2 at all ("skip entirely" is a live option for T6). Q8 tests a
*behavioral property* — correct invalidation on partial rerun — that still has to hold after S2;
S2's own before/after checkpoint (full-pipeline output equality) doesn't cover partial-rerun
invalidation, so it doesn't subsume Q8 either. Net: Q8 must wait for S2, same as T6, but Q8 is
never at risk of becoming pointless the way T6 might — it only needs to retarget whichever
function ends up doing rollback/invalidation once S2 lands.

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

## Phase S — structural simplification (before Phase F)

### S1 — Collapse model-shape detection to native scikit-learn objects
**Files:** `instancespace/instance_space.py` (`_ensure_explore_model`, `_explore_pythia`),
`instancespace/build_explore_adapter.py` (scope narrows to F7's persistence use only, if kept
at all)
**Pathway:**
1. **Decision resolved, not just confirmed:** cross-platform MATLAB-model loading is closed as
   impractical — not attempted, not "impossible" in principle, but not worth building given the
   real cost (six PYTHIA classifier types, several with no clean flattened representation —
   decision trees especially). Recorded this way specifically so it can be reopened if a real
   use case ever appears. Proceed with the steps below.
2. Rewrite `_explore_pythia` to call `svc.predict(z)`/`svc.predict_proba(z)` directly on the
   stored `SVC` objects in `model.pythia.svm`, replacing the hand-rolled decision-function
   recomputation from flattened parameters.
3. Remove `_ensure_explore_model()`'s branching entirely (not just simplify it) — there's only
   one shape to handle once native objects are the interface.
4. `adapt_for_explore()`/`build_explore_adapter.py` has no remaining purpose at all once this
   and F7's signed-pickle design (no flattening needed — `SVC` objects pickle natively) both
   land — see S3, which retires the file outright rather than narrowing its scope.
5. Re-run `test_pythia_validation.py` unmodified as a regression check — it bypasses
   `_ensure_explore_model()` via `Mock(spec=PythiaOut)` and calls `_explore_pythia` directly, so
   it shouldn't need changes, but confirm that explicitly rather than assume it.
6. New test: build with each kernel type (rbf, linear, poly) via `stages/pythia.py`, call
   `explore()`, assert predictions match calling `.predict_proba()` directly on the same stored
   `SVC` — this is also what makes the old Q1 (poly-kernel `NotImplementedError`) moot, since
   native `predict_proba()` handles poly kernels with no special-casing at all.
**Decision needed:** none — resolved above.

### S2 — Replace DAG auto-resolution with explicit stage order + prerequisites
**Files:** `instancespace/stage_builder.py` (removed or drastically reduced),
`instancespace/instance_space.py` (constructor — replace the `StageBuilder` call with a direct
list), `instancespace/stage_runner.py` (keep — the *execution* engine given a resolved order is
still useful; only the *resolution* algorithm goes)
**Pathway:**
1. Write the explicit structure MATLAB's `InstanceSpace.m` uses as a template: an ordered list
   (`[PreprocessingStage, PrelimStage, SiftedStage, PilotStage, CloisterStage, PythiaStage,
   TraceStage]`, encoding that `CloisterStage`/`PythiaStage` can run in either order relative to
   each other but both after `PilotStage`) plus an explicit prerequisite mapping.
2. Keep each stage's declared `_inputs()`/`_outputs()` NamedTuple typing — the point is to stop
   *inferring* the schedule from it, not to stop *checking* declared types against it. `mypy
   --strict` should still catch a stage declaring a field type that doesn't match what's
   actually available at that point in the explicit order.
3. Remove: the ambiguous-ordering error path, mutating-stage special-casing, wave computation
   (confirmed elsewhere — Q6/T6 — that wave-based concurrency isn't actually used).
4. Keep: `run_stage()`, `run_until_stage()`, `run_iter()` — these are the genuinely valuable
   public capabilities, and none of them require auto-resolution to work; MATLAB provides the
   equivalent (`build('stages', {...})`) off its own hardcoded structure.
5. Sequence before T6, or skip T6 entirely — no point writing edge-case tests for an
   ambiguity-detection algorithm about to be removed. **Also before Q8** (added alongside
   roadmap v1.19, distinguished from T6 in v1.20) — Q8's regression test and diagnosis target
   the same `_rollback_to_schedule_index`/wave-position mechanism this step removes, but unlike
   T6, Q8's underlying property (correct invalidation on partial rerun) still needs to hold
   post-S2 — it's not at risk of having no remaining subject matter, only of wasting an
   implementation written against the structure being replaced. See Q8's own entry for the full
   cross-reference.
**Test:** run the full 7-stage pipeline before and after, assert identical execution order and
identical output — this change should be invisible from the outside.
**Decision needed:** none blocking — this is a mechanical simplification once the team is
comfortable trading auto-inference for hand-written (but still type-checked) explicitness.

### S3 — Retire `build_explore_adapter.py` entirely
**Files:** `instancespace/build_explore_adapter.py` (deleted), `tests/build_explore_adapter/
test_adapter.py` (deleted or repurposed)
**Pathway:**
1. Sequence strictly after S1 — this is a consequence of S1 landing, not independent work.
2. Delete `build_explore_adapter.py` (`adapt_for_explore`, `_svc_to_artifact`) in full.
3. `test_adapter.py::test_unsupported_kernel_raises` has nothing left to test once
   `_svc_to_artifact` is gone — delete it along with the rest of the file, rather than leaving
   a test importing a module that no longer exists.
4. Grep the whole repo for any remaining reference to `build_explore_adapter` before considering
   this done — confirm zero, not just the call sites already known about from S1's work.
**Test:** the full existing suite passing with the module gone *is* the test here — nothing new
to verify, only an absence to confirm.
**Decision needed:** none — purely a consequence of S1 and the closed cross-platform decision.

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
4. **Sequence this after S1.** Before S1, this point would have said the build→explore adapter
   needs equivalent artifact-flattening for every new classifier type — a real scope dependency
   between F1 and F8. S1 resolves that: once `explore()` calls `.predict()`/`.predict_proba()`
   natively on whatever classifier `build()` trained, every classifier in the table above
   already works on the explore side for free, no per-classifier flattening logic needed. Doing
   F1 before S1 would mean writing that flattening logic and then deleting it — sequence S1
   first.
**Decision needed:** what should the *default* `classifier` value be once this exists — `'svm'`
(preserves today's Python behaviour exactly) or `'knn'` (matches MATLAB's default, for
cross-implementation consistency)? Recommended default: **`'svm'`**, to avoid silently changing
existing users' default output the moment this option is introduced — `'knn'` remains one
config change away.

### F2 — PILOT 3D / viewpoint / PLS alternative, plus `ntries` restart parallelism
**Status (v1.61):** the `ntries`-parallelism sub-item below, plus #301's `cost_weight`/
`precalc_alpha` semantics chunk (issues 1/3/7), are now implemented (see roadmap v1.61 and
`instancespace/stages/pilot.py`) — `numerical_solve()` runs its restarts on a
`ProcessPoolExecutor` (empirically faster than a thread pool for this CPU-bound solve, unlike
Q6/TRACE), guarded against SIFTED's GA workers via `multiprocessing.parent_process()`.
**The 3D/`viewGroups`/PLS-alternative work below (the original F2 scope) is still not started** —
`PilotOptions` has no `dims`/`view_groups`/`method` fields yet, no `pilot_viewpoint.py` exists, and
the analytic/numeric solvers are still 2D-only.
**Verified starting point:** no `dims`, `viewGroups`, or `method` handling found in `stages/
pilot.py` at all — Python's PILOT is 2D-only, single-method, with none of MATLAB's Phase 5
surface. **Added v1.25, verified directly against `core/PILOT.m`:** MATLAB's numeric/BFGS branch
also parallelises its `opts.ntries` multi-start restarts — `nworkers = gcp('nocreate').NumWorkers`
(reuses whatever pool is already open, opens none itself) then `parfor (i=1:opts.ntries,
nworkers) ... end`, each restart independent (different random `X0(:,i)`, same cost function),
picked by `[~,idx] = max(out.perf)` afterward — order of completion doesn't affect the result.
Python's `pilot.py:520` runs the equivalent `for i in range(opts.n_tries):` loop strictly
sequentially; `PilotInput` has no `parallel_options` field at all, unlike `TraceInputs`/
`PythiaInput`. Not previously tracked under Q6 (Q6's own scope only ever named TRACE's
`ThreadPoolExecutor` and PYTHIA's `n_jobs`) or anywhere else — folded into F2 since F2 already
owns PILOT parity work end to end, and this restart loop is the same code F2's `dims`/`method`
work touches regardless.
**Files:** `data/options.py` (`PilotOptions` — add `dims: int`, `view_groups: list[list[int]] |
None`, `method: str`), `stages/pilot.py` (extend the analytic/numeric solvers to `n×3` where
`dims=3`; parallelise the `ntries` restart loop), new `stages/pilot_viewpoint.py` (direct port of
`PILOTviewpoint.m`).
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
5. Parallelise the `ntries` restart loop: add `parallel_options: ParallelOptions` to
   `PilotInput` (matching `TraceInputs`/`PythiaInput`'s existing pattern) and submit each restart
   to a `ThreadPoolExecutor` instead of the plain `for` loop — Q6's `InstanceSpace._get_executor()`
   pool-reuse mechanism already exists and can be threaded through the same way it now is for
   `TraceStage`; no need to invent a second pooling scheme. Each restart is CPU-bound (`fminunc`
   equivalent — likely `scipy.optimize.minimize`), so confirm whether Python's GIL makes a thread
   pool actually faster here or whether a `ProcessPoolExecutor`/joblib `loky` backend is needed for
   a real speedup — MATLAB's `parfor` uses OS processes, not threads, so this is a real port
   decision, not just a mechanical translation. **Nested-parallelism check (added v1.27, F3's
   audit, roadmap §6.2 finding 2):** `SiftedStage._find_best_combination()` already calls
   `PilotStage.pilot(...)` from inside `pygad.GA`'s own worker processes
   (`parallel_processing=["process", n_cores]`) when `parallel_options.flag` is set. Whatever
   pool this step adds to PILOT must detect (or otherwise avoid) already running inside one of
   those worker processes, or it reintroduces MATLAB's nested-`parfor`-inside-GA bug.
**Test:** `ntries` restarts must be embarrassingly parallel and order-independent — assert
identical `out.A`/`out.Z`/`out.perf` (up to the same numerical tolerance already used elsewhere)
whether run with `parallel.flag=True` or `False`, proving parallelising the loop doesn't change
which restart wins.
**Decision needed:** should R1 (rotation canonicalisation, already scoped separately) be applied
before or after `dims=3` lands? Recommended default: **after** — R1's centroid-angle math as
written assumes 2D; generalising it to 3D orientation (which needs an axis choice, not just an
angle) is a bigger question than R1's original scope intended, so land 2D rotation first, revisit
3D rotation as its own follow-on once F2 exists. **New, added v1.25:** thread pool (matching Q6)
or process pool (matching MATLAB's actual `parfor` semantics) for the `ntries` loop? Recommended:
resolve via the GIL/CPU-bound test above rather than assuming thread-pool parity with Q6 is
automatically correct just because Q6 exists.

### F3 — SIFTED refinements
**Audit complete (roadmap §6.2, v1.27).** Checked `stages/sifted.py` directly against MATLAB
SIFTED promotion's four historical fixes. Result: 2 of 4 don't apply to Python at all (no
module-level mutable state to have inherited the thread-unsafe-cache bug into; the per-candidate
`rng` reuse is deliberate common-random-numbers, not an accidental MATLAB-style reset). 1 is a
latent risk flagged forward to F2, not a fix for F3 itself (nested-`parfor`-inside-GA — currently
safe only because `pilot.py` has no parallelism of its own yet). 1 is a real, confirmed,
still-open gap: `_compute_correlation()` (lines 1031–1076) is an unvectorised nested Python loop
calling `scipy.stats.pearsonr` once per (feature, algorithm) pair. **This section scopes that one
gap's fix. Not implemented — scoping only, per explicit instruction; do not write this code
without being separately asked.**

**The gap, precisely.** For each of `rows` features × `cols` algorithms, `_compute_correlation`
masks out rows where either column has a NaN (`valid_indices = ~np.isnan(x_col) & ~np.isnan(y_col)`,
computed independently per `(i, j)` pair — genuinely ragged, not a single shared mask), then calls
`pearsonr` on the filtered pair, or writes `(nan, nan)` if no valid rows remain. This ragged
per-pair masking is exactly why a single `np.corrcoef`-style batch call doesn't drop in as a
replacement — `np.corrcoef` needs one consistent set of valid rows for the whole matrix, not one
per `(i, j)` cell.

**Two candidate designs, in order of recommendation:**

1. **Recommended: fast-path + fallback split, not full vectorisation.** Precompute
   `x_nan = np.isnan(x)` (shape `(n, rows)`) and `y_nan = np.isnan(y)` (shape `(n, cols)`) once.
   Any `(i, j)` pair where `x_nan[:, i].any()` and `y_nan[:, j].any()` are both `False` has no
   ragged masking to do — for *that subset* of pairs, correlation and its p-value can be computed
   for every `i` and every `j` in one vectorised pass (manual Pearson formula: means/stds/
   covariance via broadcasting across the full `x`/`y` matrices, e.g. `np.corrcoef(x.T, y.T)`'s
   cross-block, or the standard `((x - x.mean(0)) / x.std(0)).T @ ((y - y.mean(0)) / y.std(0)) / n`
   form), then the matching p-value via `scipy.stats.t.sf` applied elementwise using the shared
   degrees of freedom `n - 2` (valid because every pair in this subset uses the same `n`, all rows).
   Any pair where either column actually has a NaN keeps calling the existing, already-verified
   per-pair `pearsonr` loop unchanged — the exact code path that produces today's exact output for
   the messy cases. This bounds the change's risk surface to the (likely common) all-valid case
   while leaving the ragged-NaN case's proven-correct behaviour completely untouched — no new edge
   cases to reconcile against `scipy.stats.pearsonr`'s own degenerate-input semantics (zero
   variance, `n < 3`, etc.), since those inputs never leave the old loop.
2. **Alternative: full vectorisation via masked pairwise sums.** Build the full 3D validity mask
   `valid[n, i, j] = x_nan_free[:, i, None] & y_nan_free[:, None, j]`, then compute per-pair `n`,
   sums, and sums-of-products via masked reductions (`np.where(valid, ..., 0).sum(axis=0)`), and
   derive `rho`/`pval` from those. Fully vectorised, no fallback loop at all — but it means
   re-deriving `pearsonr`'s exact degenerate-case behaviour by hand (what it returns for
   zero-variance columns, `n_valid < 2`, `n_valid == 2` giving a defined-but-degenerate `p`, etc.)
   and proving the hand-derivation matches `scipy`'s bit-for-bit or within float tolerance for
   every one of those edge cases — meaningfully more verification surface for a function that,
   per the audit, only runs once per `SiftedStage` call (not per-GA-candidate), so the performance
   upside is smaller than the risk. Prefer option 1 unless profiling shows the fast-path/fallback
   split still leaves a real hot spot (e.g. if most real datasets actually do have scattered NaNs
   across most feature/algorithm pairs, defeating the fast path's coverage).

**Files:** `instancespace/stages/sifted.py` (`_compute_correlation`, lines 1031–1076 as of this
writing).

**Test plan (either design):** `rho`/`pval` must match today's output exactly (or within a tight
float tolerance, exact preferred since this is pairwise-deletion Pearson, not an approximation)
across: no-NaN input (exercises the new fast path hardest), all-NaN-in-one-column, scattered NaN
per pair, a constant-valued column (zero variance — check what `pearsonr` actually returns for
this today before assuming), and a column pair with fewer than 3 valid rows after masking. Existing
`tests/matlab_reference/` SIFTED validation fixtures plus new unit tests targeting these specific
edge cases directly (mirroring the granular edge-case coverage already established for e.g. F14's
warning tests) — don't rely on the MATLAB fixture alone to exercise every edge case above, since
there's no guarantee that fixture's data happens to contain a zero-variance or all-NaN column.

**Decision needed:** none blocking — option 1 above is the recommendation; revisit only if
someone profiles this as an actual hot spot before implementing.

### F4 — `InstanceSpace` class & build/explore robustness
Already has its own detailed audit (roadmap §5.1, 7 findings) — see F7/F8/F9 below, which are
the concrete work items this audit produced. No further pathway needed here; F4 itself is now
"audited," not "not started."

### F5 — Output consolidation / 3D visualisation parity
**Files:** `instancespace/_serialisers.py` (the audit this pathway called for happened during a
dead-code cleanup pass: `instancespace/scripting/script_fcn.py`/`script_disc.py`, previously named
here, were deleted — every function in `script_fcn.py` was a `raise NotImplementedError` stub with
0% test coverage, and `_serialisers.py` already has working private equivalents, e.g. `_draw_
scatter`/`_draw_binary_performance`/`_draw_good_bad_footprint`/`_write_array_to_csv`, that
superseded them long before the stubs were removed. `script_disc.py`'s citation/disclaimer printer
was likewise unreferenced anywhere — that role is now served by `CITATION.cff`/`README.md`.)
**Pathway:**
1. Audit-first finding (resolved): `_serialisers.py`'s drawing functions are 2D-only throughout —
   no branch on projection dimensionality anywhere.
2. Once F2 (3D PILOT) exists, `_serialisers.py` becomes the consumer: 3D scatter/footprint
   rendering (matplotlib's `mplot3d`, or a 2D camera-angle-projected render using F2's viewpoint
   output — decide which matches the "spirit" of MATLAB's `scriptpng` 3D handling more usefully for
   a Python audience).
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
**Decision revised: signed `pickle`/`joblib` with an optional signature, superseding both the
HDF5-via-`h5py` decision and the earlier unconditional-signing decision.** Confirmed production
threat model: on the web platform, models are produced by the system and downloaded, never
re-uploaded — `load()` never receives externally-supplied input, closing pickle's core
objection, *provided this is enforced, not merely assumed* (see the non-negotiable requirement
below). A second, distinct usage mode exists alongside the server one: local/desktop
development, where a researcher saves and loads their own model file on their own machine with
no secrets-manager wired up. `secret_key` is `None` by default there — the plain-pickle risk is
no different from running any other file the user already controls, e.g. `joblib.load` itself
carries the same caveat with no signing at all.
**Files:** `instancespace/model.py` (new `Model.save()`/`Model.load()`), no new dependency
(`hmac`/`hashlib`/`pickle` are all stdlib; `joblib` is already a transitive scikit-learn
dependency)
**Pathway:**
1. `save(path, secret_key: bytes | None = None)`: `pickle.dumps(model)` (or `joblib.dump` —
   prefer `joblib` for large numpy arrays, it's more efficient than raw pickle for that case and
   already in the dependency tree via scikit-learn). If `secret_key` is given, compute
   `hmac.new(secret_key, data, hashlib.sha256).digest()` and write the signature alongside the
   serialised bytes (e.g. a sibling `.sig` file); if `secret_key` is `None`, write only the data
   — no `.sig` file at all, so an unsigned save leaves no artifact claiming to be verified.
2. `load(path, secret_key: bytes | None = None)`: read the bytes and, if a `.sig` file exists,
   the signature. Four cases, all of which must be handled explicitly (this is the actual design
   surface — not just the signed-server case as before):
   - `secret_key` given, `.sig` present: **recompute the HMAC and compare before touching
     `pickle.loads()` at all** — if it doesn't match, raise immediately and never deserialise.
     This ordering is the entire safety property for the server path; get this step first, not
     "verify then load" as two independent steps a future edit could reorder.
   - `secret_key` given, `.sig` absent: raise — a caller expecting a verified load must never
     silently fall through to an unverified one.
   - `secret_key` is `None`, `.sig` absent: desktop/dev path — `pickle.loads()` directly, no
     verification, matching today's "you already trust files you produced yourself" caveat.
   - `secret_key` is `None`, `.sig` present: **raise, do not load.** This is the downgrade-attack
     case — a server-signed file must not become loadable-unverified just because the caller
     omitted the key. Without this branch, the whole point of signing (making tampering or
     substitution detectable) is defeated by the simplest possible bypass.
3. Secret key management for the server path: needs a real answer before this ships — a
   server-side secret, rotated on whatever cadence the deployment's secret-management practice
   already uses. Not this document's call to make; flag to whoever owns the web platform's
   deployment. The desktop path has no equivalent key-management question — it has no key.
4. Once S1 lands: `pythia.svm`'s `SVC` objects pickle natively, exactly as fitted — no
   flattening step, no `SimpleNamespace` conversion at load time. `trace.good`/`trace.best`
   (shapely `Polygon`/`MultiPolygon`) also pickle natively — the vertex-array flattening this
   pathway previously specified for HDF5 is no longer needed.
5. Round-trip test: save then load, assert deep equality (`np.array_equal` for arrays;
   `SVC`/polygon objects can be compared directly now, no constituent-array comparison needed).
   Run once with `secret_key` set (signed path) and once without (desktop path).
6. Signature-tampering test: flip one byte in the serialised payload, assert `load()` refuses
   before ever calling `pickle.loads()` — this is the test that actually proves the safety
   property holds, not just an assertion in a docstring.
7. Downgrade-attack test: `save()` with a `secret_key`, then `load()` the resulting file with
   `secret_key=None` — assert this raises rather than silently deserialising unverified. This is
   the test that proves the two-mode split doesn't reopen the exact hole signing was meant to
   close.
8. Path-safety test: assert (via whatever mechanism enforces it — allowlist, storage-layer
   check) that the *server* code path always calls `load()` with a `secret_key`, and is
   structurally unreachable via any user-supplied path or parameter. This is the one thing the
   server-side design's safety depends on; it needs its own test, not just an assumption
   documented here.
**Decision needed:** where does the HMAC secret key live and get rotated, for the server path?
Not a code-design question — a deployment/ops one, flag it rather than guessing. The desktop
path's `secret_key=None` default needs no equivalent decision.

### F8 — Unify `explore()` with build-time stage code
**Implemented and verified (v1.66) — see roadmap's F8 row for the full summary.** The
"lighter" approach (recommended below) was taken for the PYTHIA half: `PythiaStage.
_determine_selections`'s weighting math split into a new shared `_weighted_selection`
static method, called by both `_determine_selections` (training) and `_explore_pythia`
(explore); `_explore_pythia` also switched to calling `PythiaStage._compute_znorm()`
instead of recomputing the same mu/sigma formula inline. The TRACE half turned out to
need no extraction at all - `_explore_trace()` was already correctly reusing the trained
`Footprint.polygon` rather than re-deriving footprint math, so there was no live
duplication to unify (see below for what the audit found instead). The `nalgos == 1`
explore-path gap this section flagged as "the likeliest place a real divergence already
exists" was real: it reproduces a separate, pre-existing bug (filed as #314), not fixed
as part of this item since it's its own `[Behavior-changing]` fix.

**Scope correction (added earlier pass, verified directly against current code, not assumed):**
this item's existing text claimed S1 "resolves the PYTHIA half of this item as a side effect...
there is no second implementation left to reconcile" and that "`_explore_pythia` is retired
entirely once S1 lands, not folded into this item." **Both claims are wrong as of the current
code.** `_explore_pythia` (`instance_space.py:959`) is very much still alive - S1 only fixed
*how it calls the classifier* (native `.predict()`/`.predict_proba()` instead of hand-rolled SVM
math), not whether its surrounding logic duplicates PYTHIA's build-time code. It does, in two
confirmed places:

1. **Precision-weighted algorithm selection.** `_explore_pythia` inlines `weighted = y_hat *
   precision; best = weighted.max(axis=1); selection0 = weighted.argmax(axis=1);
   selection0[best <= 0] = -1` — this is the *same formula* as `PythiaStage._determine_selections`
   (`pythia.py:1332`), independently reimplemented. `_determine_selections` additionally handles
   `nalgos == 1` as a special case and computes a `selection1`/`default`-fallback value
   `_explore_pythia` doesn't attempt at all - if a bug in the weighting formula were fixed in one,
   it would silently not be fixed in the other, and the `nalgos == 1` edge case is untested on the
   explore side because there's no shared code path forcing it to be.
2. **Z-score normalisation.** `_explore_pythia` inlines `mu = train_z.mean(axis=0); sigma =
   train_z.std(axis=0, ddof=1); z_norm = (z - mu) / sigma` - the same formula as
   `PythiaStage._compute_znorm` (`pythia.py:1204`). Lower-priority than (1): `_explore_pythia`'s
   own docstring already documents *why* it recomputes rather than reads back the stored
   `PythiaOutput.mu`/`sigma` (avoiding an unnecessary indirection since the two are
   mathematically identical), so this duplication is a deliberate, reasoned choice, not an
   oversight - but it's still the same formula living in two places, and still worth folding into
   whatever shared-extraction F8 does for (1), for the same reason.

TRACE's own scope (below) is unaffected by this correction - it was always real, independent of
the PYTHIA finding.

**Second correction (implementation pass, v1.66): the TRACE claim above turned out to be
wrong too, on direct inspection.** Reading `TraceStage.build()`/`contra()`/`fit_poly()` side by
side with `_explore_trace()` shows explore-time correctly reuses the already-trained
`Footprint.polygon` (built once, at training time, via DBSCAN + alpha-shape fitting) and only
performs a point-in-polygon membership test against it - there is no second implementation of
the footprint-construction math to reconcile, unlike PYTHIA's genuine formula duplication above.
The audit did find a real, separate issue while checking this: training's `Footprint.
from_polygon`/`contra()` use `.contains()` (boundary-exclusive) for their own point-in-polygon
tests, while `_explore_trace()` uses `.covers()` (boundary-inclusive, matching MATLAB's
`inpolygon`) - the same conceptual test, two different answers at the boundary. Filed as #315,
not fixed here: changing `Footprint.from_polygon`'s semantics would alter existing training-time
`density`/`purity` output for every run, a `[Behavior-changing]` fix needing its own verification,
not a side effect of F8's audit.

**Files:** `instancespace/stages/stage.py` (possibly extending the `Stage` contract),
`instancespace/instance_space.py` (`_explore_trace` *and* `_explore_pythia`, per the correction
above), `instancespace/stages/pythia.py` (`_determine_selections`, `_compute_znorm` become the
shared functions both paths call, mirroring `PrelimStage`'s extraction pattern already decided
for F9)
**Pathway — two ambition levels, pick one, applied consistently to both TRACE and PYTHIA:**
- **Lighter:** extract the *numerical core* of `TraceStage`'s footprint/alpha-shape logic, and
  separately `PythiaStage`'s `_determine_selections`/`_compute_znorm`, into shared pure functions
  that both the build-time `_run()` and the matching `explore()` method call — no change to the
  `Stage`/`StageRunner` architecture itself, just de-duplicating the math underneath it. Lower
  risk, doesn't touch the DAG scheduler. `_determine_selections`/`_compute_znorm` are already
  `@staticmethod`s taking/returning plain arrays, so this is close to a direct call-site swap for
  the PYTHIA half - lower effort than TRACE's own extraction.
- **Fuller:** extend the `Stage[IN, OUT]` contract with a second entry point (e.g. `_predict()`
  alongside `_run()`) so `TraceStage`/`PythiaStage` themselves know how to run in inference mode,
  and `explore()` dispatches to `_predict()` directly instead of maintaining `_explore_trace`/
  `_explore_pythia` separately. Higher risk (changes a core abstraction every stage implements),
  but closes the drift risk more completely — a bug fixed in `_run()` is structurally guaranteed
  to also be fixed in `_predict()` if they share the surrounding class, not just a shared helper
  function.
**Test (either way):** the drift-detection test already scoped in the roadmap's Phase T —
deliberately break something in `TraceStage`'s footprint logic, assert both the build-path test
*and* the explore-path test fail, proving they can no longer silently diverge. Add the same
pattern for PYTHIA: deliberately break `_determine_selections`'s weighting formula, assert both
`test_build_pythia.py` and an explore-path test fail together. Also add the `nalgos == 1` case to
an explore-path test specifically - it's the one behaviour `_explore_pythia`'s current inline
version doesn't attempt to handle at all, so it's the likeliest place a real divergence already
exists, not just a hypothetical future one.
**Decision needed:** lighter (shared function) or fuller (extended `Stage` contract)?
Recommended default: **lighter**, as a first step — it captures most of the "no more silent
drift" benefit for much less architectural risk, and the fuller redesign remains available later
if the lighter version proves insufficient in practice.

### F9 — Expand `explore()` to full evaluation scope
**Implemented and verified (v1.66) — see roadmap's F9 row for the full summary.** All 7
pathway steps below landed as scoped: `compute_binary_performance` extracted to `prelim.py`
(step 2, also serving F8's PYTHIA-adjacent de-duplication goal as intended), case-insensitive
algorithm reconciliation via `InstanceSpace._build_test_algo_matrix` (step 3, NaN-padding a
training algorithm absent from the test set rather than a separate branch), real accuracy/
precision/recall/confusion-matrix computation via `InstanceSpace._explore_evaluate` against
PYTHIA's already-computed `y_hat` (step 4), `ExploreResult`'s 8 new fields (step 5, plus the
"[EXPLORE] Calculating the binary measure of performance" log line coming for free from
`compute_binary_performance`'s `log_prefix` parameter, satisfying step 5's visibility
requirement), and `ExploreStage.EVALUATION`'s conditional yield in `explore_stage_iter()`
(step 7). The "new algorithm absent from training" edge case (decision point below) was
deferred exactly as recommended - not implemented.

**Decision made: Option 1 — extend `explore()` itself** (not a new method; silent branching
based on whether ground truth is present in the input).

**Scope correction (added this pass, verified directly against MATLAB source, not assumed):**
this item's existing pathway only covered PRELIM's ground-truth labels (`y_bin`/`y_best`/`p`/
`beta` for the test set) - it stopped one step short of what MATLAB's actual
`InstanceSpace.evaluateTestSet` (`InstanceSpace.m:736`) does. Read directly: after computing
those PRELIM-equivalent ground-truth fields, `evaluateTestSet` calls `out.pythia = PYTHIA(Z,
Yraw, Ybin, Ybest, algolabels, opts.pythia, model.pythia)` - the **same** `PYTHIA()` function
used at training time, given a 7th argument (the trained model) that switches it into
`PYTHIAevalMode` (`core/PYTHIA.m:305`). That eval-mode branch applies the *already-trained*
classifiers to the test set and computes real `precision`/`recall`/`accuracy`/`cvcmat`
(confusion-matrix-derived) against the newly-available ground truth, then calls the *same*
`computeSelection` subfunction training uses for the precision-weighted algorithm pick. "Full
evaluation scope" in MATLAB means algorithm-performance metrics against ground truth, not just
exposing the ground-truth labels themselves - Python's F9 scope needs to include this, which the
existing pathway didn't.

This is exactly what F8's PYTHIA-side correction (see F8 above: `_determine_selections`/
`_compute_znorm` already need extracting into shared functions to de-duplicate build/explore) is
also needed for here - F9 doesn't just benefit from that extraction, it structurally depends on
it: F9's "compute accuracy/precision/recall against ground truth using the trained classifiers"
step and F8's "de-duplicate `_explore_pythia`'s prediction/selection logic" step are two views of
the *same* code, not separable work. This sharpens why F8-before-F9 is already recorded as a soft
preference in the roadmap's dependency ordering (§6.0) - it's closer to a real dependency for the
PYTHIA half specifically, even though the ordering note there was written for the TRACE-based
reasoning only.

**Files:** `instancespace/instance_space.py` (`explore()`, `explore_stage_iter()`), `instancespace/
data/model.py` (`ExploreResult` — new optional fields), `instancespace/stages/prelim.py`
(extract shared binary-performance logic), `instancespace/stages/pythia.py` (the same
`_determine_selections`/`_compute_znorm` extraction F8 needs, reused here for real metric
computation, not just label prediction)
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
   below. MATLAB pads these with `Yhat=false`, `Pr0hat=0`, `precision`/`recall`/`accuracy=NaN`
   (`PYTHIAevalMode`, "no CV model" convention) rather than dropping them - worth matching that
   convention exactly if/when the deferred edge case (below) is picked up, rather than inventing
   a different placeholder.
4. **Compute real evaluation metrics using the trained classifiers - this is the corrected,
   previously-missing piece.** For each trained classifier: z-score the test `Z` using the
   trained model's stored `mu`/`sigma` (the same formula as `_compute_znorm`/`_explore_pythia`'s
   current inline z-scoring - one shared function per F8's correction, not a third copy), predict
   `y_hat`/`pr0_hat` (already exists in `_explore_pythia`), then additionally derive a confusion
   matrix against the newly-available `y_bin_actual` and compute `precision`/`recall`/`accuracy`
   per algorithm (`tp/(tp+fp)`, `tp/(tp+fn)`, `(tp+tn)/ninst` - MATLAB's exact formulas,
   `core/PYTHIA.m:379-381`). Feed these into the *same* precision-weighted selection function F8
   extracts (`_determine_selections`'s equivalent), using **training-time precision** for the
   selection weighting when available (MATLAB's own fallback rule: use `trained.precision` if
   present, else the freshly-computed eval precision - `core/PYTHIA.m:388-393`), not the
   just-computed eval precision by default.
5. **Extend `ExploreResult`** with new fields, all `| None`, populated only when ground truth is
   present: `y_actual: NDArray[np.bool_] | None`, `y_best_actual: NDArray[np.double] | None`,
   `p_actual: NDArray[np.int_] | None`, `beta_actual: NDArray[np.bool_] | None`, plus (per the
   correction above) `accuracy_actual: NDArray[np.double] | None`, `precision_actual:
   NDArray[np.double] | None`, `recall_actual: NDArray[np.double] | None`,
   `cvcmat_actual: NDArray[np.double] | None`. `None` in the feature-only case preserves today's
   behaviour exactly — existing callers see no change.
6. **Make the silent branch visible, even though it's automatic.** Since this is Option 1 (not
   an explicit separate call), log an info message when ground truth is detected and evaluation
   fields get populated (ties to Q3's logging work) — mirrors MATLAB's own "[EXPLORE]
   Calculating the binary measure of performance" console line, so the mode switch is
   observable, not a silent surprise, even though it's inferred from input shape rather than an
   explicit flag.
7. **`explore_stage_iter()` needs the same treatment:** add an `ExploreStage.EVALUATION`
   member (see `instancespace/instance_space.py`'s `ExploreStage` enum, added alongside the
   `explore_iter()` -> `explore_stage_iter()` rename) and yield an `AnnotatedExploreOutput`
   for it after `ExploreStage.TRACE` when ground truth is present; omit it entirely (yield
   only the original 5) when it isn't. `explore_stage_iter()`'s yielded-stage count becomes
   conditional on input shape, matching `explore()`'s own conditional field population.
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
current files don't cleanly distinguish them. The corrected scope above adds its own concrete
test target: a fixture where the trained classifiers' test-set accuracy/precision/recall are
independently known (from MATLAB's own `evaluateTestSet` output, if a MATLAB reference run is
ever obtained per T5/#278) or at minimum internally consistent (recomputing the same confusion
matrix from `y_hat`/`y_bin_actual` by hand and checking it matches `ExploreResult`'s reported
`accuracy_actual`/etc.).

### F16 — Port TRACE3, MATLAB's current default footprint algorithm
**Verified starting point, read directly against `core/TRACE.m`/`core/TRACE_legacy.m`
(`andremun/InstanceSpace`, not inferred from prose):** Python's `stages/trace.py` is a line-for-
line port of `TRACE_legacy.m` only (confirmed during F11/#302's audit). MATLAB's `TRACE.m` is a
dispatcher over **two** algorithms - `method='legacy'` (what Python has) and `method='trace3'`
(MATLAB's actual default now, `TRACE.m:20`) - and Python's own `TraceOptions.method` defaults to
`"legacy"`, the opposite of MATLAB's current default. `method='trace3'` currently raises
`NotImplementedError` (a deliberate F13-style fail-loud stub, not a silent gap) with no
implementation behind it. F11 explicitly scoped this out: "porting TRACE3 itself would be a
separate, much larger future item if ever prioritised, not part of that decision" - this item is
that future item, previously mentioned only in prose with no issue of its own.

**What TRACE3 actually does (`TRACEbuild3`, `core/TRACE.m:204-259`), per algorithm:**
1. `Zu = {zi | yhat_i=1 AND ybin_i=1}` (or just `ybin_i=1` if PYTHIA predictions are
   unavailable/skipped) - unique rows only.
2. If `|Zu| <= minInstances` (default 4) -> empty footprint, stop.
3. Build an alpha-shape over `Zu` at MATLAB's auto-computed default ("critical") alpha - the
   smallest alpha giving one connected region enclosing every point.
4. Compute `measure` (area/volume), `elements` (points inside), `goodElements` (good-labelled
   points inside), `density = elements/measure`, `purity = goodElements/elements`. If invalid
   (degenerate shape) or `measure < minAreaFrac * spaceArea` (default 0.01) -> empty, stop.
5. If `purity >= PI` (purity threshold, default 0.6) -> done, return this footprint.
6. Otherwise **iteratively tighten**: compute the shape's `alphaSpectrum` (MATLAB's built-in -
   the sorted, topologically-distinct alpha values for this point set), step alpha down through
   101 points from the current value to the spectrum's minimum, setting `RegionThreshold` (drops
   already-tiny disconnected regions - MATLAB's own noise filter) at each step, recomputing
   metrics until purity clears the threshold or the shape becomes invalid/too small. If the whole
   spectrum is exhausted without clearing purity, return the best (last-computed) footprint.

Also needed: `TRACErescore` (re-scores a *trained* footprint against new points without rebuilding
it - MATLAB's evaluation-mode counterpart, i.e. the `explore()`-time path) and the top-level
dispatcher logic (3D check, PYTHIA-availability handling, per-algorithm loop over `good`/`best`,
the `hard` beta-footprint). Python's `TraceInputs` **already has `y_hat`** wired through
(confirmed - not a blocker); MATLAB's comment that TRACE "never trains its own KNN classifier"
(PYTHIA is a mandatory upstream dependency in this pipeline) matches this repo's own stage order
already, so no fallback classifier path is needed.

**The real roadblock - no free lunch here, flagged before any implementation is attempted:**
MATLAB's `alphaShape` is a built-in, stateful computational-geometry object (part of base MATLAB,
not a toolbox function this repo could read the source of) with capabilities Python's existing
`alphashape` 1.3.1 dependency (used today only for legacy's single fixed-alpha call,
`alphashape.alphashape(polydata, 2.15)`, `trace.py:814`) does not expose as a ready-made
equivalent:
- **Default/critical alpha** - MATLAB auto-selects the smallest alpha giving single-region
  coverage. `alphashape.optimizealpha()` exists but optimises for a different goal (smallest-area
  single polygon fit); whether its output is a usable stand-in needs direct comparison, not an
  assumption.
- **`alphaSpectrum`** - no Python equivalent function exists, but the pieces to build one are
  already present: `alphashape.alphasimplices(points)` yields each Delaunay simplex and its
  circumradius (confirmed via source read - `scipy.spatial.Delaunay`-based, so it already works
  in arbitrary dimensions, 2D and 3D alike) - the sorted distinct circumradii *are* the
  alpha-spectrum values by the standard alpha-shape definition. This needs new code, and the
  radius-vs-inverse-radius convention must be reconciled against MATLAB's actual `Alpha` units
  before trusting the result.
- **`RegionThreshold`** (drops small disconnected regions as alpha shrinks) - no equivalent found
  in `alphashape`/`shapely`/`scipy`. Feasible by hand in 2D (decompose a `MultiPolygon`'s parts,
  drop ones below an area threshold), but is a fresh, unverified piece of geometry code, not a
  library call.
- **Native 3D** (`volume`, `inShape` on a 3D alpha-shape) - `alphashape`'s Delaunay-based core is
  dimension-agnostic in principle, but its higher-level convenience function and volume
  computation were not verified for 3D in this pass; a 3D-capable library (`trimesh`, `open3d`)
  may be a better fit than extending `alphashape` by hand. Ties directly to F2 - Python's PILOT
  is 2D-only until F2 lands, so there's no real 3D `Z` to test against yet regardless.

**Because of the above, exact MATLAB parity may not be achievable** the way most of this repo's
other ports are verified (bit-for-bit or tight-tolerance against a MATLAB reference fixture) -
`alphaSpectrum`/`RegionThreshold` are MATLAB toolbox internals with no published algorithm to
copy exactly, only their documented *behaviour* to approximate. This item's test plan should be
scoped around behavioural invariants (purity threshold respected once tightening stops,
`minInstances`/`minAreaFrac` respected, tightening is monotonically non-worsening) rather than
assuming a MATLAB-fixture comparison test is even meaningful here - a materially different
(weaker) parity bar than this repo's norm, worth stating explicitly rather than discovering after
the fact when a fixture comparison predictably fails to match.

**Two smaller, already-confirmed gaps, folded into this item's option-surface work (not
separately filed):**
- `TraceOptions` has no `min_instances`/`min_area_frac` fields at all (F11 deliberately left them
  out since they have no meaning for legacy - they matter starting here).
- `TraceOptions.purity` (Python's existing name for MATLAB's `PI`) defaults to `0.55`; MATLAB's
  actual `ISAdefaults.m` default is `0.6` (`0.55` traces to a specific example script's override,
  not the toolkit's own default) - worth fixing as part of wiring up `PI`'s real consumer, or
  flagging as a separate pre-existing default-value mismatch if that's preferred.

**Files:** `instancespace/stages/trace.py` (new `TRACEbuild3`/`TRACEmetrics3`/`TRACErescore`-
equivalent methods, dispatcher branch replacing today's `NotImplementedError`), `instancespace/
data/options.py` (`TraceOptions` - add `min_instances: int`, `min_area_frac: float`), possibly a
new `instancespace/utils/alpha_shape.py` for the spectrum/region-threshold logic if it doesn't
belong directly in `trace.py`.
**Pathway:**
1. Resolve the `alphaSpectrum` question first, in isolation, before touching `trace.py` itself -
   derive it from `alphashape.alphasimplices`, verify the values behave sensibly (monotonic,
   sensible range) against a few synthetic point sets, decide the radius-convention question.
2. Prototype `RegionThreshold`'s small-region-drop behaviour in 2D via `shapely` `MultiPolygon`
   part filtering.
3. Implement `TRACEbuild3`'s 2D case end to end (steps 1-6 above), then `TRACErescore`.
4. Wire the dispatcher: `method='trace3'` branch in the existing method-selection logic
   (`trace.py`'s current `NotImplementedError` site), replacing the stub.
5. Defer 3D entirely to a follow-on once F2 lands - don't build 3D alpha-shape support against a
   Z that can't exist yet.
**Test:** behavioural-invariant tests per the parity-bar note above, not a MATLAB-fixture
numeric-match test (unless/until direct comparison proves the reimplementation tracks MATLAB
closely enough for that to be meaningful).
**Decision needed:** none blocking implementation start, but two worth a call before or during
the audit above: (1) whether to fix `TraceOptions.purity`'s default (`0.55` -> `0.6`) as part of
this work given it's a "TRACE3 needs a real consumer for `PI`" fix, or leave it as a
separately-tracked pre-existing mismatch; (2) whether Python's own default `TraceOptions.method`
should flip to `"trace3"` once implemented, matching MATLAB's current default, or stay `"legacy"`
until TRACE3 has enough independent verification to trust as a new default - recommended:
**stay `"legacy"` at first**, flip only after real-world use builds confidence, since unlike most
of this repo's other behavior changes there is no MATLAB-fixture-verified bar to clear first.
**Soft preference:** implement after F8 lands, not before - `TRACErescore` is exactly the
`explore()`-time counterpart F8 is designed to unify with the build-time path; building it before
F8's shared-extraction pattern exists means writing it once now and reconciling it into that
pattern later, the same rework risk already documented for F8-before-F9.

### F17 — Preserve boundary-inclusive TRACE membership

**Resolution:** issue #315's premise is incorrect. MATLAB `polyshape.isinterior` returns true for
points in a solid region or on its boundary. Shapely `.covers()` is the matching operation.
Changing explore to `.contains()` reduced reference agreement from 100% to 92.02%.

Keep `.covers()` in explore and use pointwise `.covers()` in legacy TRACE scoring and refinement.
Boundary-exact regressions must assert `True`. The old `.contains()` pathway is superseded and
#315 should close without the proposed code change.

**Source:** [MathWorks `isinterior`](https://www.mathworks.com/help/matlab/ref/polyshape.isinterior.html).
**Compat:** build-time scoring changes where Python previously excluded boundary points; explore
behavior stays unchanged.

### F18 — Unify build/explore into single-body stage methods (all stages)
**Origin:** proposed on direct request, filed as a future item, not scoped for implementation now.
Framed explicitly as "how someone would actually implement this from scratch" - i.e., if nobody
had ever written `InstanceSpace`'s `_explore_*` method family, would today's split (`stages/*.py`
owning `build()`, `instance_space.py` separately re-deriving each stage's inference-time behaviour
in a parallel `_explore_*` method) be the design chosen? The proposal's answer is no: instead,
extend the `Stage` contract itself so each stage owns both training and inference in one place
(e.g. a `predict()` method alongside `build()`), and have `InstanceSpace.explore_stage_iter()` call
`predict()` directly rather than maintaining a second, separately-written method per stage.

**Why this isn't just F8/F17 again:** F8 already unified PYTHIA's *formula* duplication via a
shared static method (`_weighted_selection`), called from both `_determine_selections` (training)
and `_explore_pythia` (explore) - but those remain two separate methods in two separate files, each
calling into the shared piece. F18 proposes going further: collapsing the two methods themselves,
not just their shared arithmetic, so there is exactly one method body per stage that both `build()`
and `explore()` invoke, eliminating the *possibility* of the two ever drifting apart structurally
(new fields, new branches) even if every individual formula inside them is already deduplicated.

**MATLAB precedent, checked directly rather than assumed - partial, not universal:** `core/
PYTHIA.m` and `core/TRACE.m` already dispatch on a single function for both training and
`isEvalMode`/explore branches (confirmed while researching F8 and F17). `core/PRELIM.m`, `core/
SIFTED.m`, and `core/PILOT.m` do **not** - `InstanceSpace.m`'s `evaluateTestSet` hand-duplicates
PRELIM's bound/normalise logic inline instead of calling `PRELIM()` again, and never calls
`SIFTED()`/`PILOT()` at explore time at all (it reuses stored `model.featsel.idx`/`model.pilot.A`
directly). So this proposal is not "port MATLAB's own architecture" - it goes further than MATLAB
does, generalizing PYTHIA/TRACE's own precedent to every stage on the reasoning that a single-body
design is more maintainable regardless of what MATLAB happens to do.

**Benefits:**
- Removes the entire class of bug F8/F17 exist to fix (build/explore drift) at the structural
  level, not just for the two stages audited so far - no `_explore_*` method can silently omit a
  branch `build()` has, because there would be only one method.
- New stage authors (via the existing `RunBefore`/`RunAfter` plugin mechanism, post-S2) get
  train/predict symmetry "for free" from the base contract, rather than needing to separately
  remember to add an `_explore_*` counterpart on `InstanceSpace` itself.
- Matches PYTHIA/TRACE's own already-proven pattern (single dispatcher function), rather than
  leaving them as the only two stages built that way while everything else stays split.

**Costs/risks:**
- Large blast radius: touches `stages/stage.py`'s `Stage` contract itself and `stage_runner.py`'s
  orchestration, not just individual stage files - every existing stage (`PrelimStage`,
  `SiftedStage`, `PilotStage`, `PythiaStage`, `CloisterStage`, `TraceStage`) would need a `predict()`
  method added, and every corresponding `_explore_*` method on `InstanceSpace` retired.
- PRELIM's `predict()` needs read-access to parameters only known after `build()` fits them
  (`lambda_x`/`mu_x`/`sigma_x`/bounds) - the contract would need a way to expose "the fitted state
  a later `predict()` call needs" that doesn't exist today (today's split sidesteps this by having
  `InstanceSpace` read `Model.prelim.*` directly).
- SIFTED and PILOT's current `_explore_*` counterparts are already thin passthroughs (reusing
  stored `idx`/`A` with no independent logic) - folding them into a `predict()` method is close to
  mechanical, low risk, but also low value compared to PYTHIA/TRACE/PRELIM.
- Multi-session migration effort - this is not a single-sitting change across 6+ stages plus the
  contract and runner; likely needs its own phased sub-items if ever prioritised, not one shot.

**Recommended order if ever prioritised:** PYTHIA/TRACE first (F8 already did the hard part of
separating the shared formula out - folding the remaining method-level split closes the loop it
started), then PRELIM (needs the fitted-state-access design question resolved - the hardest part of
this proposal), then SIFTED/PILOT last (already-trivial passthroughs, safest to defer since there's
little drift risk to eliminate there in the first place).

**Files:** `instancespace/stages/stage.py` (the `Stage[IN, OUT]` contract), `instancespace/
stage_runner.py` (orchestration), every `instancespace/stages/*.py` file, `instancespace/
instance_space.py` (retiring the `_explore_*` family as each stage's `predict()` lands).
**Pathway:** not scoped in detail - this filing is intentionally an architecture proposal, not an
implementation plan. A real pathway would need its own design pass per the "PRELIM's fitted-state
access" open question above before any code is written.
**Test:** not scoped - would follow whatever test pattern is already established per stage
(reference-test parity for existing `build()`/`_explore_*` behaviour, unchanged by the refactor).
**Decision needed:** whether to prioritise this at all, and if so, whether to design the fitted-
state-access mechanism PRELIM's `predict()` would need before or after PYTHIA/TRACE's
lower-risk fold. Not decided - explicitly deferred per how this item was filed.
**Compat:** `[Unknown until scoped]` - a structural rewrite of the stage contract touching every
stage's public surface; blast radius and compatibility tag can't be assessed until the `predict()`
contract itself is designed, which this filing deliberately doesn't do.

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

**Implemented and verified (v1.28).** Re-checked `Ybad`'s construction directly against the
actual GitLab source rather than assuming: PyISpace's `train.py::train_is()` computes
`bad_instances = mode(Ybin * 1, axis=1, keepdims=True)[0] == 0` — the per-instance majority
vote across algorithms in `Ybin`, not `beta` (which is per-*feature*, unrelated) and not a
simple `~Ybin.all(axis=1)` (which would flag an instance as "bad" if even one algorithm fails
it, not the majority). Ported as `PilotStage._bad_instances()` using `scipy.stats.mode` for
exact fidelity to the tie-breaking behaviour of the original. `y_bin` reaches `PilotStage`
via a new `PilotInput.y_bin` field, auto-wired by `stage_runner.py`'s name-based argument
matching from `SiftedOutput.y_bin` (SIFTED already re-exports PRELIM's `y_bin` unchanged) —
no explicit plumbing needed once the field exists, confirmed by reading `run_stage()` rather
than assumed. `adjust_rotation()` and the rotation-application step (`Z`, `A = rot @ A`) are
direct ports of PyISpace's `pilot.py::adjust_rotation()` and its `train.py` call site. Both
tests scoped here are implemented in `tests/test_pilot.py` (pairwise-distance invariance and
the centroid-angle assertion), plus 4 more covering the flag's off/no-bad-instances/
cross-run-reproducibility/missing-`y_bin` cases.

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

### T6 — DAG-resolver edge-case tests — MOOT, superseded by S2 (v1.22)
S2 (implemented) removed the ambiguity-detection/mutating-stage/type-matching resolution
algorithm this item was written to test — there is no longer an "ambiguous ordering" or
"mutating stage" concept anywhere in `stage_runner.py`'s `build_stage_runner()`. Confirmed, not
just theoretical: S2's own pathway said to skip T6 entirely if S2 landed first, and it has.
`RunBefore`/`RunAfter` conflict handling (the one part of this item's original scope that still
exists post-S2) is already covered by `tests/test_stage_runner.py::test_extra_stage_without_
attachment_point_raises` and the RunAfter/RunBefore attachment tests added alongside S2 — no
further work needed here.

### T7 — Consolidate fragmented per-stage test files
**Implemented (roadmap v1.40).** Resolved as two files per stage, not three: `tests/
test_build_<stage>.py` (unchanged, already top-level) and `tests/test_explore_<stage>.py` (a
merge of the former `exploreIS/<stage>/test_<stage>_unit.py` and `..._validation.py`; TRACE's
extra `test_trace_executor_reuse.py` folded in too, a 3-way merge for that stage only). The
`exploreIS/` directory tree was removed entirely — every test file lives flat under `tests/`,
disambiguated by filename prefix instead of directory. Non-stage explore files
(`test_explore_stage_iter.py`, `test_extract_features.py` → `test_explore_extract_features.py`)
and build-only files with no explore-time counterpart (`test_build_cloister.py`,
`test_build_preprocessing.py`, `test_build_filter.py`) followed the same prefix convention.
Multi-stage build-time integration files kept their existing name plus the `test_build_` prefix
(`test_build_pilot_pythia.py`, `test_build_prepro_n_prelim.py`, `test_build_prelim_filter.py`).
`tests/exploreIS/README.md`'s content moved to a new `tests/README.md`, updated for the flat
layout. Full mapping and reasoning in the roadmap's own T7 section.

---

## Summary: decisions requiring your input

**Resolved:** F7 (HDF5 via `h5py`), Q5 (keep permissive feature-order auto-reorder), F9
(Option 1 — extend `explore()` itself, with the "new algorithm" edge case deferred by default),
Q9 (seed default — `0`, not `None`; corrected once the production/backward-compatibility
context was confirmed — see the note at the top of this document), S1 (cross-platform
MATLAB-model loading closed as impractical given no demonstrated need, not impossible —
consequently Q1 is retired in favour of S3, which retires `build_explore_adapter.py` entirely).

**Still open, each with a recommended default stated in place above:** Q7 (API shape — four
idiomatic methods vs. one MATLAB-style dispatch method),
F1 (default classifier once the registry exists — `'svm'` vs. `'knn'`), F8 (ambition level —
shared function vs. extended `Stage` contract, now scoped to TRACE only per S1), T1 (coverage
threshold, can't be set before a baseline exists), T5 (warn vs. fail on fixture staleness), T7
(three-way vs. two-way test-file split). Flag any of these you'd like to override and I'll
update this document and the roadmap accordingly.
