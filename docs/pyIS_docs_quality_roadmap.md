# pyInstanceSpace — Documentation & Quality Roadmap

**Version:** v1.15
**Date:** 2026-07-26
**Scope:** `aoxiangx/pyInstanceSpace` (branch `explore/build-explore-adapter`, tracking `andremun/pyInstanceSpace`)
**Owner:** Andrés (review) / aoxiangx (delivery)

---

## 1. Purpose and scope

`pyInstanceSpace` was forked from the MATLAB `InstanceSpace` toolkit at **v0.3.3** (Feb 2023),
before the ten-phase refactor documented in `isa_refactor_plan_v1_7.pdf` began. MATLAB is
now at **v0.9.0** (133 commits / ~44.6k lines changed since v0.3.3). The Python port's stage
architecture (`preprocessing → prelim → sifted → pilot → pythia → cloister → trace`, an
`InstanceSpace` class, a `build()`/`explore()`/`explore_iter()` API) is already independently
well-engineered — it is **not** a 1:1 port and does not need to become one.

This document intentionally **inverts** the MATLAB refactor's priority order. The MATLAB plan
front-loaded bug fixes and correctness because the code was the risk. Here, the code is in
reasonable shape but the surrounding documentation has not kept pace with either the Python
architecture or MATLAB's current quality bar. So:

- **Phases P1–P5**: documentation and process quality — bring the notebook, README, and
  release process up to the standard set by the MATLAB v0.9.0 repo. Near-term, low-risk.
- **Phase Q**: concrete code-quality ideas transferred *from* MATLAB v0.9.0 back to Python —
  the reverse of the audit above. Near-term, low-risk: each item is additive, contained, and
  doesn't touch the DAG scheduler or stage algorithms.
- **Phases F1–F6**: functionality parity against MATLAB's Phases 4–10 — deferred, slower,
  and each begins with its own audit before any fix is scoped. Long-term. F7–F9 (added in
  v1.4) are the heavier MATLAB-derived ideas that don't fit Phase Q's low-risk bar.
- **Phase R** (added v1.5): ideas from independent third-party ISA implementations
  (PyISpace/PyHard, ITA-ML group, Brazil) — not a MATLAB-vs-Python comparison, a third data
  point. Near-term, low-risk, same additive/contained bar as Phase Q.
- **Phase T** (added v1.6): testing-infrastructure audit — what's actually strong (the
  MATLAB-reference validation harness) versus genuinely thin (no end-to-end integration test,
  no coverage tooling, DAG-resolver edge cases untested), plus the specific tests every
  already-scoped item above (Q2, Q8, F7, F8, R1, R2, S1, S3) needs to be verified, not just built.
- **Phase -1** (added v1.11): a prerequisite, not a phase in sequence — merge the fork
  (`aoxiangx/pyInstanceSpace`) back into the upstream repo (`andremun/pyInstanceSpace`)
  before, or very early alongside, P0. Everything else in this document assumes work
  continues on a single, merged codebase.
- **Phase S** (added v1.14): structural simplification, sequenced before F-phase work
  specifically because it changes *how* F1/F7/F8 should be built, not just what quality bar
  they meet. Removes generality the codebase doesn't currently use (a model-shape detection
  branch with no reachable second case; a DAG auto-resolver for a pipeline whose shape has
  never actually varied), in favour of designs already proven sufficient by the MATLAB
  reference implementation.

Nothing below has been implemented yet; this is the plan, not the change.

**Backward-compatibility tagging (added v1.12):** this is heading into production behind a web
server, so every item below now carries a compact tag:
- **[Additive]** — new capability or docs/tests/tooling only; no existing caller's behaviour
  changes.
- **[Behavior-changing]** (or **[...risk]**/**[...if defaulted wrong]** where the item can be
  done safely with the right default/verification step) — an existing caller could see
  different output after this lands. Each of these states what specifically needs verifying
  before it ships, not just that caution is warranted in the abstract.
- **[Unknown until audit]** — genuinely can't be tagged yet; treat as behavior-changing by
  default until the audit resolves it.

One correction this pass made: Q9's originally-recommended seed default (`None`) was reasoned
on library-hygiene grounds without accounting for production callers — corrected to `0` (see
Q9 below) so introducing the option doesn't silently change existing behaviour.

---

## Phase -1 — Merge the fork back into the upstream repo

**Status: prerequisite, not yet done.** `aoxiangx/pyInstanceSpace`'s `explore/build-explore-
adapter` branch and `andremun/pyInstanceSpace`'s `main` are currently two diverged lines of
development. Every phase in this document (P0 onward) assumes work continues on one merged
codebase — this should happen first, not be left implicit.

**Verified, not assumed: merges cleanly.** Tested directly with a local scratch merge (not
pushed anywhere): `main` — 14 commits independently ahead since the fork point — and the fork
branch — 24 commits ahead on its side — merge with **zero conflicts**. 64 files changed,
~8,000 insertions, ~676 deletions. This is a genuine three-way merge of two diverged lines,
not a fast-forward, but git resolved it automatically without help.

**Recommended: a pull request, not a silent direct merge.** Two ways to do it:
1. **Direct** (works, confirmed clean): add the fork as a remote, fetch the branch, merge into
   `main`, push.
2. **Preferred**: open `github.com/andremun/pyInstanceSpace/compare/main...aoxiangx:
   pyInstanceSpace:explore/build-explore-adapter` and create a PR from there directly — no
   action needed from aoxiangx's side, since the fork is public. Verified `andremun/
   pyInstanceSpace` already has the same `validation-tests.yml` CI as the fork, so this runs
   the test suite against the merge automatically, and it's a natural place to attach the
   already-catalogued follow-up work (S3's planned retirement of `build_explore_adapter.py`,
   F4's audit findings, and everything else in this document) as a visible checklist rather
   than leaving it undiscoverable in a silent merge.

**Checkpoint:** PR merged, CI green, `main` contains the stage architecture and `build()`/
`explore()`/`explore_iter()` API the rest of this roadmap assumes already exists.

---

## 2. Current-state audit findings

Findings from a documentation-focused pass over the `aoxiangx` fork (README.md,
`liveDemoExploreIS.ipynb`, `CLIDocs.txt`, `.github/workflows/`, `LICENSE`) compared against
`andremun/InstanceSpace` at v0.9.0 (README.md, RELEASE_NOTES.md, CITATION.cff, Contents.m).

| # | File | Finding |
|---|---|---|
| 1 | `README.md` — Contact section | Links to `andremun/InstanceSpace`'s issue tracker, not the Python repo's own. |
| 2 | `README.md` — citation block | `Also, if you specifically use this code, please cite as follows: TBD` — unresolved placeholder. |
| 3 | `README.md` — Options section | Describes PYTHIA in terms of MATLAB's Statistics and Machine Learning Toolbox / LIBSVM, and documents `opts.pythia.uselibsvm` as current. The Python code itself already treats `uselibsvm` as a **legacy alias** for `use_grid_search` (`instancespace/data/options.py`) — the README documents the old surface, not the actual one. |
| 4 | Root directory | No `CITATION.cff` (MATLAB has one, with DOI/author/licence metadata). |
| 5 | Root directory | No `RELEASE_NOTES.md` / `CHANGELOG.md` — no versioned record of what changed and why. |
| 6 | `README.md` structure | Missing sections MATLAB's README has: *Repository layout*, *Working with the code* (class walkthrough), *The metadata file*. |
| 7 | `liveDemoExploreIS.ipynb` | 17 cells / 9 markdown headers, reasonable stage-by-stage structure, actively being improved (recent commits added per-stage diagnostics and `explore_iter` notes). Gap is narrative depth — MATLAB's manual explains *why* and *how to read the output* at each stage; the notebook should be audited cell-by-cell against that bar, not rewritten from scratch. |
| 8 | `docs/explore_validation.ipynb` | Exists but its relationship to the main demo notebook (user-facing manual vs. validation artefact) isn't documented anywhere. |
| 9 | `.github/workflows/` | Only `validation-tests.yml`. The `docs-passing` badge in `README.md` has no corresponding CI job — it's either stale or manually maintained. |
| 10 | `LICENSE` | Already PolyForm Noncommercial 1.0.0 — matches MATLAB. No action needed here. |

### 2.1 Dependency security audit (upstream `andremun/pyInstanceSpace`)

Ran `pip-audit` against the locked dependency set (`poetry.lock`, 81 packages). No dangerous
code patterns found in the project's own source (`pickle`, `eval`/`exec`, `os.system`,
`shell=True`, unsafe `yaml.load`, zip-slip, hardcoded secrets, private-key material — all clean).
All findings below are **outdated pinned dependency versions**, not bugs in this project's code.

| Package | Locked | Patched | Reachability in this repo | Advisory summary |
|---|---|---|---|---|
| `pillow` | 12.2.0 | 12.3.0 | Transitive via `matplotlib`; project only *writes* PNGs (`_serialisers.py`), never calls `Image.open()` on untrusted input | Multiple decompression-bomb / heap-overflow issues in font & image parsers (PCF, BDF, GD, TGA, JPEG2000, PDF) |
| `tornado` | 6.5.5 | 6.5.6 / 6.5.7 | Transitive via `ipykernel`/Jupyter (dev-only, not imported by the library itself) | Credential leakage on redirect/proxy reuse; unbounded gzip decompression |
| `click` | 8.1.7 | 8.3.3 | Direct dependency (CLI), but codebase never calls `click.edit()` — the vulnerable function | Command injection in `click.edit()` |
| `jupyter-core` | 5.7.2 | 5.8.1 | Dev-only dependency; issue is Windows-only (`%PROGRAMDATA%` permissions) | Local config-file tampering on shared Windows machines |

**Assessment:** none of these are currently exploitable through this project's own code paths —
each vulnerable function is either unreached or the package is dev/transitive-only. Practical
risk is low today, but worth closing because (a) it's a one-line version bump per package, no
logic changes, and (b) the README states an ambition to power MATILDA's **web** backend, at
which point "not currently reachable" stops being true by default.

**Update, confirmed while verifying the fork-merge PR (§Phase -1):** re-fetched `main` directly
and read its commit history rather than re-running `pip-audit` blind. `main` has picked up 14
Dependabot commits since the fork point — all seven are dependency bumps, and two of them are
exactly the packages flagged above: `pillow` 11.0.0→**12.2.0** and `tornado` 6.4.1→**6.5.5**.
These land at precisely the "Locked" versions in the table — i.e. `main` and the fork already
agree on these two, and both are still short of the patched target (12.3.0 / 6.5.7
respectively). **`click` and `jupyter-core` show zero Dependabot movement** despite being
flagged — direct confirmation of the theory below (security-alerts-only, not full
version-update automation): Dependabot evidently surfaced alerts for pillow/tornado but not for
click/jupyter-core, rather than systematically checking every package against latest.
Remaining P0 scope, updated: bump `pillow` the rest of the way (12.2.0→12.3.0), `tornado`
(6.5.5→6.5.7), and `click`/`jupyter-core` in full (neither has started).

Also checked `.github/workflows/validation-tests.yml`: already scoped to `permissions: contents: read`
(good baseline), no `pull_request_target` or unsanitised script-injection patterns. Gap: no
`.github/dependabot.yml` — the pygments/fonttools/pytest/jinja2/black/pillow/tornado bumps
visible in the commit history came from GitHub's security-only alerts, not a configured
version-update schedule — now directly confirmed, not just inferred, by the click/jupyter-core
gap above.

**[DECISION] Topic:** Future MATILDA web-upload surface restricted to CSV only
**Rationale:** Metadata input is already CSV-only (`from_csv_file`); no format-detection or
multi-type dispatch needs to be built. Restricting uploads to `.csv` at the boundary means the
Pillow image/font-parsing attack surface (§2.1) is never reachable via untrusted uploads — the
"stops being true" caveat above no longer applies, regardless of when web integration is built.
**Alternatives rejected:** Accepting arbitrary file types with server-side content sniffing —
rejected as unnecessary complexity for no functional benefit; this toolkit's only documented
input format is CSV.
**Impact:** Applies to any future upload-handling code (not yet built — no F-phase currently
covers it). Recorded here so the constraint is a documented requirement before that work starts,
not a retrofit. CSV parsing itself still warrants standard hygiene when that work begins (size
limits against memory exhaustion, and CSV-injection awareness — leading `=`/`+`/`-`/`@` in cells —
if any output CSV is likely to be opened in Excel downstream).

---

## 3. Phase plan — documentation & quality (near-term)

### P0 — Dependency security hygiene
**[Additive]** — verify numeric outputs unchanged post-bump before release (dependency patches can occasionally shift floating-point results).
- Bump pinned versions past the audited vulnerabilities. Updated per §2.1's confirmed
  Dependabot history: `pillow` is already at 12.2.0 on `main`, needs the last step to
  ≥12.3.0; `tornado` is already at 6.5.5, needs the last step to ≥6.5.7; `click` (still
  8.1.7) and `jupyter-core` (still 5.7.2) haven't moved at all and need the full bump to
  ≥8.3.3 / ≥5.8.1 respectively.
- Add `.github/dependabot.yml` (pip/poetry ecosystem, weekly) so future version drift is caught automatically rather than manually.
- Optional, low-cost: add a `pip-audit` step to `validation-tests.yml` so a regression is a CI failure, not a future manual re-audit.
- No application code changes — pure lockfile/CI config. Zero-risk, can go first, independent of P1–P5.

### P1 — Baseline hygiene
**[Additive]** — docs only.
- Fix the Contact section to point at the Python repo's own issue tracker.
- Resolve the citation placeholder (either a real citation entry, or defer explicitly to P1's `CITATION.cff`).
- Add `CITATION.cff`, mirroring MATLAB's schema (title, version, date-released, DOI, authors, licence, `repository-code` pointed at the Python repo).
- Rewrite the PYTHIA section of the Options documentation to describe the actual Python implementation and current option names, not the MATLAB/legacy surface.

### P2 — Notebook parity
**[Additive]** — docs/notebook only.
- Cell-by-cell audit of `liveDemoExploreIS.ipynb` against the MATLAB manual: does each stage explain *why* it exists and *how to interpret* its output, or only *how to call it*?
- Confirm the recent per-stage diagnostics work reads as prose, not just printed output.
- Add a short pointer (in the notebook and/or README) distinguishing it from `docs/explore_validation.ipynb`.

### P3 — README structural parity
**[Additive]** — docs only.
- Add *Repository layout*, *Working with the code* (referencing `integration_demo.py` and `example_plugin.py`), and *The metadata file* sections, matching MATLAB's structure.
- Defer an *AI-assisted analysis* section until (if) a Claude Code skill is written for this repo, mirroring `andremun/InstanceSpace`'s `.claude/skills/`.

### P4 — Release notes discipline
**[Additive]** — docs/process only.
- Introduce `RELEASE_NOTES.md`, seeded with a baseline entry describing the current state (stage architecture, `build()`/`explore()`/`explore_iter()`, licence), using MATLAB's section convention: *New functionality* / *Better engineering* / *Bug fixes* / *Licence*.
- Process rule going forward: every version bump gets an entry before merge, not after.

### P5 — Docs CI honesty
**[Additive]** — docs/CI config only.
- Decide whether to wire a real `pdoc` build/deploy job, or replace the `docs-passing` badge with language that doesn't imply CI-gated freshness.

**Checkpoint for P0:** `pip-audit` (or equivalent) against the refreshed lock file reports zero known vulnerabilities; `pytest` suite still passes after the version bumps.

**Checkpoint for P1–P5:** existing `pytest` suite still passes; notebook runs end-to-end without error; every badge and cross-reference in `README.md` resolves to something true.

---

## 4. Phase plan — quality ideas transferred from MATLAB (near-term, low-risk)

The reverse of §6.1's audit: concrete MATLAB v0.9.0 behaviours worth adopting in Python.
Filtered specifically for low risk — each item below is additive or narrowly contained, doesn't
touch the DAG scheduler (`stage_builder.py`/`stage_runner.py`) or change any stage's algorithm,
and is independently testable. Heavier ideas that didn't clear that bar are in §6 as F7–F9
instead.

### Q1 — RETIRED, superseded by S3
**Originally:** fix `build_explore_adapter.py`'s missing polynomial-kernel branch (it handled
`"rbf"`/`"linear"` only, raising `NotImplementedError` for poly-kernel models). **Superseded,
not fixed:** with S1 (native scikit-learn objects in `explore()`) and the closed decision that
cross-platform MATLAB-model loading isn't worth building, the adapter this bug lived in has no
remaining reason to exist at all — see S3, which retires the whole file rather than patching
one branch of it. Kept this heading as a pointer rather than deleting it outright, since other
parts of this document (and the drafted MATLAB issue batches) reference "Q1" by name.

### Q2 — Out-of-distribution warning in `explore()`
**[Additive]** — adds a logged warning only; no change to any returned value.
MATLAB warns when >5% of test instances get clipped to training bounds. Add the equivalent
check to `_explore_prelim()` — same threshold, same spirit, no new option needed.

### Q3 — Standardise console output + add a `general.verbose`-style option
**[Additive]** — verify nothing downstream parses today's stdout output before switching `print`→`logger`.
121 unconditional `print()` calls across `instancespace/stages/*.py` today, no verbose gate
exists anywhere in `InstanceSpaceOptions` (confirmed: no `general`/`verbose` field). `loguru` is
already a dependency but only used in 10 places. Mirrors a lesson MATLAB already learned in its
own refactor (RELEASE_NOTES.md: "console output standardised to a consistent `[STAGE] message`
format... detailed per-trial/per-iteration output gated behind `opts.general.verbose`"). Proposed:
adopt the same `[STAGE] message` convention, add `general.verbose` to `InstanceSpaceOptions`,
gate per-trial/per-iteration detail behind it.

### Q4 — Recursive, compact options printer
**[Additive]** — changes only a debug-print helper's output format, not a return value.
`instance_space_from_files()` prints only top-level `InstanceSpaceOptions` fields (nested
dataclasses print as raw reprs, e.g. `parallel: ParallelOptions(flag=True, n_cores=4)`). Mirror
MATLAB's `InstanceSpace.printOptions()`/`formatOptionValue()` — recurse into nested dataclasses,
one leaf setting per line.

### Q5 — Feature-order handling (confirmed: keep permissive)
**[Additive]** — no behaviour change; this confirms current behaviour as intentional and permanent.
**Decided:** Python's auto-reorder-by-name in `_extract_features()` is the intended, permanent
behaviour — not an accidental divergence from MATLAB's stricter `featureOrderMismatch` error.
Document it explicitly in `explore()`'s docstring and add the regression test below.

### Q6 — Reuse thread/process pools across staged calls
**[Additive if implemented correctly]** — this is a concurrency change, not just a resource optimisation; verify computed output is bit-identical before/after, not just "faster."
MATLAB's `ensurePool()` opens a parallel pool once and reuses it across successive staged
`build()` calls in the same session, only tearing it down if it opened it. Python currently
creates a fresh `ThreadPoolExecutor`/joblib backend per stage call. Pure resource-management
change — no correctness implications, easy to test (assert no new pool created on a second
`run_stage()` call).

**Interaction with F7 (added v1.18) — undocumented until now, must be handled in Q6's own
implementation:** a `ThreadPoolExecutor` is not picklable (`threading.Lock`/`Thread` objects
raise `TypeError` from `pickle.dumps`). If Q6's pool-holder attribute (e.g. `self._executor`)
sits on the same object F7's `save()` pickles, this produces a scenario-dependent failure: save
crashes outright if a pool is live (e.g. a session that ran TRACE, then saved, without an
intervening `close()`), or succeeds only by caller discipline (remembering to `close()` first)
otherwise — and forgetting once surfaces as an opaque threading-internals traceback, not a
domain-relevant error. The fix belongs to Q6: exclude the pool from pickled state via
`__getstate__`/`__setstate__` (or `__reduce__`) so `save()` never attempts to serialise it
regardless of caller discipline, and `load()` always comes back with the pool attribute unset —
consistent with Q6's own "created lazily on first use" design, so the next `run_stage()` call
after a load simply recreates the pool from scratch. This is not a regression: OS threads from a
previous process can't meaningfully survive a save/load round-trip anyway, and MATLAB's own
`gcp`/`ensurePool()` pool handles are session-local too — `.mat` save/load never attempted to
serialise a parallel pool either. Net effect on Q6's own open decision (explicit `close()` vs
`__del__`): keep `close()` for live-session resource cleanup, but treat the pickle-exclusion as
the actual correctness mechanism for the save/load path, not `close()` discipline.

### Q7 — Add `plot()` convenience methods
**[Additive]** — new methods only; nothing existing calls them yet.
Mirror MATLAB's `InstanceSpace.plot('sources' | 'portfolio' | 'good' | 'footprint', algoIdx)` —
thin matplotlib wrappers around `model.pilot.z` and friends. Additive only, no pipeline logic
touched. Complements P2 (notebook parity): a `plot()` method means the notebook needs less
inline matplotlib boilerplate to demonstrate the same views MATLAB's manual shows.

### Q8 — Regression test for stage-rerun invalidation (verification, not yet a fix)
**[Additive]** — this is a test. If it reveals a real fix is needed, that fix (in `stage_runner.py`) inherits its own **[Behavior-changing]** tag — don't assume it's free just because the test itself is.
§6.1 flagged that Python's `_rollback_to_schedule_index()` invalidates by schedule-wave position
rather than by real dependency (MATLAB's `invalidateDownstream()` BFS). Write a test: build,
re-run `cloister` only via `run_stage()`, and check whether `pythia`'s output is unnecessarily
marked stale. If the test confirms over-invalidation, promote the fix to F-phase work (it touches
`stage_runner.py`, so it doesn't clear this phase's low-risk bar) — see the note on F4 in §6.

**Sequencing gap with S2 (added v1.19, sharpened v1.20) — related to, but not identical in
shape to, the already-recorded S2→T6 sequencing:** verified directly in `stage_runner.py:256-267`
— `_rollback_to_schedule_index()` invalidates by iterating `self._stage_order[index+1:]`, i.e.
it operates directly on the wave-grouped schedule list. S2 (§5) explicitly removes "wave
computation" as part of replacing DAG auto-resolution with an explicit stage order. Running Q8
before S2 doesn't just risk a stale test — it risks **implementing a fix twice**: if Q8's test
confirms over-invalidation, the promoted F-phase fix (per Q8's own pathway, "replacing
schedule-index comparison with a real dependency-graph walk") would be written against the
wave-grouped `_stage_order`, a data structure S2 then deletes — S2 would have to re-derive the
same dependency-graph walk against its own new structure regardless, making the first
implementation wasted work, not just a wasted test.

**Where this differs from T6, precisely:** T6 tests the *ambiguity-detection/resolution
algorithm itself* — S2 deletes that algorithm outright, so post-S2 there may be no remaining
subject matter at all, which is why S2's own pathway says "or skip T6 entirely." Q8 tests a
*behavioral property* — rerunning one stage must not wrongly invalidate unrelated downstream
stages — that still has to hold after S2 lands; S2 changes how the schedule is represented, not
whether correct invalidation matters. S2's own checkpoint ("run the full 7-stage pipeline before
and after, assert identical output") only covers a full initial run, not partial-rerun
invalidation, so it doesn't accidentally subsume Q8 either. **Net: sequence Q8 after S2 (same as
T6), but unlike T6, Q8 is not at risk of becoming pointless — it only needs to target whatever
`_rollback_to_schedule_index`-equivalent S2 leaves behind, not be abandoned.**

### Q9 — Centralise RNG seeding via a `general.seed` option
**[Behavior-changing if defaulted wrong — corrected below]** Every current build/explore call is
*implicitly* deterministic (the hardcoded `0` everywhere means identical input always produces
identical output). This document originally recommended defaulting the new option to `None`
(non-deterministic unless the caller opts in) on library-hygiene grounds — that reasoning didn't
account for production callers, and this is going into a production web service. **Corrected
default: `0`**, not `None` — exactly matching today's hardcoded value everywhere, so introducing
the option changes nothing for any existing caller, production or otherwise. `None` remains
available as an explicit opt-in for genuinely non-deterministic runs, just isn't the default.
No `seed` field exists anywhere in `InstanceSpaceOptions` — verified directly, not inferred.
Randomness is still seeded, but as the literal `0` hardcoded in at least 8 separate call sites
across `pilot.py`, `sifted.py` (×3), `prelim.py`, and `pythia.py` (×4), mixing two disconnected
mechanisms (`np.random.default_rng(seed=0)` in some places, scikit-learn's `random_state=0` in
others). Every run is reproducible, but not *configurably* so — there's no way to run
independent replications. MATLAB has a documented `general.seed` (default 42), threaded through
via a single `rng(opts.general.seed, 'twister')` call, and goes further: PYTHIA.m deliberately
reseeds per-fold/per-trial (`rng(opts.seed + i, ...)`) specifically because parallel workers
don't inherit the client's RNG stream and candidates being compared need "common random
numbers" — this reasoning is written directly into MATLAB's own comments. This is the same
pre-refactor pattern MATLAB's own history eliminated ("every previously-hardcoded magic number
is now a documented `opts` field" — RELEASE_NOTES.md).

**Proposed change:** add `general.seed` to `InstanceSpaceOptions` with **default `0`**, thread it
through to replace every hardcoded `0`. When doing so, follow MATLAB's per-fold/per-trial
reseeding discipline rather than reusing one naive global seed everywhere — scikit-learn's own
parallel backends have the same "workers don't inherit RNG" problem MATLAB already solved, so a
naive single seed won't compose correctly once real parallelism is added.

### Q10 — Add `SECURITY.md` and `CONTRIBUTING.md`
**[Additive]** — docs only.
Verified: neither file exists. Given the dependency audit (§2.1) and the CSV-only upload
[DECISION] already made, a short `SECURITY.md` (how to report a vulnerability) is a natural,
cheap next step — especially since the README states an ambition to become a web backend.
`pyproject.toml` lists nine historical student authors — real contributor turnover, now
essentially solo-maintained — so a short `CONTRIBUTING.md` (mostly pointing at the dev-setup
steps already in `README.md`, plus test-running and code-style expectations) would lower the
bar for the next contributor. Both are small, additive, no code touched.

### Q11 — Remove or enable the dead commented-out CI step
**[Additive]** — CI config only. Note: enabling the dormant lint/format check may surface pre-existing violations elsewhere in the codebase; fix those as part of this change, don't treat them as separate scope creep.
Found while cross-checking both repos against Wilson et al.'s "Best Practices for Scientific
Computing" (PLOS Biology, 2014) — practice 2h ("do not comment and uncomment sections of code to
control a program's behavior") is violated directly in this repo's own CI config.
`.github/workflows/validation-tests.yml` has a formatting/lint/type-check step
(`black`/`ruff`/`mypy` via `poe --fix test`) sitting commented out rather than either enabled or
removed. One-line decision, not a design question: either uncomment it (folding it in alongside
whatever T4 lands as `poe test`'s real content) or delete the dead lines outright — leaving
commented-out CI steps in place is exactly the anti-pattern the practice above names.

**Checkpoint for Phase Q:** existing `pytest` suite passes (Q1 is retired — see S3's checkpoint
instead); `pytest` covers the new out-of-distribution warning (Q2), the feature-order
regression test (Q5), and the new seed option (Q9) producing identical output across repeated
runs with the same seed and different output across different seeds; no change to any stage's
numerical output for the reference dataset when run with the same seed as before.

---

## 5. Phase S — structural simplification (before F-phase)

Two candidates surfaced during an architecture discussion, not from a code-pattern audit like
Q/R — both remove generality the codebase carries but doesn't currently use, in favour of
designs the MATLAB reference implementation already proves sufficient. Sequenced *before*
F-phase specifically because S1 changes how much new code F1/F7/F8 actually need.

### S1 — Collapse model-shape detection to the one reachable path
**[Behavior-changing risk — but see native-object recommendation below, which changes the risk shape]**
`instance_space.py`'s `_ensure_explore_model()` branches on whether `self._model.pythia.svm[0]`
has an `.alphas` attribute — real scikit-learn `SVC` objects don't, the flattened
("MATLAB-artifact") shape does. Verified directly: there is no `Model.load()` (F7 gap) and no
MATLAB-`.mat`-into-Python loader anywhere in this repo, so the "already flattened" branch is not
reachable by any real, documented, working path today — the only place it's exercised is
`test_pythia_validation.py`, which bypasses this method entirely via `Mock(spec=PythiaOut)` and
calls `_explore_pythia` directly with a hand-built `SimpleNamespace`.

**Recommended design, not just a branch removal:** go further than collapsing the detection —
have `explore()` operate on live scikit-learn objects natively (`.predict()`/`.predict_proba()`)
instead of a hand-rolled recomputation of the SVM decision function from flattened parameters.
Scikit-learn's Estimator API already gives every classifier type the same calling convention,
which is what F1's classifier registry needs on the *build* side anyway — this makes the
*explore* side of F1 close to free, and removes the reimplemented-SVM-math half of F8 outright
(there's no second implementation left to reconcile with the first). See F1/F8 below, revised
accordingly.

**Superseded by F7's pickle decision:** this section originally argued for keeping
`adapt_for_explore()`/the flattened shape alive as F7's persistence format once it stopped
being `explore()`'s primary interface. That's no longer the plan — F7's persistence format was
separately decided as signed `pickle`/`joblib` (see F7 below), under which `SVC` objects
round-trip natively with no flattening step at all. That leaves no remaining caller for the
adapter in any form, which is what S3 below acts on: full deletion, not a narrowing of scope.

**[DECISION] Topic:** is the "accept an externally-produced model" branch a dead path or a
planted seam? **Closed.** Cross-platform MATLAB-model loading is impractical, not attempted —
not "impossible" in principle, but not worth building given the actual cost: MATLAB's PYTHIA
registry has six classifier types, each needing its own converter (the SVM one already required
non-trivial kernel-scale and Platt-parameter handling that doesn't generalise), and at least one
type (decision trees — fitted state is splitting rules and node structure, not a few named
arrays) has no clean flattened representation at all. Six converters, several without a good
solution, for a capability with no demonstrated need, isn't worth it. Recorded as "impractical
given no demonstrated need" rather than "impossible" specifically so this can be reopened later
if a real use case ever appears — this is a closed decision under current circumstances, not a
permanent architectural wall.

**Test impact:** `test_pythia_validation.py` bypasses `_ensure_explore_model()` via mocking, so
it is not directly affected by this change — but re-verify it explicitly rather than assuming,
since it's the one place today that exercises the artifact shape at all.

### S2 — Replace DAG auto-resolution with explicit stage order + prerequisites
**Implemented and verified (v1.22)** on `v0.9.0/development-branch-QSF`. Audit while
implementing found a real gap this section's design didn't account for: `InstanceSpace`'s
`stages` constructor parameter is a documented extension point (`example_plugin.py`,
referenced by name in the README), not just internal plumbing for the fixed 7 — the
auto-resolver was also handling arbitrary plugin-stage placement by type-matching, which a
naively hardcoded order would have silently broken. Resolved (discussed and decided directly,
not guessed): the built-in 7 stages get the hardcoded explicit order below; any additional
stage must declare an explicit `RunBefore[X]`/`RunAfter[X]` field instead of relying on
input/output type-matching — `example_plugin.py` updated to add one (`RunAfter[PythiaStage]`).
Verified the resulting schedule is byte-identical to the pre-S2 auto-resolved one for the real
7-stage pipeline (`[[Preprocessing], [Prelim], [Sifted], [Pilot], [Pythia, Cloister],
[Trace]]`), and that `integration_demo.py`/`example_plugin.py` both still run end-to-end.
Two latent bugs surfaced and fixed along the way, both from `RunBefore`/`RunAfter` never
having a real caller before this: their `TypeVar` bound in `stages/stage.py` was `type[Stage[
Any, Any]]` where it needed to be `Stage[Any, Any]` for `RunAfter[SomeStage]` to type-check;
and both `StageRunner.run_stage()` and `_check_stage_order_is_runnable()` used
`isinstance(x, type)` to detect these fields, which is `False` for a subscripted generic like
`RunAfter[SomeStage]` (a `_GenericAlias`, not a plain class) — `get_origin()` is required
instead. Full test suite: 265 passed, 0 failed. Following a direct request, `stage_builder.py`
was subsequently folded entirely into `stage_runner.py` (as `build_stage_runner()` plus two
private helpers) and deleted — once reduced to attaching extras to a caller-supplied base
order, it had shrunk to one call site with no remaining reason to be a separate module; see
§10 v1.22 for the line-count accounting. `tests/test_stage_builder_runner.py` renamed to
`tests/test_stage_runner.py` and rewritten against the new function-based API.

**[Additive if done carefully — see the type-safety tradeoff below, which is the real cost]**
`stage_builder.py` (414 lines) infers stage dependencies from NamedTuple field-name/type
matching, with cycle detection, ambiguous-ordering errors, mutating-stage handling, and
`RunBefore`/`RunAfter` overrides. MATLAB's `InstanceSpace.m` does the equivalent with a
~10-line hardcoded `StageOrder` cell array and `StagePrereq` map. This pipeline's shape
(prelim → sifted → pilot → {cloister, pythia} → trace) has been stable across everything
read in this whole project — no stage insertions or reorderings in evidence, and even F2/F5
extend existing stages' internals rather than proposing new ones.

**Proposed change:** explicit `StageOrder`-equivalent list + explicit prerequisite mapping,
still typed against each stage's declared input/output NamedTuples so `mypy --strict` can
verify the *literal* prerequisites as written — just not *auto-infer* them. This keeps most of
the type-safety benefit the current design provides while dropping the resolution algorithm
and the edge cases (ambiguous-ordering detection, mutating-stage special-casing, wave
computation — the last of which Q6/T6 already found isn't even used for real parallelism)
that nothing currently exercises.

**Real cost, not a formality:** losing auto-*inference* is losing something — a stage
declaring the wrong input type today gets caught by the resolver at typecheck-adjacent time
(via the NamedTuple matching); a hand-written prerequisite map, MATLAB's way, only fails at
runtime if someone gets it wrong. This is a genuine tradeoff to make consciously, not an
incidental side effect to discover later.

**Sequencing:** do this *before* T6 (DAG-resolver edge-case tests), or not at all — there's no
point writing tests for an ambiguity-detection algorithm about to be deleted. **Also before Q8
(added v1.19, distinguished from T6 in v1.20)**: Q8's regression test (and the diagnosis its own
pathway states) targets `_rollback_to_schedule_index()`'s schedule-wave-position invalidation —
exactly the "wave computation" this item removes. Unlike T6, though, Q8 doesn't risk becoming
pointless here — the invalidation *property* Q8 checks still needs to hold post-S2, only the
*data structure* it's implemented against changes. The risk with Q8 specifically is wasted
*implementation*, not wasted *test-writing*: if Q8's test is run and fixed before S2, that fix
gets written against the wave-grouped structure this item deletes, and S2 ends up re-deriving
the same dependency-graph walk against its own new structure anyway. See Q8's entry (§4) for the
full reasoning; recorded there and here since neither phase heading would surface it to someone
reading only the other one.

### S3 — Retire `build_explore_adapter.py` entirely
**[Additive]** — deleting code nothing calls once S1 lands and cross-platform loading is
closed; the risk here is entirely in confirming that precondition, not in the deletion itself.
Formerly tracked as Q1 ("fix the missing polynomial-kernel branch"). Superseded, not fixed:
once S1 makes `explore()` operate on native `SVC` objects directly, and cross-platform
MATLAB-model loading is closed as impractical (see S1's decision above), there is no remaining
caller for `adapt_for_explore()`/`_svc_to_artifact()` at all — not a Python-build consumer (S1
removed the need), not a MATLAB-model consumer (never had one reachable in practice, per S1's
verification, and now formally not being built).

**Pathway:**
1. Confirm S1 has landed and the model-shape branching is gone before starting this — this is
   a consequence of S1, not independent work.
2. Delete `build_explore_adapter.py` (`adapt_for_explore`, `_svc_to_artifact`) entirely.
3. Delete or repurpose `test_adapter.py` — its `test_unsupported_kernel_raises` test (the one
   that confirmed the poly-kernel gap fails loudly rather than silently) has nothing left to
   test once the function it's testing is gone.
4. Grep the whole repo for any remaining import of `build_explore_adapter` before considering
   this done — confirm zero, not just the call sites already known about.
**Test:** the existing full test suite passing with the module gone *is* the test — there's no
new behaviour to verify, only an absence to confirm.

**Checkpoint for Phase S:** S1 — `test_pythia_validation.py` still passes unmodified;
`_ensure_explore_model()`'s branching is gone, not just simplified. S2 — the same 7-stage
pipeline resolves to the identical execution order as before; `mypy --strict` still passes
against the explicit prerequisite declarations. S3 — `build_explore_adapter.py` no longer
exists; full test suite passes without it.

---

## 6. Phase plan — functionality parity (long-term, deferred)

These map loosely to MATLAB's Phases 4–10 but are **not scoped yet** — each starts with its
own audit (read the relevant `stages/*.py` + tests) before any specific fix is committed to.
Order is a starting suggestion, not a commitment.

### 6.0 Consolidated execution order — remaining Q/S/F items (added v1.21)

Q1–Q5, Q7, Q9–Q11, S1/S3, and — as of v1.22 — S2 and F1 and F6 are already implemented. What
follows orders everything still pending — Q6, Q8, and every remaining F item — by actual
dependency, not just by letter. Compiled from every cross-item finding recorded in this document
(v1.17–v1.20) plus two dependencies already stated in each item's own pathway that hadn't been
pulled into one place before: F5's hard block on F2, and F9's shared-extraction pattern mirroring
F8's.

**Hard dependencies (violating these means redoing real work, not just resequencing):**
- **Q8 → after S2** — Q8's fix targets `_rollback_to_schedule_index()`'s wave-grouped
  `_stage_order`, which S2 deletes; fixing Q8 first means implementing the same dependency-graph
  walk twice (§4 Q8, §5 S2).
- **Q8 → after T2** (Phase T, not Q/S/F, flagged because it's a real blocker regardless) — Q8's
  own pathway requires a real end-to-end `build()` fixture that doesn't exist yet.
- **F5 → after F2** — F5's pathway states this outright: "genuinely blocked on F2 landing
  first — no further detail useful until then."
- **F1, F7, F8 → after S1** — already satisfied; all three are unblocked now.

**Soft preferences (no correctness risk, but reduce rework or converging conventions
independently):**
- **Q6 after S2** — no dependency, but both touch `stage_runner.py`, which S2 substantially
  rewrites; building Q6's pool cache against the post-S2 structure avoids redoing it.
- **Q6 and F7, whichever lands second, must add the pickle-exclusion check** (§4 Q6) — no order
  requirement between the two, just a shared checklist item neither should skip.
- **F8 before F9** — not blocking, but F9's own pathway extracts `PrelimStage`'s shared logic
  explicitly "to serve F8's goal at the same time," and F8 has its own open ambition-level
  decision (lighter shared-function vs. fuller `Stage` contract extension) that determines what
  that extraction pattern looks like in this codebase. Deciding it once via F8 first means F9
  mirrors an established pattern instead of the two converging on one independently.
- **Within F2:** land R1's 2D rotation canonicalisation (Phase R, tangential to Q/S/F) before
  F2's `dims=3` work — already F2's own recommended default.

**No dependency either way, ready any time:** F3's *audit* (its fix stays unscoped until the
audit runs, but nothing blocks running the audit itself).

**Recommended order:**

| Order | Item | Why here |
|---|---|---|
| 1 | ~~S2~~ | **Done (v1.22).** Unblocked Q8, de-risked Q6. |
| 2 | ~~F1~~ | **Done (v1.22).** Fully unblocked, additive at default. |
| 3 | ~~F6~~ | **Done (v1.22).** Trivial, mechanical. |
| 4 | Q6 | Targets S2's final `stage_runner.py` (now `build_stage_runner()`); build the pickle-exclusion in from the start |
| 5 | F7 | S1 already done; if Q6 landed, its round-trip tests exercise the pool-exclusion case directly |
| 6 | Q8 | S2 done (no wasted implementation) — confirm T2 exists first or this stays blocked regardless |
| 7 | F8 | S1 already resolved the PYTHIA half; decide lighter-vs-fuller here; behavior-changing — full `tests/matlab_reference` suite before/after |
| 8 | F9 | Mirrors F8's just-decided extraction pattern for `PrelimStage`; fully additive |
| 9 | F2 | Independent but higher-risk (bit-for-bit verification burden) — do with full attention once lower-risk items are clear; land R1 first internally |
| 10 | F5 | Direct consumer of F2, natural next step |
| — | F3 (audit) | No dependency — run whenever, ideally early, so its fix scope stops being unknown sooner rather than later |

F4 doesn't appear above — it's already "audited," not an actionable item; F7/F8/F9 are its
concrete derivatives and are already in the table.

| Phase | Maps to MATLAB | Focus | Status | Compat |
|---|---|---|---|---|
| F1 | Phase 4 | PYTHIA classifier registry — confirm whether `stages/pythia.py` supports a pluggable classifier set or is fixed | **Implemented and verified (v1.22)** — training-side registry (`instancespace/utils/get_classifier_fcn.py`) dispatches to `svm`/`knn`/`tree`/`nb`/`linear`/`ensemble`; explore-side already handled by S1. Only `svm` is tuned via the existing `C`/`gamma` search — the other five fit with scikit-learn's own defaults, not a MATLAB-verified tuning range (no MATLAB reference exists for them). `PythiaOutput.svm`'s type widened to `list[ClassifierMixin]` (field name kept for backward compatibility). Full suite + 18 new dedicated tests (registry unit tests + one per registered classifier trained end-to-end) all pass. | **[Additive at default]** — new `classifier` option defaults to `'svm'`, matching today's only behaviour verified via the existing MATLAB-reference tests unchanged. New registry entries themselves are new production surface — validated by dedicated tests, not just "it runs," but without MATLAB-verified hyperparameter tuning for the five non-`svm` entries; flagged in code and docs, not assumed. |
| F2 | Phase 5 | PILOT 3D / viewpoint optimisation parity in `stages/pilot.py` | Not started | **[Behavior-changing risk]** — generalising the 2D-specific solver to n-dims can shift 2D output even at `dims=2` if not done carefully (different array shapes can trigger different BLAS code paths). Verify bit-for-bit or tolerance-verified identical 2D output before shipping — this touches existing code, not just adding an independent new path. |
| F3 | Phase 6 | SIFTED promotion refinements | Not started | **[Unknown until audit]** — F3's own pathway starts with "audit first" for exactly this reason. Treat any fix the audit finds as **[Behavior-changing]** by default until proven otherwise, since it touches SIFTED's core computation. |
| F4 | Phases 7–8 | `InstanceSpace` class & `build`/`explore` robustness | **Audited (v1.3)** — see §6.1 for findings; Q8 (§4) verifies one open question before F4's invalidation-fix work is scoped | — (audit only; see F7/F8/F9 for the actionable, taggable derivatives) |
| F5 | Phase 9 | Output consolidation / 3D visualisation parity (MATLAB's `scriptpng.m`) | Not started | **[Additive]** — new rendering paths; doesn't change any existing 2D output function. |
| F6 | Phase 10 | Namespace & per-file licence headers — licence itself already matches MATLAB | **Implemented (v1.22)** — `SPDX-License-Identifier: LicenseRef-PolyForm-Noncommercial-1.0.0` + copyright header added to all 27 `instancespace/**/*.py` files. | **[Additive]** — comments only. |
| F7 | — | Model save/load round-trip (`Model.save()`/`InstanceSpace.load()`), matching MATLAB's persistence | **Format revised: signed `pickle`/`joblib`, with signing optional (`secret_key: bytes \| None`)** — see design constraint below (supersedes the earlier HDF5-via-`h5py` decision and the earlier unconditional-signing decision) | **[Additive]** — brand-new capability; nothing existing depends on it. |
| F8 | — | Unify `explore()` with build-time stage code (predict-mode dispatch on `PythiaStage`/`TraceStage`, matching MATLAB calling the same `PYTHIA()`/`TRACE()` in both modes) | **Narrowed by S1**: the PYTHIA half is resolved as a side effect of calling native `.predict_proba()` instead of reimplementing SVM math — nothing left there to reconcile. Remaining scope is TRACE only (footprint/alpha-shape membership testing is a genuinely different computation S1's insight doesn't extend to) | **[Behavior-changing risk]** — this refactors existing, working code. The full `tests/matlab_reference/` validation suite must pass identically before/after; treat any tolerance-threshold change during this work as a red flag to investigate, not a "close enough" adjustment. |
| F9 | — | Expand `explore()` to full evaluation scope: algorithm reconciliation + ground-truth performance metrics, matching MATLAB's `evaluateTestSet` | **Decided: extend `explore()` itself** (silent branch on whether ground truth is present) — see companion implementation-pathways document for the full pathway | **[Additive]** — new fields default to `None`; existing feature-only callers see no change. Add explicit test coverage for the "no ground truth present" path specifically, to lock this in rather than assume it. |

**F7 design constraint — revised:** the original "no `pickle`" constraint assumed model files
could arrive from an untrusted source, following §2.1's CSV-only upload [DECISION]. Revisited
given the confirmed production threat model: on the web platform, models are produced by the
system and downloaded by users — never re-uploaded. There is no code path where `load()` is
called on anything other than a file the system itself wrote, *as long as this stays true and
is enforced, not merely assumed* — hence the signing requirement below, which makes that
enforcement real rather than a claim about the future that could quietly become false. A second
usage mode — local/desktop development, with no secrets-manager available — is deliberately
allowed to skip signing entirely; see the [DECISION] below for why that doesn't reopen the risk
the signing requirement exists to close.

**[DECISION] Topic:** F7 persistence format — signed `pickle`/`joblib`, signing optional via
`secret_key: bytes | None` (supersedes the v1.9 HDF5-via-`h5py` decision, and revises the v1.14
unconditional-signing decision to add a second mode)
**Rationale:** the "never re-uploaded" threat model removes pickle's core objection (arbitrary
code execution from *untrusted* input) — but rather than rely on that assumption holding
forever across every future change to this codebase, add an HMAC signature: sign the serialised
bytes with a server-held secret at `save()` time, verify the signature *before* ever
unpickling at `load()` time; refuse to deserialise on mismatch. This converts "we're confident
this file is trustworthy" from an architectural assumption into something checked at the
moment it matters — if the never-re-upload assumption is ever accidentally violated later (a
debug endpoint, a path parameter, a storage misconfiguration), the signature check is what
actually stops it from mattering, not a design note nobody re-reads. Also resolves a problem
the HDF5 approach never solved: `DecisionTreeClassifier`/ensemble estimators (needed once F1
adds them to the registry) don't have a small set of named arrays to flatten the way `SVC`
does — pickle round-trips them natively, no custom serialiser required.

A second, narrower use case doesn't fit the server threat model at all: local/desktop
development, where a researcher saves and loads a model on their own machine and has no
server-managed secret to sign with. Rather than force that caller to invent a throwaway key (or
block desktop use entirely), `secret_key` defaults to `None`, in which case `save()` writes no
signature and `load()` performs no verification — the risk is identical to running any other
file the caller already possesses and trusts, the same caveat every unsigned `pickle`/`joblib`
user already lives with today. This is a genuine second reachable mode serving a real,
distinct caller (unlike the model-shape branch S1 removed, which had no second reachable
caller) — not an accidental generality regression.

The one new risk this introduces is a **downgrade attack**: a file `save()`-d *with* a
`secret_key` must never become loadable *without* one just because a caller omits the key at
`load()` time — that would silently defeat the entire signing mechanism. This is closed
structurally, not by convention: `load()` must raise if a `.sig` file (or equivalent signed
marker) is present but no `secret_key` was given, and must equally raise if a `secret_key` was
given but no `.sig` marker exists. Only "signed key + signed file" and "no key + unsigned file"
are valid, verified combinations; the two mismatched combinations are both refused.
**Alternatives rejected:** HDF5 via `h5py` (the previous decision) — still viable, still safe,
but adds a new dependency and requires hand-written flattening for every estimator type,
including ones (trees, ensembles) that don't flatten cleanly; no longer justified once the
threat model is confirmed to make pickle safe. `skops` (a library built specifically for
pickle-free sklearn persistence) — worth a look if the signing approach ever proves
insufficient, but not needed given the signing approach already closes the real risk.
Unconditional signing (the v1.14 decision) — rejected as the sole mode because it has no answer
for the desktop/no-secrets-manager caller other than "invent a key nobody manages," which is
security theatre, not a control.
**Impact:** no new third-party dependency (`hmac`/`hashlib` are stdlib) — a smaller dependency
footprint than the HDF5 option, not just a safer-by-assumption one. **Non-negotiable
implementation requirements:** (1) every server code path that calls `load()` must be audited to
guarantee it always passes `secret_key` and never receives a user-supplied path or file — this
is the one place the server-side design's safety actually lives, and it needs to be a checked
invariant (e.g. a path-allowlist or a storage-layer guarantee), not an assumption held only in
this document; (2) `load()` must refuse both mismatched signed/unsigned combinations described
above — this is what prevents the desktop mode's existence from becoming a bypass for the server
mode. Once S1 lands, `SVC` objects round-trip through pickle exactly as trained — no
adapter/flattening step needed at load time at all, since `explore()` will already be operating
on native objects. **Depends on Q6 handling its own pickle-exclusion (added v1.18):** if Q6
lands first, its pool-holder attribute must already be excluded from pickled state (see Q6's
entry above) — otherwise F7's round-trip test can fail intermittently depending on whether a
pool-using stage ran before `save()`, which would look like an F7 bug but is actually an
un-isolated Q6 gap.

### 6.1 F4 audit findings — class architecture deep dive

Line-by-line comparison of `instancespace/instance_space.py` (+ `stage_builder.py`,
`stage_runner.py`, `build_explore_adapter.py`) against MATLAB's `InstanceSpace.m` (1030 lines).

**Architecture differences (neither is "wrong", both verified):**
- MATLAB uses a hand-written `StageOrder`/`StagePrereq` dependency map; Python auto-infers the
  DAG from each stage's declared `_inputs()`/`_outputs()` NamedTuples, with its own cycle/
  ambiguity detection — more rigorous, more machinery.
- MATLAB's `invalidateDownstream()` does a BFS over the real dependency graph when an earlier
  stage is re-run; Python's `_rollback_to_schedule_index()` invalidates by schedule-wave
  position instead. Worth a concrete test: rerun `cloister` only and confirm `pythia` isn't
  forced to redo unnecessarily.
- Parallelism is intra-stage in both (MATLAB `parfor` + shared `parpool`; Python
  `ThreadPoolExecutor` in TRACE, scikit-learn `n_jobs` in PYTHIA/SIFTED) — functional parity.
  `StageRunner.run_many_stages_parallel` is `NotImplementedError` but unused dead code, not a
  regression, since neither implementation runs independent stages concurrently.

**Concrete gaps (verified against source):**
1. **No save/load round-trip.** `Model` has `save_to_csv/for_web/graphs/to_mat/zip` but no
   `load`. No Python equivalent of `InstanceSpace.load()` or `ISAmigrateModel` (415 lines of
   legacy-field migration in MATLAB). Every Python session must `build()` from scratch.
2. **`explore()` duplicates rather than reuses build-time code.** MATLAB's `evaluateTestSet`
   calls the same `PYTHIA()`/`TRACE()` functions used at training time in a different mode.
   Python's `_explore_pythia`/`_explore_trace`/etc. are independent reimplementations in
   `instance_space.py` — a future fix to `stages/pythia.py` won't automatically propagate.
3. **Build→explore adapter doesn't cover polynomial-kernel models.** `stages/pythia.py` can
   train `poly` or `rbf` SVMs; `build_explore_adapter.py::_svc_to_artifact()` only handles
   `"rbf"` and `"linear"` — calling `.explore()` after building with a polynomial kernel raises
   `NotImplementedError`. Reproducible, not hypothetical.
4. **`explore()` scope is narrower than MATLAB's.** MATLAB's version reconciles test-set
   algorithms against training and computes actual performance labels — it's an evaluation
   tool. Python's `explore()` returns predictions/footprint membership only.
5. **Feature-order handling diverges.** MATLAB errors (`featureOrderMismatch`) on mismatched
   column order; Python's `_extract_features()` silently reorders by name. Confirm this was a
   deliberate choice.
6. **No out-of-distribution warning.** MATLAB warns if >5% of test instances are clipped to
   training bounds; Python's `_explore_prelim` clips silently.
7. **No `plot()` convenience method** on the Python class.

**Where Python is ahead:** the self-validating DAG scheduler is more rigorous than MATLAB's
hand-maintained `StagePrereq` map; frozen dataclasses give real immutability MATLAB structs
can't; CSV ingestion is cleanly separated into `PreprocessingStage` versus MATLAB's monolithic
`runPrelim` (file I/O + selvars filtering + calling `PRELIM()` all in one method).

---

## 7. Ideas from independent implementations — PyISpace / PyHard

Not a MATLAB-vs-Python comparison — a third data point. **PyISpace** (`gitlab.com/ita-ml/pyispace`)
is a deliberately lean, partial Python reimplementation of ISA (PRELIM-equivalent inline, PILOT,
TRACE only — no SIFTED/CLOISTER/PYTHIA), built by the ITA-ML group (Instituto Tecnológico de
Aeronáutica, Brazil — primarily Pedro Yuri Arbs Paiva with Ana Carolina Lorena's group). **PyHard**
(`gitlab.com/ita-ml/pyhard`) builds on it for a different purpose (instance-hardness analysis, not
general ISA) and isn't directly comparable architecturally, but its CLI surfaced a couple of
reusable small ideas. Both installed from PyPI (`pyispace==0.3.7`, `pyhard==2.2.4`) and read
directly — no GitLab access available in this session, but PyPI sdist/wheel contents are the
actual source.

### R1 — Canonicalise PILOT's 2D projection orientation
**[Additive by design]** — ships as an opt-in flag (`adjust_rotation`, default `False`), specifically because the rotated output differs from today's default orientation.
PyISpace's `pilot.py` has an `adjust_rotation()` function: rotates the trained `Z` so a reference
group's centroid (bad-performing instances) lands at a fixed angle (135° = upper-left), then
returns the rotation matrix alongside it. Pure post-processing — rotation preserves all pairwise
distances, so error/R²/footprint areas are unaffected. Neither MATLAB's `PILOT.m` nor Python's
`stages/pilot.py` canonicalises orientation today, so the same or similar datasets can come out
mirrored/rotated relative to each other across runs, making plots hard to visually compare.
Low-risk, self-contained (~10 lines), additive-only. Worth adding to **both** MATLAB and Python.

### R2 — Alpha-shape auto-retry for TRACE footprint construction
**[Behavior-changing for the specific edge case it fixes]** — output changes from a wrong partial boundary to a correct complete one for any dataset that hits the multi-region alpha-shape case. This is a correctness fix, but call it out explicitly in release notes — a downstream consumer may have unknowingly built something around the old (wrong) boundary shape.
PyISpace's TRACE tries `alphashape.alphashape(points, alpha)`, and if that doesn't yield a clean
`Polygon` (e.g. produces a `MultiPolygon` instead), retries with `alphashape.optimizealpha()` to
find a better alpha automatically. This is the same failure mode behind the still-open MATLAB
finding: `traceAlphaBoundary` silently exporting partial boundaries for multi-region alpha shapes
(flagged by Copilot, tracked as a Phase 9 follow-up, never fixed). Worth checking whether this
retry pattern — or the underlying idea of validating the alpha-shape result and re-attempting
with a different alpha before giving up — closes that gap in both `stages/trace.py` (Python) and
`traceAlphaBoundary.m` (MATLAB). Corresponding MATLAB issue drafted separately (see the MATLAB
v0.9.1 issue set).

### R3 — Small CLI ideas (lower priority)
From PyHard's `cli.py` (`typer`-based): `typer.confirm()` for interactive yes/no prompts on
consequential choices, and path validation at startup (resolve relative→absolute, clear
`NotADirectoryError`/`FileNotFoundError` before any computation starts, rather than failing deep
inside a stage). Only relevant if/when the Python port's CLI work (`CLIDocs.txt`) continues —
not urgent, no corresponding issue drafted yet.

### Corroboration, not new action
Two things PyISpace does independently confirm decisions already in this roadmap rather than
add new ones:
- **Proper `logging` module use throughout, never raw `print()`** — independent validation that
  Q3 (§4) is the right call, since a separate implementation reached the same conclusion.
- **Raw `pickle.dump()` straight to `model.pkl` for persistence** — a real-world example of
  exactly the anti-pattern F7's design constraint (§6) already rules out. Cited here as evidence
  for *why* that constraint exists, not as something to adopt.

**Checkpoint for Phase R:** R1 — a rotation-adjustment unit test confirms `Z`'s pairwise distances
are unchanged before/after rotation, and that a reference dataset's rotated output is visually
consistent across two independent runs. R2 — a regression test using a known multi-region
alpha-shape case (if one is available) confirms the retry path is exercised and produces a
complete boundary rather than a partial one.

---

## 8. Phase T — testing infrastructure quality & additions

### 8.1 Audit findings

`tests/` is 6,678 lines across ~35 files. One genuine strength, several concrete gaps —
verified against source, not inferred from file names.

**Strength:** `tests/matlab_reference/` is a real cross-implementation golden-reference harness
— actual MATLAB-trained artifacts (projection matrix, SVM support vectors, footprint polygons)
checked in, with per-stage validation tests comparing Python's output against them under
documented tolerance thresholds (e.g. `test_pilot_matches_matlab`'s docstring states the 1%
threshold's rationale: PILOT inference is a pure linear projection, so Python should match
MATLAB to floating-point precision). `test_adapter.py::test_unsupported_kernel_raises` was good
discipline while the adapter existed — tested the poly-kernel gap failed loudly, not silently.
Now moot: S3 retires `build_explore_adapter.py` (and this test with it) entirely, once S1 makes
`explore()` operate on native `SVC` objects, which handle poly kernels with no special-casing
at all.

**Gaps, verified:**
1. **No true end-to-end integration test.** Every `InstanceSpace(` construction outside
   `exploreIS/` returns zero hits — `explore_iter` tests use `InstanceSpace.__new__` with every
   method manually stubbed. Nothing constructs a real `InstanceSpace` and calls `.build()`
   through the actual 7-stage pipeline. No Python equivalent of MATLAB's `test_integration.m`.
2. **The DAG resolver's hard logic is untested.** `test_stage_builder_runner.py` (3 tests) uses
   two trivial synthetic stages (`int→str→str`). Mutating-stage handling, `RunBefore`/
   `RunAfter`, and ambiguous-ordering error paths (all found during the §6.1 audit) have no
   test touching them. Moot if S2 lands — no point testing a resolver about to be replaced.
3. **No coverage tooling.** `pytest-cov` isn't even a dev dependency — no visibility into what
   percentage of the codebase is actually exercised.
4. **`poe test` doesn't run pytest.** The `[tool.poe.tasks]` `test` sequence is
   `ruff → mypy → black` only; pytest only runs via a separate direct call in CI
   (`validation-tests.yml`). A contributor running the local "test" command would reasonably
   assume tests ran.
5. **No `conftest.py`.** Zero shared fixtures across ~35 files.
6. **Test-file fragmentation.** PILOT alone is covered by `test_pilot.py`,
   `exploreIS/pilot/test_pilot_unit.py`, *and* `exploreIS/pilot/test_pilot_validation.py`, with
   no documented rule for what belongs where. Same pattern for sifted/trace/pythia/prelim.
7. **Historical note, now moot.** Every `svm_<algo>.csv` reference artifact has
   `kernel_fn = gaussian` — there was no MATLAB reference data for a polynomial-kernel PYTHIA
   model, which would have blocked validating Q1's original fix. No longer relevant: S3 retires
   the code this would have validated rather than fixing it, so no new fixture is needed for
   this purpose. The general fixture-provenance problem this finding also raised (no recorded
   MATLAB commit/tag, no staleness detection) stands independently — see §8.3.

### T1 — Add `pytest-cov` + track a coverage threshold in CI
**[Additive]** — tooling only.
Mechanical, additive. Can't manage what isn't measured.

### T2 — Add a real end-to-end `build()` integration test
**[Additive]** — a test.
Construct a real `InstanceSpace` with real metadata + options, call `.build()`, assert it
completes and produces a `Model` with every expected stage output populated. The Python
equivalent of MATLAB's `test_integration.m` — currently doesn't exist in any form.

### T3 — Add `conftest.py` with shared fixtures
**[Additive]** — test infrastructure only.
Reference-dataset loader, common `Mock` builders — reduce duplication across ~35 files.

### T4 — Fix `poe test` to actually include pytest
**[Additive]** — CI/tooling config only.
One-line change to `[tool.poe.tasks]`. Makes the local dev command match what CI actually does.

### T5 — Version-pin `tests/matlab_reference/`'s MATLAB provenance
**[Additive]** — test-fixture metadata only.
Record the exact MATLAB commit/tag the fixtures were generated from. See §8.3 for the fuller
cross-repo data-sharing proposal this connects to.

### T6 — DAG-resolver edge-case tests with representative stages
**[Additive]** — tests only.
Two synthetic linear stages can't exercise mutating-stage handling, `RunBefore`/`RunAfter`, or
the ambiguous-same-output-at-same-schedule-step error path. Add cases that actually trigger each.

### T7 — Consolidate fragmented per-stage test files
**[Additive]** — test-file reorganisation only, no production code touched.
Decide and document what belongs in top-level `test_<stage>.py` vs. `exploreIS/<stage>/
test_<stage>_unit.py` vs. `..._validation.py`, then merge or clearly demarcate — starting with
PILOT as the worst-fragmented case.

### 8.2 Test debt tied to already-scoped items

Specific tests each already-scoped item (Q, F, R) needs to actually be verified, not just built:

| Item | Test needed |
|---|---|
| Q1 | Retired — see S3's checkpoint (confirm the module is genuinely unreferenced anywhere, full suite passes without it) instead of a poly-kernel validation test |
| Q2 (OOD warning) | Fires above 5% clipped; silent below it — both directions need a test |
| Q8 (rerun-invalidation regression test) | Must use the real 7-stage pipeline once T2 exists — the current synthetic 2-stage setup can't exercise the cloister/pythia sibling-branch question at all |
| F7 (save/load) | Round-trip equality test, run once signed and once unsigned; a signature-tampering test (flip one byte, assert `load()` refuses to deserialise rather than raising deep inside `pickle`); a downgrade-attack test (`save()` with `secret_key`, `load()` without it, assert refusal rather than silent unverified deserialisation — proves the two-mode split doesn't reopen the hole signing closes); a test asserting the *server* code path always passes `secret_key` and is never reachable from any user-supplied path/parameter (the actual safety invariant the server-side design depends on) |
| F8 (explore/build code reuse, now TRACE-only per S1) | A deliberately-introduced bug in `TraceStage`'s footprint logic should break both the build-path and explore-path tests once they share code — proves the drift risk F8 is meant to close is actually closed. (The PYTHIA half of this row is resolved by S1 instead — see S1's own checkpoint.) |
| R1 (rotation canonicalisation) | Not just "pairwise distances preserved" — assert the target group's centroid angle lands within tolerance of 135° post-rotation, or the test doesn't prove the feature does what it's for |
| R2 (alpha-shape retry) | A constructed point cloud engineered to produce a `MultiPolygon` at the naive alpha, asserting a complete (non-partial) boundary after the fix |
| S1 (model-shape collapse) | `test_pythia_validation.py` re-run unmodified as a regression check (it bypasses `_ensure_explore_model()` via mocking, so it shouldn't need changes — confirm that stays true rather than assume it); a new test asserting `explore()` on a real `build()`-trained model of each kernel type (rbf, linear, poly) produces predictions matching direct `.predict_proba()` calls on the stored `SVC` |
| S2 (explicit stage order) | Same 7-stage pipeline resolves to the identical execution order as the auto-resolver produced before the change — a before/after comparison, not just "it doesn't crash" |

### 8.3 Cross-repo test-data sharing proposal

The root problem: `tests/matlab_reference/` was produced by a one-off manual MATLAB run, with no
recorded provenance and no repeatable process. As MATLAB keeps evolving (this document alone has
logged a dozen-plus MATLAB-side changes worth making), the fixture set can only get further out
of sync, silently, with no signal when it happens. Proposed, layered by effort:

1. **Now, low-risk:** a MATLAB export script (new file, or extend `test_integration.m`) that
   dumps training artifacts + explore outputs in exactly the CSV interchange format
   `tests/matlab_reference/` already documents. Turns "regenerate the fixtures" from a bespoke
   manual copy-paste into one command. No longer needed for the poly-kernel case specifically
   (§8.1 finding 7 is moot now that S3 retires the code it would have validated) — but the
   general problem (no recorded provenance, no staleness detection as MATLAB keeps evolving)
   stands on its own regardless, and this remains the right first fix for it.
2. **Now, low-risk:** add a `provenance.json` (or similar) alongside the fixtures recording the
   exact MATLAB commit SHA/tag/version used, mirroring the `Contents.m`/`CITATION.cff` version
   fields already established in the MATLAB refactor. Cheap, and it's the difference between
   "we don't know if this is stale" and "we know exactly which MATLAB version this validates
   against."
3. **Later, process-level:** MATLAB has no CI at all right now (verified — no
   `.github/workflows/` in the MATLAB repo, only an `ISSUE_TEMPLATE/` folder). Once that's
   addressed, a release-triggered job could run the export script from (1) and publish the
   fixtures as a tagged release asset, which Python's own CI can then pull down and diff against
   the committed copy — turning "did the reference data go stale" into an automated check
   instead of something that only surfaces when someone happens to compare by hand.

Deliberately not proposing a shared submodule or a new dedicated repo for this — the fixture set
is small (well under the size where that complexity pays for itself) and the export-script +
provenance-stamp approach gets most of the benefit for a fraction of the process overhead.

**Checkpoint for Phase T:** `pytest-cov` reports a baseline coverage number (T1); the new
end-to-end test (T2) passes and is added to `poe test` (T4); each Q/F/R item in the table above
ships with its listed test, not just its implementation.

---

## 9. Outstanding / deferred items

| Item | Status |
|---|---|
| CITATION.cff DOI | Repo already has a Zenodo concept DOI (`10.5281/zenodo.15562567`); confirm whether to reuse it or mint a version-specific one |
| `docs-passing` badge | Needs a decision in P5 — real CI or honest wording |
| README "AI-assisted analysis" section | Deferred pending a Claude Code skill for this repo |
| F1–F9 scoping | Deferred until each phase's own audit is done (F1–F6, F8's ambition level) — F9 resolved (see below) |
| `pillow`/`tornado` remaining bump | Both partially bumped by Dependabot already (confirmed on `main`, §2.1) — `pillow` 12.2.0→12.3.0 and `tornado` 6.5.5→6.5.7 are the remaining steps, not full bumps from scratch |
| `click`/`jupyter-core` bump | Confirmed zero Dependabot movement on either (§2.1) — full bump still needed, `.github/dependabot.yml`'s absence is the likely reason full version-update checks aren't happening |
| MATILDA web-upload format | Decided: CSV only (see [DECISION] in §2.1). No F-phase owns upload-handling code yet — flag this constraint when one is scoped. |
| Q8 outcome | Determines whether F4's invalidation-fix scope includes `stage_runner.py` changes or closes as "verified fine" |
| R2 / `traceAlphaBoundary` | Same underlying question on both sides now: does an alpha-retry pattern fix the known multi-region alpha-shape gap? Corresponding MATLAB issue tracks the MATLAB side; R2 (§6) tracks the Python side. |
| Missing poly-kernel reference fixture | No longer relevant — was blocking validation of Q1's original fix, which is retired (see S3) rather than fixed. The general fixture-provenance problem this finding also raised stands independently, tracked via T5/§8.3. |
| MATLAB has no CI | Verified (no `.github/workflows/`) — outside this document's scope (MATLAB repo), but relevant context for §8.3's phased data-sharing proposal, phase 3 of which depends on it existing |
| S1's "planted seam" question | **Resolved (v1.15)** — closed as impractical given no demonstrated need, not impossible; see S1's `[DECISION]` block. Reopenable if a real cross-platform use case appears later. |
| F7's `load()` path-safety invariant | The server-side half of the revised F7 design depends on `load()` never receiving a user-supplied path, and always passing `secret_key` — needs to be an enforced, checked invariant (allowlist, storage-layer guarantee) once implementation starts, not left as a design-doc assumption |
| F7's desktop/unsigned mode — downgrade-attack invariant | `secret_key=None` is a deliberate second mode (v1.17), not a loophole — but only if `load()` refuses both "signed file, no key given" and "no signed file, key given" cases. Needs its own enforced test (see §8.2), since this is the one place the two-mode split could silently regress into a bypass of the server mode's signing. |
| Q6/F7 pickle-exclusion for pooled executors | (Added v1.18) If Q6 lands before F7, its pool-holder attribute must exclude itself from pickled state (`__getstate__`/`__setstate__`) or F7's save/load round-trip fails intermittently depending on whether a pool-using stage ran before `save()`. Whichever of Q6/F7 implements second should verify this explicitly rather than discover it via a flaky test. |
| Q8/S2 sequencing | (Added v1.19, sharpened v1.20) Q8's test and diagnosis target `_rollback_to_schedule_index()`'s wave-position invalidation (verified in `stage_runner.py:256-267`); S2 removes wave computation entirely. Sequence Q8 after S2, same reasoning already recorded for S2→T6 — but unlike T6, Q8 isn't at risk of becoming pointless (the invalidation property it checks still matters post-S2), only of wasting *implementation* effort if a fix is written against the pre-S2 wave-grouped structure and then has to be re-derived against S2's replacement. |

---

## 10. Document history

| Version | Date | Change |
|---|---|---|
| v1.0 | 2026-07-26 | Initial roadmap: documentation-first phases (P1–P5) scoped from audit; functionality phases (F1–F6) listed as deferred backlog |
| v1.1 | 2026-07-26 | Added §2.1 dependency security audit (`pip-audit` against `andremun/pyInstanceSpace`'s `poetry.lock`) and new Phase P0 (dependency security hygiene) — confirmed low-risk, mechanical, and safe to run first, ahead of P1–P5 |
| v1.2 | 2026-07-26 | Recorded [DECISION]: future MATILDA web uploads restricted to CSV only, closing the "stops being true" web-deployment caveat on the Pillow findings in §2.1 |
| v1.3 | 2026-07-26 | Added §5.1 — deep class-architecture audit of `InstanceSpace`/`StageRunner`/`build_explore_adapter` vs. MATLAB's `InstanceSpace.m`; F4 status changed from "audit only" to "audited," 7 concrete findings recorded |
| v1.4 | 2026-07-26 | Reverse direction: added Phase Q (§4) — 8 low-risk MATLAB→Python quality ideas, filtered specifically for additive/contained/no-scheduler-changes; added F7–F9 (§5) for the 3 heavier MATLAB-derived ideas (save/load, explore/build code reuse, full evaluation scope) that didn't clear Phase Q's bar; renumbered §4→§5 (functionality parity), §5→§6 (outstanding items), §6→§7 (document history) to make room |
| v1.5 | 2026-07-26 | Added §6 — Phase R: 2 actionable ideas (R1 rotation canonicalisation, R2 alpha-shape auto-retry) plus corroborating evidence from independent third-party implementations PyISpace/PyHard (ITA-ML, Brazil); R2 connects to the pre-existing, still-open MATLAB `traceAlphaBoundary` multi-region finding; renumbered §6→§7 (outstanding items), §7→§8 (document history) to make room |
| v1.6 | 2026-07-26 | Added §7 — Phase T: testing-infrastructure audit (1 real strength — the MATLAB golden-reference harness — and 7 verified gaps, including a missing polynomial-kernel reference fixture that blocks validating Q1), 7 low-risk additions (T1–T7), a test-debt cross-reference table for Q1/Q2/Q8/F7/F8/R1/R2 (§7.2), and a phased cross-repo test-data sharing proposal (§7.3); noted MATLAB has no CI at all (verified); renumbered §7→§8 (outstanding items), §8→§9 (document history) to make room |
| v1.7 | 2026-07-26 | Added Q9 (§4) — centralise RNG seeding via a `general.seed` option, replacing 8+ hardcoded `seed=0`/`random_state=0` literals across `pilot.py`/`sifted.py`/`prelim.py`/`pythia.py`; verified no `seed` field exists in `InstanceSpaceOptions` at all. Added Q10 (§4) — `SECURITY.md` + `CONTRIBUTING.md`, both verified absent. Corresponding MATLAB-side repo-hygiene issue drafted separately (batch 4) — RNG seeding does not apply to MATLAB, which already does this correctly. |
| v1.8 | 2026-07-26 | Resolved two open decisions from the companion implementation-pathways document: Q5 confirmed as permissive (keep auto-reorder-by-name, document + test it); F7's persistence format decided as HDF5 via `h5py` (§5, design constraint updated with rationale and the SVM/polygon flattening detail). F9 remains open — further explanation requested before deciding same-method vs. new-method vs. out-of-scope. |
| v1.9 | 2026-07-26 | Resolved F9: extend `explore()` itself (Option 1), branching silently on whether ground truth is present in test metadata — confirmed this is free to detect (`Metadata.from_csv_file` already parses `algo_*` columns unconditionally). Pathway extracts `PrelimStage`'s binary-performance logic into a shared function, serving F8's drift-reduction goal at the same time; "new algorithm absent from training" edge case deferred by default. Full pathway in the companion implementation-pathways document. |
| v1.10 | 2026-07-26 | Added Q11 (§4) — found while cross-checking both repos against Wilson et al.'s "Best Practices for Scientific Computing" (2014): `validation-tests.yml` has a formatting/lint/type-check CI step sitting commented out rather than enabled or removed, a direct instance of that paper's practice 2h. Corresponding MATLAB-side finding (adopt `matlab.unittest` for its built-in coverage plugin, per the same cross-check) drafted separately as batch 5. |
| v1.11 | 2026-07-26 | Added Phase -1 (before §2) — merging `aoxiangx/pyInstanceSpace`'s fork branch back into `andremun/pyInstanceSpace` upstream, as an explicit prerequisite ahead of P0 rather than an implicit assumption. Verified via a real (unpushed) scratch merge: clean, no conflicts, despite both sides having diverged independently (main +14 commits, fork branch +24). Recommended path is a PR via GitHub's cross-fork compare view rather than a silent direct merge, since upstream already has matching CI. |
| v1.12 | 2026-07-26 | Production/delegation context: Claude Code will have write access, work is delegated, this ships behind a production web server. Added a **[Additive]**/**[Behavior-changing]**/**[Unknown until audit]** compatibility tag to every P/Q/F/R/T item (new "Compat" column on the F-phase table; inline tags elsewhere). Corrected Q9's recommended seed default from `None` to `0` — the original recommendation didn't account for production callers expecting deterministic output; `0` exactly preserves today's hardcoded behaviour. |
| v1.13 | 2026-07-26 | Re-verified §2.1's dependency findings against `main`'s actual current commit history (prompted by a direct question about whether "main +14 commits" was still accurate while drafting the fork-merge PR description) — confirmed accurate, and confirmed all 14 are Dependabot dependency bumps, not manual development. Two of them (`pillow`→12.2.0, `tornado`→6.5.5) partially address two of P0's four flagged CVEs, landing exactly at the versions already audited — both still short of the fully-patched target. `click`/`jupyter-core` show zero Dependabot movement, directly confirming (not just inferring) that only security-alerts-only automation is active, not full version-update checks. P0 and the outstanding-items table updated to reflect partial-vs-untouched status per package. |
| v1.14 | 2026-07-26 | Added Phase S (§5) — structural simplification, sequenced before F-phase: S1 (collapse model-shape detection to the one reachable path, moving to native scikit-learn objects for `explore()`) and S2 (replace DAG auto-resolution with explicit stage order + prerequisites, keeping mypy verification of the literal declarations). Revised F1 (explore-side classifier dispatch resolved as a side effect of S1, narrowing remaining scope to the training-side registry), F7 (persistence format changed from HDF5-via-`h5py` to signed `pickle`/`joblib`, given the confirmed production threat model — models are system-produced/downloaded only, never re-uploaded — with an HMAC signature as the actual enforced control rather than relying on that assumption holding forever), and F8 (narrowed to TRACE only, since S1 resolves the PYTHIA half by removing the second implementation entirely rather than needing to reconcile it with the first). Renumbered §5→§6 (functionality parity), §5.1→§6.1, §6→§7 (PyISpace/PyHard), §7→§8 (Phase T, and its own §7.1–§7.3→§8.1–§8.3), §8→§9 (outstanding items), §9→§10 (document history) to make room. |
| v1.15 | 2026-07-26 | Closed S1's open decision: cross-platform MATLAB-model loading recorded as impractical (six classifier types, several — decision trees especially — with no clean flattened representation), not impossible, and not attempted given no demonstrated need — reopenable later if that changes. Consequence: added S3, retiring `build_explore_adapter.py` entirely, since nothing calls it once S1 lands and cross-platform loading is closed. Q1 (was "fix the poly-kernel gap") retired in favour of S3 — kept as a pointer rather than deleted outright since other parts of this document and the drafted MATLAB issue batches reference it by name. Updated every downstream reference: Phase Q's checkpoint, §8.1's audit findings (the poly-kernel reference-fixture gap and the `test_unsupported_kernel_raises` note are both now historical/moot), §8.2's test-debt table, §8.3's data-sharing proposal (kept on its own merits, decoupled from Q1), the outstanding-items table, and Phase -1's example PR text. |
| v1.16 | 2026-07-28 | S1 and S3 implemented and verified on `v0.9.0/development-branch-S` (issues #282, #284 closed). Fixed a stale internal contradiction in §5's S1 section, caught during a post-implementation risk/payoff review: it still argued for keeping `adapt_for_explore()` alive as F7's persistence format, a position F7's own pickle/joblib decision (already recorded in v1.14) had already superseded — under signed pickle, `SVC` objects round-trip natively with no flattening step, so there was never a remaining caller to keep the adapter for. S3's full-deletion pathway (already correct) is what was actually implemented; this entry just brings §5's prose back in sync with it. |
| v1.17 | 2026-07-28 | Revised F7's persistence decision (still design-only, not yet implemented) from unconditional HMAC signing to signing-optional-via-`secret_key: bytes \| None`, to serve a second reachable caller the v1.14 decision didn't account for: local/desktop development with no server-managed secret. `secret_key=None` skips signing entirely for that caller (equivalent risk to any other unsigned `pickle`/`joblib` use of a file the caller already trusts); `secret_key` given enforces the original signed-and-verified path unchanged. Added the one new risk the split introduces — a downgrade attack, where a server-signed file is loaded unverified by omitting the key — and closed it structurally: `load()` must refuse both "signed file, no key" and "unsigned file, key given" mismatches, not just the already-covered "signed file, wrong key" case. Updated the F7 table row, `[DECISION]` block, §8.2 test-debt row (added a downgrade-attack test), and §9 outstanding-items (added a dedicated downgrade-invariant entry) in this document, plus F7's pathway steps and `Decision needed` note in the companion implementation-pathways document. No code changed — F7 remains unimplemented. |
| v1.18 | 2026-07-28 | Recorded a previously-undocumented Q6↔F7 interaction, found while discussing a start-pool → run stage → save → restart session → load → run next stage scenario: `ThreadPoolExecutor` (Q6's pool-holder attribute) is not picklable, so if Q6 lands without excluding it from pickled state, F7's `save()` either crashes outright (pool live) or succeeds only by caller discipline (pool closed first) — a scenario-dependent failure that would surface as a flaky F7 round-trip test rather than an obvious Q6 gap. Fix recorded as belonging to Q6: exclude the pool via `__getstate__`/`__setstate__`, letting `load()` come back with the pool unset for lazy recreation on next use — consistent with both Q6's own "lazily created" design and MATLAB's own non-serialised, session-local parallel-pool handles. Updated Q6's entry (new interaction note), F7's `[DECISION]` block (cross-reference), and §9 outstanding-items (new row) in this document, plus Q6's pathway in the companion implementation-pathways document. No code changed — both Q6 and F7 remain unimplemented. |
| v1.19 | 2026-07-28 | Swept the remaining pending Q/S items (Q6, Q8, S2 — everything else in both phases is already implemented) for the same shape of cross-item gap just found in Q6/F7. Found one: Q8's regression test and its own stated diagnosis target `stage_runner.py`'s `_rollback_to_schedule_index()` — verified directly (`stage_runner.py:256-267`) to invalidate by iterating the wave-grouped `_stage_order` list by position. S2's own pathway (step 3) already removes "wave computation" as part of replacing DAG auto-resolution, and already sequences itself against T6 for the identical reason ("no point testing an algorithm about to be deleted") — but nothing sequenced S2 against Q8, even though both operate on the same function, just because they sit under different phase headings. Recorded the same before-S2-or-reconcile-after reasoning for Q8 that already existed for T6, in both directions (Q8's entry and S2's entry) so it surfaces regardless of which one someone reads first. No further gaps found in Q6 beyond the one already recorded in v1.18. No code changed. |
| v1.20 | 2026-07-28 | Sharpened v1.19's Q8/S2 sequencing note: it had treated Q8 and T6 as fully parallel cases, but they differ in what "sequence after S2" actually protects against. T6 tests the resolution *algorithm* S2 deletes outright — post-S2 it may have no remaining subject matter, hence S2's own "or skip T6 entirely." Q8 tests a *behavioral property* (correct invalidation on partial rerun) that still has to hold post-S2 — S2's own before/after checkpoint (full-pipeline output equality) doesn't cover partial-rerun invalidation, so it doesn't subsume Q8 either. The real risk in doing Q8 before S2 isn't a stale test, it's wasted *implementation*: a promoted F-phase fix for over-invalidation would be written against the wave-grouped `_stage_order` S2 then deletes, forcing S2 to re-derive the same dependency-graph walk against its own replacement structure. Net correction: Q8 must wait for S2 same as T6, but unlike T6 it is never at risk of becoming pointless. Updated Q8's entry, S2's entry, and the Q8/S2 outstanding-items row in this document, plus both cross-references in the companion implementation-pathways document. No code changed. |
| v1.21 | 2026-07-28 | Added §6.0 — a single consolidated execution order for every remaining Q/S/F item (Q6, Q8, S2, F1–F3, F5–F9; F4 excluded, already audited), compiled from every cross-item dependency recorded so far (v1.17–v1.20) plus two more pulled from each item's own pathway that hadn't been assembled in one place before: F5's hard block on F2 ("genuinely blocked on F2 landing first") and F9's shared-extraction step explicitly mirroring F8's pattern. Recommended order: S2 → F1/F6 → Q6 → F7 → Q8 → F8 → F9 → F2 → F5, with F3's audit runnable independently at any point. CLAUDE.md's phase-gate section updated to point at §6.0 instead of restating a partial version of it. No code changed. |
| v1.22 | 2026-07-28 | S2, F1, and F6 implemented and verified on `v0.9.0/development-branch-QSF` (full suite: 265 passed, 0 failed; F1 alone adds 18 new dedicated tests). S2's audit found a real gap in its own §5 design — the `stages` constructor parameter is a documented plugin-extension point (`example_plugin.py`), not just internal plumbing — resolved by requiring explicit `RunBefore`/`RunAfter` declarations for any non-built-in stage rather than type-matching inference; two latent bugs in the never-before-exercised `RunBefore`/`RunAfter` mechanism were found and fixed in the same pass (a wrong `TypeVar` bound, and `isinstance()` used where `get_origin()` was needed for subscripted generics). Following a direct request, `stage_builder.py` was then folded entirely into `stage_runner.py` (`build_stage_runner()` plus two private helpers) and deleted, since post-S2 it had shrunk to one call site with no remaining reason to be a separate module — net line-count for that fold alone: 322 deleted, 122 added. `tests/test_stage_builder_runner.py` renamed to `tests/test_stage_runner.py`. F1 added `PythiaOptions.classifier` (default `'svm'`, zero behaviour change verified against the existing MATLAB-reference tests) and a training-side registry dispatching to 6 scikit-learn classifiers, explicitly scoped to *not* claim MATLAB-verified hyperparameter tuning for the 5 non-`svm` entries (no MATLAB reference exists for them) — flagged in both code and this document rather than silently assumed. Fixed a real, separate bug found during cleanup: `instancespace/__init__.py`'s `__all__` still listed the now-deleted `stage_builder`, which made `from instancespace import *` raise `AttributeError` — confirmed by reproducing the crash before fixing it. README and RELEASE_NOTES.md's baseline section updated to describe the current (not pre-S2) DAG/scheduler behaviour, following the same v1.16 precedent for stale "describes current state" prose. §6.0's recommended-order table and status line updated to mark S2/F1/F6 done. |
