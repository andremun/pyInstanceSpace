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
`InstanceSpace` class, a `build()`/`explore()`/`explore_stage_iter()` API) is already independently
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
`explore()`/`explore_stage_iter()` API the rest of this roadmap assumes already exists.

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
| 7 | `liveDemoExploreIS.ipynb` | 17 cells / 9 markdown headers, reasonable stage-by-stage structure, actively being improved (recent commits added per-stage diagnostics and `explore_stage_iter` notes). Gap is narrative depth — MATLAB's manual explains *why* and *how to read the output* at each stage; the notebook should be audited cell-by-cell against that bar, not rewritten from scratch. |
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
- Introduce `RELEASE_NOTES.md`, seeded with a baseline entry describing the current state (stage architecture, `build()`/`explore()`/`explore_stage_iter()`, licence), using MATLAB's section convention: *New functionality* / *Better engineering* / *Bug fixes* / *Licence*.
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
**Implemented and verified (v1.24)** — `InstanceSpace._get_executor()`/`close()` cache a lazily
-created `ThreadPoolExecutor`, recreated only if the worker count changes, threaded through
`TraceInputs.executor`/`TraceStage`'s `compute_algorithm_qualities()` instead of the previous
per-call `with ThreadPoolExecutor(...) as executor:`. 10 new unit tests (pool identity reuse,
recreation on worker-count change, `close()`/lazy-recreate-after-close, `run_stage()` injecting
the cached pool by default without overriding a caller-supplied one, and `compute_algorithm_
qualities()` submitting to a supplied pool vs. creating its own with identical output either
way). Output verified bit-identical: `test_trace.py`'s existing MATLAB-reference tests pass
unmodified. See T2/Q8's entry below for a real crash this change caused that only a full
end-to-end build caught, and the resulting fix in `stage_runner.py`.
**[Additive if implemented correctly]** — this is a concurrency change, not just a resource optimisation; verify computed output is bit-identical before/after, not just "faster."
MATLAB's `ensurePool()` opens a parallel pool once and reuses it across successive staged
`build()` calls in the same session, only tearing it down if it opened it. Python currently
creates a fresh `ThreadPoolExecutor`/joblib backend per stage call. Pure resource-management
change — no correctness implications, easy to test (assert no new pool created on a second
`run_stage()` call).

**Interaction with F7 (added v1.18) — corrected v1.23, was based on a premise that doesn't hold
in the actual code:** a `ThreadPoolExecutor` is not picklable, so if Q6's pool-holder attribute
sat on the same object F7's `save()` pickles, that would be a real problem. Checked directly
against `instancespace/model.py`: F7's target is `Model.save()`/`Model.load()`, and `Model` is a
frozen dataclass (`data, data_dense, feat_sel, prelim, sifted, pilot, cloister, pythia, trace,
opts`) with **no field referencing `InstanceSpace` or `StageRunner`** — the only two places Q6's
own pathway proposes putting the pool. `Model.save()` never touches either, so there is nothing
for a `__getstate__`/`__setstate__` pickle-exclusion to guard against, as currently scoped. The
shared-checklist item in §6.0 ("Q6 and F7, whichever lands second, must add the pickle-exclusion
check") is retracted for that reason — verified moot, not just deprioritised. If a future change
ever gives `Model` a live reference back to `InstanceSpace`/`StageRunner` (it doesn't today),
this would need revisiting; until then, don't build unused `__getstate__`/`__setstate__`
machinery on `InstanceSpace`/`StageRunner` on this note's authority. Q6's own open decision
(explicit `close()` vs `__del__`) stands on its own merits, unaffected by this correction:
default to explicit `close()`, per the reasoning already given below.

### Q7 — Add `plot()` convenience methods
**[Additive]** — new methods only; nothing existing calls them yet.
Mirror MATLAB's `InstanceSpace.plot('sources' | 'portfolio' | 'good' | 'footprint', algoIdx)` —
thin matplotlib wrappers around `model.pilot.z` and friends. Additive only, no pipeline logic
touched. Complements P2 (notebook parity): a `plot()` method means the notebook needs less
inline matplotlib boilerplate to demonstrate the same views MATLAB's manual shows.

### Q8 — Regression test for stage-rerun invalidation (verification, not yet a fix)
**Implemented and verified (v1.24)**, negative result: on `v0.9.0/development-branch-QSF`,
against a real full-7-stage build (T2's fixture, `tests/test_build_integration.py`) — rerunning
`CloisterStage` via `run_stage()` does **not** wrongly invalidate `PythiaStage`'s output (they
share a schedule wave, and `_rollback_to_schedule_index()` only invalidates *later* waves, never
wave-mates) and does not block `run_stage(TraceStage)` either, even though `TraceInputs` has no
field CloisterStage produces (`z_edge`/`z_ecorr`) — confirmed by reading `TraceInputs`'s full
field list directly, not inferred. Traced the general case, not just this one pair: every
"later wave depends on an earlier wave" relationship in the built-in 7-stage order is a real
dependency except this one (Cloister→Trace, non-adjacent-in-dependency but adjacent-in-schedule),
and that one is already correctly not order-adjacent in a way that triggers invalidation, since
Cloister and Trace aren't in the same wave and Trace's own rollback point only depends on the
wave it's actually in. §6.1's speculative concern does not reproduce for the *current* built-in
order — this is a scoped, verified finding about this specific 7-stage pipeline, not a claim
that `_rollback_to_schedule_index()`'s wave-position mechanism is correct in general (a future
plugin stage attached via `RunBefore`/`RunAfter` with a real cross-wave dependency gap could
still hit it). No fix promoted to F-phase; nothing to promote. **Real bug found and fixed along
the way, unrelated to the invalidation question**: writing T2 as this test's real-build fixture
surfaced a genuine crash — `StageRunner.run_stage()` unconditionally deep-copies every stage's
resolved inputs, and Q6's `executor: ThreadPoolExecutor | None` field (added to `TraceInputs`)
isn't deepcopy-safe (`TypeError: cannot pickle '_queue.SimpleQueue' object`) — none of Q6's own
unit tests caught this because they called `TraceStage.compute_algorithm_qualities()`/`_run()`
directly, bypassing `StageRunner.run_stage()`'s deepcopy step entirely. Fixed in
`stage_runner.py` via a new `_deepcopy_stage_inputs()` helper that pre-seeds `copy.deepcopy`'s
memo with any live `ThreadPoolExecutor` values so they pass through by reference instead of
being copied — correct for two independent reasons, not just to silence the crash: copying a
pool wouldn't just fail, it would (if it somehow succeeded) silently create a redundant pool per
stage call, defeating Q6's entire reuse purpose. Added a fast, targeted regression test
(`tests/test_stage_runner.py::test_run_stage_does_not_deepcopy_a_live_executor`) so this doesn't
require another 8-minute full build to catch a regression. This is exactly why T2 was sequenced
as Q8's prerequisite rather than skipped — a synthetic 2-stage fixture would never have exercised
this.
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

**Superseded at the top by §6.3, added v1.43, per direct instruction:** the 43-finding external
audit batch (#297 and its 6 sub-issues, all UNVERIFIED) is now first in priority — ahead of every
row in the table below. "First" there means *triage/independent verification first*, not
"implement before F8" — see §6.3 for the full instruction and status. The ordering below is still
the right sequencing for the F8 → F9 → F2 → F5 items themselves, once/if any of them end up
following from the audit batch's verification.

Q1–Q5, Q7, Q9–Q11, S1/S3, and — as of v1.22 — S2 and F1 and F6, and — as of v1.24 — Q6, F7, T2
(Phase T), and Q8 are already implemented. What follows orders everything still pending — F8, F9,
F2, F5, F3's audit — by actual dependency, not just by letter. Compiled from every cross-item
finding recorded in this document (v1.17–v1.20) plus two dependencies already stated in each
item's own pathway that hadn't been pulled into one place before: F5's hard block on F2, and F9's
shared-extraction pattern mirroring F8's.

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
- ~~**Q6 and F7, whichever lands second, must add the pickle-exclusion check**~~ — **retracted,
  v1.23**: verified `Model` (F7's actual pickle target) has no reference to `InstanceSpace`/
  `StageRunner` (Q6's only proposed pool-holder locations), so there is nothing to exclude. See
  §4 Q6's corrected note for the full check.
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
| 4 | ~~Q6~~ | **Done (v1.24).** No pickle-exclusion needed (corrected §4 Q6 note, v1.23). |
| 5 | ~~F7~~ | **Done (v1.24).** Independent of Q6 once the pickle-exclusion link was retracted. |
| 5.5 | ~~T2~~ (Phase T) | **Done (v1.24).** Real full-7-stage `.build()` fixture; found and fixed a real deepcopy/`ThreadPoolExecutor` crash Q6's own unit tests hadn't caught. |
| 6 | ~~Q8~~ | **Done (v1.24), negative result.** No over-invalidation reproduces for the built-in 7-stage order — see §4 Q8. |
| 7 | F8 | S1 already resolved the PYTHIA half; decide lighter-vs-fuller here; behavior-changing — full `tests/matlab_reference` suite before/after |
| 8 | ~~F9~~ | **Deferred (v1.36).** Not a quick follow-on to F8 as this row implies — F8 is itself unstarted, behavior-changing, with its own unresolved ambition-level decision. F9's pathway stays fully decided; not queued as near-term work until F8 actually lands. |
| 9 | F2 | Independent but higher-risk (bit-for-bit verification burden) — do with full attention once lower-risk items are clear; land R1 first internally |
| 10 | F5 | Direct consumer of F2, natural next step |
| — | ~~F3 (audit)~~ | **Closed, won't-fix (v1.36).** Audit ran (v1.27); the one confirmed gap was scoped (v1.35) then closed without implementing — performance-only, no correctness impact, not worth the new edge-case test surface a safe fix needs. See §6.2 and the roadmap's F3 row for full detail. |

F4 doesn't appear above — it's already "audited," not an actionable item; F7/F8/F9 are its
concrete derivatives and are already in the table.

| Phase | Maps to MATLAB | Focus | Status | Compat |
|---|---|---|---|---|
| F1 | Phase 4 | PYTHIA classifier registry — confirm whether `stages/pythia.py` supports a pluggable classifier set or is fixed | **Implemented and verified (v1.22)** — training-side registry (`instancespace/utils/get_classifier_fcn.py`) dispatches to `svm`/`knn`/`tree`/`nb`/`linear`/`ensemble`; explore-side already handled by S1. Only `svm` is tuned via the existing `C`/`gamma` search — the other five fit with scikit-learn's own defaults, not a MATLAB-verified tuning range (no MATLAB reference exists for them). `PythiaOutput.svm`'s type widened to `list[ClassifierMixin]` (field name kept for backward compatibility). Full suite + 18 new dedicated tests (registry unit tests + one per registered classifier trained end-to-end) all pass. | **[Additive at default]** — new `classifier` option defaults to `'svm'`, matching today's only behaviour verified via the existing MATLAB-reference tests unchanged. New registry entries themselves are new production surface — validated by dedicated tests, not just "it runs," but without MATLAB-verified hyperparameter tuning for the five non-`svm` entries; flagged in code and docs, not assumed. |
| F2 | Phase 5 | PILOT 3D / viewpoint optimisation parity in `stages/pilot.py`, **plus `ntries` restart parallelism** (added v1.25 — verified directly against MATLAB's `PILOT.m`, not inferred: `parfor (i=1:opts.ntries, nworkers)` runs the BFGS multi-start restarts in parallel, reusing an existing pool via `gcp('nocreate')` rather than opening a new one; Python's `pilot.py` runs the equivalent `for i in range(opts.n_tries):` loop sequentially, no parallelism at all, no `parallel_options` field on `PilotInput`) | Not started | **[Behavior-changing risk]** — generalising the 2D-specific solver to n-dims can shift 2D output even at `dims=2` if not done carefully (different array shapes can trigger different BLAS code paths). Verify bit-for-bit or tolerance-verified identical 2D output before shipping — this touches existing code, not just adding an independent new path. Parallelising `ntries` is additive on top of that (independent restarts, order of completion doesn't affect which is picked — `out.perf`'s `argmax` is order-invariant), but do it in the same pass as the dims/PLS work since both touch `PilotInput`'s options surface and the same `for`/`parfor` loop. **Nested-parallelism caution (added v1.27, found during F3's audit, §6.2 finding 2):** `SiftedStage._find_best_combination()` already calls `PilotStage.pilot(...)` from inside its own GA fitness function, which itself runs in separate OS worker processes when `parallel_options.flag` is set — adding a pool to PILOT's `ntries` loop without detecting this (e.g. skipping/serialising PILOT's own pool when already running inside a worker process) reintroduces the exact nested-`parfor`-inside-GA bug MATLAB's SIFTED promotion fixed. |
| F3 | Phase 6 | SIFTED promotion refinements | **Closed, won't-fix (v1.36)** — audited (v1.27, §6.2) against MATLAB's 4 historical fixes: 2 don't apply to Python at all, 1 is a cross-item risk flagged to F2, 1 is a real gap (`_compute_correlation`'s unvectorized loop), scoped in detail (v1.35, companion pathways doc) but not implemented. Closed on direct instruction (#263) — the gap runs once per `SiftedStage` call, not per-GA-candidate, so the performance upside doesn't justify the new edge-case test surface (zero-variance columns, `n_valid<3`) a safe fix would need. Pathway kept in docs in case a future profiling run changes the calculus. | — (closed; no further work planned) |
| F4 | Phases 7–8 | `InstanceSpace` class & `build`/`explore` robustness | **Audited (v1.3)** — see §6.1 for findings; Q8 (§4) verifies one open question before F4's invalidation-fix work is scoped | — (audit only; see F7/F8/F9 for the actionable, taggable derivatives) |
| F5 | Phase 9 | Output consolidation / 3D visualisation parity (MATLAB's `scriptpng.m`) | Not started | **[Additive]** — new rendering paths; doesn't change any existing 2D output function. |
| F6 | Phase 10 | Namespace & per-file licence headers — licence itself already matches MATLAB | **Implemented (v1.22)** — `SPDX-License-Identifier: LicenseRef-PolyForm-Noncommercial-1.0.0` + copyright header added to all 27 `instancespace/**/*.py` files. | **[Additive]** — comments only. |
| F7 | — | Model save/load round-trip (`Model.save()`/`Model.load()`), matching MATLAB's persistence | **Implemented and verified (v1.24)** — `instancespace/model.py`, signed `joblib`-based round-trip with `secret_key: bytes \| None`. 7 new tests: round-trip (signed/unsigned, incl. a genuinely fitted `SVC` and a real shapely `Polygon` compared directly, not flattened), wrong-key rejection, byte-tampering rejection before deserialising, the downgrade-attack guard, missing-signature rejection, and stale-`.sig`-cleanup on a subsequent unsigned save to the same path. `joblib` promoted from a transitive to a direct dependency (`pyproject.toml`) since it's now imported directly. | **[Additive]** — brand-new capability; nothing existing depends on it. |
| F8 | — | Unify `explore()` with build-time stage code (predict-mode dispatch on `PythiaStage`/`TraceStage`, matching MATLAB calling the same `PYTHIA()`/`TRACE()` in both modes) | **Narrowed by S1**: the PYTHIA half is resolved as a side effect of calling native `.predict_proba()` instead of reimplementing SVM math — nothing left there to reconcile. Remaining scope is TRACE only (footprint/alpha-shape membership testing is a genuinely different computation S1's insight doesn't extend to) | **[Behavior-changing risk]** — this refactors existing, working code. The full `tests/matlab_reference/` validation suite must pass identically before/after; treat any tolerance-threshold change during this work as a red flag to investigate, not a "close enough" adjustment. |
| F9 | — | Expand `explore()` to full evaluation scope: algorithm reconciliation + ground-truth performance metrics, matching MATLAB's `evaluateTestSet` | **Deferred (v1.36)** — pathway is fully decided (extend `explore()` itself, silent branch on ground truth; see companion pathways doc), but transitively blocked on F8 landing first (§6.0), and F8 is itself **[Behavior-changing risk]** with an unresolved ambition-level decision (#268) — not a quick sequencing gap. Not queued as near-term work until F8 actually lands. | **[Additive]** — new fields default to `None`; existing feature-only callers see no change. Add explicit test coverage for the "no ground truth present" path specifically, to lock this in rather than assume it. |
| F10 | — | PYTHIA hyperparameter-tuning *strategy* parity (`opts.pythia.tuning`) — added v1.25, found during a full-repo MATLAB audit, not previously tracked | **Implemented and verified (v1.30)** — `PythiaOptions.tuning`/`n_tuning_iter`, `PythiaStage._sobol_search()`; `use_grid_search` removed (superseded); 6 new tests + 12 pre-existing golden tests pinned to `tuning="bayes"`. | **[Behavior-changing]** — default tuning strategy changes from Bayes to Sobol for every caller not setting `tuning` explicitly; `use_grid_search` field removed entirely (breaking for any caller setting it). Verified: all pre-existing MATLAB-reference golden tests still pass once pinned to their intended strategy explicitly. |
| F11 | — | TRACE option-surface parity: `method`, `contra` — added v1.25, corrected + implemented v1.29 (see §6 for the correction: Python's `trace.py` already ports MATLAB's *legacy* algorithm, not `trace3` as originally claimed) | **Implemented and verified (v1.29)** — `TraceOptions.method`/`contra`; 2 new tests. `minInstances`/`minAreaFrac`/`pythia.skip` are `TRACE3`-only, out of scope. | **[Additive]** — new fields default to exactly the prior behaviour (`method='legacy'`, `contra=True`); both MATLAB-reference golden tests verified unchanged. |
| F12 | — | `utils/filter.py` performance (naive `O(n²)` nested-loop with per-pair `scipy.spatial.distance.cdist` calls) and a missing degenerate-uniformity guard — added v1.25 | **Fully implemented and verified (v1.54)** — guard landed v1.34; `O(n²)`→KD-tree performance rewrite (`scipy.spatial.cKDTree`) landed v1.54, ~980x speedup measured at n=2000 (7.87s→0.008s), bit-identical output confirmed via differential testing against the old algorithm. | **[Additive]** — both the guard and the KD-tree rewrite; the rewrite's differential tests (20 new cases across coincident points, exact-boundary distances, dense clusters, no-neighbours-at-all, and very small n) confirm identical output to the old `O(n²)` algorithm, and all pre-existing MATLAB-reference golden tests pass unchanged. |
| F13 | — | No Python equivalent of `ISAvalidateOpts.m` — eager, comprehensive type/range/membership validation of every recognised option field at load time, with clear `ISA:ISAvalidateOpts:*`-style errors, instead of a bad value surfacing as a confusing crash deep inside a stage — added v1.25 | **Implemented and verified (v1.34)** — `InstanceSpaceOptions.__post_init__()`; 14 new tests. | **[Additive]** — a new validation pass that only ever rejects what would already have failed later (or silently produced wrong output); doesn't change behaviour for any currently-valid options file. |
| F14 | — | PRELIM's missing "more than 5% of instances have a best-algorithm performance of exactly zero" data-quality warning (`ISA:PRELIM:manyZeroBest`) — added v1.25 | **Implemented and verified (v1.27)** — `PrelimStage._warn_many_zero_best()`; 2 new tests | **[Additive]** — a warning only; the underlying eps-substitution computation it warns about is already correctly ported (verified directly in `prelim.py`). |
| F15 | — (no MATLAB counterpart — production/infra capability) | Resumable `InstanceSpace` save/load (mid-pipeline checkpoint, not just F7's finished-`Model` snapshot) + stage-progress-reporting callbacks — added v1.31, found while auditing `feature/staged-matilda-support` for salvageable work | **Implemented and verified (v1.42)** — `instancespace/progress_reporter.py` (ported); `InstanceSpace.save()`/`load()` (signed `joblib`, mirrors F7); `__getstate__` on both `StageRunner` and `InstanceSpace` dropping the live cached executor before pickling; `run_stage()` auto-seeds initial inputs and reports progress/completion. 38 new tests; full suite (366, excluding the independently-slow pythia-tuning/T2 tests) passing. | **[Additive]** — new `InstanceSpace.save()`/`load()` and new `progress_reporter` module; no existing caller's output changes. The `StageRunner` half of the prerequisite work is tracked separately as #293 (bug, not new capability, closed). |

**F15 finding (v1.31, tracked as #294, folds in #293):** while auditing `feature/staged-matilda-support`
(2 commits ahead of `main`, otherwise stale — see the branch-audit note this same session) for
anything worth salvaging, found that branch's actual motivation exposes a real, previously
undocumented gap in F7's scope. The roadmap's own v1.18 entry recorded a "start pool → run stage
→ save → restart session → load → run next stage" scenario as the reason Q6 and F7 needed to
interact — that scenario only makes sense if `save()`/`load()` operate on the *in-progress*
`InstanceSpace` (so a partially-run pipeline can resume), not on a finished `Model`. v1.23's
retraction of that interaction is correct as a statement about what got built (`Model.save()`'s
actual target, `Model`, has no field referencing `InstanceSpace`/`StageRunner`, so nothing about
Q6's pool is part of what F7 pickles) — but it quietly answers a different question than the one
that motivated the interaction: it confirms F7 doesn't *break* on Q6's pool, not that F7 still
*does* what the save-and-resume scenario needed. Verified directly: `Model.load()` returns a
finished `Model` suitable for `explore()`/export, never a resumable `InstanceSpace` on which
`run_stage()` could be called for the next stage — every one of `Model.from_stage_runner_output()`'s
per-stage builders (`PrelimOut`, `SiftedOut`, `PilotOut`, ...) does unconditional dict-bracket
access against the stage-runner output dict, so it would raise `KeyError` on any run where a
later stage hasn't executed yet. The save-and-resume requirement is genuinely unimplemented.

Separately, that same branch adds job-queue-style progress reporting
(`instancespace/progress_reporter.py`, 528 lines) — an abstract `ProgressReporter` with
`HttpProgressReporter` (HTTP-POST callbacks per stage via stdlib `urllib`, no new dependency),
`FileProgressReporter` (JSON status files), `CompositeProgressReporter`, and a default
`NullProgressReporter`, wired into `InstanceSpace.build()`/`run_iter()` (confirmed compatible
with this repo's current `run_iter`/`_available_arguments` shape after S2's rewrite — no
structural conflict). This has no counterpart anywhere in this repo. **Not portable as-is**,
however: that module uses raw `pickle` as its own transport/persistence format —
`pickle.dumps(instance_space)` base64-encoded into HTTP callback bodies
(`OutputDetail.FULL`/`include_pickle_on_completion`), and `pickle.dump(instance_space, f)` written
to disk for both intermediate per-stage snapshots and the final model (`FileProgressReporter`'s
`stages_dir`). Unlike F7's signed-`joblib` scheme (§ above), these payloads carry no integrity
check at all. Folding the two findings together: the resumable-checkpoint mechanism this item
adds *is* the correct, signed replacement for what the progress reporter was doing with raw
pickle — for both the per-stage checkpoint case (previously unaddressed by anything in this repo)
and the final-completion case (previously addressed only for the narrower `Model`, not the full,
resumable `InstanceSpace`). Port the reporter interface/HTTP-transport/JSON-metadata parts;
replace its pickle-dumping paths with calls to this item's new `InstanceSpace.save()` instead.

**F15 — Implemented and verified (v1.42).** Ported `progress_reporter.py` from
`feature/staged-matilda-support` (`ProgressReporter` ABC, `Http`/`File`/`Composite`/
`NullProgressReporter` implementations, same HTTP callback/payload shape as the source branch),
adapted to this repo's conventions (`loguru`, `datetime.now(UTC)`, `mypy --strict` clean). Wired
into `InstanceSpace.__init__` (new optional `progress_reporter` param, defaults to
`NullProgressReporter()` — fully additive), `build()`, and `run_stage()`.

Two things needed beyond a mechanical port, both because the source branch's single-process
`build()`-loop design doesn't match the actual production usage this item exists to serve: a
SLURM job runs **one stage per invocation**, triggered by a scheduler, with the next stage's job
submitted separately and started at an unknown later time.

1. **A real pickling gap, found and fixed.** `StageRunner`/`InstanceSpace` didn't previously
   need to survive being pickled *after a stage had actually run* — Q6's cached `ThreadPoolExecutor`
   ends up embedded in `StageRunner._available_arguments`/`_schedule_output_data` (via
   `_InstanceSpaceInputs.executor`) and in `InstanceSpace._final_output` (an aliased reference to
   the same dict, not a copy) as soon as any stage runs. Added `__getstate__` on both classes to
   strip the live executor before pickling; `_get_executor()` recreates it lazily on the next call,
   same as after `close()` (Q6's existing pattern). Confirmed via a real reproduction: checkpointing
   an `InstanceSpace` after running just `PreprocessingStage`+`PrelimStage` crashed with
   `TypeError: cannot pickle '_queue.SimpleQueue' object` before this fix.
2. **`InstanceSpace.save()`/`load()`** — whole-class checkpoint via signed `joblib`, reusing F7's
   exact HMAC-SHA256 scheme (`ModelSignatureError` reused directly rather than duplicated). This is
   the actual, supported checkpoint/resume mechanism a driver script should call between SLURM
   invocations — as opposed to `FileProgressReporter`'s own raw, unsigned per-stage pickle snapshots
   (kept from the source branch, unchanged), which remain a debugging/observability convenience,
   not the supported resume path.
3. **`run_stage()` auto-seeds initial inputs on a truly fresh `InstanceSpace`** (checked via
   `not self._runner._available_arguments`) — needed so a SLURM job's *first* invocation can call
   `run_stage()` directly with no prior `build()`/`run_until_stage()` call to seed
   `_InstanceSpaceInputs`. A checkpoint loaded partway through never hits this branch (its
   `_available_arguments` is never empty). `run_stage()` also now reports `report_stage_completed`
   per call and fires `report_job_completed` once the whole schedule finishes
   (`_current_schedule_item >= len(_stage_order)`).

Found one regression via the full test suite (364/366 passing on first run): `_stage_report_name()`
assumed `stage.__name__` exists, but two pre-existing tests in `test_instance_space_executor.py`
intentionally stub `stage` as a plain string. Fixed with `getattr(stage, "__name__", str(stage))`;
updated those tests' bare `_runner` stub to also carry the fields `run_stage()` now reads
(`_available_arguments`/`_current_schedule_item`/`_stage_order`) — both fixes verified by a second
full run: 366 passed, 0 failed. 38 new tests total (progress-reporter unit tests against stubbed
`InstanceSpace`s; a real 2-stage checkpoint round-trip against `test_data/preprocessing/`'s
fixture, including signed/tamper/downgrade-attack cases mirroring F7's own suite; pickling
regression tests for both `StageRunner` and `InstanceSpace`).

**F10 finding, verified directly against `core/PYTHIA.m` and `utils/ISAdefaults.m` (v1.25, tracked as #287):**
MATLAB exposes `opts.pythia.tuning` with three modes — `'sobol'` (the *default*: quasi-random
Sobol-sequence sampling of the hyperparameter space, `nsobol` candidates evaluated in parallel
per CV fold via `parfor`, same "common random numbers" per-fold seeding discipline Q9 already
threads through Python), `'bayes'` (Bayesian optimisation — MATLAB's own comment calls `'bayes'`
and `'sobol'` "directly comparable tuning strategies"), and `'none'` (skip tuning, use
`opts.params` directly). Python's `stages/pythia.py` implements **only** the Bayes-style strategy
(`skopt.BayesSearchCV`) — MATLAB's actual default (`'sobol'`) has no Python implementation at
all, and there is no `tuning` option field to choose between strategies. This is a different,
narrower gap than F1 (already shipped): F1 covers *which classifier*; this covers *how its
hyperparameters get searched*, and Python currently only offers the MATLAB non-default. Scoping
this needs a decision on whether `'sobol'` is worth porting (a new quasi-random search, likely
`scipy.stats.qmc.Sobol`) purely for parity, or whether documenting Python's Bayes-only approach
as an intentional, permanent divergence is the better call — not decided here.

**F10 — Implemented and verified (v1.30).** Decided directly: port `'sobol'` and make it the new
default, matching MATLAB exactly (a **[Behavior-changing]** default swap, since Python's only
prior tuning strategy was Bayes). `PythiaOptions.tuning: str = 'sobol'` (`'sobol'`/`'bayes'`/
`'none'`) and `PythiaOptions.n_tuning_iter: int = 20` (MATLAB's own default budget) added.
`PythiaStage._sobol_search()` evaluates `n_tuning_iter` scrambled-Sobol `(C, gamma)` candidates
(`scipy.stats.qmc.Sobol`, mapped to MATLAB's own `2^[-10,4]` range per `sobolToParams`) via k-fold
CV, keeping the lowest-error candidate — a direct, deliberately lighter-weight port of MATLAB's
`sobolSearch`, replacing sklearn's `BayesSearchCV` machinery for this path entirely rather than
wrapping it. `tuning='none'` requires `opts.params`; `tuning='bayes'` reproduces the prior
(pre-F10) behaviour exactly, unchanged.

**Follow-on decision, made in the same pass:** asked directly whether `PythiaOptions.use_grid_search`
(the pre-existing `RandomizedSearchCV` "grid search" alternative to Bayes) should still exist now
that Sobol covers the same lightweight/random-ish search role — decided **remove it entirely**,
matching MATLAB (which has no grid-search mode at all, only sobol/bayes/none). Removed the field,
`RandomizedSearchCV` import, and its branch in `_fit_classifier`; `tuning='bayes'` is now pure
`BayesSearchCV`. This is itself behavior-changing for any caller that set `use_grid_search`
explicitly - a public option field removal, not just a default change. The MATLAB-legacy
`uselibsvm` JSON alias (previously silently mapped onto `use_grid_search`, itself a pre-existing
mismatch since MATLAB's `uselibsvm` has never meant "grid search") now maps to `"_"` (the
options-loader's existing, previously-unused "genuinely ignore this key" convention), matching
its actual deprecated status.

**Discovered, not fixed, out of scope:** while writing a test for `tuning='none'` with real
pre-calculated `opts.params`, found that supplying non-`None` `params` has *always* crashed
(`ValueError: Invalid dimension 1.0`, from `skopt`/`RandomizedSearchCV` rejecting scalar
search-space values) - for both `use_grid_search` settings, predating F10 entirely and never
previously exercised by any test. Not fixed here: unrelated to F10's actual scope (the tuning
*strategy* dimension, not the precalculated-params feature itself), and fixing it properly needs
its own scoping and verification pass. Tracked as a new finding rather than silently left
undiscovered - see #292.

6 new tests in `tests/test_pythia.py` (default is `'sobol'`; Sobol produces a valid C/gamma pair
and a fitted `SVC`; Sobol is seed-reproducible; `tuning='none'` without params raises; an
unrecognised `tuning` value raises) plus explicit `tuning="bayes"` pins added to the 12 pre-existing
MATLAB-reference golden tests across `tests/test_pythia.py`/`tests/test_pilot_pythia.py` that
validate the Bayes/grid-search paths specifically - without those pins they would have silently
started exercising the new Sobol default instead of what they were written to test. The 4 tests
whose entire subject was grid-search-vs-MATLAB-grid-search-CSV comparison
(`test_gridsearch_opts_gaussian`/`_poly` and their two `test_pilot_*_grid_*` analogues x2 = 6
total across both files) were deleted rather than repointed, since there is no grid-search code
path left on either side to validate.

**F11 finding — CORRECTED (v1.29), original write-up below was factually wrong.** The v1.25
audit claimed "Python's `trace.py` already implements the `'trace3'`-equivalent behaviour as its
only path." Re-verified directly against `core/TRACE.m` and `core/TRACE_legacy.m` before acting
on F11 (per this document's own "verify, don't guess" rule) and that claim does not hold:
`instancespace/stages/trace.py`'s `run_dbscan()`/`epsilon()`/`dist()`/`dbscan()`/`fit_poly()`/
`tight()`/`contra()` are a line-for-line port of `TRACE_legacy.m`'s `TRACEdbscan`/`TRACEepsilon`/
`TRACEdist`/`TRACEfitpoly`/`TRACEtight`/`TRACEcontra` — DBSCAN clustering followed by
`polyshape`/alpha-shape construction per cluster, contradiction removal between best-algorithm
footprints — not `TRACE3`'s alpha-shape-with-iterative-purity-tightening algorithm at all. This
is consistent with this repo predating MATLAB's ten-phase refactor that introduced the
`trace3`/`legacy` split in the first place (CLAUDE.md) — Python's `trace.py` was ported from
before that split existed, when "the DBSCAN algorithm" was simply the only TRACE there was.

**F11 — Implemented and verified (v1.29).** Given the corrected finding above, F11's real scope
was the option surface Python's *already-legacy* implementation was missing, not a new algorithm
to port. Added `TraceOptions.method: str = 'legacy'` (the only value this port can honour —
requesting anything else raises `NotImplementedError` with a clear message rather than silently
running legacy anyway, matching F13's "fail loud, not deep" philosophy) and
`TraceOptions.contra: bool = True` (matching MATLAB legacy's own default, and reproducing
Python's previous unconditional behaviour exactly), now actually gating the
contradiction-removal step instead of always running it. `minInstances`/`minAreaFrac` are
`TRACE3`-only parameters with no meaning in the legacy algorithm Python implements, and
MATLAB's `opts.pythia.skip`/`pythiaSkip` interaction lives inside `TRACE3`'s own PYTHIA-
availability branch — neither applies here and both are left out of scope; porting `TRACE3`
itself would be a separate, much larger future item if ever prioritised, not part of this
decision. 2 new tests in `tests/test_trace.py`: `method='trace3'` raises `NotImplementedError`;
`contra=False` measurably skips the contradiction-removal log step (this fixture's footprints
don't happen to overlap, so asserting on the log trace rather than on summary-table equality is
what actually proves the gate works, not incidental numeric difference). Both pre-existing
MATLAB-reference golden tests (`test_trace_pythia`, `test_trace_simulation`) still pass
unchanged, confirming the new fields' defaults reproduce prior behaviour exactly.

**F12 finding, verified directly against `core/FILTER.m` and `instancespace/utils/filter.py`
(v1.25, tracked as #289):** MATLAB avoids the `O(ninst²)` pairwise-distance cost entirely — `rangesearch(X, X,
opts.mindistance)` builds a KD-tree once and returns only the pairs actually within
`mindistance` (MATLAB's own comment: "at ninst ~ 20000 a dense Dx alone needs ~3GB"), and the
uniformity computation uses `knnsearch(Xkept, Xkept, 'K', 2)` the same way. Python's
`filter_instance()` is a **plain nested Python `for i in range(n_insts): for j in range(i+1,
n_insts):` loop**, calling `scipy.spatial.distance.cdist([x[i,:]], [x[j,:]])` per pair — not just
algorithmically `O(n²)` but paying full per-call Python/scipy overhead for every one of those
pairs, with no vectorisation at all. Separately, MATLAB's `unif` computation has an explicit
guard (`nkept < 2`, or all-NaN, or `mean(nearest)==0`) that returns `NaN` with a clear
`ISA:FILTER:degenerateUniformity` warning; Python's `compute_uniformity()` has no such guard —
those same degenerate cases (e.g. every retained instance filtered down to fewer than 2, or all
coincident in feature space) fall through to `numpy`'s own `RuntimeWarning`s (`invalid value
encountered`, `divide by zero`) and a silently-produced `NaN`/`inf`, not a domain-meaningful
message.

**F12 — Guard implemented and verified (v1.34); performance rewrite implemented and verified
(v1.54).** `compute_uniformity()` mirrors `core/FILTER.m`'s exact guard: fewer than 2 retained
instances, all-NaN nearest-neighbour distances, or a zero mean distance all return `NaN` with a
`logger.warning` (matching MATLAB's `ISA:FILTER:degenerateUniformity` message) instead of letting
`numpy` silently produce a `RuntimeWarning`-riddled `NaN`/`inf`. 2 new tests (fewer-than-2-kept,
all-coincident).

The `O(n²)`→KD-tree rewrite (v1.54): `filter_instance()`'s per-pair `cdist()` loop replaced with a
`scipy.spatial.cKDTree(x)` built once, then `query_ball_point(x, min_distance)` mirroring MATLAB's
own one-shot `rangesearch(X, X, opts.mindistance)` (verified both use inclusive `<=` distance
semantics). The elimination itself stays a genuinely sequential Python loop, not vectorised away -
`subset_index`'s running state means which instances end up marked redundant depends on
processing order, matching `core/FILTER.m`'s own comment on why this can't be parallelised either.
`compute_uniformity()`'s `pdist`/`squareform` replaced with `cKDTree(x_kept).query(x_kept, k=2)`
(column 1 = nearest *other* point, matching MATLAB's `knnsearch(Xkept, Xkept, 'K', 2)`). Verified,
not just plausible: all 10 pre-existing MATLAB-reference golden tests pass unchanged (bit-identical
output across all 4 `selvars_type` values); 20 new differential tests compare the KD-tree rewrite
directly against the old `O(n²)` algorithm (kept only in the test file as a reference oracle) across
5 edge cases a KD-tree can plausibly behave differently on - coincident/duplicate points, a pair
exactly at the `min_distance` boundary, a dense cluster where every pair is within `min_distance`,
instances with no neighbours at all, and very small `n` - crossed with all 4 `selvars_type` values.
Performance measured directly, not assumed: ~980x speedup at n=2000 (old: 7.87s, new: 0.008s), old
implementation confirmed to scale quadratically (0.18s at n=300, 0.72s at n=600, 1.88s at n=1000).
Full suite re-run: 416 passed (up from 396 pre-change, the +20 differential tests), 91.37%
coverage, `filter.py` itself now 98% covered (up from 42%).

**F13 — Implemented and verified (v1.34).** `InstanceSpaceOptions.__post_init__()` validates
every recognised, currently-ported option field (logical/unit-range/positive/pos-int/member/
text-list checks matching `ISAvalidateOpts.m`'s corresponding rules) at construction time, for
every construction path (`default()`, `from_dict()`, or direct construction), not just the two
entry points a `validate()`-style method called explicitly would have covered. Deliberately
scoped to fields that exist in this port today - MATLAB fields with no Python equivalent yet
(`pilot.method`/`dims`/`topoWeight`/`viewGroups`, `pythia.skip`/`ensembleMethod`,
`trace.minInstances`/`minAreaFrac`, `sifted.pval`, `cloister.maxFeatures`, `outputs.fig`) are out
of scope until those options themselves are ported (F2/F5/F8/F9) - adding validation for an
option that doesn't exist yet would be inventing scope, not porting it. Surfaced and fixed three
genuine pre-existing test-fixture data-quality issues while verifying against the full test
suite (not weakened to accommodate them, per this document's own "root cause, not alias" rule):
two JSON fixtures (`options.json`, `options_dataclass_names.json`, `options_dropped.json`) had a
deliberately-distinctive but semantically-invalid `betaThreshold`/`selvars.type` sentinel value
used purely to verify field-name round-tripping, corrected to an equally-distinctive *valid*
value instead of loosening the check; a real MATLAB-produced `.mat` fixture
(`test_data/serialisers/input/workspace.mat`) stored several `logical` fields as raw numeric 0/1
doubles (which MATLAB's own `ISAvalidateOpts.m` would also reject - `islogical(0)` is `false` in
MATLAB) and one field (`selvars.type`) as the literal typo `'Ftr&&Good'`, both fixed at the test
helper's consumption site (`bool(...)` casts; a `.replace("&&", "&")` normalisation) rather than
by touching the binary fixture. 14 new tests in `tests/test_options_validation.py` (representative
coverage across every check type, `from_dict()` hitting the same validation, `None`
seed/feats/algos correctly not rejected, and confirming the pre-existing unknown-field-name check
still fires before this new value-level one does).

**F13 finding, verified directly against `utils/ISAvalidateOpts.m` (v1.25, tracked as #290):** MATLAB validates
~35 individual option fields (logical scalars, positive-integer, unit-range `[0,1]`, membership
in an enumerated set, cell-of-text, etc.) immediately after loading `options.json` and before any
stage runs, erroring clearly on the first invalid one (`opts.pythia.classifier must be one of
{...}; got 'xyz'`). `InstanceSpaceOptions._validate_fields()` (`instancespace/data/options.py`)
only checks that JSON keys map to *known field names* — it never validates a value's type or
range. A bad value (negative `epsilon`, out-of-range `dims`, an unregistered `classifier` string)
currently passes straight through and either crashes confusingly deep inside a stage, or — worse
— silently produces a numerically-valid-looking but wrong result with no error at all.

**F14 — Implemented and verified (v1.27).** `PrelimStage._warn_many_zero_best()` warns when more
than 5% of instances have a best-algorithm performance of exactly zero, matching MATLAB's message
and threshold exactly. 2 new tests (fires above threshold, silent at/below it).

**F14 finding, verified directly against `core/PRELIM.m` lines 78-83/99-104 (v1.25, tracked as #291):** MATLAB
warns (`ISA:PRELIM:manyZeroBest`) when more than 5% of instances have a best-algorithm
performance of exactly zero, since the relative-performance matrix becomes uninformative (close
to 1 everywhere) for those instances once the `eps`-substitution kicks in. Verified Python's
`prelim.py` already performs the identical `eps`-substitution (`y_best[y_best == 0] =
np.finfo(float).eps`, both branches) — the *computation* is correct — but emits no equivalent
warning at all, so a dataset that would trigger this exact MATLAB diagnostic runs silently in
Python. Smallest of the five findings; mirrors the pattern Q2 already established (surface a
MATLAB data-quality diagnostic Python was computing the trigger condition for but not reporting).

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
on native objects. ~~**Depends on Q6 handling its own pickle-exclusion (added v1.18)**~~ —
**retracted, v1.23**: `Model.save()`/`Model.load()` operate on `Model`
(`instancespace/model.py`), which has no field referencing `InstanceSpace`/`StageRunner` (where
Q6's pool actually lives), so nothing about Q6's pool state is ever part of what F7 pickles.
Verified directly rather than assumed — see Q6's corrected §4 entry.

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

### 6.2 F3 audit findings — SIFTED vs. MATLAB's 4 historical fixes (added v1.27)

Checked `instancespace/stages/sifted.py` directly against each of the four fixes MATLAB's SIFTED
promotion made (see F3's own description, §6 table). No fix implemented here — audit only, per
CLAUDE.md's F3 audit-first rule.

1. **Thread-unsafe global `containers.Map` → persistent variable.** **Not applicable to
   Python.** Verified via grep: zero module-level mutable state anywhere in `sifted.py` — no
   shared cache of any kind for Python to have inherited this bug into.
2. **Nested-`parfor`-inside-GA bug.** **Not currently present, but only by accident, not by
   design — a real forward-looking risk for F2.** `_find_best_combination()`'s `pygad.GA`
   uses real OS-process parallelism (`parallel_processing=["process", n_cores]`) when
   `parallel_options.flag` is set. Its fitness function, `cost_fcn` (a `@staticmethod`, so it
   runs inside those worker processes), calls `PilotStage.pilot(...)` internally on every
   candidate evaluation. This is safe *today* only because `pilot.py` currently has zero
   parallelism of its own (confirmed separately during the v1.25 MATLAB audit, now part of
   F2's scope). **If F2 adds a thread/process pool for PILOT's `ntries` restart loop without
   checking whether `pilot()` is already running inside one of SIFTED's own worker processes,
   it will reintroduce this exact MATLAB bug** — a pool spawning its own nested sub-pool.
   Recorded here and cross-referenced in F2's own entry so whoever implements F2's parallelism
   sees it before, not after. Separately verified `cost_fcn`'s own `cross_val_score(...)` call
   passes no `n_jobs` (defaults to sequential) — no nested-parallelism hazard from that call
   specifically, today.
3. **`rng('default')` reset inside the per-candidate cost function, discarding any user
   seed.** **Not present.** `rng = np.random.default_rng(seed=self.general_opts.seed)` is
   created once per `SiftedStage` run and passed by reference through
   `evaluate_cluster`/`select_features_by_clustering`/`_find_best_combination` — never
   recreated or reset inside a loop. `ga_instance.general_options = self.general_opts` (a
   direct reference, not a fresh `GeneralOptions.default()`) threads the same user-configured
   seed into `cost_fcn`, so every GA candidate's internal `PilotStage.pilot(...)` call uses the
   identical seed — this is the deliberate "common random numbers" pattern already established
   elsewhere in this codebase (e.g. PYTHIA's `sobolSearch` per-fold reseeding), needed for a
   fair comparison across candidates, not an accidental MATLAB-style global-state reset (Python's
   `np.random.default_rng()` creates an isolated local generator regardless; there is no global
   RNG state for a stray reset to silently corrupt the way MATLAB's `rng('default')` can).
4. **Unvectorised correlation-selection loop.** **Present, confirmed.** `_compute_correlation()`
   is a plain nested Python `for i in range(rows): for j in range(cols):` loop calling
   `scipy.stats.pearsonr` once per (feature, algorithm) pair — not vectorised. Runs once per
   `SiftedStage` call (not per-GA-candidate, so lower urgency than a hot-loop would be), but a
   real, still-open, bounded-scope gap matching MATLAB's fix exactly. Per-column-pair NaN
   filtering (`valid_indices` can differ for every (i, j) pair) is what makes this non-trivial
   to vectorise fully — `np.corrcoef` alone doesn't handle ragged NaN patterns per pair.

**Net result:** 2 of 4 MATLAB issues don't apply to Python at all (items 1, 3); 1 is a real,
scoped, low-urgency performance gap (item 4); 1 is a latent risk for a *different*, not-yet-done
item to avoid reintroducing (item 2, flagged to F2).

---

### 6.3 External audit findings — PYTHIA/CLOISTER/SIFTED/PILOT/TRACE/PRELIM (added v1.43, updated v1.44-v1.45, v1.53, v1.58)

**Status as of v1.58: all six stages verified; CLOISTER (#299) fully resolved, no findings
remain open.** PRELIM, SIFTED, PILOT, PYTHIA, and CLOISTER were
triaged in v1.44-v1.45 — confirmed findings either fixed and shipped, or explicitly deferred with
a stated reason (never silently dropped). **TRACE (#302) is now verified too** — per direct
instruction, documented only (all 7 findings confirmed or confirmed-with-correction against
`instancespace/stages/trace.py`/`core/TRACE_legacy.m`, GitHub comment posted), no fixes
implemented in this pass. A batch of 43 findings from an
externally-sourced audit document was uploaded directly by the user on 2026-07-30, with an
explicit instruction: log all of them, place them first in priority order, and do not act on any
of them until independently verified. This is the same audit-first discipline already applied to
F1–F9, F3, and F10–F14 in this document, extended to a source whose reliability/provenance is
itself unestablished (unlike F10–F14, which a prior session verified directly against the MATLAB
source before recording).

**Per-stage status (updated v1.45):**
- **PRELIM (#303, 5 findings)** — verified; fixes shipped (see commit `7b96097`). GitHub comment
  on #303 has the fixed/deferred breakdown; a KNN-based tie-breaking improvement (deliberately
  *not* implemented — the audit's own tie-break fix was rescoped to detection-only per direct
  instruction, keeping "pick first") is tracked as a documented future-feature note, flagged as
  potentially relevant to the MATLAB repo too.
- **SIFTED (#300, 9 findings)** — verified; fixes shipped (see commit `a3e2859`, then issues 2/4/6
  in a follow-up commit `6731aa6` + `feat(options)` commit `6eab45b`). 7 fixed (issues 1, 2, 3, 4,
  6, 8 partial, 9): issue 2 (`opts.pval` now a real `SiftedOptions.pval` field, previously a
  hardcoded `PVAL_THRESHOLD` constant), issue 4 (GA fitness's internal PILOT call now uses
  `analytic=True, ntries=5` matching MATLAB's hardcoded `costfcn` constants — a genuine bug fix,
  since `PilotOptions.default()`'s own default is `analytic=False`; KNN neighbour count is now
  `dims + 1` via the new `SiftedOptions.dims` field), issue 6 (GA fitness now caches by
  feature-selection bitmask, scoped per-`ga_instance`/per-SIFTED-call, correctly isolated per
  worker under `parallel_processing`). 2 deferred (issues 5, 7 — GA fitness metric (MSE vs.
  classification loss) and clustering distance metric (Euclidean vs. correlation), explicitly held
  back for a separate design decision per direct instruction, not mechanical fixes). Full
  breakdown in the GitHub comments on #300.
- **PILOT (#301, 7 findings)** — verified; fixes shipped (see commit `f29dbbe`). 4 fixed (issues
  2, 4 partial, 5, 6); 3 deferred as one coherent `cost_weight`/`alpha` semantics chunk (issues 1,
  3, 7), overlapping F2. Full breakdown in the GitHub comment on #301.
- **PYTHIA (#298, 10 findings)** — verified; fixes shipped (see commits `b6508cb`, `1fe551f`, then
  issue 6 in a follow-up commit `b43b71e`). 9 fixed (issues 1, 2, 3, 4, 5, 6, 7, 8, 9): issue 6
  (sample weights now thread into every CV fold's fit during Sobol/Bayes candidate ranking, not
  just the final full-data fit, via a new `_cv_fit_params` helper — matches MATLAB's `Wtrain`
  threading through `evalFoldClassifier`, confirmed the ranking metric itself stays unweighted in
  both MATLAB and Python). 1 deferred (issue 10 — eval/skip mode, overlaps F8/F9). Full breakdown
  in the GitHub comments on #298.
  - Issue 4 (`n_tuning_iter` ignored for `tuning='bayes'`; SVM's discrete Bayes search space) was
    initially triaged as "deliberate prior design decisions, not bugs" — that framing was wrong,
    corrected in a follow-up comment on #298 and fixed in commit `1fe551f`: MATLAB's
    `core/PYTHIA.m` uses `opts.nTuningIter` identically for `'sobol'` and `'bayes'`, and gives
    `'svm'` the same continuous Bayes search space as every other classifier — Python's old
    hardcoded-30-iteration Bayes budget and SVM-only discrete 30-point LHS candidate list were
    both genuine deviations, not preserved parity. Fixing the search space exposed a new,
    separately-tracked quality gap (**#304**): `skopt`'s `BayesSearchCV` converges measurably
    slower than MATLAB's `bayesopt` at the shared default `n_tuning_iter=20` on at least one
    MATLAB fixture (24/30 tolerance-gated metrics vs. the 90% bar) — not resolved by raising
    Python's default, since 20 is MATLAB's own default too. Test-level impact contained (raised
    `test_build_pilot_pythia.py`'s `BAYES_N_ITER_FOR_TESTS` 15 → 40); the production-default
    question remains open on #304.
- **CLOISTER (#299, 5 findings)** — verified; fixes shipped (see commit `8cf1d1a`, then issue 5 in
  a follow-up commit `fb4ad1c`). All 5 fixed (issues 1-5: `max_features` guard + convex-hull
  fallback, NaN-robust correlation/bounds, correct convex-hull failure semantics distinguishing a
  genuine `z_edge` failure from a threshold-driven `z_ecorr` one, the "weakely"→"weakly" typo that
  shared issue 3's root cause, and issue 5's new `CloisterOptions.hull_dims` option letting
  `_compute_convex_hull` restrict its geometry computation to the first N projected columns while
  still returning full-dimensional vertices — currently dormant in practice since PILOT's
  projection matrix is hard-coded to 2 rows everywhere in this repo, so `hull_dims="all"` and
  `hull_dims=2` are equivalent until F2 (unshipped) changes that). **#299 has no remaining open
  findings** and can be closed. Full breakdown in the GitHub comments on #299.
- **TRACE (#302, 7 findings)** — verified, per direct instruction **documented only, not
  implemented** (stopping one step earlier than the stages above). Against
  `instancespace/stages/trace.py` and MATLAB's `core/TRACE_legacy.m` (the only TRACE variant this
  port implements - `TRACE3` is out of scope per F11): 6 of 7 confirmed exactly as described,
  several reproduced empirically rather than taken on the audit's word - issue 1
  (`tight()`'s `.contains(MultiPoint(...))` returns a single bool, not a per-point mask; the
  follow-on `polydata[boundary]` indexing raises a real `IndexError`), issue 2 (`tight()`'s
  `return None` path has no guard before `contra()`'s unconditional `.is_empty` check), issue 4
  (`fit_poly` only removes low-purity triangles, not zero-element ones - MATLAB's `elements == 0 OR
  purity < threshold` narrowed to just the second disjunct), issue 5 (`build()` can return an
  empty-but-non-`None` `Footprint.from_polygon(Polygon())` instead of `throw()`'s `polygon=None`
  when no cluster ever fits a polygon - two representations of "empty footprint" that downstream
  `is None` guards don't both catch), issue 6 (`dist()` raises `TypeError` on 1D data, reproduced
  directly - currently unreachable since `Z` is always 2D everywhere in this repo, same hard-coded
  constraint CLOISTER's own audit noted in #299 issue 5), issue 7 (DBSCAN labels are `float64` by
  construction, not integer - confirmed via source, no functional bug demonstrated). Issue 3
  (division by zero in `contra()`'s purity calculation) is confirmed as a real, live defect but
  **not** as described - it does not raise `ZeroDivisionError` (NumPy scalar division by zero
  silently warns and returns `nan`/`inf` instead of raising); reproduced live by running
  `tests/test_build_trace.py::test_trace_simulation` with `RuntimeWarning` promoted to an error,
  which fails at `trace.py:700` on every run. Cross-cutting finding beyond the 7: `tight()`'s
  entire body (issues 1-2's code) has 0% line coverage in the existing suite, and `contra()`'s
  differing-purity branches (where issues 1-3's fixes would actually execute) are never taken
  either - every contradiction resolution in the current suite falls into the "purity equal,
  ignore" branch. TRACE's contradiction-refinement path is a complete test-coverage blind spot,
  independent of whether/how the 7 findings get fixed. Full breakdown in the GitHub comment on
  #302.

Logged verbatim as GitHub issues rather than summarised here, so nothing is lost or paraphrased
before an actual review happens: parent tracking issue **#297**, with one sub-issue per stage —
**#298** (PYTHIA, 10 findings), **#299** (CLOISTER, 5 findings), **#300** (SIFTED, 9 findings),
**#301** (PILOT, 7 findings), **#302** (TRACE, 7 findings), **#303** (PRELIM, 5 findings). Full
original text preserved in each sub-issue body — see those issues, not this section, for the
actual claims and proposed fixes.

**Priority:** per direct instruction, this verification work was first in the queue — ahead of
§6.0's previously-recorded F8 → F9 → F2 → F5 execution order. "First priority" meant *triage and
independent verification first*, not "implement the proposed fixes first." As of v1.53 all six
stages are verified, so this batch no longer blocks §6.0's order — but the confirmed-and-deferred
findings above (PYTHIA issue 10, SIFTED issues 5/7, PILOT issues 1/3/7, and all 7 of TRACE's, none
yet implemented) remain a real backlog, tracked on their respective GitHub issues, not silently
closed out by "verification is done." CLOISTER's #299 has no remaining deferred findings — see the
CLOISTER bullet above.

**Known overlaps flagged for the eventual verification pass (not resolved here):**
- PYTHIA finding #4 (`n_tuning_iter` ignored for `tuning='bayes'`) — resolved, not stale: F10
  (v1.30) added `PythiaOptions.n_tuning_iter` but only wired it into the new Sobol path, not
  Bayes; fixed in commit `1fe551f` (see the PYTHIA bullet above and #304 for the follow-on
  search-convergence question it exposed).
- SIFTED findings #1/#2 (correlation-threshold logic, hard-coded p-value) touch code adjacent to,
  but distinct from, what F3's own audit (§6.2 above) already reviewed in `_compute_correlation`
  — F3 only checked the unvectorised-loop performance question, not this threshold-logic claim.
- CLOISTER has no existing F-item in this document at all (F1–F15) — if any of its 5 findings
  verify as real, they need a brand-new F-item, not a slot in an existing one.
- TRACE findings #1–#7 (`tight()`/`contra()`/`fit_poly()`/`build()`/`dist()`/`dbscan()`) are
  distinct from F11 (v1.29), which only added the `method`/`contra` *option surface*, not the
  internal correctness of `contra()`'s purity computation or the other methods this batch names.
- PRELIM findings #1–#5 are distinct from F14 (v1.27), which only added the `manyZeroBest`
  warning.

None of the above resolves which side (this audit vs. this document's own prior audits) is
correct where they might conflict — that's exactly what the independent assessment needs to
determine, not something to guess at while logging.

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

**R1 — Implemented and verified (v1.28).** Re-read directly from the actual GitLab source
(`gitlab.com/ita-ml/pyispace`, cloned at commit `a5ee7f3a02e`), not the PyPI sdist this section
was originally written against, and not from memory — `pyispace/pilot.py`'s `adjust_rotation(Z,
Ybad, theta=135.0)` plus its call site in `train.py::train_is()` (`bad_instances =
mode(Ybin*1, axis=1)[0] == 0`, i.e. instances where the majority-vote across algorithms in
`Ybin` is "not good"). Ported into `PilotStage`: `PilotOptions.adjust_rotation: bool = False`
(new field, defaulted so every existing `PilotOptions(...)` call site — production and test —
is unaffected); `PilotInput` gained a `y_bin` field, auto-wired by the stage runner's
name-based matching from `SiftedOutput.y_bin` (no explicit plumbing needed, confirmed by
reading `stage_runner.py`'s `run_stage()`); `PilotStage.adjust_rotation()` and
`PilotStage._bad_instances()` are direct ports of PyISpace's two pieces; `pilot()` applies the
rotation to `Z` and `A` (`A = rot @ A`) only when `adjust_rotation=True` and at least one bad
instance exists, matching PyISpace's own guard. Six new tests in `tests/test_pilot.py`: pairwise
distances and orthonormality preserved by the rotation itself; the bad-instance centroid lands
at 135° within tolerance (the §8.2 test-debt row's explicit requirement, not just "distances
preserved"); the flag reproduces the same `Z` up to rotation as the flag off; two independent
runs on identical input rotate identically (the roadmap's Phase R checkpoint's "visually
consistent across two independent runs", verified as exact equality since PILOT's numerical
solve is itself seed-deterministic); `adjust_rotation=True` without `y_bin` raises `ValueError`
rather than failing silently; no bad instances present leaves `Z` unchanged. MATLAB-side
`PILOT.m` change is explicitly out of scope for this repo (see CLAUDE.md) — tracked separately
for the MATLAB v0.9.1 issue set, not done here.

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

**T1 — Implemented and verified (v1.30).** `pytest-cov` added as a dev dependency. Measured a
real baseline before picking a threshold (not a guess): a full local run —
`poetry run pytest --cov=instancespace --cov-report=term-missing` — reports **79% total coverage**
(3101 statements, 641 missed; 306 passed, ~70 minutes since it includes the genuinely slow
T2/PYTHIA-tuning tests). `[tool.coverage.run]`/`[tool.coverage.report]` added to `pyproject.toml`
with `fail_under = 75` (a few points below the measured baseline, enough to catch a real
regression without tripping on normal fluctuation). `poe`'s `test_pytest` task and
`.github/workflows/validation-tests.yml`'s pytest step both updated to
`pytest --cov=instancespace --cov-report=term-missing`, which reads `fail_under` from
`pyproject.toml` automatically - confirmed this actually gates (not just reports) by running a
deliberately partial test subset and observing `FAIL Required test coverage of 75.0% not
reached` before restoring the full suite.

### T2 — Add a real end-to-end `build()` integration test
**Implemented and verified (v1.24)** — `tests/test_build_integration.py`, built against
`tests/test_data/preprocessing/`'s 213-instance/10-algorithm real fixture (same dataset
`instance_space_from_files`'s existing partial tests already use). A genuinely slow test
(~8.5 minutes, serial - `parallel.flag: false` in that fixture's options) but the only one in
the repo that calls `.build()` end to end; found a real crash (see Q8's entry) that no other
test in the repo was positioned to catch.
**[Additive]** — a test.
Construct a real `InstanceSpace` with real metadata + options, call `.build()`, assert it
completes and produces a `Model` with every expected stage output populated. The Python
equivalent of MATLAB's `test_integration.m` — currently doesn't exist in any form.

### T3 — Add `conftest.py` with shared fixtures
**[Additive]** — test infrastructure only.
Reference-dataset loader, common `Mock` builders — reduce duplication across ~35 files.

### T4 — Fix `poe test` to actually include pytest
**Implemented and verified (v1.27)** — added `test_pytest = "pytest"` to `[tool.poe.tasks]` and
appended it to `test.sequence`. Verified `poe --dry-run test` now runs `ruff check --no-fix` →
`mypy --strict .` → `black . --check` → `pytest`, matching CI's `poetry run pytest` step plus the
lint/type/format checks CI doesn't currently run at all (confirmed: `validation-tests.yml` has no
ruff/mypy/black step, only pytest — Q11 removed the dead commented-out lint step rather than
re-enabling it).
**[Additive]** — CI/tooling config only.
One-line change to `[tool.poe.tasks]`. Makes the local dev command match what CI actually does.

### T5 — Version-pin `tests/matlab_reference/`'s MATLAB provenance
**[Additive]** — test-fixture metadata only.
Record the exact MATLAB commit/tag the fixtures were generated from. See §8.3 for the fuller
cross-repo data-sharing proposal this connects to.

**Design + script written (v1.49), unverified.** §8.3 proposal 1 (the MATLAB export
script) designed and written as `tests/matlab_export/pyis_export_reference_data.m` +
`README.md`, on direct instruction — living in this repo (not pushed to
`andremun/InstanceSpace`, a deliberate deviation from #278's original "implementation
lives on the MATLAB side" framing). Coverage matrix, transfer-format choice (CSV by
default, matching `output/scriptcsv.m`'s existing convention), and the staged-`build()`
design (prelim→sifted→pilot→cloister once, then re-running only `{pythia,trace}` per
option variant) are all verified directly against `andremun/InstanceSpace`'s actual source
(commit `a0197ee3`), not guessed — including catching two real mistakes a naive
field-name grep produced (SIFTED's `out.Z` doesn't exist on the real return value, a
nested GA-fitness-function's own local `PILOT(...)` call reuses the name `out`; PYTHIA's
`best`/`good` belong to TRACE, a stray match came from a comment) and one real bug in the
script itself (`isinterior` is `polyshape`'s containment method, not `alphaShape`'s —
TRACE3, the toolkit's actual default, needs `inShape` instead; fixed before finalizing).
`provenance.json` (T5's actual ask) is written by the script itself on every run. **Not
executed against real MATLAB** — no MATLAB installation available in this session, so
this is written-carefully-but-unconfirmed, not verified; #278 stays open with this
recorded as the concrete next step (run once, diff against the existing committed
fixtures, fix whatever surfaces) rather than marking T5 done on unexecuted code.

### T6 — DAG-resolver edge-case tests with representative stages
**[Additive]** — tests only.
Two synthetic linear stages can't exercise mutating-stage handling, `RunBefore`/`RunAfter`, or
the ambiguous-same-output-at-same-schedule-step error path. Add cases that actually trigger each.

**T6 — Closed, not implemented (v1.29).** Confirmed directly (not assumed): S2 replaced the
type-matching DAG auto-resolution algorithm with a hardcoded explicit 7-stage order (see S2's
own entry and v1.20's note above — this was already anticipated as the likely outcome, just not
yet confirmed against the landed code). Grepped `stage_runner.py` for `ambiguous`/`mutating`
special-casing: neither exists anywhere in the file post-S2. T6's actual subject matter —
"ambiguous-same-output-at-same-schedule-step" detection and mutating-stage handling — was part
of the deleted auto-resolver and has no equivalent in the new explicit-order design; there is
nothing left to write a test against. `RunBefore`/`RunAfter` (the one piece of T6's scope that
*does* still exist post-S2) already has dedicated coverage in `tests/test_stage_runner.py`
(`test_extra_stage_run_after`, `test_extra_stage_run_before`,
`test_multiple_extra_stages_share_a_wave`, `test_extra_stage_without_attachment_point_raises`),
added incidentally during S2's own implementation. Closed as `not_planned` (#279) rather than
implemented — this is S2's own predicted "or skip T6 entirely" outcome, not a new decision.

### T7 — Consolidate fragmented per-stage test files
**[Additive]** — test-file reorganisation only, no production code touched.
Decide and document what belongs in top-level `test_<stage>.py` vs. `exploreIS/<stage>/
test_<stage>_unit.py` vs. `..._validation.py`, then merge or clearly demarcate — starting with
PILOT as the worst-fragmented case.

**T7 — Deferred (v1.29).** Explicitly left for later per direct instruction: consolidating
`test_pilot.py`/`exploreIS/pilot/test_pilot_unit.py`/`exploreIS/pilot/test_pilot_validation.py`
(and the equivalent trio for sifted/trace/pythia/prelim) risks touching test cases that other
in-flight work (F10, F11, and any future stage-behaviour change) will also need to touch —
better done once those land, not concurrently with them. No code or test files changed; issue
#280 left open.

**Layout decided (v1.37, direct instruction) — not yet implemented.** Resolves #280's open
"two-way or three-way split" question in favour of two, and removes the `exploreIS/` directory
tree entirely — every test file lives directly under `tests/`, distinguished purely by a
`test_build_*`/`test_explore_*` filename prefix rather than by directory location.

- **Per-stage trios collapse to a pair, both at `tests/` root:** `test_build_<stage>.py` (today's
  top-level `test_<stage>.py`, unchanged) and `test_explore_<stage>.py` — a merge of today's
  `exploreIS/<stage>/test_<stage>_unit.py` (stubbed-dependency orchestration tests) *and*
  `exploreIS/<stage>/test_<stage>_validation.py` (MATLAB-reference numerical validation) into one
  file, not two. Applies to all five stages `explore()` covers: pilot, prelim, sifted, pythia,
  trace. (CLOISTER and preprocessing have no explore-time counterpart — see below.)
  - TRACE has a third existing file, `exploreIS/trace/test_trace_executor_reuse.py` (Q6's
    pool-reuse unit tests) — folds into the same merge, so `test_explore_trace.py` is a 3-way
    merge for TRACE specifically, not 2-way.
- **Non-stage-specific explore-time files** (`exploreIS/test_explore_stage_iter.py`,
  `exploreIS/test_extract_features.py` — orchestration across the whole pipeline, not one
  stage) move to `tests/` root with an `explore_` prefix for consistency:
  `test_explore_stage_iter.py` (already named right, just relocates) and
  `test_explore_extract_features.py` (renamed, since `_extract_features` is an `explore()`-only
  method with no build-time equivalent to pair against).
- **Build-time files with no explore-time counterpart** get the `test_build_` prefix too, for
  naming consistency even though there's no pair to disambiguate from: `test_cloister.py` →
  `test_build_cloister.py` (CLOISTER isn't in `ExploreStage`, build-only), `test_preprocessing.py`
  → `test_build_preprocessing.py`, `test_filter.py` → `test_build_filter.py` (training-only
  feature/instance filtering, no explore-time equivalent).
- **Multi-stage build-time integration files keep their existing name, just gain the prefix:**
  `test_pilot_pythia.py` → `test_build_pilot_pythia.py`, `test_prepro_n_prelim.py` →
  `test_build_prepro_n_prelim.py`, `test_prelim_filter.py` → `test_build_prelim_filter.py`. These
  span more than one stage, so they don't get folded into any single stage's file.
- **Everything else in `tests/` is untouched** — genuinely cross-cutting infrastructure with no
  single-stage build/explore distinction to make (`test_instance_space_executor.py`,
  `test_model_save_load.py`, `test_options_validation.py`, `test_general_options.py`,
  `test_print_options.py`, `test_verbose_logging.py`, `test_stage_runner.py`,
  `test_serialisers.py`, `test_metadata.py`, `test_load_file.py`, `test_manual_selection.py`,
  `test_get_classifier_fcn.py`, `test_remove_all_nan_row.py`, `test_basic.py`,
  `test_plotting.py`, `test_build_integration.py` — already matches the convention as a
  whole-pipeline build-time integration test, `conftest.py`, `utils/`).
- **`tests/exploreIS/` (and its five stage subfolders, each with an `__init__.py`) is deleted
  entirely** once every file above has moved — nothing should remain under it.

**Implemented (v1.40).** Every move above landed exactly as decided: 5 stage pairs
(`test_build_<stage>.py` + merged `test_explore_<stage>.py`, TRACE's 3-way merge included), the
2 non-stage explore files, the 3 build-only renames, and the 3 multi-stage integration renames.
`tests/exploreIS/` (5 subfolders, 6 `__init__.py` files, and its own `README.md`) deleted
entirely; its README's content moved to a new `tests/README.md`, rewritten for the flat layout.
Stale `tests/exploreIS/` references also fixed in the root `README.md`, `tests/matlab_reference/
README.md`, `test_instance_space_executor.py`'s docstring, and this document's own companion
pathways doc. `mypy --strict .`: clean (67 files, down from 79 — the trio-merges reduced file
count as T8's own entry anticipated). `pytest --collect-only`: 345 tests collected, identical to
the pre-move count — confirms no test function was lost or duplicated during the merges. Full
suite re-run to confirm zero regressions before committing, per this document's own
verification rule.

**T8 before T7 (decided v1.33, direct instruction).** T8's own file-by-file error breakdown
(above) cites specific current file paths; if T7 merges/renames those fragmented per-stage
trios first, that breakdown goes stale and needs re-auditing against wherever the code ends
up. Doing T8 first avoids that - annotations travel with the functions when files are later
merged, so nothing done for T8 is wasted by T7 landing afterward. Not a hard blocker either
way, but whichever lands second should re-verify against the other's result rather than trust
what's already written; T7's own trio-consolidation might also shrink T8's remaining work by
eliminating duplicate/dead test functions along the way, which is a reason to keep T7 as the
second step rather than skip it.

### T8 — Close the `tests/` `mypy --strict` annotation gap (added v1.32, tracked as #295)
**[Additive]** — type-annotation-only changes to test files; no behaviour change to any test's
assertions or to production code.

`pyproject.toml`'s `[tool.mypy]` only sets `disallow_untyped_defs = 'True'`; the project's
actual configured check (`poe check`'s `check_mypy`) runs `mypy --strict .`, layering on many
more rules beyond the config. Verified directly: 160 errors across 21 files under that real
command, all in `tests/` except one in `instancespace/plotting.py` - `instancespace/` itself is
otherwise clean. Confirmed pure pre-existing debt, unaffected by this session's F10/F11/T1/#65
batch: ran the identical command against both the commit before that batch and the current HEAD
- both give exactly 160 errors. Breakdown: 102 `no-untyped-def` (missing `-> None`/parameter
types, mechanical), 31 `no-untyped-call` (cascades from the above), 27 spread across `misc`/
`assignment`/`unused-ignore`/`arg-type`/`type-arg`/`method-assign`/`comparison-overlap`/
`var-annotated`/`return-value`/`no-any-return`/`attr-defined` - these look like genuine
type-mismatch findings, not just missing annotations, and need individual review. Worst files:
`exploreIS/trace/test_trace_unit.py` (19), `test_plotting.py` (18), `test_instance_space_executor.py`
(18), `exploreIS/prelim/test_prelim_unit.py` (16), `exploreIS/sifted/test_sifted_unit.py` (15).
Doesn't block anything today - CI (`validation-tests.yml`) only runs `pytest`, no lint/type step
at all - so this is low-priority hygiene, not urgent; flagged mainly because the ~27
non-annotation findings could be masking real bugs in test fixtures/mocks.

**Implemented and verified (v1.38).** `mypy --strict .`: 160 → 0 errors across 21 files (plus
`instancespace/plotting.py`'s one production-code error). Mechanical fixes (missing `-> None`,
`NDArray[np.double]` in place of bare `np.ndarray`) applied via a script keyed off mypy's own
suggested-fix text. The ~27 non-mechanical findings were resolved individually, not blanket-
ignored — most turned out to be genuinely-untyped duck-typed test doubles (`SimpleNamespace`/
`Mock` standing in for a real dataclass), fixed by introducing `typing.cast(RealType, ...)` as
this suite's now-established pattern for that. One finding was a real production bug in the type
signature, not the test: `_explore_trace`'s return type carried a vestigial `tuple[...] | None`
that the function body could never actually produce and no caller (production or test) ever
checked for — removed the `| None` at the source rather than adding `assert is not None` at
every one of the 3 affected call sites, per this document's own root-cause-not-alias rule. Two
files (`test_explore_stage_iter.py`, `test_instance_space_executor.py`) monkeypatch bound methods
on real class instances with intentionally-wrong-signature stand-ins; kept the relevant helper
parameter deliberately unannotated (with a comment recording why) so mypy doesn't check a body
meant to violate the real signatures, using targeted `# type: ignore` only at the few call sites
where a concretely-typed local object still needed one. Full `pytest` suite re-run after all
fixes: 345 passed, 0 failed — confirms this was genuinely annotation/type-only with zero
behavioural change, not just plausible.

### T9 — ruff/black debt survey (added + partially fixed, v1.39)
**[Additive]** — the 4 fixed rule categories are genuine bug-adjacent fixes with no behaviour
change to any test assertion; the remaining, deliberately-undone categories are pure style/
documentation debt with zero runtime effect either way.

`poe test`'s `test_ruff` step (`ruff check --no-fix`) would fail today if actually run — CI
(`validation-tests.yml`) doesn't run it at all (Q11's decision not to re-enable a lint gate), so
nothing currently depends on this being zero. Surveyed the full `ruff check .` output (263 errors
at the time of this survey, now lower after the fixes below) on direct instruction, to separate
what's worth fixing from what's just volume. Verified via `git stash` during T8 that essentially
all of it predates this session's work (409 errors in the untouched tree vs. 263 after T8's
annotation fixes — T8's real type annotations incidentally resolved some of ruff's own weaker
`ANN001` checks as a side effect, not a deliberate ruff-focused pass).

**Fixed (cherry-picked as genuinely bug-adjacent, not cosmetic):**
- **`NPY002`, 11 instances, all in `tests/`** — legacy `np.random.rand()`/global RNG state
  replaced with `np.random.default_rng()` instances. Matters because Q9 already did real work
  threading explicit seeds through PILOT/SIFTED/PYTHIA for reproducibility; a test still using
  the legacy *shared, global* RNG is inconsistent with that guarantee and could behave
  differently under parallel test execution, since the legacy RNG is process-wide mutable state
  shared across every test that touches it.
- **`DTZ005`, 2 instances, both in `instancespace/instance_space.py`** (`explore()`'s
  `dataset_id`/`timestamp` generation) — naive `datetime.now()` replaced with
  `datetime.now(tz=timezone.utc)`. A naive timestamp is a latent bug waiting for a multi-timezone
  deployment (this ships behind a production server per CLAUDE.md) to make it visible as
  inconsistent `dataset_id`s or misleading `ExploreResult.timestamp` ordering.
- **`RUF100`, 3 instances, all in `tests/test_plotting.py`** — three `# noqa: E402` comments
  (module-level imports after `matplotlib.use("Agg")`) that ruff's current version simply
  doesn't flag for this pattern, so the suppressions were dead weight; confirmed by running
  ruff against the file with the comments stripped before removing them, not assumed. Two
  pre-dated this session (from before T8); one was T8's own addition, copied from the
  surrounding (already-stale) convention without re-checking whether it was still needed —
  a small instance of the same "copied an existing pattern without verifying its premise still
  holds" risk this document warns about elsewhere.
- **`F401`, 1 instance, `tests/exploreIS/pilot/test_pilot_unit.py`** — an unused `import pytest`
  (the file has no `@pytest.fixture`/`pytest.raises` usage), removed as part of the same file's
  T8 edit.

**Documented, deliberately not fixed** (the remaining ~248 errors as of this entry):
- **`SLF001`, 125 instances — the largest single category, entirely in `tests/`.** This is
  private-attribute access (`instance_space._model`, `runner._available_arguments`,
  `stage._inputs()`, etc.) from test code reaching into implementation internals to set up
  fixtures or assert on internal state. This is *intentional* test design, not an accident —
  these are unit tests deliberately checking internal bookkeeping (stage-rerun invalidation,
  pool reuse, schedule ordering) that has no public API surface to assert on instead. "Fixing"
  this would mean either ~125 individual `# noqa: SLF001` comments (pure busywork, no behaviour
  or readability improvement) or restructuring tests to avoid touching internals (which would
  make several of them *worse* tests — e.g. T2/Q8's stage-rerun tests specifically need to
  inspect `StageRunner._available_arguments` to prove invalidation happened, there's no
  public-API equivalent to check the same thing). If this is ever revisited, the right fix is a
  `[tool.ruff.lint.per-file-ignores]` entry silencing `SLF001` for `tests/*`, not touching each
  call site — recorded here as the recommended shape of a future fix, not implemented now.
- **`D103`/`D104`, 62 instances** — missing docstrings on public test functions/`__init__.py`
  packages. Low value here specifically: this suite's test names are already descriptive
  (`test_prelim_ood_warning_fires_above_threshold`), so a docstring would mostly restate the
  name rather than add information.
- **`COM812` (15), `PT001` (7), `E501` (8), `PT018` (8), `ANN001` (3), `ICN001`/`RET504`/`F541`
  (1 each)** — pure style/formatting (trailing commas, fixture-decorator parens, line length,
  composite asserts, import alias convention, needless assign-then-return, f-string with no
  placeholder). Cosmetic; most are auto-fixable (`ruff check --fix`) if this is ever revisited
  wholesale, but nothing here indicates a behavioural risk.
- **`PLR2004`, 15 instances** — bare literals (`2`, `10`) in test assertions instead of named
  constants. Marginal value in test code where the literal is usually self-explanatory in
  context (`assert x_transformed.shape == (5, 3)` after five rows were constructed two lines
  above) — a named constant would mostly add indirection, not clarity, for most of these.

**`black --check .`:** 9 files would be reformatted, all confirmed pre-existing (via the same
`git stash` comparison used for the ruff survey) and not touched here — a separate, smaller
formatting-only pass if this is ever prioritized.

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

**Added v1.51:** a full audit of every file under `tests/` (not only `tests/matlab_reference/`)
found dead/duplicate/stale data, a test-run scratch directory committed to git, and a structural
split where build-time and explore-time MATLAB-comparison fixtures live in two incompatible
directory layouts, with most stages missing explore-time coverage entirely. Full findings and a
phased remediation proposal are in `docs/test_data_audit.md` — read that before touching any file
under `tests/test_data/` or the other stray top-level data directories it identifies
(`tests/fileidx/`, `tests/Prelim_out/`, etc.). Two open decisions there need a direct answer
before the layout-unification step (§7 of that document) can start; everything else in it is
either already-verified fact or a proposal awaiting sign-off, not a completed action.

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
| v1.23 | 2026-07-28 | Before starting Q6/F7/Q8 (the next three items in §6.0's order), evaluated their scope against the post-fold code rather than assuming the existing text still applied. Two findings: (1) the Q6↔F7 pickle-exclusion interaction (v1.18) rests on a premise that doesn't hold — verified `instancespace/model.py`'s `Model` (F7's actual pickle target, per its own pathway) is a frozen dataclass with no field referencing `InstanceSpace`/`StageRunner` (Q6's only proposed pool-holder locations), so there is nothing for a `__getstate__`/`__setstate__` exclusion to guard against; retracted the shared-checklist item in §6.0 and corrected Q6's §4 entry accordingly — this is a scoping correction, not new work. (2) Q8's hard T2 dependency, already flagged in §6.0's "Hard dependencies" list, was confirmed concrete rather than theoretical: grepped `tests/` for any full-7-stage `.build()` call and found none — the only `instance_space_from_files` callers (`test_prepro_n_prelim.py`, `test_load_file.py`, `test_preprocessing.py`) stop at preprocessing/prelim. T2 does not exist in any form, so Q8 cannot be written yet. User confirmed (asked directly, not guessed): insert T2 as Q8's direct prerequisite rather than deferring Q8 to later in the order or substituting a different third item. §6.0's recommended-order table updated (new row 5.5 for T2). No code changed. |
| v1.24 | 2026-07-28 | Q6, F7, T2, and Q8 implemented and verified on `v0.9.0/development-branch-QSF`. Q6: a lazily-created, reused `ThreadPoolExecutor` cached on `InstanceSpace`, threaded through `TraceStage` in place of a fresh per-call pool; 10 new unit tests. F7: `Model.save()`/`Model.load()`, signed `joblib` round-trip with optional `secret_key`, following the format decided in v1.14/v1.17; 7 new tests covering both modes plus the tampering/downgrade-attack/missing-signature guards; `joblib` promoted to a direct dependency. T2: `tests/test_build_integration.py`, the repo's first true end-to-end `.build()` test, against the real 213-instance/10-algorithm fixture already used by other partial tests — genuinely slow (~8.5 min, serial) but the only test positioned to catch what it caught. Q8: verified negative result — rerunning `CloisterStage` neither invalidates `PythiaStage`'s output (same schedule wave) nor blocks `run_stage(TraceStage)` (no real dependency on `CloisterStage`, and not wave-position-blocked either); §6.1's speculative over-invalidation concern does not reproduce for the current built-in 7-stage order — scoped to that order, not a general correctness claim about `_rollback_to_schedule_index()`. **Real bug found by T2, unrelated to the invalidation question**: `StageRunner.run_stage()`'s unconditional `deepcopy(inputs)` crashed (`TypeError: cannot pickle '_queue.SimpleQueue' object`) on Q6's new `TraceInputs.executor` field — Q6's own unit tests missed this because they called `TraceStage` directly, bypassing `run_stage()`'s deepcopy step. Root-caused and fixed via a new `_deepcopy_stage_inputs()` helper in `stage_runner.py` that pre-seeds `deepcopy`'s memo with any live `ThreadPoolExecutor` so it passes through by reference — necessary for two independent reasons (executors aren't deepcopy-safe at all, and even a successful copy would silently defeat Q6's pool-reuse purpose), not just to silence the crash. Added a fast, dedicated regression test (`test_run_stage_does_not_deepcopy_a_live_executor`) so this doesn't require another 8-minute build to re-catch. Full test suite (excluding the two genuinely slow PYTHIA-tuning/T2 tests, which were run and verified separately): 224 passed, 0 failed. §6.0's recommended-order table, status line, and Q6/F7/T2/Q8 entries updated to reflect all four items done. |
| v1.25 | 2026-07-28 | Two-part follow-up requested directly: (1) folded a PILOT-parallelism finding into F2 — verified directly against `core/PILOT.m`, MATLAB parallelises its `ntries` BFGS multi-start restarts via `parfor` reusing an existing pool (`gcp('nocreate')`), while Python's `pilot.py` runs the equivalent loop strictly sequentially with no `parallel_options` field at all; added as pathway step 5 and a new decision point (thread pool matching Q6, or process pool matching MATLAB's actual `parfor`/OS-process semantics — a real port decision, not just translation, given the GIL). (2) A full pass through every `.m` file in `andremun/InstanceSpace` (cloned to `/workspace/instancespace`) not already referenced anywhere in this document, cross-checked against the corresponding Python module, per an explicit "do not change the MATLAB code, this is read-only" instruction. Five new, previously-untracked findings recorded as F10–F14 (all "Not started," none implemented): F10 (PYTHIA's actual default tuning strategy, Sobol-sequence search, has no Python implementation at all — Python only implements the non-default Bayes strategy, with no `tuning` option to choose either way); F11 (TRACE's `method`/`contra`/`minInstances`/`minAreaFrac` option surface and the `pythia.skip` interaction are entirely absent from Python — `TraceOptions` only has `use_sim`/`purity`); F12 (`utils/filter.py` is a naive nested-Python-loop `O(n²)` with per-pair `cdist()` calls where MATLAB uses a KD-tree/`rangesearch`, and has no guard for the degenerate-uniformity cases MATLAB explicitly checks and warns on); F13 (no Python equivalent of `ISAvalidateOpts.m`'s eager type/range/membership validation of ~35 option fields — Python's `_validate_fields` only checks field *names*, never values); F14 (PRELIM's `manyZeroBest` 5%-threshold data-quality warning has no Python equivalent, though the underlying eps-substitution computation it warns about is already correctly ported). Explicitly not implemented, listed for the user to prioritise, per direct instruction. No MATLAB code touched (read-only clone). No Python code changed beyond F2's documentation update. |
| v1.26 | 2026-07-28 | Registered F10–F14 as GitHub sub-issues under the Phase F parent (#260), matching the existing F1–F9 issue format and milestone (v0.5.0, milestone 3): F10 → #287, F11 → #288, F12 → #289, F13 → #290, F14 → #291. Each of F10–F14's finding write-ups above (§6) updated with a "tracked as #NNN" cross-reference. No code changed. |
| v1.27 | 2026-07-28 | Following a direct risk/reward review of all open issues, worked T4, F14, T1 (in progress), F3, and R1 as a batch, deferring T5 (needs an as-yet-nonexistent MATLAB-data collection script, left open). T4: `pyproject.toml`'s `test` poe task was missing `pytest` entirely (`test.sequence` ran only `ruff`/`mypy`/`black`) — added `test_pytest = "pytest"`; also surfaced that CI (`validation-tests.yml`) runs only `pytest`, no lint/type/format step at all, consistent with Q11's prior decision not to re-enable one. F14: `PrelimStage._warn_many_zero_best()` — warns when more than 5% of instances have a best-algorithm performance of exactly zero, matching MATLAB's `ISA:PRELIM:manyZeroBest`; 2 new tests. F3: audited `stages/sifted.py` against MATLAB SIFTED's 4 historical bug fixes (thread-unsafe global cache, nested-`parfor`-inside-GA, RNG-reset losing the seed, unvectorized correlation loop) — full result in the new §6.2. Two of the four confirmed not present in Python at all (no module-level mutable state anywhere in the file; the per-candidate `rng` reuse across GA fitness evaluations is deliberate common-random-numbers, not an accidental reset — Python's `rng` is threaded by reference throughout, never reseeded mid-search). One real gap confirmed and left as F3's own future fix target: `_compute_correlation()` is an unvectorized nested Python loop calling `scipy.stats.pearsonr` per feature/algorithm pair. One genuinely new, forward-looking finding with no current bug: `SiftedStage._find_best_combination()`'s GA already runs `PilotStage.pilot(...)` inside `pygad.GA`'s own OS worker processes when `parallel_options.flag` is set — F2's planned PILOT `ntries` parallelism must detect (or otherwise avoid) already running inside one of those workers, or it reintroduces the exact nested-`parfor`-inside-GA bug MATLAB's SIFTED promotion fixed; cross-referenced into F2's entry here and in the companion pathways document. Audit only, per the audit-first rule — no SIFTED code changed. T1 (pytest-cov): dependency added (`pytest-cov` 7.1.0); coverage config and CI wiring carried into v1.28 below once a real baseline number was in hand. |
| v1.28 | 2026-07-28 | R1 implemented and verified: PyISpace's `adjust_rotation()` read directly from the actual GitLab source (`gitlab.com/ita-ml/pyispace`, cloned at commit `a5ee7f3a02e`) rather than the PyPI sdist v1.25's original write-up was based on, per direct instruction not to port from memory. Full detail in the R1 section (§7) above; summary: `PilotOptions.adjust_rotation: bool = False` (new, defaulted field — no existing call site changes), `PilotInput.y_bin` (auto-wired by the stage runner's name-based matching from `SiftedOutput.y_bin`, confirmed by reading `stage_runner.py::run_stage()` rather than assumed), `PilotStage.adjust_rotation()`/`_bad_instances()` as direct ports, applied inside `pilot()` only when the flag is set and at least one poorly-solved instance exists. 6 new tests in `tests/test_pilot.py`, including the centroid-angle assertion §8.2's test-debt row explicitly required (not just "distances preserved") and the Phase R checkpoint's cross-run reproducibility check. MATLAB-side `PILOT.m` change explicitly left out of scope for this repo. |
| v1.29 | 2026-07-28 | Decided F10, F11, T6, T7 on direct instruction. T6: confirmed its "or skip entirely" premise actually holds now that S2 has landed — no `ambiguous`/`mutating` special-casing remains anywhere in `stage_runner.py`, and `RunBefore`/`RunAfter` (the one piece of its scope still relevant) already has dedicated tests added during S2 itself. Closed #279 as `not_planned` rather than implemented. T7: left open/deferred per direct instruction — consolidating the fragmented per-stage test files risks touching test cases F10/F11 (this same batch) and any future stage-behaviour change will also touch; better done once those settle. No code changed for either. F11's premise was independently re-verified against the actual MATLAB source before acting on it, per this document's own "verify, don't guess" rule — see §6's corrected F11 finding: the v1.25 audit's claim that Python's `trace.py` already implements `trace3` was wrong; `trace.py`'s `run_dbscan`/`epsilon`/`dbscan`/`fit_poly`/`tight`/`contra` are a line-for-line port of `TRACE_legacy.m`'s `TRACEdbscan`/`TRACEepsilon`/`TRACEdist`/`TRACEfitpoly`/`TRACEtight`/`TRACEcontra` (DBSCAN clustering, not trace3's alphaShape-iterative-tightening), consistent with this repo predating the MATLAB ten-phase refactor that introduced the trace3/legacy split at all (CLAUDE.md). F10 and F11 implementation follow in this same version — see their own sections below for what changed. |
| v1.30 | 2026-07-28 | F11 and F10 implemented; T1 completed. F11: `TraceOptions.method`/`contra` added (§6's corrected finding); `method` only accepts `'legacy'`, raising `NotImplementedError` for anything else rather than silently running legacy anyway; `contra` (default `True`, matching prior behaviour) now actually gates the contradiction-removal step. 2 new tests; both pre-existing MATLAB-reference golden tests unchanged. F10: `PythiaOptions.tuning`/`n_tuning_iter` added, default `'sobol'` (matching MATLAB, a deliberate default change); `PythiaStage._sobol_search()` is a direct, lighter-weight port of MATLAB's `sobolSearch`. Follow-on decision made in the same pass (asked directly, not guessed): `PythiaOptions.use_grid_search` removed entirely now that Sobol supersedes its role, matching MATLAB's own option surface (no grid-search mode there at all) — a second, independent behavior-changing/breaking edge beyond the default swap. `uselibsvm`'s JSON mapping corrected to genuinely ignore the key (previously silently aliased onto `use_grid_search`, itself already a mismatch). Discovered, filed separately rather than fixed under F10: pre-calculated `opts.params` has always crashed for the tunable classifier, unrelated to F10's own scope (#292). 6 new Sobol tests; 12 pre-existing golden tests across two files pinned to `tuning="bayes"` explicitly so they keep validating what they were written for; 6 tests whose entire subject was the now-removed grid-search path deleted. T1: real baseline measured (79%, full suite, 306 passed) before setting `fail_under = 75` in `pyproject.toml`; `poe`'s `test_pytest` task and CI's pytest step both updated to `--cov=instancespace --cov-report=term-missing`; confirmed the gate actually fails a build (not just reports) by observing it trip on a deliberately partial test subset. |
| v1.31 | 2026-07-29 | Audited `feature/staged-matilda-support` (a stale, non-mainline branch, 2 commits ahead of `main`) for salvageable work, on direct instruction. Found three things: (1) its `idx`/`selvars` SIFTED fix is the same bug this repo already root-cause-fixed (removed the redundant field) — theirs instead aliases consumers onto the correct field while leaving the redundant one in place, the exact anti-pattern CLAUDE.md warns against; nothing to pull from there. (2) A real, currently-unfixed bug: `StageRunner.__init__`'s `defaultdict(lambda: False)` is unpicklable (verified directly) — filed as #293 (bug, not new capability), linked under the Phase F parent. My first two comments on #293 incorrectly claimed F7's design constraint bans pickle outright; corrected in-thread — F7's *actual*, current decision (v1.14/v1.17, superseding the v1.9 text I'd misquoted from #267's stale issue body) is signed `pickle`/`joblib` with optional HMAC signing, already implemented. (3) That branch's `progress_reporter.py` (528 lines, job-queue-style HTTP/file progress callbacks) has no counterpart in this repo, but isn't portable as-is — it uses raw `pickle` as its own transport format (base64-encoded in HTTP bodies, dumped to disk per-stage), with no integrity check, unlike F7's signed scheme. Discussion with the user surfaced a genuine, previously undocumented scope gap: F7 only covers a *finished* `Model`, never a resumable mid-pipeline `InstanceSpace` — the exact "save → restart → load → continue" scenario that originally motivated the v1.18 Q6↔F7 interaction note, which v1.23's retraction answered around rather than closed (see the new F15 finding, §6). Filed as **F15** (#294), folding in the progress-reporter port, with #293 as an explicit prerequisite/sub-issue. No code changed. |
| v1.32 | 2026-07-29 | Scoped a new Phase T item on direct instruction, after explaining a `tests/` `mypy --strict` annotation gap the user asked about. Filed as **T8** (#295), linked under the Phase T parent (#273). Corrected an earlier mistake in the same discussion: initially reported this gap shrank from 160→105 errors as a result of this session's F10/F11/T1/#65 batch, but that compared two different, incompatible `mypy` invocations (a weaker `mypy instancespace tests` without `--strict`, against the actual configured `mypy --strict .`). Re-verified properly: ran the identical `mypy --strict .` command against both the commit immediately before that batch and the current HEAD — both give exactly 160 errors, confirming the gap is untouched pre-existing debt, not something this session's work affected either way. No code changed. |
| v1.33 | 2026-07-29 | Recorded a sequencing decision on direct instruction: T8 (#295) before T7 (#280). T8's file-by-file error breakdown cites current file paths; if T7's test-file consolidation lands first, that breakdown goes stale and needs re-auditing against the new structure. Doing T8 first avoids that (annotations travel with functions through a later merge); T7 second may also benefit by inheriting a smaller remaining annotation surface if consolidation removes duplicate/dead test functions. Not a hard blocker either direction, but recorded so whichever lands second knows to re-verify rather than trust stale numbers. No code changed. |
| v1.34 | 2026-07-29 | Following a direct risk/reward re-evaluation of all pending items, worked a low-risk batch: #292 (PYTHIA precalc-params crash), #293 (StageRunner unpicklable lambda), F12's degenerate-uniformity guard (performance rewrite deferred), and F13 (ISAvalidateOpts.m eager validation). #292: `PythiaStage._fit_precalculated()` now bypasses search entirely for pre-calculated `opts.pythia.params`, matching MATLAB's own `crossValPredict`/`trainFinalClassifier` branch, instead of feeding scalars into `BayesSearchCV` (which requires real search-space `Dimension`s); `ParamSpec.from_precalc()` added as `reported()`'s inverse, handling KNN's categorical `Distance` index correctly; 7 new tests across all 6 classifiers plus a dedicated KNN-category round-trip test. #293: `StageRunner`'s `defaultdict(lambda: False)` replaced with a module-level `_default_false()`; 1 new pickle round-trip test. F12: `compute_uniformity()`'s degenerate-case guard added (2 new tests); performance rewrite deliberately left for later. F13: `InstanceSpaceOptions.__post_init__()` added, validating every recognised, currently-ported option field; found and fixed three genuine pre-existing test-fixture data-quality issues along the way (two JSON fixtures' distinctive-but-invalid sentinel values, corrected to distinctive-but-valid ones; a real MATLAB `.mat` fixture's numeric-not-logical bool fields and one literal-typo string value, both fixed at the test consumption site, not the binary fixture) - full detail in each item's own §6 section above. 14 new tests in `tests/test_options_validation.py`. Full suite (excluding the two genuinely slow PYTHIA-tuning/T2/pilot-pythia test files, each re-run and verified separately): all passing. |
| v1.35 | 2026-07-29 | On direct instruction ("scope F3 but don't implement"), replaced the companion pathways doc's stale pre-audit F3 section (which still said "fix whatever the audit finds — can't be scoped further until the audit runs," dating from before v1.27's audit actually ran) with a concrete, unimplemented pathway for the one confirmed gap: `_compute_correlation()`'s unvectorised per-`(feature, algorithm)`-pair loop (`stages/sifted.py:1031-1076`). Recommends a fast-path/fallback split (vectorise the common no-NaN case via a manual Pearson formula + `scipy.stats.t.sf` for the p-value; keep the existing, already-verified per-pair `pearsonr` loop unchanged for any pair with a ragged NaN mask) over a full masked-pairwise-sums vectorisation, specifically to avoid having to hand-re-derive `scipy.stats.pearsonr`'s own degenerate-case behaviour (zero-variance columns, `n_valid < 3`) for a function that only runs once per `SiftedStage` call, not per-GA-candidate — the performance upside doesn't justify that extra verification surface. Test plan specified (no-NaN, all-NaN-column, scattered-NaN, zero-variance, `n_valid<3` cases) but no tests written and no code changed — scoping only, per instruction. Also added F9 (#269) to the local task queue, explicitly gated on F8 (#268, still open) per the existing §6.0 F8→F9 ordering — not started. |
| v1.36 | 2026-07-29 | Two decisions on direct instruction, following v1.35's scoping pass. **F3 closed, won't-fix** (#263, `state_reason: not_planned`): the one confirmed gap scoped in v1.35 runs once per `SiftedStage` call, not per-GA-candidate, so its performance upside doesn't justify the new edge-case test surface (zero-variance columns, `n_valid<3`) a safe fix needs — explained in the closing comment; the scoped pathway stays in the companion doc in case a future profiling run changes that calculus. No code changed. **F9 corrected from "queued" to deferred** (comment on #269): v1.35's "add F9 to the queue" framed it as next-up-after-F8, which understated the dependency — F9 only unblocks once F8 lands, and F8 is itself **[Behavior-changing risk]** with its own unresolved ambition-level decision (#268: lighter shared-function extraction vs. fuller `Stage` contract change), not a quick sequencing gap. F9 is not near-term work until F8 actually lands; roadmap §5 table and #269 both updated to say so. |
| v1.37 | 2026-07-29 | Resolved #280's open "two-way or three-way split" question for T7 on direct instruction: two files per stage (`test_build_<stage>.py`/`test_explore_<stage>.py`), not three — today's `..._unit.py`/`..._validation.py` explore-time pair merges into one `test_explore_<stage>.py` file (TRACE's extra `test_trace_executor_reuse.py` folds in too, a 3-way merge for that stage only). Also resolves a question #280 didn't originally ask: the `exploreIS/` directory tree is removed entirely, every file living flat under `tests/`, disambiguated by filename prefix instead of directory. Confirmed via a clarifying question (not guessed): non-stage-specific explore files get an `explore_` prefix for consistency (`test_extract_features.py` → `test_explore_extract_features.py`); build-only stage files (`cloister`, `preprocessing`, `filter` — no explore-time counterpart in `ExploreStage`) get `test_build_` for the same reason; multi-stage build-time integration files (`test_pilot_pythia.py`, `test_prepro_n_prelim.py`, `test_prelim_filter.py`) keep their existing name and just gain the `test_build_` prefix rather than being folded into any single stage's file. Full file-by-file mapping recorded in T7's own section above. Decision only — no files moved or renamed yet; T7 stays gated behind T8 landing first (v1.33), and T8's own pytest verification was still running in the background when this was recorded. |
| v1.38 | 2026-07-29 | **T8 implemented and verified** (#295): closed all 160 `mypy --strict` errors across `tests/` (21 files) plus `instancespace/plotting.py`'s one production-code error, down to zero. ~133 mechanical (missing `-> None`, bare `np.ndarray` → `NDArray[np.double]`) fixed via a script keyed off mypy's own suggested-fix text; the ~27 non-mechanical findings resolved individually rather than blanket-ignored. Introduced `typing.cast(RealType, SimpleNamespace(...))` as this suite's now-established pattern for duck-typed test doubles (no prior `cast()` usage existed). Found and root-cause-fixed one real production bug in the process, not just a test annotation: `InstanceSpace._explore_trace`'s return type carried a vestigial `tuple[...] \| None` that the function body could never produce and no caller ever checked for — removed the `\| None` at the source (`instance_space.py`) instead of adding `assert is not None` at each of the 3 affected call sites, per this document's own root-cause-not-alias rule. Two test files that deliberately monkeypatch bound methods with wrong-signature stand-ins (`test_explore_stage_iter.py`, `test_instance_space_executor.py`) kept one helper parameter intentionally unannotated, documented inline, so mypy doesn't check a body that's meant to violate the real signatures. Full `pytest` suite re-run after every fix: 345 passed, 0 failed, ~62 minutes — confirms zero behavioural change, matching the `[Additive]` tag. Committed in 3 parts (T7/F3/F9 doc-only decisions first, since they needed no runtime verification; the mypy/test-file diff held back until the full suite actually finished and confirmed green, per this document's own "never leave verification as a promise" rule). |
| v1.39 | 2026-07-29 | Filed **T9**: surveyed the full `ruff check .` output (263 errors) on direct instruction, following a question about what the ruff/black debt actually means. Fixed 4 categories confirmed genuinely bug-adjacent rather than cosmetic: `NPY002` (11, legacy global-RNG calls in tests replaced with `np.random.default_rng()` instances, for consistency with Q9's seed-threading reproducibility work), `DTZ005` (2, in `instance_space.py`'s `explore()` — naive `datetime.now()` replaced with `datetime.now(tz=timezone.utc)`, a real latent bug for a multi-timezone server deployment), `RUF100` (3, unused `# noqa: E402` in `test_plotting.py`, verified unused by testing with the comments stripped before removing them), and `F401` (1, an unused `import pytest`). Documented, but deliberately left unfixed, the remaining ~248: `SLF001` (125, the largest category — private-attribute access from tests, which is intentional test design reaching into internals with no public-API equivalent to assert on instead, not a mistake; recommended fix if ever revisited is a `per-file-ignores` config entry for `tests/*`, not 125 individual suppressions) and the rest (`D103`/`D104`, `COM812`, `PLR2004`, `PT001`/`PT018`, `E501`, `ANN001`, `ICN001`/`RET504`/`F541`) as low-value cosmetic/documentation debt not worth the effort, per the same risk/reward logic used to close F3. `black --check .`: 9 files, all confirmed pre-existing, left untouched. Full detail in T9's own section (§8). |
| v1.40 | 2026-07-29 | **T7 implemented** (#280), per the v1.37-decided layout. All 5 per-stage trios collapsed to pairs (`test_build_<stage>.py` + a merged `test_explore_<stage>.py`; TRACE's extra `test_trace_executor_reuse.py` folded in as a 3-way merge); 2 non-stage explore files relocated/renamed; 3 build-only files and 3 multi-stage integration files gained the `test_build_` prefix; `tests/exploreIS/` (5 subfolders, 6 `__init__.py`, its own `README.md`) deleted entirely, its README content moved to a new `tests/README.md` rewritten for the flat layout. Fixed stale `tests/exploreIS/` path references discovered along the way in the root `README.md` (2 places), `tests/matlab_reference/README.md` (2 places), `test_instance_space_executor.py`'s docstring, and the companion pathways doc's own T7 entry — all found via a repo-wide grep before declaring the move complete, not assumed clean. Also cleaned up two small pieces of dead boilerplate uncovered while merging PRELIM's and SIFTED's `_unit.py` files: a `sys.path.insert` hack and an `if __name__ == "__main__": pytest.main(...)` block, both redundant since T3's `conftest.py` landed and not present in any sibling test file. Caught and fixed a `UP017` finding (`datetime.UTC` alias preferred over `timezone.utc`) introduced by T9's own `DTZ005` fix earlier this session — new debt from this session's own prior edit, not pre-existing, so fixed rather than left for T9. `mypy --strict .`: clean, 67 files (down from 79 — expected, trio merges reduce file count). `ruff check .`: 244 errors, consistent with T9's documented categories (no new findings beyond the transient `UP017` already fixed). `pytest --collect-only`: 345 tests, unchanged from before the move. Full suite re-run once, alone (a `--collect-only` check run concurrently against an earlier full-suite background run raced on `test_serialisers.py`'s module-level output-directory cleanup and caused a spurious `FileNotFoundError` — diagnosed as self-inflicted process interference, not a T7 regression, and re-run cleanly) before committing. |
| v1.41 | 2026-07-29 | Bookkeeping sweep on direct instruction ("reevaluate all remaining tasks... complete any bookkeeping left"): pulled every open GitHub issue in the repo and cross-checked each against this document's actual status. Found and fixed three gaps. (1) **#248 (Phase Q parent) closed** — GitHub's own sub-issue tracker already showed 11/11 (100%) complete, but the parent issue itself had been left open; closed with `completed`, matching the precedent already set when Phase S's parent (#281) was closed the same way. (2) **T9 filed as #296** and immediately closed — every other roadmap item touched this session (F3, F9, T7) got a GitHub issue, but T9's survey-and-cherry-pick work (v1.39) never did; filed retroactively with the same fixed/documented breakdown already in this document, then closed since the fixable part is done and the rest is a deliberate won't-fix, same reasoning as F3. (3) **§6.0's "Recommended order" table corrected** — row 8 (F9) and the F3-audit row still read as if both were live, undecided pending items; both now show their actual v1.36 resolution (F9 deferred behind F8, F3 closed) with a strikethrough, matching the style already used for the table's other completed rows. No code changed. Confirmed via the same sweep that Phase F (#260, 5 real open items: F2, F5, F8, F9, F12's remaining perf-rewrite, F15) and Phase R (#270, 1 open: R2) parents are correctly still open — nothing else needed closing. |
| v1.42 | 2026-07-30 | **F15 implemented and verified** (#294, closed): ported `progress_reporter.py` from `feature/staged-matilda-support` (`ProgressReporter` ABC, `Http`/`File`/`Composite`/`Null` implementations, same HTTP callback/payload shape), wired into `InstanceSpace.__init__`/`build()`/`run_stage()`. Beyond a mechanical port — needed for the real production usage (a SLURM job runs one stage per invocation, next stage triggered separately at an unknown later time), which the source branch's single-process `build()`-loop design didn't support: (1) found and fixed a real pickling gap — `StageRunner`/`InstanceSpace` didn't survive being pickled once a stage had actually run (Q6's cached `ThreadPoolExecutor` ends up embedded in `_available_arguments`/`_schedule_output_data` and in `InstanceSpace._final_output`, an aliased reference to the same dict); added `__getstate__` to both classes, dropping the live executor, recreated lazily via existing `_get_executor()`. (2) `InstanceSpace.save()`/`load()` — whole-class checkpoint via signed `joblib`, reusing F7's exact HMAC-SHA256 scheme (`ModelSignatureError` reused directly). (3) `run_stage()` now auto-seeds initial inputs on a truly fresh `InstanceSpace` (so a SLURM job's first invocation needs no prior `build()`/`run_until_stage()` call) and reports stage/job completion. Found and fixed one regression via the full suite: `_stage_report_name()` assumed `stage.__name__` exists, breaking two pre-existing tests that stub `stage` as a plain string — fixed with a `getattr` fallback, those tests' bare `_runner` stub updated to carry the fields `run_stage()` now reads. 38 new tests; full suite verified twice (364/366 then, after the fix, 366/366 — excluding the independently-slow pythia-tuning/T2 tests). Full detail in F15's own §6 write-up. Committed as `4be2332` on `v0.9.0/development-branch-QSF`. |
| v1.43 | 2026-07-30 | Logged, but explicitly did **not** act on, a batch of 43 findings from an externally-sourced audit document uploaded directly by the user, per direct instruction ("log all of them and place them first... do not act upon them as they require independent assessment"). Filed as GitHub issue **#297** (parent) + 6 stage-specific sub-issues — **#298** PYTHIA (10), **#299** CLOISTER (5), **#300** SIFTED (9), **#301** PILOT (7), **#302** TRACE (7), **#303** PRELIM (5) — each preserving the original upload's text verbatim rather than summarising it, so nothing is lost or paraphrased before an actual review happens. New §6.3 added, cross-referencing all 6 sub-issues and flagging plausible overlaps with already-tracked items (F10's `n_tuning_iter`, F3's `_compute_correlation` audit, F11's TRACE option surface, F14's PRELIM warning) without resolving which side is correct — that's for the independent assessment, not this logging pass. §6.0 updated to note this batch now sits ahead of the F8→F9→F2→F5 order for priority purposes (triage first, not implementation first). No code changed — explicitly out of scope for this pass. Also closed out unrelated stale bookkeeping found in the same session: F15 (#294, see v1.42) marked implemented in its own table row and write-up. |
| v1.44 | 2026-07-31 | Verified and triaged v1.43's audit batch for PRELIM, SIFTED, PILOT, and PYTHIA (CLOISTER and TRACE remain unverified — see updated §6.3). Confirmed findings were fixed and shipped, one stage at a time, each with full regression tests and a fresh MATLAB-reference-test pass before commit: **PRELIM** (`7b96097`) — minimisation formula, zero-tie detection, NaN-aware statistics, configurable IQR multiplier; a KNN-based tie-break improvement added to the backlog as a future feature (not implemented) per direct instruction, keeping "pick first" for now. **SIFTED** (`a3e2859`) — correlation-threshold logic, GA's wrong (unfiltered) feature matrix, silhouette `min_clusters`/suggested-K indexing, density-filter column slicing; 5 of 9 findings deferred (p-value option, KNN `dims+1`/analytic PILOT call, GA fitness metric, fitness caching, correlation-distance clustering) — each needs its own design/verification pass, not a mechanical fix. **PILOT** (`f29dbbe`) — precalculated-`alpha` crash, rank-deficient `analytic_solve()` (switched `inv`→`pinv`), numerical-branch R² (the analytic branch's own R² was independently re-derived and found already correct, contradicting the audit's claim both were broken), premature float16 downcast; 3 findings (`cost_weight`/`alpha` scalar-weighting semantics) deferred as one coherent chunk overlapping F2. **PYTHIA** (`b6508cb`) — znorm mu/sigma (matching a bug `InstanceSpace._explore_pythia` had already independently discovered and worked around, whose stale docstring is now corrected), degenerate-weights crash, degenerate-label `StratifiedKFold` crash (new `_ConstantClassifier` sentinel, shaped to need zero changes in `_explore_pythia`), precalc-params column count, CV metrics using `Yhat` instead of `Ysub`, selection-index off-by-one (fixed via a `-1` "no selection" sentinel matching a convention `_explore_pythia` already established, *not* the audit's proposed 1-based indices, which would have silently broken `TraceStage`'s existing 0-based consumption), selector precision/recall formula (matched to MATLAB's per-instance `any(...)` computation, verified directly against `core/PYTHIA.m`), and `pr0` reversed (P(class 1) instead of P(class 0)); 3 findings deferred (`n_tuning_iter`/SVM Bayes domain — deliberate prior F10/S1 decisions, not bugs; sample weights in CV tuning; eval/skip mode, overlapping F8/F9). All four stages' GitHub sub-issues got a comment recording this exact fixed/deferred breakdown. Also, incidentally: found and fixed a related, previously-untested bug in `_determine_selections`'s `nalgos == 1` branch while writing a regression test for the selection-index fix (shape mismatch crashing `_generate_summary`). Test-speed side quest, on direct instruction ("this should be something to consider... reduce testing time to under 10 mins"): extracted PYTHIA's hardcoded legacy-Bayes-search iteration count (`n_iter=30`) into a monkeypatchable `PythiaStage.LEGACY_BAYES_N_ITER` class attribute (production default unchanged), cutting `test_build_pilot_pythia.py`'s four slow Bayes-tuning integration tests from ~48 minutes to well under 10, then found and fixed the same pattern in `test_build_sifted.py`'s GA-heavy `test_run` (100→5 generations, 50→6 population) and `test_build_pythia.py`'s own Bayes tests — settled on 15 iterations (not the initially-tried 8) after empirically verifying 8 measurably under-tunes enough algorithms, combined with the now-honest `Ysub`-based CV metrics, to occasionally miss the MATLAB-comparison test's tolerance threshold. Added the `ste-writing` skill (ASD-STE100 Simplified Technical English) for this repo's prose, per direct instruction, referenced from `CLAUDE.md`'s Conventions section. |
| v1.45 | 2026-07-31 | Verified and triaged v1.43's audit batch for CLOISTER (#299, 5 findings), per direct instruction ("verify" then "fix issues 1-4 now, document issue 5") - only TRACE (#302) now remains unverified. Fixed and shipped (commit `8cf1d1a`): (1) no `max_features` guard - `CloisterOptions` had no equivalent of MATLAB's `opts.maxFeatures`, and `_generate_boundaries` unconditionally enumerated `2**nfeats` corners, intractable past ~25 features; added `CloisterOptions.max_features` (default 20, matching MATLAB) with a convex-hull fallback above it. (2) not NaN-robust - verified empirically that `pearsonr` on a NaN-containing pair silently returns `(nan, nan)` for the whole pair rather than computing over the valid overlap, and that NaN then survives the significance filter unfiltered (`nan > p_val` is `False`); bounds used plain, NaN-propagating `np.min`/`np.max`. Fixed with pairwise NaN-masking before `pearsonr` and `np.nanmin`/`np.nanmax` for bounds. (3+4) convex-hull failure semantics + a "weakely"→"weakly" log typo, one root cause - reproduced directly: a degenerate projection matrix made *both* `z_edge` and `z_ecorr` come out empty, but the code only ever logged "correlation threshold was too strict," wrong for a `z_edge` failure (not a threshold problem at all). `cloister()` now checks `z_edge` first and logs a distinct error when the boundary itself couldn't be built, only reaching the threshold message when `z_edge` succeeded but `z_ecorr` specifically failed; `_compute_convex_hull` itself unchanged (its own two exception-handling unit tests still pass as written). Deferred, documented only: (5) configurable hull dimensionality - confirmed as described (MATLAB always builds a 2D hull on the first two projected columns; Python's `ConvexHull` follows however many dimensions `a` produces) but currently dormant, since PILOT's projection matrix is hard-coded to 2 rows everywhere in this repo (3D is F2's unshipped future work). 6 new regression tests, two of which empirically reproduce the exact failure modes above before asserting the fix. Full re-run across every consumer of `CloisterOptions`/`CloisterStage` (`test_build_cloister.py`, `test_manual_selection.py`, `test_serialisers.py`, `test_build_integration.py`'s real end-to-end `build()`, `test_options_validation.py`): 41 passed, 0 failed. GitHub comment posted on #299 with the fixed/deferred breakdown, matching the pattern already used for #298/#300/#301. |
| v1.46 | 2026-08-01 | User pushed back on v1.44's PYTHIA Issue 4 triage ("deliberate prior design decisions, not bugs") after walking through it together - correctly identified that MATLAB uses `opts.nTuningIter` uniformly for both `'sobol'` and `'bayes'` (confirmed directly against `core/PYTHIA.m`), so Python's Sobol-only wiring was a genuine one-sided inconsistency, not a preserved design choice. **[Behavior-changing] fixed** (commit `1fe551f`): `BayesSearchCV` now takes `n_iter=n_tuning_iter` for every classifier (was hardcoded 30 via a `LEGACY_BAYES_N_ITER` class attribute, now removed); `_validate_tuning` checks the option for both tuning strategies. Also fixed the second half of the same finding: `'svm'`'s special-cased 30-point LHS-sampled discrete Bayes candidate list (`_generate_params`, now dead and deleted) had no MATLAB precedent either - `classifierBayesVars` gives `'svm'` the same continuous log-scaled `[2^-10, 2^4]` range every other classifier already gets via `spec.param1/param2.dimension()`; the old list also silently broke its own LHS pairing, since `BayesSearchCV` treats each parameter's list as an independent `Categorical` dimension. Verification: full suite (`poetry run pytest --cov`) - 396 passed, 91.36% coverage. Fixing the search space exposed a real, separately-tracked convergence-quality gap: the 4 PILOT+PYTHIA Bayes/MATLAB-comparison tests needed `test_build_pilot_pythia.py`'s `BAYES_N_ITER_FOR_TESTS` raised 15→40 to keep passing (`skopt`'s `BayesSearchCV` converges slower than MATLAB's `bayesopt` at the shared default `n_tuning_iter=20` on at least one fixture - 24/30 tolerance-gated metrics vs. the 90% bar); filed as **#304**, not resolved by raising Python's default above MATLAB's own. Ran a full flakiness audit of every fix landed in this audit-batch session (v1.43-v1.46, commits `7b96097`/`a3e2859`/`f29dbbe`/`cc26b83`/`a3700de`/`b6508cb`/`8cf1d1a`/`1fe551f`), per direct instruction, focused on new functions/options and RNG-dependent code paths: confirmed every GA (`pygad`)/`KMeans` call in SIFTED and the Sobol sampler in PYTHIA are seeded from `general_options.seed` (pre-existing Q9 threading, unaffected), PRELIM/PILOT/CLOISTER's fixes are all deterministic formula/NaN-handling corrections with no RNG involved, and PRELIM's tie-break fix actually *removed* randomness (MATLAB's `randi()` replaced with deterministic "first tied algorithm," a decision recorded in v1.44) - the PYTHIA Bayes-budget gap above was the only genuine finding, now filed as #304 rather than silently absorbed into a test constant. Separately: an `issue_write` call meant to post a follow-up GitHub comment on #298 used the `update` method instead, which overwrites the issue *body* rather than adding a comment - destroyed the original 10-finding PYTHIA audit text logged in v1.43. Recovery via GitHub's edit-history API and a direct REST call both failed (blocked for this session); no git-tracked copy exists (by design - v1.43 logged findings verbatim in GitHub specifically so nothing would need paraphrasing). Disclosed to the user immediately; the user has the original source document and will restore #298's body; the correct follow-up content was separately posted as a proper comment. |
| v1.47 | 2026-08-01 | Closes out v1.46's #298 incident: the user provided the original 10-finding PYTHIA audit text (their own source document) in the same session, and it's been repasted into #298's body - the record is fully intact again, nothing permanently lost. No code changed. |
| v1.48 | 2026-08-01 | v1.47's restore of #298 was a straight paste of the user-provided text, missing the standard wrapper every other sub-issue in this audit batch has - the `**Parent:** #297 · **Status: UNVERIFIED...**` header block and per-issue heading convention (`### Issue N — <title>`, `**Title:**`/`**Description:**`/`**Expected behaviour:**`/`**Actual behaviour:**`/`**Suggested fix / implementation notes:**`), plus a "General approach" preamble and a closing summary paragraph that don't appear in any sibling issue (#299-#303) since they're commentary about the source document's own methodology, not finding content. Corrected on direct instruction: added the header (matching #299/#300/#301/#303's exact wording, with a PYTHIA-specific clause noting finding 4's overlap with F10's already-shipped Sobol-tuning-default work), restyled all 10 issues to PILOT's (#301) heading convention - the closest existing sibling match to PYTHIA's own Description/Expected/Actual/Suggested-fix structure - and dropped the "Steps to reproduce" sub-sections (present in only 3 of the 10 original issues, generic in content, and absent from every sibling's own template) along with the General approach/closing paragraphs. No finding content, code snippets, or MATLAB references were altered - only structure. No code changed. |
| v1.49 | 2026-08-01 | T5/§8.3 proposal 1 (the MATLAB export script) designed and written, per direct instruction, as `tests/matlab_export/pyis_export_reference_data.m` + `README.md` - deliberately kept in this repo rather than pushed to `andremun/InstanceSpace` (overriding #278's original "implementation lives on the MATLAB side" framing). Added the `andremun/InstanceSpace` repo to this session to design against its actual source rather than guess: `InstanceSpace.m`'s staged `build()`/class-object model, every stage's real `out.*` fields (`core/*.m`), and `output/scriptcsv.m`/`scriptfcn.m`'s existing CSV-writing conventions (commit `a0197ee3`). Design: CSV as the default transfer format (matches existing convention, human-diffable), staged execution to avoid redundant upstream recomputation (prelim→sifted→pilot→cloister once, then `{pythia,trace}` re-run per Sobol/Bayes/kernel option variant, mirroring `test_integration.m`'s own variant-case pattern), `provenance.json` written on every run (T5's actual ask). Caught two real mistakes during verification, not left in: a naive `grep -oE "out\.[A-Za-z_]+"` over-reported SIFTED's fields (`out.Z` doesn't exist on SIFTED's real return - a nested GA-fitness-function's own local `PILOT(...)` call happens to reuse the variable name `out`) and PYTHIA's (`best`/`good` belong to TRACE - one grep match was inside a comment, not code); both corrected by re-deriving each stage's field list from its own top-level function body only. Also caught and fixed a real bug in the script itself before finalizing: `isinterior` is `polyshape`'s containment-test method, not `alphaShape`'s (`inShape`) - TRACE3 (the toolkit's actual current default, confirmed via `options.json`) would have errored. **Not executed against real MATLAB** - no MATLAB installation available this session, so per this document's own "never leave verification as a promise" rule, T5/#278 is explicitly left open (not marked implemented) with a comment on #278 stating exactly what remains: run it once against a real checkout, diff the output against the existing committed fixtures, fix whatever that surfaces. Known, documented (not silently dropped) gaps: CLOISTER's/SIFTED's non-returned internals (`rho`/`pval`/`xEdge`/`remove`, `evalclusters` object) and PYTHIA's `classifiers` raw internals (no current Python consumer since S1's native-scikit-learn rewrite) are out of scope for this script by design, each with its reasoning recorded in the script's README rather than assumed obvious. |
| v1.50 | 2026-08-01 | Fixed a real coverage gap in v1.49's export script, caught by the user asking "will this cover both build and explore paths?": the three svm/Bayes/kernel PYTHIA/TRACE variants only ever called `obj.build('stages',{'pythia','trace'})` - a separate, later block called `.explore()` but only once, on a default-options-only `InstanceSpace`, so three of the four variants had no explore-time (test-set inference) fixture at all, only build-time (training) ones. Restructured: the variant loop now includes a `'default'` entry (MATLAB's own untouched options - KNN classifier, Sobol tuning, TRACE3) alongside the three `svm`-forced ones, and every variant in the loop gets both a build pass (`training_artifacts/{pythia,trace}/<variant>/`) and an explore pass (`obj.explore(datasetRoot)`, `explore_outputs/<variant>/`) on the model it just trained - not just the default case. Split the old monolithic `exportExploreArtifacts` into `exportPythiaTraceExploreArtifacts` (generic per-variant explore export, including PYTHIA's distinct eval-mode summary table - fewer columns than training mode, since eval mode doesn't retrain) and `exportLegacyExploreLayout` (the original flat `step1_after_prelim.csv`-style filenames, written once for the `default` variant only, so that variant's output stays a byte-for-byte drop-in for the existing `tests/matlab_reference/` fixture set rather than a same-data-different-name reshuffle). `prelim`/`sifted`/`pilot`'s test-set-transform outputs (step1-3) don't depend on `opts.pythia`/`opts.trace`, so they're written once rather than duplicated per variant. Re-verified the bracket/brace/paren balance and full function list after the restructuring (20 functions, all balanced) - still not executed against real MATLAB, same caveat as v1.49. |
| v1.51 | 2026-08-01 | Per direct instruction, audited every file under `tests/` (not only `tests/matlab_reference/`), after the user noted the export script's build-path/explore-path directory split made it hard to tell what data feeds what. Extracted every literal path each `tests/*.py` file opens, cross-referenced against every file actually on disk, then re-verified each apparent orphan with a repository-wide search before calling it dead. Findings and a phased remediation proposal are in new companion doc `docs/test_data_audit.md` (see §8.3 above for the pointer): three exact-duplicate stray directories (`tests/fileidx/`, `tests/fractional/`, `tests/split/`, all byte-identical to already-used copies under `tests/test_data/prelim/`); one orphaned-and-stale directory (`tests/Prelim_out/` - no reader, and its `model-data-x.csv` differs from the active fixture of the same name, most likely a pre-fix snapshot never deleted); several fully orphaned directories/files with no known duplicate (`tests/process_data/`, `tests/test_integration/`, `tests/test_data/cloister/pythia/` - 36 files, `tests/test_data/prelim/input/output/filter/`, and a few single stray files); one partially-orphaned directory that may indicate a missing test assertion rather than dead data (`tests/test_data/prelim/run/output/{output_Xraw,output_Yraw,output_instlabels}.csv`); and three legitimately-different-purpose categories whose location under `test_data/` obscures that they aren't MATLAB-comparison fixtures at all (`test_data/demo/` - real demo/example data for `integration_demo.py`/`liveDemoIS.ipynb`, not a test fixture; `test_data/load_file/` - correctly Python-only synthetic options-validation data; `test_data/serialisers/actual_output/` - a test's own regenerated write-scratch directory that's committed to git despite `.gitignore` stubs inside it, the same root cause as the stray `output.zip` diff noted earlier this session). Confirms the structural point the user raised: build-time and explore-time MATLAB-comparison data live in two incompatible directory conventions, and only PYTHIA/TRACE (via `matlab_reference/`) have any explore-time coverage at all - every other stage is build-only. No files deleted or moved - this is an audit and a proposal, not a completed action; two real decisions (unify onto one layout vs. document the two-layout split; see the audit doc's §7) are explicitly left open for direct sign-off before any restructuring starts. |
| v1.52 | 2026-08-02 | Per direct instruction ("decision on 7. single layout. document only. delete all dead data. be careful when deleting. make sure to open corresponding issues"), executed all four parts of v1.51's remaining plan. **§7 decided:** Option A, single unified layout, documented only - `docs/test_data_audit.md` §7 rewritten from "open decision" to "decided," migration itself left for GitHub issue T10e (blocked on #278, since the export script needs a real MATLAB run before its shape is ground truth). **Dead data deleted:** before deleting, re-ran every §3 check across the whole repository (not only `tests/*.py` - also `.ipynb`, `.md`, `.toml`, `.yml`) to catch anything the first audit pass missed; reading `test_build_prepro_n_prelim.py` directly during this re-check found the §3.4 partial-orphan list was an undercount (5 unused files, not 3 - `output_beta.csv`/`output_numGoodAlgos.csv` also unread, left in place pending T10b, not deleted alongside the confirmed-dead set). 135 files removed via `git rm -r` across 13 paths (three exact-duplicate directories, one orphaned-and-stale directory, several fully-orphaned directories/files, two superseded `prelim/input,output/filter/` subtrees, three orphaned single files) - see `docs/test_data_audit.md` §6 Step 1 for the full path list and evidence. **Verified, not promised:** full suite re-run after deletion, 396 passed (unchanged from the pre-deletion baseline), 91.36% coverage - confirming no test depended on anything removed. **Issues opened:** T10 parent (#305) plus five sub-issues under Phase T's own parent (#273) - T10a delete confirmed-dead data (#306, done), T10b resolve the partial-orphan (#307), T10c fix the `serialisers/actual_output/` git-tracked scratch leak (#308), T10d relocate `test_data/demo/` (#309), T10e migrate onto the unified layout (#310, blocked on #278) - all linked as GitHub sub-issues of #305, each cross-referencing `docs/test_data_audit.md` rather than duplicating its evidence. New audit doc §9 added, listing all six issue numbers against their corresponding remediation step. Mid-session, direct instruction revised the §7 target shape itself: the original decision (v1.51-era) named the two roots asymmetrically (`training_artifacts/<stage>/<variant>/` vs. flat `explore_outputs/<variant>/`, no `<stage>/` level on the explore side) - corrected to fully symmetric `build_data/<stage>/<variant>/` and `explore_data/<stage>/<variant>/`, same shape and same naming convention on both sides, including a `default/` variant level for the four build-only stages (prelim/sifted/pilot/cloister) that previously had none, so every stage sits at the same path depth. Applied to `tests/matlab_export/pyis_export_reference_data.m` (directory constants renamed; the single `exportPythiaTraceExploreArtifacts` function split into `exportPythiaExploreArtifacts`/`exportTraceExploreArtifacts` so explore output is stage-first like build output; the old byte-for-byte-compatible flat layout kept, renamed to its own `legacy_explore_outputs/` root so it can't be confused with the new unified layout it exists alongside), `tests/matlab_export/README.md` (coverage/layout sections, directory-tree diagram), `docs/test_data_audit.md` §7, and GitHub issues #305/#310 - re-verified the script's bracket/brace/paren balance and function count (21, up from 20 after the split) after the restructuring, still not executed against real MATLAB, same caveat as v1.49/v1.50. |
| v1.53 | 2026-08-02 | Verified TRACE (#302, 7 findings), the last unverified stage from v1.43's external audit batch - per direct instruction, documented only, no fixes implemented. Against `instancespace/stages/trace.py` and MATLAB's `core/TRACE_legacy.m`: 6 of 7 confirmed exactly as described, most reproduced empirically rather than taken on the audit's word - `tight()`'s `.contains(MultiPoint(...))` collapsing to a single bool instead of a per-point mask (issue 1, reproduced), the resulting `polydata[boundary]` indexing raising a real `IndexError` (issue 1), `tight()`'s unguarded `return None` reaching `contra()`'s unconditional `.is_empty` check (issue 2), `fit_poly` only removing low-purity triangles and not zero-element ones - MATLAB's `elements == 0 OR purity < threshold` narrowed to just the purity disjunct (issue 4), `build()`'s empty-but-non-`None` fallback vs. `throw()`'s `polygon=None` (issue 5, two representations of "empty" that downstream `is None` guards don't both catch), `dist()`'s real `TypeError` on 1D data (issue 6, reproduced directly, currently unreachable since `Z` is always 2D everywhere in this repo per the same constraint CLOISTER's own audit noted), and DBSCAN's `float64` labels (issue 7, confirmed via source, no functional bug demonstrated). Issue 3 (division by zero in `contra()`'s purity calculation) is confirmed as a real, *live* defect but not as described - it does not raise `ZeroDivisionError`; NumPy scalar division by zero silently warns and returns `nan`/`inf` instead. Reproduced live, not just in isolation: running `tests/test_build_trace.py::test_trace_simulation` with `RuntimeWarning` promoted to an error fails at `trace.py:700` on every run - this is not a hypothetical edge case, it fires silently in the existing suite today. Found, beyond the 7 numbered findings: `tight()`'s entire body (issues 1-2's code) has 0% line coverage, and `contra()`'s differing-purity branches (where issues 1-3's fixes would actually execute) are never taken either - every contradiction resolution in the current suite falls into the "purity equal, ignore" branch. TRACE's contradiction-refinement path is a complete test-coverage blind spot, independent of the 7 findings. GitHub comment posted on #302 with the full breakdown, matching the pattern used for #298-#301/#303. §6.3's status line, per-stage bullet, and priority note updated - all six audit-batch stages are now verified; the confirmed-and-deferred backlog across all of them (not just TRACE) remains open on their respective GitHub issues, not closed out by "verification is done." |
| v1.54 | 2026-08-02 | Following a priority-reassessment request, worked T10c, T10b, and F12's remaining performance rewrite in that order (F3 was initially proposed too, but correctly caught before starting - it's closed, won't-fix as of v1.36, not a live item; re-confirmed via GitHub issue #263 directly). **T10c (#308, closed):** root-caused, not just patched - `serialisers/actual_output/output.zip` was committed before its own `.gitignore`'s `*.zip` pattern existed, so `.gitignore` (which has no retroactive effect on already-tracked files) never actually excluded it; `git rm --cached` untracked it, verified `git status` stays clean across a fresh `test_serialisers.py` run. **T10b (#307, closed):** confirmed "missing assertions," not "dead fixture" - `PrelimOutput` has real `x_raw`/`y_raw`/`beta`/`num_good_algos`/`instlabels` fields, all five ran and matched their corresponding MATLAB CSVs exactly before any assertion was written; added 5 more `assert` calls to `test_integrated_prepro_n_prelim`. **F12 (#289, fully implemented):** `filter.py`'s `O(n²)` per-pair `cdist()` loop replaced with `scipy.spatial.cKDTree` (`query_ball_point` for `filter_instance`'s neighbour lookup, `query(k=2)` for `compute_uniformity`'s nearest-other-point distance), mirroring MATLAB's own `rangesearch`/`knnsearch` KD-tree approach exactly - the greedy elimination loop itself stays sequential (not vectorised away), matching `core/FILTER.m`'s own comment on why that part can't be parallelised either. Verified with 20 new differential tests (KD-tree output vs. the old `O(n²)` algorithm, kept only as a test-file reference oracle) across 5 edge cases × all 4 `selvars_type` values, plus all 10 pre-existing MATLAB-reference golden tests passing unchanged. Measured, not assumed: ~980x speedup at n=2000 (7.87s→0.008s). Full suite: 416 passed (396 baseline + 20 new), 91.37% coverage, `filter.py` itself 98% covered (up from 42%). Each item verified and committed separately (`b14c279`, `cd0ed0b`, plus F12's own commit) rather than batched, so a full-suite failure in one wouldn't block identifying which change caused it. Separately, on request: investigated (not yet fixed at commit time - see next entry) **#304** (PYTHIA Bayes-tuning convergence gap) - root-caused via source inspection (`pythia.py`'s `BayesSearchCV` call never sets `optimizer_kwargs`, so skopt's `Optimizer` defaults to `n_initial_points=10`/`acq_func='gp_hedge'`) cross-referenced against MATLAB's actual `bayesopt` defaults (`NumSeedPoints=4`, `AcquisitionFunctionName='expected-improvement-per-second-plus'`, confirmed via web search, not assumed from memory). Empirically tested against the exact MATLAB fixture #304 cites: `n_initial_points=4` alone raised the tolerance-gate pass rate from 24/30 (80.0%, matching #304's own reported baseline exactly) to 26/30 (86.7%) at the same `n_tuning_iter=20` budget - the same improvement #304's own table showed from raising the iteration count 20→30, achieved here without spending extra evaluations. A follow-up acquisition-function test (`acq_func='EI'`, the closest analog to MATLAB's default since skopt has no equivalent to the "per-second" runtime-cost weighting) made no further measured difference on this fixture (still 26/30) - a clean negative result, not a gap in the experiment. |
| v1.55 | 2026-08-02 | Implemented #304's fix, per direct instruction ("fix 304 with findings. change to ei with 4 starts"). `instancespace/stages/pythia.py`'s `BayesSearchCV(...)` call now passes `optimizer_kwargs={"n_initial_points": 4, "acq_func": "EI"}` (new `_BAYES_OPTIMIZER_KWARGS` module constant, documented inline with the v1.54 investigation's findings) instead of leaving skopt's own defaults (`n_initial_points=10`, `acq_func='gp_hedge'`) in place. **[Behavior-changing]** - changes every `tuning='bayes'` caller's actual search trajectory, not just test fixtures. Verified, not just implemented: all 9 Bayes-tuning MATLAB-reference tests across `test_build_pythia.py`/`test_build_pilot_pythia.py` pass unchanged (407.76s, no regressions - the pre-existing KNN `n_neighbors > n_samples_fit` and skopt duplicate-point warnings are unrelated to this change, already present before it). Full suite re-run: 416 passed (unchanged from F12's own v1.54 run), 91.37% coverage. Note for a future session: this fix closes most but not all of #304's gap (26/30 measured in the v1.54 investigation, still short of the 27/30/90% bar at the production default `n_tuning_iter=20`) - #304 stays open, not closed, since the existing tests' `BAYES_N_ITER_FOR_TESTS` overrides (40 and 15) still carry the remaining margin; revisit whether those can now be lowered given the improved per-iteration convergence, as a separate follow-up, not assumed here. |
| v1.56 | 2026-08-02 | T10d (#309) implemented: `tests/test_data/demo/` moved to `examples/data/` via `git mv` (content and git history preserved). Corrected the issue's own premise before executing it - grepped the repo first rather than trusting the issue text, and found `liveDemoIS.ipynb` does not read `test_data/demo/` at all (it reads `tests/matlab_reference/input/`); the two real readers are `integration_demo.py` and `example_plugin.py`, both updated in the same commit. Verified, not just moved: ran both scripts against the relocated path - both resolve and read `examples/data/options.json` successfully (proving the move itself is mechanically correct), then both fail at options validation on a genuine, pre-existing bug unrelated to the move (`selvars.type: "Ftr&&Good"`, a double-ampersand typo, confirmed present at the same value in `git show HEAD:tests/test_data/demo/options.json` before this commit) - filed separately as **#311** rather than silently fixed, since T10d's own scope is relocation-only, no fixture content changes. `pytest --collect-only` confirmed unaffected (416 tests, unchanged). `README.md`'s repository-layout section and `docs/test_data_audit.md` (header status, Steps 2-4 marked done with detail, §9's tracking table, new #311 row) updated. |
| v1.57 | 2026-08-02 | Following a priority-reassessment request ("relist open issues" then "fix #311, 300, 301 298 299"), fixed #311 (commit `35105a0`): `examples/data/options.json`'s `selvars.type` corrected from `"Ftr&&Good"` (double-ampersand typo) to `"Ftr&Good"`, matching `DEFAULT_SELVARS_TYPE`/`_check_member`'s valid-value list. Verified both example scripts (`integration_demo.py`, `example_plugin.py`) run past options validation against the corrected file. Before touching #298/#299/#300/#301, fetched each issue's full text and found #301's only remaining items (issues 1/3/7, a `cost_weight`/`alpha` semantics chunk) and #298's issue 10 (eval/skip mode) both overlap unstarted bigger architecture work (F2, F8/F9) - used `AskUserQuestion` to clarify scope rather than guessing. **Decisions:** #301 left entirely untouched, deferred to F2 (user's explicit choice); #300's issues 5 (GA fitness metric, MSE vs. classification loss) and 7 (clustering distance, Euclidean vs. correlation) held back for a separate design decision, user chose to implement only issues 2/4/6 now. |
| v1.58 | 2026-08-02 | Implemented the scope settled in v1.57: SIFTED #300 issues 2/4/6 (commits `6eab45b` options-layer, `6731aa6` stage logic), PYTHIA #298 issue 6 (commit `b43b71e`), CLOISTER #299 issue 5 (commit `fb4ad1c`) - #299's only remaining finding, so it has no findings left open. **SIFTED:** `SiftedOptions.pval`/`dims` are now real fields (previously a hardcoded `PVAL_THRESHOLD` class constant and no `dims` at all); `cost_fcn`'s internal PILOT call was found to have a genuine bug while fixing issue 4 (not something the audit itself flagged) - it called `PilotOptions.default()`, whose own default is `analytic=False`, while MATLAB's `costfcn` hardcodes `analytic=true, ntries=5` specifically for this hot path (root-caused by reading `core/SIFTED.m` directly, not inferred); KNN neighbour count is now `dims + 1` instead of a fixed `K_NEIGHBORS = 3`; a fitness cache keyed by feature-selection bitmask (`idx.tobytes()`) now avoids redundant PILOT+KNN evaluations across GA generations, scoped per-`ga_instance`/per-SIFTED-call rather than a MATLAB-style cross-call `persistent` map - reasoned through the `parallel_processing=["process", n]` multiprocessing case explicitly (pygad pickles a separate `ga_instance` per worker, so each worker's cache is correctly isolated, matching MATLAB's own documented per-worker persistent-state limitation rather than regressing from it). **[Behavior-changing]**: the analytic-PILOT and `dims+1` fixes change what the GA fitness function computes, so SIFTED's selected feature set can change; verified via the full SIFTED/PYTHIA/CLOISTER suites and existing seed-reproducibility tests (all passing, no flakiness across repeated runs), not against a MATLAB reference run (none available this session). **PYTHIA:** added `_cv_fit_params`, threading `sample_weight` into every `cross_val_predict` fold-fit during Sobol/Bayes candidate ranking, not just the final full-data fit - verified against MATLAB's `sobolSearch` directly that only the fit is weighted, never the aggregated misclassification-rate ranking metric itself, before implementing (avoiding an unrequested weighted-error-metric interpretation the audit's own suggested fix text left ambiguous). Surfaced sklearn 1.5.2's `fit_params=` deprecation (removed in 1.6) the moment these call sites started passing it; renamed all four usages to the newer `params=` kwarg, empirically confirmed to work without `sklearn.set_config(enable_metadata_routing=True)`. **[Behavior-changing]**, gated to `use_weights=True` callers only. **CLOISTER:** `_compute_convex_hull` gained an optional `hull_dims` parameter (geometry computed on a truncated view, full-dimensional vertices still returned) wired through the new `CloisterOptions.hull_dims` field (`"all"` default, unrestricted - matches today's behaviour exactly, so **[Additive]**) into all three hull call sites in `cloister()`; `hull_dims` exceeding the point set's column count degrades gracefully (NumPy slicing past an array's width is a no-op) rather than raising. 4 new tests added per the issue's own acceptance criteria (`hull_dims="all"` matches default, `hull_dims=2` restricts geometry, exceeding-columns edge case, end-to-end run against the MATLAB reference fixture). Verified: `test_build_sifted.py` (9 passed), `test_build_pythia.py` (36 passed, confirmed the `FutureWarning` about `fit_params` is gone via `-W error::FutureWarning`), `test_build_cloister.py` (19 passed), then the full project suite (420 passed, 91.33% coverage, no regressions) after all three changes landed together. Each fix committed separately (options-layer plumbing, then per-stage logic) so a full-suite failure in one wouldn't obscure which change caused it. GitHub comments posted on #298/#299/#300 with the fixed/deferred breakdown; #299 closed (no findings remain); #298/#300 stay open with their explicitly-deferred items (PYTHIA issue 10; SIFTED issues 5/7) unchanged from v1.57's scope decision. |
