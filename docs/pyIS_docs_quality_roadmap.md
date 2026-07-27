# pyInstanceSpace — Documentation & Quality Roadmap

**Version:** v1.13
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
  already-scoped item above (Q1, Q2, Q8, F7, F8, R1, R2) needs to be verified, not just built.
- **Phase -1** (added v1.11): a prerequisite, not a phase in sequence — merge the fork
  (`aoxiangx/pyInstanceSpace`) back into the upstream repo (`andremun/pyInstanceSpace`)
  before, or very early alongside, P0. Everything else in this document assumes work
  continues on a single, merged codebase.

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
   already-catalogued follow-up work (Q1's poly-kernel gap, F4's audit findings, and everything
   else in this document) as a visible checklist rather than leaving it undiscoverable in a
   silent merge.

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

The reverse of §5.1's audit: concrete MATLAB v0.9.0 behaviours worth adopting in Python.
Filtered specifically for low risk — each item below is additive or narrowly contained, doesn't
touch the DAG scheduler (`stage_builder.py`/`stage_runner.py`) or change any stage's algorithm,
and is independently testable. Heavier ideas that didn't clear that bar are in §5 as F7–F9
instead.

### Q1 — Fix build→explore adapter's missing polynomial-kernel branch
**[Additive]** — fixes a path that currently always raises `NotImplementedError`; no working caller could have depended on the old failure.
`build_explore_adapter.py::_svc_to_artifact()` handles `"rbf"` and `"linear"` only; `stages/pythia.py`
can train `"poly"` (per `opts.pythia.ispolykrnl` parity with MATLAB). Calling `.explore()` after a
poly-kernel `build()` raises `NotImplementedError` today. **This is a bug fix, not a hygiene
item** — recommend doing this first, ahead of Q2–Q8, since it's the one item on this list with
a reproducible failure case rather than a missing nicety.

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

### Q7 — Add `plot()` convenience methods
**[Additive]** — new methods only; nothing existing calls them yet.
Mirror MATLAB's `InstanceSpace.plot('sources' | 'portfolio' | 'good' | 'footprint', algoIdx)` —
thin matplotlib wrappers around `model.pilot.z` and friends. Additive only, no pipeline logic
touched. Complements P2 (notebook parity): a `plot()` method means the notebook needs less
inline matplotlib boilerplate to demonstrate the same views MATLAB's manual shows.

### Q8 — Regression test for stage-rerun invalidation (verification, not yet a fix)
**[Additive]** — this is a test. If it reveals a real fix is needed, that fix (in `stage_runner.py`) inherits its own **[Behavior-changing]** tag — don't assume it's free just because the test itself is.
§5.1 flagged that Python's `_rollback_to_schedule_index()` invalidates by schedule-wave position
rather than by real dependency (MATLAB's `invalidateDownstream()` BFS). Write a test: build,
re-run `cloister` only via `run_stage()`, and check whether `pythia`'s output is unnecessarily
marked stale. If the test confirms over-invalidation, promote the fix to F-phase work (it touches
`stage_runner.py`, so it doesn't clear this phase's low-risk bar) — see the note on F4 in §5.

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

**Checkpoint for Phase Q:** existing `pytest` suite passes; a new poly-kernel build→explore
round-trip test passes (Q1); `pytest` covers the new out-of-distribution warning (Q2), the
feature-order regression test (Q5), and the new seed option (Q9) producing identical output
across repeated runs with the same seed and different output across different seeds; no change
to any stage's numerical output for the reference dataset when run with the same seed as before.

---

## 5. Phase plan — functionality parity (long-term, deferred)

These map loosely to MATLAB's Phases 4–10 but are **not scoped yet** — each starts with its
own audit (read the relevant `stages/*.py` + tests) before any specific fix is committed to.
Order is a starting suggestion, not a commitment.

| Phase | Maps to MATLAB | Focus | Status | Compat |
|---|---|---|---|---|
| F1 | Phase 4 | PYTHIA classifier registry — confirm whether `stages/pythia.py` supports a pluggable classifier set or is fixed | Not started | **[Additive at default]** — new `classifier` option defaults to `'svm'`, matching today's only behaviour. New registry entries themselves are new production surface, though — need their own validation before being trusted in production, not just "it runs." |
| F2 | Phase 5 | PILOT 3D / viewpoint optimisation parity in `stages/pilot.py` | Not started | **[Behavior-changing risk]** — generalising the 2D-specific solver to n-dims can shift 2D output even at `dims=2` if not done carefully (different array shapes can trigger different BLAS code paths). Verify bit-for-bit or tolerance-verified identical 2D output before shipping — this touches existing code, not just adding an independent new path. |
| F3 | Phase 6 | SIFTED promotion refinements | Not started | **[Unknown until audit]** — F3's own pathway starts with "audit first" for exactly this reason. Treat any fix the audit finds as **[Behavior-changing]** by default until proven otherwise, since it touches SIFTED's core computation. |
| F4 | Phases 7–8 | `InstanceSpace` class & `build`/`explore` robustness | **Audited (v1.3)** — see §5.1 for findings; Q8 (§4) verifies one open question before F4's invalidation-fix work is scoped | — (audit only; see F7/F8/F9 for the actionable, taggable derivatives) |
| F5 | Phase 9 | Output consolidation / 3D visualisation parity (MATLAB's `scriptpng.m`) | Not started | **[Additive]** — new rendering paths; doesn't change any existing 2D output function. |
| F6 | Phase 10 | Namespace & per-file licence headers — licence itself already matches MATLAB | Header audit only | **[Additive]** — comments only. |
| F7 | — | Model save/load round-trip (`Model.save()`/`InstanceSpace.load()`), matching MATLAB's persistence | **Format decided: HDF5 via `h5py`** — see design constraint below | **[Additive]** — brand-new capability; nothing existing depends on it. |
| F8 | — | Unify `explore()` with build-time stage code (predict-mode dispatch on `PythiaStage`/`TraceStage`, matching MATLAB calling the same `PYTHIA()`/`TRACE()` in both modes) | Not started | **[Behavior-changing risk]** — this refactors existing, working code. The full `tests/matlab_reference/` validation suite must pass identically before/after; treat any tolerance-threshold change during this work as a red flag to investigate, not a "close enough" adjustment. |
| F9 | — | Expand `explore()` to full evaluation scope: algorithm reconciliation + ground-truth performance metrics, matching MATLAB's `evaluateTestSet` | **Decided: extend `explore()` itself** (silent branch on whether ground truth is present) — see companion implementation-pathways document for the full pathway | **[Additive]** — new fields default to `None`; existing feature-only callers see no change. Add explicit test coverage for the "no ground truth present" path specifically, to lock this in rather than assume it. |

**F7 design constraint:** must not use raw `pickle` or any other unsafe-deserialisation format.
§2.1 confirmed this codebase is currently clean of `pickle`/`eval`/`exec`/unsafe deserialisation,
and §2.1's [DECISION] already restricts future web uploads to CSV specifically to avoid
reintroducing untrusted-deserialisation risk — adding a pickle-based model format would undercut
both.

**[DECISION] Topic:** F7 persistence format — HDF5 via `h5py`
**Rationale:** handles nested structure and large numpy arrays natively (`Model`'s tree of
dataclasses maps onto HDF5 groups/datasets directly), without the manual flattening a JSON+
`.npz` approach would need. Doesn't execute arbitrary code on load — reading an HDF5 file means
reading arrays and attributes, not deserialising Python objects — so the no-pickle constraint
above still holds.
**Alternatives rejected:** JSON manifest + `.npz` (no new dependency, but more manual flattening
work for no real safety or capability advantage); a fully custom versioned binary schema
(unjustified implementation/maintenance cost).
**Impact:** adds `h5py` as a new dependency — goes through the same P0-style scrutiny as any
other dependency before merging (check its own transitive tree, not just add and forget).
Non-trivial serialisation shapes to decide during implementation: `pythia.svm`'s per-algorithm
SVM objects and `trace.good`/`trace.best`'s shapely polygons both need flattening into arrays
(constituent SVM parameters; vertex lists with the NaN-separator convention already used for CSV
export) rather than attempting to store the objects themselves — full detail in the companion
implementation-pathways document.

### 5.1 F4 audit findings — class architecture deep dive

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

## 6. Ideas from independent implementations — PyISpace / PyHard

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
  exactly the anti-pattern F7's design constraint (§5) already rules out. Cited here as evidence
  for *why* that constraint exists, not as something to adopt.

**Checkpoint for Phase R:** R1 — a rotation-adjustment unit test confirms `Z`'s pairwise distances
are unchanged before/after rotation, and that a reference dataset's rotated output is visually
consistent across two independent runs. R2 — a regression test using a known multi-region
alpha-shape case (if one is available) confirms the retry path is exercised and produces a
complete boundary rather than a partial one.

---

## 7. Phase T — testing infrastructure quality & additions

### 7.1 Audit findings

`tests/` is 6,678 lines across ~35 files. One genuine strength, several concrete gaps —
verified against source, not inferred from file names.

**Strength:** `tests/matlab_reference/` is a real cross-implementation golden-reference harness
— actual MATLAB-trained artifacts (projection matrix, SVM support vectors, footprint polygons)
checked in, with per-stage validation tests comparing Python's output against them under
documented tolerance thresholds (e.g. `test_pilot_matches_matlab`'s docstring states the 1%
threshold's rationale: PILOT inference is a pure linear projection, so Python should match
MATLAB to floating-point precision). `test_adapter.py::test_unsupported_kernel_raises` is good
discipline too — the poly-kernel gap (Q1) is tested to fail loudly, not silently.

**Gaps, verified:**
1. **No true end-to-end integration test.** Every `InstanceSpace(` construction outside
   `exploreIS/` returns zero hits — `explore_iter` tests use `InstanceSpace.__new__` with every
   method manually stubbed. Nothing constructs a real `InstanceSpace` and calls `.build()`
   through the actual 7-stage pipeline. No Python equivalent of MATLAB's `test_integration.m`.
2. **The DAG resolver's hard logic is untested.** `test_stage_builder_runner.py` (3 tests) uses
   two trivial synthetic stages (`int→str→str`). Mutating-stage handling, `RunBefore`/
   `RunAfter`, and ambiguous-ordering error paths (all found during the §5.1 audit) have no
   test touching them.
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
7. **The reference harness itself has a real gap.** Checked several `svm_<algo>.csv` artifacts
   directly — every one has `kernel_fn = gaussian`, regardless of which portfolio algorithm it
   predicts for. There is **no MATLAB reference data for a polynomial-kernel PYTHIA model at
   all** — Q1 can't be validated against MATLAB until new reference data exists. The reference
   README also never states which MATLAB commit/tag produced it, so there's no way to detect if
   it's gone stale as MATLAB keeps moving (see §7.3).

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
Record the exact MATLAB commit/tag the fixtures were generated from. See §7.3 for the fuller
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

### 7.2 Test debt tied to already-scoped items

Specific tests each already-scoped item (Q, F, R) needs to actually be verified, not just built:

| Item | Test needed |
|---|---|
| Q1 (poly-kernel adapter fix) | New MATLAB reference artifact with a real poly-kernel PYTHIA SVM (doesn't exist — prerequisite per §7.1 finding 7) + a build→adapt→explore round-trip test |
| Q2 (OOD warning) | Fires above 5% clipped; silent below it — both directions need a test |
| Q8 (rerun-invalidation regression test) | Must use the real 7-stage pipeline once T2 exists — the current synthetic 2-stage setup can't exercise the cloister/pythia sibling-branch question at all |
| F7 (save/load) | Round-trip equality test, plus a malformed/adversarial-file test proving the safe format (per F7's design constraint) can't execute anything on load |
| F8 (explore/build code reuse) | A deliberately-introduced bug in `PythiaStage`'s logic should break both the build-path and explore-path tests once they share code — proves the drift risk F8 is meant to close is actually closed |
| R1 (rotation canonicalisation) | Not just "pairwise distances preserved" — assert the target group's centroid angle lands within tolerance of 135° post-rotation, or the test doesn't prove the feature does what it's for |
| R2 (alpha-shape retry) | A constructed point cloud engineered to produce a `MultiPolygon` at the naive alpha, asserting a complete (non-partial) boundary after the fix |

### 7.3 Cross-repo test-data sharing proposal

The root problem: `tests/matlab_reference/` was produced by a one-off manual MATLAB run, with no
recorded provenance and no repeatable process. As MATLAB keeps evolving (this document alone has
logged a dozen-plus MATLAB-side changes worth making), the fixture set can only get further out
of sync, silently, with no signal when it happens. Proposed, layered by effort:

1. **Now, low-risk:** a MATLAB export script (new file, or extend `test_integration.m`) that
   dumps training artifacts + explore outputs in exactly the CSV interchange format
   `tests/matlab_reference/` already documents. Turns "regenerate the fixtures" from a bespoke
   manual copy-paste into one command. Use it immediately to generate the missing poly-kernel
   case (§7.1 finding 7 / Q1's blocker) — the first real use of the new tool closes an existing
   gap rather than being speculative infrastructure.
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

## 8. Outstanding / deferred items

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
| Missing poly-kernel reference fixture | Blocks validating Q1 against MATLAB — needs the T5/§7.3 export tool run once before Q1 can be closed with confidence |
| MATLAB has no CI | Verified (no `.github/workflows/`) — outside this document's scope (MATLAB repo), but relevant context for §7.3's phased data-sharing proposal, phase 3 of which depends on it existing |

---

## 9. Document history

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
