# CLAUDE.md — pyInstanceSpace

Persistent context for Claude Code sessions working on this repo. Read this before touching
anything. Full detail lives in `docs/` (see below) — this file is a map, not the content.

## What this is

Python port of the MATLAB Instance Space Analysis (ISA) toolkit (`andremun/InstanceSpace`,
currently v0.9.0). **Not** a 1:1 port and not required to become one — the stage architecture
(`preprocessing → prelim → sifted → pilot → pythia → cloister → trace`, an `InstanceSpace`
class, `build()`/`explore()`/`explore_stage_iter()`) is independently engineered. Forked from MATLAB
v0.3.3 (Feb 2023) — MATLAB has since moved through a ten-phase refactor this repo doesn't have.

## Production status — read this before changing any existing behaviour

**This ships behind a production web server.** Every change must be tagged, in its PR
description, as one of:
- **Additive** — new capability, docs, tests, or tooling; no existing caller's output changes.
- **Behavior-changing** — an existing caller could see different output. Requires: the
  `docs/pyIS_docs_quality_roadmap.md` entry for this item already carries this tag with a
  stated verification step (e.g. "matches the reference test suite output before/after") —
  do that verification and report the result in the PR, don't skip it because the change
  looks small.

If a task's roadmap entry doesn't have a compatibility tag yet, don't guess — flag it and ask
rather than assuming "probably additive."

## Source of truth, in order

1. **`docs/pyIS_docs_quality_roadmap.md`** — the living plan: phases, decisions already made
   (read the "Document history" table at the end first, it's append-only and tells you what's
   changed and why), open questions, backward-compatibility tags.
2. **`docs/python_implementation_pathways.md`** — implementation-level detail for every roadmap
   item: files to touch, concrete steps, resolved and open design decisions.
3. This file — process/context only, not technical detail. If something here conflicts with
   the roadmap, the roadmap wins; update this file to match, don't silently follow the stale
   version.

## Current phase — do not skip ahead

**Phase -1 (merge the fork into upstream) must land before anything else.** If you're reading
this from a session working directly in `andremun/pyInstanceSpace`, confirm the fork merge
(see roadmap Phase -1) has actually happened — if not, that's the task, not whatever else you
were asked to do.

After that: **P0, Phase Q, and Phase S before F-phase (functionality parity).** Phase S
("structural simplification") is not optional pre-work you can defer — F1, F7, and F8's
current scope in the roadmap is written *assuming Phase S has already landed* (S1
specifically: `explore()` calling native scikit-learn objects instead of a flattened
artifact).

**Status: Phase S is fully done — S1, S2, and S3 have all landed** (S1/S3 on
`v0.9.0/development-branch-S`, roadmap v1.16; S2 on `v0.9.0/development-branch-QSF`, roadmap
v1.22, commit `71c852b`) — `explore()` already calls native scikit-learn objects,
`build_explore_adapter.py` is gone, and the DAG auto-resolver has been replaced with an explicit
hardcoded stage order (`RunBefore`/`RunAfter` for plugin stages). F1/F7/F8 are unblocked on all
of this. GitHub issue #281 (the Phase S tracking issue) previously said S2 was deferred — that
was stale bookkeeping the code had already outgrown; corrected and closed 2026-07-29.

Also done since the last time this section was written: P0, Q1–Q11, T1–T4, T6 (closed
not-implemented — its subject matter was S2's own DAG resolver, which S2 deleted), F1, F6, F7,
F10, F11, F14, F15 (progress reporting + `InstanceSpace` checkpoint/resume, roadmap v1.42), Q6,
Q8, R1, and the "extend real tuning to non-SVM classifiers" F1 follow-on.
Check `docs/pyIS_docs_quality_roadmap.md`'s document-history table (append-only, read newest
entries first) for the actual current state before assuming anything below is still accurate —
this file gets stale between sessions; that table doesn't.

**Read this before picking up any F-item below: an external 43-finding audit batch (roadmap §6.3,
GitHub #297 + sub-issues #298–#303, added 2026-07-30) sits ahead of everything else in this
section, per direct user instruction.** As of roadmap v1.46 (2026-08-01): PRELIM (#303), SIFTED
(#300), PILOT (#301), PYTHIA (#298), and CLOISTER (#299) are all verified and triaged — confirmed
findings fixed and shipped (commits `7b96097`, `a3e2859`, `f29dbbe`, `b6508cb`, `8cf1d1a`,
`1fe551f`), the rest explicitly deferred with a stated reason on each stage's GitHub issue, never
silently dropped. PYTHIA's Issue 4 was initially mis-triaged as "deliberate, not a bug" — corrected
and fixed in `1fe551f` after direct pushback; its follow-on convergence-quality question is tracked
separately as **#304**, not silently absorbed. **TRACE (#302) is the only stage still unverified**
— that verification pass (document findings only, per direct instruction; do not implement) is the
next thing to pick up here, not F8.

Note (resolved): issue #298's original body (the verbatim 10-finding PYTHIA audit text) was
accidentally overwritten by a `mcp__github__issue_write` `update` call on 2026-08-01 (that method
replaces the issue body, not adds a comment — `add_issue_comment` is the right tool for a
follow-up). GitHub-side recovery failed (edit-history API blocked for this session; no git-tracked
copy exists by design), but the user had the original source document and repasted it into #298's
body the same session — the record is intact again. Lesson for future sessions: use
`add_issue_comment` for follow-ups, never `issue_write update`'s `body` field unless intentionally
replacing an issue's own content.

**Full remaining order (every F item, T5/T7/T8) is worked out in roadmap §6.0** — don't
re-derive it from scratch or guess an order from the phase letters. Short version, per the
roadmap's own dependency notes: F8 → F9 (F9 mirrors a pattern F8's own decision establishes),
then F2 → F5 (F5 is hard-blocked on F2), with F3's one remaining confirmed gap
(`_compute_correlation`'s unvectorized loop) and F12's remaining `O(n²)`→KD-tree performance
rewrite runnable independently. T8 is mechanical and low-risk; T7 is sequenced after T8 (T8's
file-by-file breakdown would go stale if T7's consolidation landed first). T5's export
script now exists (`tests/matlab_export/`, roadmap v1.49) but is **unverified** — it has
never been run against real MATLAB (none available in the session that wrote it) — so
don't treat T5 as done; #278 stays open until someone actually runs it and confirms its
output. If you're about to start any of these and haven't read §6.0's actual reasoning,
read it before picking an order.

F-phase items are long-term beyond that dependency too — several are explicitly "audit first,
no implementation until the audit resolves what's actually there" (F3 especially) — don't
start implementing an F-item whose roadmap entry says "not started" without doing that audit
first and reporting back, even if the fix seems obvious.

## Deferred — do not act on these without being asked

- F9's "new algorithm absent from training" edge case — explicitly deferred, ship the common
  case first.
- R3 (CLI ideas) — not scoped, no active work item.
- Any MATLAB-side change — that's a different repo (`andremun/InstanceSpace`), tracked in its
  own issue batches, not this one.

## `build_explore_adapter.py` is deleted — do not recreate it

S3 deleted this file (and its test) in full — `grep`-confirmed zero remaining `.py` references
anywhere in the repo. `explore()` now calls native scikit-learn objects directly, with no
flattening step. Q1 (originally: fix its missing polynomial-kernel branch) was retired for this
reason, not fixed. If you ever find yourself wanting to add a "flatten a trained model into an
artifact" step back in — for a new classifier type, for cross-platform loading, or anything
else — stop: cross-platform loading of externally-produced (e.g. MATLAB) models was evaluated
and closed as impractical, and native pickling (see roadmap F7) already handles every
scikit-learn classifier type F1's registry would add, with no flattening required. Check the
roadmap's S1 `[DECISION]` block before re-litigating either point; both are recorded, reasoned
decisions, not open questions.

## Before considering any bug fix complete

A fix that makes the reported symptom go away is not automatically a correct or complete fix.
Run every fix through this before moving on — don't skip it because the change looks small:

- **Root cause, not alias.** If the fix works by aliasing one value to another, adding a
  special-case branch, or a null/bounds check around something that shouldn't have been
  null/out-of-bounds — ask directly: would fixing *why* that value was wrong eliminate the
  need for this patch entirely? Real example from this exact repo's history: a stale `idx`
  field causing a shape-mismatch crash was fixed by aliasing `idx = selvars` everywhere. The
  values became correct. But it left two fields permanently identical with no distinct
  purpose, because nobody asked why the redundant field existed in the first place. Don't
  repeat that shape of fix.
- **Redundancy check.** After the fix, are there now two or more variables/fields/outputs
  that are always identical or overlapping in purpose? That's a concrete signal the fix
  addressed the symptom, not the structure.
- **Consumer-completeness.** Grep the *whole repo* for every consumer of the changed value,
  not just the one that happened to crash.
- **Fresh-reader test.** Would someone with no memory of this bug's history immediately ask
  "why are there two of these?" reading the result? If yes, fix that in the code (ideally by
  removing the redundancy), not just in your own head.
- **Verified, not just plausible.** See the rule below — don't let "this should work" stand
  in for actually checking it.

## Never leave verification as a promise

A commit message saying "verification pending" or "will confirm in a follow-up" is not
allowed to be the final state of a session — this has happened before on this exact repo
(two consecutive commits, neither ever followed up) and cost real time to untangle later.
Either verify before committing — run it in the background and wait if it's slow, don't skip
it because it's inconvenient — or, if it genuinely can't be verified this session, open a
GitHub issue stating exactly what's unverified and reference it in the commit. An open issue
is tracked; commit-message prose is not.

## Testing

`poetry run pytest`. `poe test`'s sequence includes `test_pytest` (T4, fixed — see roadmap
v1.27) and runs with `--cov=instancespace --cov-report=term-missing` against a 75% coverage
gate (T1, roadmap v1.30/v1.34). If `poe test` doesn't actually run pytest when you read this,
that's a regression in this repo's own tooling, not the expected state — don't assume it's
still broken without checking `pyproject.toml`'s `[tool.poe.tasks]` first.

## Conventions

ruff + mypy (strict) + black, already configured. PolyForm Noncommercial 1.0.0 licence — match
it in any new file headers (F6). Commit messages: conventional-commit style (`fix:`, `feat:`,
`chore:`), matching what's already used in this repo's history.

## Writing documentation and other prose

Use the `ste-writing` skill (`.claude/skills/ste-writing/SKILL.md`) for prose you write in this
repo — docs, READMEs, PR descriptions, error messages, release notes, and comments. It does not
apply to code or commit messages. Use STE-flavored mode for docs/READMEs/PR text; use strict
mode for procedures and error messages.
