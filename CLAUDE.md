# CLAUDE.md — pyInstanceSpace

Persistent context for Claude Code sessions working on this repo. Read this before touching
anything. Full detail lives in `docs/` (see below) — this file is a map, not the content.

## What this is

Python port of the MATLAB Instance Space Analysis (ISA) toolkit (`andremun/InstanceSpace`,
currently v0.9.0). **Not** a 1:1 port and not required to become one — the stage architecture
(`preprocessing → prelim → sifted → pilot → pythia → cloister → trace`, an `InstanceSpace`
class, `build()`/`explore()`/`explore_iter()`) is independently engineered. Forked from MATLAB
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

**Status: S1 and S3 have landed** (verified on `v0.9.0/development-branch-S`, see roadmap
v1.16) — `explore()` already calls native scikit-learn objects, and `build_explore_adapter.py`
is gone. F1/F7/F8 are unblocked on that front; nothing about S1/S3 remains to redo. **S2**
(replace DAG auto-resolution with an explicit stage order) is still not started, and its own
roadmap entry sequences it before both T6 and Q8 — but per the roadmap's own F1/F7/F8 pathways,
none of them actually depend on S2, only on S1. Don't stop F-phase work to do S2 on this file's
authority alone; if a specific F-item's pathway says otherwise, that overrides this summary.

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

`poetry run pytest`. `poe test` should include pytest (Q11/T4 in the roadmap) — if it still
doesn't when you read this, that's an open bug in this repo's own tooling, not a signal to skip
running pytest directly.

## Conventions

ruff + mypy (strict) + black, already configured. PolyForm Noncommercial 1.0.0 licence — match
it in any new file headers (F6). Commit messages: conventional-commit style (`fix:`, `feat:`,
`chore:`), matching what's already used in this repo's history.
