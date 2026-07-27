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

After that: **P0 (dependency security) and Phase Q (low-risk quality items) before F-phase
(functionality parity).** F-phase items are long-term, several are explicitly "audit first, no
implementation until the audit resolves what's actually there" (F3 especially) — don't start
implementing an F-item whose roadmap entry says "not started" without doing that audit first
and reporting back, even if the fix seems obvious.

## Deferred — do not act on these without being asked

- F9's "new algorithm absent from training" edge case — explicitly deferred, ship the common
  case first.
- R3 (CLI ideas) — not scoped, no active work item.
- Any MATLAB-side change — that's a different repo (`andremun/InstanceSpace`), tracked in its
  own issue batches, not this one.

## Testing

`poetry run pytest`. `poe test` should include pytest (Q11/T4 in the roadmap) — if it still
doesn't when you read this, that's an open bug in this repo's own tooling, not a signal to skip
running pytest directly.

## Conventions

ruff + mypy (strict) + black, already configured. PolyForm Noncommercial 1.0.0 licence — match
it in any new file headers (F6). Commit messages: conventional-commit style (`fix:`, `feat:`,
`chore:`), matching what's already used in this repo's history.
