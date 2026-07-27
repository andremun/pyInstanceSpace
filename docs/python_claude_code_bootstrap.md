# pyInstanceSpace — Claude Code Bootstrap Runbook

**Audience:** a Claude Code session with write access to `andremun/pyInstanceSpace` (post-merge)
or `aoxiangx/pyInstanceSpace` (pre-merge). Not for a chat session without write access — this
runbook assumes commands actually execute.

**Do not run this speculatively.** Confirm write access first (`git push --dry-run` against a
throwaway branch, or check `gh auth status` + repo permissions) before starting step 1. If
access isn't confirmed, stop and report that instead of proceeding.

Run the steps below **in order** — later steps assume earlier ones landed.

---

## Step 0 — Confirm the fork-merge prerequisite (roadmap Phase -1)

Check whether `andremun/pyInstanceSpace`'s `main` already contains
`aoxiangx/pyInstanceSpace`'s `explore/build-explore-adapter` branch. If not, that merge is the
actual first task — see the roadmap's "Phase -1" section for the verified-clean merge details
and the two ways to do it (direct merge vs. cross-fork PR). Do not proceed past this step until
it's resolved; everything below assumes a single merged codebase.

## Step 1 — Commit the planning documents

Add, at repo root / in `docs/`:
- `docs/pyIS_docs_quality_roadmap.md`
- `docs/python_implementation_pathways.md`
- `CLAUDE.md` (repo root, not `docs/` — mirrors where `andremun/InstanceSpace`'s own
  `CLAUDE.md` lives)

These currently exist only as chat-delivered files, not repo content — committing them is what
makes them a shared source of truth instead of something that has to be re-pasted into every
session. Commit message: `docs: add roadmap, implementation pathways, and CLAUDE.md`.

## Step 2 — Create milestones

Two, matching the near-term/long-term split already in the roadmap:
- **`fork-merge-and-quality`** — Phase -1 (if not already done), P0–P5, Phase Q, Phase T.
  Everything additive or low-risk-behavior-changing-with-a-clear-verification-step.
- **`functionality-parity`** — F1–F9, R1–R3. Long-term, several explicitly "audit first."

```bash
gh api repos/{owner}/{repo}/milestones -f title="fork-merge-and-quality" -f state="open"
gh api repos/{owner}/{repo}/milestones -f title="functionality-parity" -f state="open"
```

## Step 3 — Create labels

```bash
gh label create "phase:P" --color "0E8A16" --description "Documentation & quality"
gh label create "phase:Q" --color "1D76DB" --description "MATLAB-derived quality ideas"
gh label create "phase:F" --color "5319E7" --description "Functionality parity (long-term)"
gh label create "phase:R" --color "FBCA04" --description "Third-party implementation ideas"
gh label create "phase:T" --color "D93F0B" --description "Testing infrastructure"
gh label create "compat:additive" --color "0E8A16" --description "No existing caller's output changes"
gh label create "compat:behavior-changing" --color "B60205" --description "Existing output may change — verification required before merge"
gh label create "compat:unknown" --color "BFD4F2" --description "Needs its own audit before it can be tagged"
```

## Step 4 — Create one parent issue per phase, with sub-issues underneath

For each phase (P, Q, F, R, T), create a parent issue summarizing the phase (pull the
phase-level prose straight from the roadmap document — don't rewrite it), then create one
sub-issue per numbered item (P0, P1, ... Q1, Q2, ... etc.), each with:
- Title: the item's heading from the roadmap (e.g. "Q9 — Centralise RNG seeding via a
  `general.seed` option")
- Body: the item's full text from the roadmap, plus its implementation pathway from
  `python_implementation_pathways.md`
- Labels: the relevant `phase:X` and `compat:Y` labels
- Milestone: `fork-merge-and-quality` or `functionality-parity` per the item's phase

```bash
# Parent issue example (Phase Q)
gh issue create --title "Phase Q — quality ideas transferred from MATLAB" \
  --body-file <(extract phase Q's intro prose from the roadmap) \
  --label "phase:Q" --milestone "fork-merge-and-quality"
# note the returned issue number, e.g. 42

# Sub-issue example (Q9), linked under the parent
gh issue create --title "Q9 — Centralise RNG seeding via a general.seed option" \
  --body-file <(extract Q9's section from the roadmap + pathways doc) \
  --label "phase:Q,compat:behavior-changing" --milestone "fork-merge-and-quality"
# note the returned issue number, e.g. 43
```

**Link sub-issues to their parent.** GitHub's sub-issues feature is the right mechanism (shows
a progress bar on the parent automatically) — check current syntax before running, since this
API has changed since this runbook was written:
```bash
gh api graphql -f query='
  mutation { addSubIssue(input: {issueId: "PARENT_NODE_ID", subIssueId: "CHILD_NODE_ID"}) {
    subIssue { number }
  }}'
```
If that mutation doesn't work as shown (check `gh api graphql --help` and GitHub's current
REST/GraphQL docs for sub-issues), fall back to a plain task-list in the parent issue's body —
`- [ ] #43` — which GitHub auto-links and checks off when the referenced issue closes. Simpler,
well-established, no API-version risk.

## Step 5 — Do not create issues for deferred items

Skip: F9's "new algorithm absent from training" edge case, R3 (not scoped). Note them in the
relevant parent issue's body as "deferred, not tracked as a sub-issue yet" instead of creating
issues nobody's meant to pick up.

## Step 6 — Report back

Summarize what was created (milestone URLs, parent issue numbers, sub-issue count per phase)
rather than assuming success from command exit codes alone — spot-check at least one created
issue renders correctly with its labels and milestone attached.
