# Test data audit and remediation proposal

**Status:** Audit complete. Remediation not yet started — this document proposes a plan
and lists the decisions it needs before any file moves or deletes.
**Scope:** every file under `tests/` in this repository, not only
`tests/matlab_reference/`. Companion to `docs/pyIS_docs_quality_roadmap.md` §8.3 and
`tests/matlab_export/README.md`.

## 1. Why this audit

A question about the MATLAB export script's directory layout exposed a larger problem.
Test data lives in several places under `tests/`, in several incompatible layouts, and
some of it has no reader at all. Nobody had checked, file by file, which fixture feeds
which test, or which files are dead weight. This document is that check.

## 2. Method

For every `tests/*.py` file, this audit extracted each literal file path the code opens.
It then compared that list against every file that actually exists under `tests/`. A
file counts as used when a test opens it directly, or when a test opens its parent
directory as a base path. Every path claim below comes from a direct `grep` across the
whole repository, not from a single test file in isolation. Where a first pass looked
like an orphan, a second, repository-wide search confirmed it before this document
calls it dead.

## 3. Findings: dead data

Three tiers, by how confident the evidence is.

### 3.1 Exact duplicates (byte-identical to data a test already reads)

| Path | Size | Duplicate of | Confirmed reader of the duplicate |
|---|---|---|---|
| `tests/fileidx/` | 22 files | `tests/test_data/prelim/fileidx/` | `test_build_prelim_filter.py` |
| `tests/fractional/` | 21 files | `tests/test_data/prelim/fractional/` | `test_build_prelim.py`, `test_build_prelim_filter.py` |
| `tests/split/` | 21 files | `tests/test_data/prelim/split/` | `test_build_prelim_filter.py` |

`diff -rq` between each pair returns no differences. No test anywhere opens the
top-level copy. These three directories are pure leftover copies, most likely left
behind when the data moved under `test_data/` during an earlier reorganization.

### 3.2 Orphaned and stale (no reader, and the content itself is out of date)

`tests/Prelim_out/` has no reader anywhere in the repository. Its file
`model-data-x.csv` differs in content from the file of the same name at
`tests/test_data/prelim/output/model-data-x.csv`, which `test_build_prelim.py` does
read. `tests/Prelim_out/` is also missing most of the per-feature parameter files
(`model-prelim-hibound.csv`, `model-prelim-lambdaX.csv`, and eight more) that the
active fixture set has.

This is the riskiest kind of dead file. A reader could open `tests/Prelim_out/` by
hand, assume it holds current PRELIM output, and draw a wrong conclusion from stale
numbers. The likely origin: a snapshot taken before a PRELIM fix landed, never deleted
after `tests/test_data/prelim/output/` replaced it.

### 3.3 Orphaned, no active duplicate found

| Path | Size | Note |
|---|---|---|
| `tests/process_data/` | 8 files | No reader anywhere. |
| `tests/test_integration/` | 6 files | No reader anywhere. Not the same thing as `tests/test_data/preprocessing/`, which `test_build_integration.py` actually reads. |
| `tests/test_data/cloister/pythia/` | 36 files | Two full input+output cases (`fitlibsvm_gaussian`, `fitlibsvm_polynomial`). No reader in any `.py` file. The only repository-wide hit for `fitlibsvm` is a one-line, unrelated mention in the roadmap's prose. |
| `tests/test_data/prelim/input/filter/`, `tests/test_data/prelim/output/filter/` | 11 files | Superseded by `tests/test_data/filter/`, which `test_build_filter.py` reads instead. |
| `tests/test_data/prelim/input/model-data-x.csv` | 1 file | A same-purpose file, `model-data-x-input.csv`, is the one `test_build_prelim.py` actually reads. |
| `tests/test_data/prelim/input/options.json` | 1 file | No reader found. |
| `tests/test_data/preprocessing/X.mat`, `Y.mat` | 2 files | The CSV siblings `X.csv`/`Y.csv` are the ones in use. |

### 3.4 Partially orphaned — a possible test-coverage gap, not just dead data

`tests/test_data/prelim/run/output/output_Xraw.csv`, `output_Yraw.csv`, and
`output_instlabels.csv` sit next to files that `test_build_prepro_n_prelim.py` reads
(`output_X.csv`, `output_Y.csv`, `output_P.csv`, and others), but that test never opens
these three. Either the fixture generator wrote three extra files nobody asked for, or
the test is missing an assertion it was meant to have. This needs a look at
`test_build_prepro_n_prelim.py` itself before deciding which.

## 4. Findings: data that is not dead, but is not what its location implies

### 4.1 Two incompatible layouts for MATLAB-comparison data

`tests/matlab_reference/` groups by pipeline phase first: `input/`,
`training_artifacts/<stage>/`, `explore_outputs/step<N>_<name>.csv`. `tests/test_data/
<stage>/` groups by stage first: `<stage>/input/`, `<stage>/output/`, with no separate
concept of build time versus explore time at all.

The practical effect: build-time (training) fixtures live under `test_data/<stage>/`.
Explore-time (test-set inference) fixtures live only under `matlab_reference/`, and
only for one option configuration (MATLAB's plain defaults). PRELIM, SIFTED, PILOT,
CLOISTER, and PYTHIA each have build-time coverage under `test_data/` but no
stage-specific explore-time fixture of their own. A reader who wants to know "does this
directory hold build data or explore data" has to already know which of the two
top-level conventions they are standing in. Nothing in the data says so.

### 4.2 Data that legitimately is not MATLAB-derived

Three groups exist for reasons that have nothing to do with numerical parity against
MATLAB. Their location under `tests/test_data/` next to the MATLAB-comparison fixtures
obscures that difference.

- **`tests/test_data/demo/`** — a multi-dataset collection (BBO, JSS, KP, MFP, and more)
  plus 20 `options_*.json` files. `integration_demo.py` and `liveDemoIS.ipynb` use it as
  worked examples. No pytest file reads it. It is not test data at all in the sense the
  rest of this document uses that term.
- **`tests/test_data/load_file/`** — hand-built JSON and CSV, including deliberately
  broken files (`illegal.json`, `options_invalid.json`). This checks Python's own
  options-loading and validation code. MATLAB has no equivalent concept to compare
  against, so this data is correctly Python-only, and should stay that way.
- **`tests/test_data/serialisers/actual_output/`** — not a fixture. It is a test's own
  write target, regenerated on every run of `test_serialisers.py`. A `.gitignore` file
  sits inside each of its subdirectories, but git still tracks `output.zip` inside it —
  the same file that kept showing up as "modified" in this session's `git status`
  output after running the test suite. `expected_output/` (the golden comparison
  target) is the fixture that actually belongs in version control here.

## 5. Category framework

Every file under `tests/` that holds data falls into one of five categories. Naming the
category is the fix for the "hard to know what is consumed by what" problem — once each
directory declares its category, the reader does not have to guess.

| Category | Origin | Purpose | Example |
|---|---|---|---|
| MATLAB build-path parity | MATLAB, training run | Check one stage's Python output against MATLAB, same inputs | `test_data/pythia/output/BO_gaussian/gaussian.csv` |
| MATLAB explore-path parity | MATLAB, `explore()` run | Check test-set inference against MATLAB | `matlab_reference/explore_outputs/step4_pythia_predictions.csv` |
| Python-only synthetic | Hand-built or generated in Python | Exercise a code path no real MATLAB dataset reaches (degenerate input, invalid options, edge cases) | `test_data/load_file/illegal.json` |
| Example / demo | Real-world, not a golden-value comparison | Show a working usage pattern, not verify numeric correctness | `test_data/demo/metadata/metadata_BBO.csv` |
| Test-run scratch output | Written by the test itself, not read by anything | Debugging artifact only, should not need version control | `test_data/serialisers/actual_output/` |

Answering the question in your message directly: MATLAB-parity data (the first two
rows) should keep coming from MATLAB, and only from MATLAB — hand-editing a
MATLAB-comparison fixture to make a test pass would defeat the point of the comparison.
Python-only synthetic data (row three) should keep coming from Python, since it tests
Python-side logic that has no MATLAB counterpart to derive from. The problem this audit
found is not that both kinds of origin exist. It is that nothing in the current
directory layout tells you which kind you are looking at.

## 6. Remediation proposal

Ordered by risk, cheapest and safest first. Each step names its compatibility tag per
this repository's convention.

### Step 1 — Delete confirmed-dead data
**[Additive]** (removes files nothing reads; no behavior change for any passing test).

Delete, in one commit, with the reasoning from §3 in the commit message:
`tests/fileidx/`, `tests/fractional/`, `tests/split/`, `tests/Prelim_out/`,
`tests/process_data/`, `tests/test_integration/`, `tests/test_data/cloister/pythia/`,
`tests/test_data/prelim/input/filter/`, `tests/test_data/prelim/output/filter/`,
`tests/test_data/prelim/input/model-data-x.csv`, `tests/test_data/prelim/input/
options.json`, `tests/test_data/preprocessing/X.mat`, `tests/test_data/preprocessing/
Y.mat`.

Run the full suite before and after. A pass count that does not change is the
verification this step needs — CLAUDE.md's own rule against treating "should be safe"
as a substitute for a checked result applies here too.

### Step 2 — Resolve the partial-orphan question (§3.4)
**[Additive] if the fix is a new assertion, [Behavior-changing] if the fix changes what
the fixture generator writes.**

Read `test_build_prepro_n_prelim.py` to decide whether `output_Xraw.csv`,
`output_Yraw.csv`, and `output_instlabels.csv` are dead files or a missing assertion.
Fix whichever it is. Do this before Step 1's cleanup in the same directory, so the
decision is made deliberately rather than swept up in a bulk delete.

### Step 3 — Fix the serialisers scratch-output leak (§4.1, third bullet)
**[Additive]** — tooling only.

Check why `.gitignore` inside `serialisers/actual_output/` does not stop `output.zip`
from being tracked. Either the pattern is wrong, or the file was committed once before
the `.gitignore` existed and now needs `git rm --cached`. Confirm `expected_output/` (not
`actual_output/`) is the one under version control as a real fixture.

### Step 4 — Rename or relocate the non-parity categories (§4.2)
**[Additive]** — path changes to non-MATLAB-comparison data only, with every reader
updated in the same commit.

Move `tests/test_data/demo/` to something outside `test_data/` entirely — for example
`examples/data/`, next to `integration_demo.py` — so its name stops implying it is a
test fixture. Leave `tests/test_data/load_file/` where it is, since "test data for
loading tests" is an accurate name for exactly what it is. No numeric fixture moves in
this step, only the demo directory and its two readers' path strings.

### Step 5 — Decide the target layout for MATLAB-parity data (open decision, see §7)
**[Behavior-changing] if it moves any fixture a currently-passing test reads** — needs
the full `tests/matlab_reference`-style verification pass before and after, same as any
other change to existing test data.

This is the big one, and it is a decision, not a mechanical step. §7 lays out the
options. Do not start moving `test_data/<stage>/` fixtures until one option is chosen.

## 7. Open decision: one layout, or two on purpose?

Two real options, not a right-versus-wrong choice.

**Option A — one unified layout, by pipeline phase then stage.** Adopt
`tests/matlab_export/pyis_export_reference_data.m`'s own output shape
(`training_artifacts/<stage>/<variant>/`, `explore_outputs/<variant>/`) as the
target for every MATLAB-comparison fixture, including the ones now under
`test_data/<stage>/`. Every stage gains explore-path coverage it does not have today,
not only PYTHIA and TRACE. One convention, one place to look. Cost: every
`test_build_*.py`/`test_explore_*.py` file's fixture paths need updating, and this can
only happen for real once the export script is actually run against MATLAB and its
output reviewed — §8's open item.

**Option B — keep two layouts, but document the split.** Leave `test_data/<stage>/`
as build-path-only and `matlab_reference/` as the phase-first layout, and add a short
`tests/test_data/README.md` stating that split explicitly, so a future reader does not
have to reverse-engineer it the way this audit did. Lower cost, but the "no
explore-path fixture for five of six stages" gap in §4.1 stays open.

This document takes no position between the two. Both are legitimate. The choice
affects how much of the new export script's design gets used, so it needs your call
before Step 5 starts.

## 8. What this audit did not do

It did not run the MATLAB export script (`tests/matlab_export/`) against a real MATLAB
checkout — that dependency is already tracked on GitHub issue #278 and is unchanged by
this document. It did not delete or move a single file — Step 1 above is a proposal,
not a completed action. It did not audit `docs/`, `output/`, or any directory outside
`tests/` — a repository-root sweep, if wanted, is separate work.
